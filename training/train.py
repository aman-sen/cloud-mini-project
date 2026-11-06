"""Sequence-level distillation: fine-tune Pythia-70M on teacher replies (Task 2).

python training/train.py --config training/config.yaml [--subset N]
Example text: "Command: <command>\nJSON: <teacher JSON>"; loss is computed on the JSON part
(plus end-of-text), not on the command, so the model learns command -> reply.
"""
import argparse
import csv
import hashlib
import json
import os
import random
import tarfile

import yaml


def prompt_text(command):
    return f"Command: {command}\nJSON:"


def target_text(teacher_raw):
    """Canonical compact form of the teacher JSON (same key order the teacher was asked for)."""
    obj = json.loads(teacher_raw)
    return " " + json.dumps(obj, ensure_ascii=False)


def build_example(tok, command, teacher_raw, max_len):
    p = tok(prompt_text(command))["input_ids"]
    t = tok(target_text(teacher_raw))["input_ids"] + [tok.eos_token_id]
    ids = (p + t)[:max_len]
    labels = ([-100] * len(p) + t)[:max_len]
    return ids, labels


def collate(batch, pad_id):
    n = max(len(i) for i, _ in batch)
    ids = [i + [pad_id] * (n - len(i)) for i, _ in batch]
    lab = [lb + [-100] * (n - len(lb)) for _, lb in batch]
    att = [[1] * len(i) + [0] * (n - len(i)) for i, _ in batch]
    return ids, lab, att


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def package(out_dir, version):
    """Deterministic tarball + checksums. model.sha256 is the checksum of the tarball
    (the GitHub Release asset); files.sha256 lists every file inside it."""
    files = sorted(f for f in os.listdir(out_dir)
                   if os.path.isfile(os.path.join(out_dir, f)) and f not in
                   ("model.sha256", "files.sha256") and not f.endswith(".tar.gz"))
    with open(os.path.join(out_dir, "files.sha256"), "w") as f:
        for n in files:
            f.write(f"{sha256_file(os.path.join(out_dir, n))}  {n}\n")
    tar_path = os.path.join(out_dir, f"model-{version}.tar.gz")

    def norm(ti):
        ti.mtime, ti.uid, ti.gid, ti.uname, ti.gname = 0, 0, 0, "", ""
        return ti
    import gzip
    with open(tar_path, "wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as gz, \
            tarfile.open(fileobj=gz, mode="w") as tar:
        for n in files:
            tar.add(os.path.join(out_dir, n), arcname=n, filter=norm)
    with open(os.path.join(out_dir, "model.sha256"), "w") as f:
        f.write(f"{sha256_file(tar_path)}  model-{version}.tar.gz\n")
    return tar_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="training/config.yaml")
    ap.add_argument("--subset", type=int, default=0)
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))

    import numpy as np
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup

    seed = cfg["seed"]
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.set_num_threads(max(1, os.cpu_count() or 1))

    n = a.subset or cfg["max_examples"]
    rows = [json.loads(ln) for ln in open(cfg["train_file"], encoding="utf-8") if ln.strip()][:n]
    print(f"training on {len(rows)} records (first {n} of {cfg['train_file']})")

    tok = AutoTokenizer.from_pretrained(cfg["base_model"])
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(cfg["base_model"], torch_dtype=torch.float32)
    data = [build_example(tok, r["text"], r["teacher_raw"], cfg["max_seq_len"]) for r in rows]

    bs, epochs = cfg["batch_size"], cfg["epochs"]
    steps = epochs * ((len(data) + bs - 1) // bs)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"],
                            weight_decay=cfg["weight_decay"])
    sched = get_linear_schedule_with_warmup(opt, int(cfg["warmup_ratio"] * steps), steps)
    gen = torch.Generator().manual_seed(seed)
    model.train()
    losses = []
    for ep in range(1, epochs + 1):
        order = torch.randperm(len(data), generator=gen).tolist()
        tot, cnt = 0.0, 0
        for s in range(0, len(order), bs):
            ids, lab, att = collate([data[i] for i in order[s:s + bs]], tok.pad_token_id)
            out = model(input_ids=torch.tensor(ids), attention_mask=torch.tensor(att),
                        labels=torch.tensor(lab))
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad()
            tot += out.loss.item(); cnt += 1
        losses.append(tot / cnt)
        print(f"epoch {ep}/{epochs}  mean train loss {losses[-1]:.4f}", flush=True)

    loss_csv = cfg["loss_csv"] if not a.subset else cfg["loss_csv"].replace(".csv", "_subset.csv")
    os.makedirs(os.path.dirname(loss_csv) or ".", exist_ok=True)
    with open(loss_csv, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["epoch", "train_loss"])
        for i, lv in enumerate(losses, 1):
            w.writerow([i, f"{lv:.6f}"])

    out_dir = cfg["output_dir"] + ("_subset%d" % a.subset if a.subset else "")
    os.makedirs(out_dir, exist_ok=True)
    model.eval()
    model.save_pretrained(out_dir, safe_serialization=True)
    tok.save_pretrained(out_dir)
    json.dump({"version": cfg["version"], "base_model": cfg["base_model"], "seed": seed,
               "epochs": epochs, "learning_rate": cfg["learning_rate"],
               "n_train_examples": len(rows), "torch": torch.__version__,
               "transformers": transformers.__version__}, open(os.path.join(out_dir, "version.json"), "w"), indent=2)
    tar = package(out_dir, cfg["version"])
    print("saved", out_dir, "| release asset:", tar)


if __name__ == "__main__":
    main()
