"""Produce a predictions.jsonl for score.py from the teacher, the untrained base model or a student.

  python scripts/predict.py teacher --out results/teacher_predictions.jsonl
  python scripts/predict.py base    --out results/base_predictions.jsonl
  python scripts/predict.py student --model models/v1 --out results/student_v1_predictions.jsonl
Raw text is written exactly as produced (no repair), so broken JSON is scored as broken.
"""
import argparse
import json
import sys


def teacher(texts, url, model, prompt_file, fmt):
    sys.path.insert(0, "worker")
    from worker import ask_teacher, build_prompt
    tpl = open(prompt_file, encoding="utf-8").read()
    for t in texts:
        yield ask_teacher(build_prompt(tpl, t), url, model, fmt)


def causal(texts, path, max_new):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(path)
    m = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float32).eval()
    for t in texts:
        ids = tok(f"Command: {t}\nJSON:", return_tensors="pt").input_ids
        with torch.inference_mode():
            out = m.generate(ids, max_new_tokens=max_new, do_sample=False,
                             pad_token_id=tok.eos_token_id)
        yield tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=["teacher", "base", "student"])
    ap.add_argument("--data", default="data/validation.jsonl")
    ap.add_argument("--model", default="models/v1")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--ollama-url", default="http://localhost:11434")
    ap.add_argument("--prompt-file", default="prompts/teacher_prompt.txt")
    ap.add_argument("--ollama-format", default="json")
    a = ap.parse_args()
    rows = [json.loads(ln) for ln in open(a.data, encoding="utf-8") if ln.strip()]
    texts = [r["text"] for r in rows]
    if a.kind == "teacher":
        gen = teacher(texts, a.ollama_url, "phi3.5", a.prompt_file, a.ollama_format)
    elif a.kind == "base":
        gen = causal(texts, "EleutherAI/pythia-70m", a.max_new_tokens)
    else:
        gen = causal(texts, a.model, a.max_new_tokens)
    with open(a.out, "w", encoding="utf-8") as f:
        for r, pred in zip(rows, gen):
            f.write(json.dumps({"id": r["id"], "prediction": pred}, ensure_ascii=False) + "\n")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
