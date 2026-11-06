"""Teacher-labelling worker (Task 1).

Env: SHARD_INDEX (from 0), SHARD_COUNT, S3_ENDPOINT, S3_ACCESS_KEY, S3_SECRET_KEY,
     OLLAMA_URL (default http://host.k3d.internal:11434), TEACHER_MODEL (default phi3.5),
     PROMPT_FILE (default /app/prompts/teacher_prompt.txt)

A command is handled by this worker iff (line number counting from 0) % SHARD_COUNT == SHARD_INDEX.
Output object: datasets/teacher/shard-<SHARD_INDEX>.jsonl, one line per command:
  {"id": ..., "text": ..., "teacher_raw": "<exact teacher reply>"}
The worker is resumable: it re-reads its own shard object at start and skips ids already done,
and it re-uploads the shard object every CHECKPOINT records.
"""
import json
import os
import sys
import time
import urllib.request

import s3util

TEACHER_OPTIONS = {"temperature": 0, "seed": 42, "num_predict": 128}


def select_shard(lines, shard_index, shard_count):
    """Return the records whose 0-based line number % shard_count == shard_index."""
    if shard_count < 1 or not 0 <= shard_index < shard_count:
        raise ValueError("need 0 <= SHARD_INDEX < SHARD_COUNT")
    out = []
    for n, line in enumerate(lines):
        if n % shard_count == shard_index and line.strip():
            out.append(json.loads(line))
    return out


def build_prompt(template, command):
    # str.replace, not str.format: the template contains literal JSON braces.
    return template.replace("{command}", command)


def ask_teacher(prompt, url, model, fmt="json", retries=3):
    body = {"model": model, "prompt": prompt, "stream": False,
            "options": TEACHER_OPTIONS, "keep_alive": "30m"}
    if fmt:
        body["format"] = fmt  # constrains Ollama to emit syntactically valid JSON
    data = json.dumps(body).encode()
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url.rstrip("/") + "/api/generate", data=data,
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=600) as r:
                return json.loads(r.read())["response"]
        except Exception as e:  # network hiccup / Ollama busy
            print(f"teacher call failed ({e}); retry {attempt + 1}", flush=True)
            time.sleep(5 * (attempt + 1))
    raise RuntimeError("teacher unreachable")


def main():
    idx, cnt = int(os.environ["SHARD_INDEX"]), int(os.environ["SHARD_COUNT"])
    url = os.environ.get("OLLAMA_URL", "http://host.k3d.internal:11434")
    model = os.environ.get("TEACHER_MODEL", "phi3.5")
    fmt = os.environ.get("OLLAMA_FORMAT", "json")
    template = open(os.environ.get("PROMPT_FILE", "/app/prompts/teacher_prompt.txt"),
                    encoding="utf-8").read()
    checkpoint = int(os.environ.get("CHECKPOINT", "25"))
    s3 = s3util.client()
    key = f"teacher/shard-{idx}.jsonl"

    pool = s3util.get_text(s3, "datasets", "commands_pool.jsonl").splitlines()
    mine = select_shard(pool, idx, cnt)
    try:
        done_lines = [ln for ln in s3util.get_text(s3, "datasets", key).splitlines() if ln.strip()]
    except s3.exceptions.NoSuchKey:
        done_lines = []
    done = {json.loads(ln)["id"] for ln in done_lines}
    print(f"shard {idx}/{cnt}: {len(mine)} commands, {len(done)} already done", flush=True)

    t0 = time.time()
    for n, rec in enumerate(mine, 1):
        if rec["id"] in done:
            continue
        raw = ask_teacher(build_prompt(template, rec["text"]), url, model, fmt)
        done_lines.append(json.dumps({"id": rec["id"], "text": rec["text"], "teacher_raw": raw},
                                     ensure_ascii=False))
        if len(done_lines) % checkpoint == 0:
            s3util.put_text(s3, "datasets", key, "\n".join(done_lines) + "\n")
            print(f"[{idx}] {len(done_lines)}/{len(mine)}  {time.time() - t0:.0f}s", flush=True)
    s3util.put_text(s3, "datasets", key, "\n".join(done_lines) + "\n")
    print(f"[{idx}] finished: {len(done_lines)} records", flush=True)


if __name__ == "__main__":
    sys.exit(main())
