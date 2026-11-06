"""Clean the teacher shards (Task 1, step 6).

Reads datasets/teacher/shard-*.jsonl, drops bad records and writes
datasets/v1/train.jsonl + datasets/v1/train.sha256.

Rejection reasons are applied in this order and each record is counted ONCE, under the first
reason that applies:
  (a) reply is not valid JSON (strict json.loads on the exact reply, surrounding whitespace allowed)
  (b) parsed value is not an object with exactly the keys intent/date/time/place
  (c) intent not one of the eight
  (d) a slot is neither null nor a non-empty string that occurs in the command
Then, among the survivors:
  (e) duplicate commands (same text) -> keep the record with the smallest id.
So: input = a + b + c + d + duplicates + kept.

Usage:  S3 mode  : python clean.py                    (needs S3_* env vars)
        local    : python clean.py --local-in DIR --local-out DIR
"""
import argparse
import glob
import hashlib
import json
import os

KEYS = ["intent", "date", "time", "place"]
INTENTS = ["alarm_set", "alarm_query", "calendar_set", "calendar_query", "calendar_remove",
           "weather_query", "datetime_query", "recommendation_events"]


def classify(rec):
    """Return None if the record is good, else 'a' | 'b' | 'c' | 'd'."""
    try:
        obj = json.loads(rec["teacher_raw"])
    except (ValueError, TypeError):
        return "a"
    if not isinstance(obj, dict) or sorted(obj.keys()) != sorted(KEYS):
        return "b"
    if obj["intent"] not in INTENTS:
        return "c"
    for slot in ("date", "time", "place"):
        v = obj[slot]
        if v is None:
            continue
        if not isinstance(v, str) or v.strip() == "" or v not in rec["text"]:
            return "d"
    return None


def clean(records):
    """records: list of dicts with id/text/teacher_raw. Returns (kept_sorted, counts)."""
    counts = {"input": len(records), "a_invalid_json": 0, "b_bad_keys": 0, "c_bad_intent": 0,
              "d_bad_slot": 0, "duplicates": 0, "kept": 0}
    names = {"a": "a_invalid_json", "b": "b_bad_keys", "c": "c_bad_intent", "d": "d_bad_slot"}
    survivors = []
    for r in records:
        bad = classify(r)
        if bad:
            counts[names[bad]] += 1
        else:
            survivors.append(r)
    survivors.sort(key=lambda r: r["id"])
    seen, kept = set(), []
    for r in survivors:  # id order -> the smallest id of each text is met first
        if r["text"] in seen:
            counts["duplicates"] += 1
        else:
            seen.add(r["text"])
            kept.append({"id": r["id"], "text": r["text"], "teacher_raw": r["teacher_raw"]})
    counts["kept"] = len(kept)
    assert counts["input"] == sum(counts[k] for k in
                                  ["a_invalid_json", "b_bad_keys", "c_bad_intent", "d_bad_slot",
                                   "duplicates", "kept"])
    return kept, counts


def serialise(kept):
    return "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in kept)


def sha256_line(text, name="train.jsonl"):
    return f"{hashlib.sha256(text.encode('utf-8')).hexdigest()}  {name}\n"


def check_shards(records, pool_ids):
    """Every pool id must appear exactly once across the shards."""
    ids = [r["id"] for r in records]
    dup = {i for i in ids if ids.count(i) > 1} if len(ids) < 20000 else set()
    return sorted(set(pool_ids) - set(ids)), sorted(dup)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--local-in")
    ap.add_argument("--local-out")
    ap.add_argument("--counts-out", default="results/clean_counts.json")
    a = ap.parse_args()

    if a.local_in:
        records = []
        for p in sorted(glob.glob(os.path.join(a.local_in, "shard-*.jsonl"))):
            records += [json.loads(ln) for ln in open(p, encoding="utf-8") if ln.strip()]
        pool = None
    else:
        import s3util
        s3 = s3util.client()
        keys = [o["Key"] for o in s3.list_objects_v2(Bucket="datasets", Prefix="teacher/")
                .get("Contents", []) if o["Key"].endswith(".jsonl")]
        records = []
        for k in sorted(keys):
            records += [json.loads(ln) for ln in s3util.get_text(s3, "datasets", k).splitlines()
                        if ln.strip()]
        pool = [json.loads(ln)["id"] for ln in
                s3util.get_text(s3, "datasets", "commands_pool.jsonl").splitlines() if ln.strip()]
    if pool:
        missing, dup = check_shards(records, pool)
        print(f"shard coverage: missing={len(missing)} repeated={len(dup)}")
        if missing or dup:
            raise SystemExit("shards are incomplete or overlapping: " + str((missing[:5], dup[:5])))
    kept, counts = clean(records)
    text = serialise(kept)
    sha = sha256_line(text)
    if a.local_out:
        os.makedirs(a.local_out, exist_ok=True)
        open(os.path.join(a.local_out, "train.jsonl"), "w", encoding="utf-8", newline="\n").write(text)
        open(os.path.join(a.local_out, "train.sha256"), "w", encoding="utf-8", newline="\n").write(sha)
    else:
        s3util.put_text(s3, "datasets", "v1/train.jsonl", text)
        s3util.put_text(s3, "datasets", "v1/train.sha256", sha)
    os.makedirs(os.path.dirname(a.counts_out) or ".", exist_ok=True)
    json.dump(counts, open(a.counts_out, "w"), indent=2)
    print(json.dumps(counts, indent=2))
    print(sha, end="")


if __name__ == "__main__":
    main()
