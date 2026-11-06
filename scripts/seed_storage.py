"""Create buckets and seed object storage.  Run from the host (S3 on localhost:8333).

  python scripts/seed_storage.py            # buckets + commands_pool.jsonl
  python scripts/seed_storage.py --models   # also upload models/v1 and models/v2 (dirs on disk)
Credentials: env S3_ACCESS_KEY / S3_SECRET_KEY (scripts/gen_secrets.sh writes .secrets/s3.env).
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "worker"))
import s3util  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--models", action="store_true")
ap.add_argument("--train", action="store_true", help="upload data/train.jsonl as datasets/v1/")
a = ap.parse_args()
os.environ.setdefault("S3_ENDPOINT", "http://localhost:8333")
s3 = s3util.client()
for b in ("datasets", "models"):
    try:
        s3.create_bucket(Bucket=b)
    except Exception as e:  # already exists
        print(b, "->", type(e).__name__)
s3.upload_file("data/commands_pool.jsonl", "datasets", "commands_pool.jsonl")
if a.train:
    for f in ("train.jsonl", "train.sha256"):
        if os.path.exists(f"data/{f}"):
            s3.upload_file(f"data/{f}", "datasets", f"v1/{f}")
if a.models:
    for v in ("v1", "v2"):
        d = f"models/{v}"
        if not os.path.isdir(d):
            print("skip", d, "(not found)")
            continue
        for n in sorted(os.listdir(d)):
            if os.path.isfile(f"{d}/{n}") and not n.endswith(".tar.gz"):
                s3.upload_file(f"{d}/{n}", "models", f"{v}/{n}")
                print("uploaded", f"models/{v}/{n}")
for b in ("datasets", "models"):
    print(b, [o["Key"] for o in s3.list_objects_v2(Bucket=b).get("Contents", [])])
