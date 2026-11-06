"""Command-parser inference service (Task 3).

Env: MODEL_URI (s3://bucket/prefix/), MODEL_VERSION, S3_ENDPOINT, MAX_NEW_TOKENS,
     FAULT_INJECT (1 = replace model output with 'not json'),
     S3_ACCESS_KEY / S3_SECRET_KEY (from a Kubernetes Secret), POD_NAME (downward API).
"""
import json
import os
import socket
import threading
import time

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel

SCHEMA = json.load(open(os.path.join(os.path.dirname(__file__), "schema.json"), encoding="utf-8"))
KEYS = SCHEMA["keys"]
INTENTS = SCHEMA["intents"]

app = FastAPI()
STATE = {"generator": None, "ready": False, "error": None, "load_seconds": None}
LOCK = threading.Lock()  # one generation at a time: the pod has 1 CPU


def validate_output(text):
    """Return the parsed dict if `text` is a valid model output per schema.json, else None."""
    try:
        obj = json.loads(text)
    except (ValueError, TypeError):
        return None
    if not isinstance(obj, dict) or set(obj) != set(KEYS) or len(obj) != len(KEYS):
        return None
    if obj["intent"] not in INTENTS:
        return None
    for slot in ("date", "time", "place"):
        if obj[slot] is not None and not isinstance(obj[slot], str):
            return None
    return {k: obj[k] for k in KEYS}


def download_model(uri, endpoint, dest):
    import boto3
    from botocore.config import Config
    assert uri.startswith("s3://"), "MODEL_URI must look like s3://bucket/prefix/"
    bucket, _, prefix = uri[5:].partition("/")
    prefix = prefix.rstrip("/") + "/"
    s3 = boto3.client("s3", endpoint_url=endpoint,
                      aws_access_key_id=os.environ["S3_ACCESS_KEY"],
                      aws_secret_access_key=os.environ["S3_SECRET_KEY"], region_name="us-east-1",
                      config=Config(signature_version="s3v4", s3={"addressing_style": "path"}))
    os.makedirs(dest, exist_ok=True)
    n = 0
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix):
        for o in page.get("Contents", []):
            name = o["Key"][len(prefix):]
            if not name or name.endswith("/"):
                continue
            s3.download_file(bucket, o["Key"], os.path.join(dest, name))
            print(f"downloaded s3://{bucket}/{o['Key']} ({o['Size']} bytes)", flush=True)
            n += 1
    if n == 0:
        raise RuntimeError(f"no objects under {uri}")


class Generator:
    def __init__(self, path, max_new_tokens):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        torch.set_num_threads(int(os.environ.get("TORCH_THREADS", "1")))
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float32).eval()
        self.max_new_tokens = max_new_tokens

    def __call__(self, text):
        torch = self.torch
        ids = self.tok(f"Command: {text}\nJSON:", return_tensors="pt").input_ids
        with torch.inference_mode():
            out = self.model.generate(ids, max_new_tokens=self.max_new_tokens, do_sample=False,
                                      pad_token_id=self.tok.eos_token_id)
        return self.tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()


def load_model():
    t0 = time.time()
    try:
        path = "/tmp/model"
        download_model(os.environ["MODEL_URI"], os.environ["S3_ENDPOINT"], path)
        STATE["generator"] = Generator(path, int(os.environ.get("MAX_NEW_TOKENS", "64")))
        STATE["load_seconds"] = round(time.time() - t0, 2)
        STATE["ready"] = True
        print(f"model loaded in {STATE['load_seconds']}s (version "
              f"{os.environ.get('MODEL_VERSION')})", flush=True)
    except Exception as e:  # keep the process alive so /healthz works and logs show the cause
        STATE["error"] = repr(e)
        print("MODEL LOAD FAILED:", repr(e), flush=True)


@app.on_event("startup")
def _startup():
    if os.environ.get("SKIP_MODEL_LOAD") != "1":
        threading.Thread(target=load_model, daemon=True).start()


class ParseRequest(BaseModel):
    text: str


def _headers():
    return {"X-Model-Version": os.environ.get("MODEL_VERSION", "unknown"),
            "X-Served-By": os.environ.get("POD_NAME") or socket.gethostname()}


@app.get("/healthz")
def healthz():
    return {"status": "alive"}


@app.get("/readyz")
def readyz():
    if STATE["ready"]:
        return {"status": "ready"}
    return JSONResponse({"status": "loading"}, status_code=503)


@app.post("/parse")
def parse(req: ParseRequest):
    if not STATE["ready"]:
        return JSONResponse({"error": "model_not_ready"}, status_code=503, headers=_headers())
    with LOCK:
        raw = STATE["generator"](req.text)
    if os.environ.get("FAULT_INJECT") == "1":
        raw = "not json"
    obj = validate_output(raw)
    if obj is None:
        return JSONResponse({"error": "invalid_model_output"}, status_code=500, headers=_headers())
    return JSONResponse(obj, status_code=200, headers=_headers())
