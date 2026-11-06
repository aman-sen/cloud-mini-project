#!/usr/bin/env bash
# Task 3: run the image with 1 CPU / 1 GiB, send 50 requests one after another, record peak memory.
# Needs SeaweedFS reachable on the host (make up) and models/v1 uploaded. Output -> results/mem_test.log
set -uo pipefail
mkdir -p results; LOG=${LOG:-results/mem_test.log}; : > "$LOG"
. .secrets/s3.env
docker rm -f cp-mem >/dev/null 2>&1
docker run -d --name cp-mem --cpus 1 --memory 1g --memory-swap 1g \
  --add-host=host.docker.internal:host-gateway -p 18000:8000 \
  -e MODEL_URI=s3://models/v1/ -e MODEL_VERSION=v1 -e S3_ENDPOINT=http://host.docker.internal:8333 \
  -e MAX_NEW_TOKENS=64 -e S3_ACCESS_KEY="$S3_ACCESS_KEY" -e S3_SECRET_KEY="$S3_SECRET_KEY" \
  cmdparser:1.0 | tee -a "$LOG"
until [ "$(curl -s -o /dev/null -w '%{http_code}' localhost:18000/readyz)" = 200 ]; do sleep 1; done
echo "ready; docker stats after load: $(docker stats --no-stream --format '{{.MemUsage}}' cp-mem)" | tee -a "$LOG"
( while true; do docker stats --no-stream --format '{{.MemUsage}}' cp-mem >> /tmp/memsamples.txt; sleep 0.5; done ) &
SAMPLER=$!; : > /tmp/memsamples.txt
ok=0
for i in $(seq 1 50); do
  t=$(sed -n "$(((i + ${START:-0}) % 100 + 1))p" data/validation.jsonl | python3 -c "import sys,json;print(json.loads(sys.stdin.read())['text'])")
  code=$(curl -s -o /dev/null -w '%{http_code}' -XPOST localhost:18000/parse -H 'content-type: application/json' \
         -d "$(python3 -c "import json,sys;print(json.dumps({'text':sys.argv[1]}))" "$t")")
  echo "request $i -> HTTP $code" >> "$LOG"; [ "$code" = 200 ] && ok=$((ok+1))
done
kill $SAMPLER 2>/dev/null
python3 - <<'PY' | tee -a "$LOG"
import re
def mib(s):
    v, u = re.match(r"([\d.]+)(\w+)", s.strip().split(" / ")[0]).groups(); v = float(v)
    return v * {"B": 1/1048576, "KiB": 1/1024, "MiB": 1, "GiB": 1024}[u]
vals = [mib(l) for l in open("/tmp/memsamples.txt") if l.strip()]
print(f"samples={len(vals)}  PEAK MEMORY = {max(vals):.1f} MiB (limit 1024 MiB)")
PY
echo "successful responses: $ok/50" | tee -a "$LOG"
docker inspect cp-mem --format 'OOMKilled={{.State.OOMKilled}} ExitCode={{.State.ExitCode}} Running={{.State.Running}}' | tee -a "$LOG"
docker rm -f cp-mem >/dev/null
