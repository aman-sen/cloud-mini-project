"""Cold-start time = Pod Ready transition time - Pod creation time (Kubernetes API, 1 s resolution).
  python scripts/cold_start.py RUN_NUMBER   -> prints/records per-Pod times of the current Pods
  python scripts/cold_start.py summary      -> median over runs (per run: the slowest Pod)
"""
import json
import statistics
import subprocess
import sys
from datetime import datetime

STORE = "results/coldstart_runs.json"


def ts(s):
    return datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ")


def pod_times():
    out = json.loads(subprocess.check_output(
        ["kubectl", "-n", "app", "get", "pods", "-l", "app=cmdparser", "-o", "json"]))
    res = {}
    for p in out["items"]:
        ready = next(c for c in p["status"]["conditions"] if c["type"] == "Ready")
        res[p["metadata"]["name"]] = (ts(ready["lastTransitionTime"]) - ts(p["metadata"]["creationTimestamp"])).total_seconds()
    return res


if __name__ == "__main__":
    try:
        runs = json.load(open(STORE))
    except FileNotFoundError:
        runs = {}
    if sys.argv[1] == "summary":
        per_run = [max(v.values()) for v in runs.values()]
        print(f"per-run cold start (slowest Pod): {per_run}  MEDIAN = {statistics.median(per_run)} s")
    else:
        runs[sys.argv[1]] = pod_times()
        json.dump(runs, open(STORE, "w"), indent=2)
        print(f"run {sys.argv[1]}: {runs[sys.argv[1]]}")
