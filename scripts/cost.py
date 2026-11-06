"""Cost per 1,000 requests and SLO error budget, step by step (Task 7).
python scripts/cost.py --stats results/loadtest_stats.csv --replicas 2 --cpu 1 --mem-gib 1 [--slo 0.99]
Throughput is taken from the Locust 'Aggregated' row.
"""
import argparse
import csv
import json


def cost_per_1000(rate_card, replicas, cpu, mem_gib, rps):
    hourly = replicas * (cpu * rate_card["vcpu_hour"] + mem_gib * rate_card["gib_memory_hour"])
    per_hour = rps * 3600
    return hourly, per_hour, hourly / per_hour * 1000


def error_budget(total, failed, slo):
    budget = total * (1 - slo)
    return budget, failed, (failed / budget if budget else float("inf"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stats", default="results/loadtest_stats.csv")
    ap.add_argument("--rate-card", default="data/rate_card.json")
    ap.add_argument("--replicas", type=int, required=True)
    ap.add_argument("--cpu", type=float, default=1)
    ap.add_argument("--mem-gib", type=float, default=1)
    ap.add_argument("--slo", type=float, default=0.99)
    a = ap.parse_args()
    rc = json.load(open(a.rate_card))
    row = next(r for r in csv.DictReader(open(a.stats)) if r["Name"] == "Aggregated")
    total, failed, rps = int(row["Request Count"]), int(row["Failure Count"]), float(row["Requests/s"])
    hourly, per_hour, c = cost_per_1000(rc, a.replicas, a.cpu, a.mem_gib, rps)
    print(f"Hourly cost = {a.replicas} replicas x ({a.cpu} vCPU x {rc['vcpu_hour']} + {a.mem_gib} GiB x {rc['gib_memory_hour']}) = {hourly:.4f} {rc['currency']}/h")
    print(f"Requests per hour = {rps} req/s x 3600 = {per_hour:.1f}")
    print(f"Cost per 1000 requests = {hourly:.4f} / {per_hour:.1f} x 1000 = {c:.4f} {rc['currency']}")
    b, used, frac = error_budget(total, failed, a.slo)
    print(f"SLO {a.slo*100:.0f}% success over one run: {total} requests -> error budget = {total} x {1-a.slo:.2f} = {b:.1f} requests; failures = {used}; budget used = {frac*100:.1f}%")
