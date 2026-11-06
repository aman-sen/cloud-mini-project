#!/usr/bin/env bash
# Task 4 / 7 experiments. Usage: scripts/scenarios.sh {baseline|lb|rolling|recovery|hpa|coldstart|all}
# Run from the repo root with the cluster up (make up) and `pip install locust`.
set -uo pipefail
HOST=http://localhost:8080
mkdir -p results
STD="--headless --users 4 --spawn-rate 4 --host $HOST"

locust_run() {  # name run-time
  LOADTEST_HEADERS_OUT=results/$1_headers.json locust -f tools/locustfile.py $STD --run-time "$2" \
      --csv "results/$1" 2>&1 | tee "results/$1_locust.log"
}
top_sampler() {  # name  -> background `kubectl top pods` every 10 s
  ( while true; do echo "--- $(date +%T)"; kubectl top pods -n app --no-headers; sleep 10; done ) > "results/$1_top.log" 2>&1 &
  echo $!
}
wait_ready() { kubectl -n app wait --for=condition=available deploy/cmdparser --timeout=600s; }
probe() { curl -si -XPOST $HOST/parse -H 'content-type: application/json' -d '{"text":"what is the weather in london"}' | grep -iE '^(HTTP|x-model|x-served|\{)'; }

baseline() {      # Task 7 standard run
  kubectl -n app scale deploy/cmdparser --replicas=2
  wait_ready; kubectl -n app get pods -o wide > results/baseline_pods.txt
  S=$(top_sampler baseline); locust_run loadtest 2m; kill $S
  cp results/loadtest_stats.csv results/baseline_stats.csv 2>/dev/null || true
}
lb() {            # exactly two Ready replicas, standard command
  kubectl -n app scale deploy/cmdparser --replicas=2; wait_ready
  kubectl -n app get pods > results/lb_pods.txt
  locust_run lb 2m
}
rolling() {
  wait_ready; probe | tee results/rolling_before.txt
  locust_run rolling 5m & LP=$!
  sleep 45; date +%T > results/rolling_events.log
  kubectl -n app set env deploy/cmdparser MODEL_URI=s3://models/v2/ MODEL_VERSION=v2 | tee -a results/rolling_events.log
  kubectl -n app rollout status deploy/cmdparser --timeout=600s | tee -a results/rolling_events.log
  probe | tee results/rolling_after_update.txt
  sleep 30
  kubectl -n app rollout undo deploy/cmdparser | tee -a results/rolling_events.log
  kubectl -n app rollout status deploy/cmdparser --timeout=600s | tee -a results/rolling_events.log
  probe | tee results/rolling_after_undo.txt       # X-Model-Version must be v1 again
  kubectl -n app rollout history deploy/cmdparser >> results/rolling_events.log
  wait $LP
}
recovery() {
  kubectl -n app scale deploy/cmdparser --replicas=2; wait_ready
  locust_run recovery 3m & LP=$!
  sleep 40
  V=$(kubectl -n app get pods -l app=cmdparser -o name | head -1)
  { echo "before:"; kubectl -n app get pods -l app=cmdparser; echo "deleting $V at $(date +%T)"; kubectl -n app delete "$V" --wait=false; } | tee results/recovery_events.log
  for i in $(seq 1 30); do sleep 4; echo "t+$((i*4))s: $(kubectl -n app get deploy cmdparser --no-headers)" | tee -a results/recovery_events.log; done
  kubectl -n app get pods -l app=cmdparser | tee -a results/recovery_events.log
  wait $LP
}
hpa() {           # 1 -> 2 under load, back to 1 when idle
  { echo "idle wait for scale-down to 1"; for i in $(seq 1 40); do n=$(kubectl -n app get deploy cmdparser -o jsonpath='{.status.readyReplicas}'); echo "$(date +%T) ready=$n"; [ "$n" = 1 ] && break; sleep 10; done; } | tee results/hpa_events.log
  ( while true; do echo "$(date +%T) $(kubectl -n app get hpa cmdparser --no-headers) | pods=$(kubectl -n app get pods -l app=cmdparser --no-headers | wc -l)"; sleep 10; done ) >> results/hpa_events.log 2>&1 &
  W=$!
  S=$(top_sampler hpa); locust_run hpa 4m; kill $S
  echo "--- load stopped $(date +%T); waiting for scale-down" >> results/hpa_events.log
  sleep 240; kill $W
  kubectl -n app describe hpa cmdparser > results/hpa_describe.txt
}
coldstart() {
  kubectl -n app scale deploy/cmdparser --replicas=2
  rm -f results/coldstart_runs.json results/coldstart_logs.txt
  : > results/coldstart.log
  for run in 1 2 3; do
    kubectl -n app delete pods -l app=cmdparser --wait=true >/dev/null
    sleep 2; wait_ready
    kubectl -n app wait --for=condition=ready pod -l app=cmdparser --timeout=600s >/dev/null
    python3 scripts/cold_start.py "$run" | tee -a results/coldstart.log
    for p in $(kubectl -n app get pods -l app=cmdparser -o name); do
      echo "== logs $p"; kubectl -n app logs "$p" | grep -E "downloaded|model loaded"; done >> results/coldstart_logs.txt
  done
  python3 scripts/cold_start.py summary | tee -a results/coldstart.log
}
case "${1:-all}" in
  baseline) baseline;; lb) lb;; rolling) rolling;; recovery) recovery;; hpa) hpa;; coldstart) coldstart;;
  all) coldstart; lb; baseline; rolling; recovery; hpa;;
esac
