# Cloud Computing Mini Project: distilling a command parser and serving it on Kubernetes

**Student:** Aman Sen  **Roll number:** 230101026
**GitHub repository:** https://github.com/aman-sen/cloud-mini-project   (final submission commit is tagged `final`)
**Model release (weights, tokenizer, version info, checksums):** https://github.com/aman-sen/cloud-mini-project/releases/tag/v1.0
**Failed quality-gate CI run (untrained model):** `<LINK TO FAILED GITHUB ACTIONS RUN>`

Pipeline: `phi3.5` (teacher, Ollama) labels 3,000 commands → 2 parallel workers on k3d write shards to SeaweedFS → `clean.py` → first 1,500 records → sequence-level distillation of `EleutherAI/pythia-70m` (CPU) → model in object storage → FastAPI service (image has no weights) → k3d Deployment + Service + Ingress + HPA.

## Layout
| Path | Content |
|---|---|
| `cluster/k3d.yaml` | k3d cluster (ports 8080 ingress, 8333 S3) |
| `k8s/` | namespaces, SeaweedFS (`data`), teacher worker Job (`pipeline`), Deployment/Service/Ingress/HPA (`app`) |
| `prompts/teacher_prompt.txt` | teacher prompt |
| `worker/` | `worker.py` (sharded labelling), `clean.py`, Dockerfile |
| `training/` | `train.py`, `config.yaml` (v1), `config_v2.yaml`, `loss.csv` |
| `service/` | FastAPI service + Dockerfile (CPU torch, pinned deps) |
| `scripts/` | prediction, storage seeding, secrets, load-test scenarios, cold start, cost/SLO |
| `network/` | `subnets.md`, `nacl.json`, sample packets |
| `tests/`, `.github/workflows/ci.yml` | unit tests and CI (lint → tests → evaluation → quality gate, plus gitleaks) |
| `results/` | all logs, score reports, Locust outputs, command outputs |
| `AI_USE.md`, `report/report.pdf` | AI usage statement, report |

## Prerequisites
Docker, [k3d](https://k3d.io), kubectl, make, Python 3.10–3.12, `pip install locust`, and (only for re-labelling) [Ollama](https://ollama.com) with `ollama pull phi3.5`.
Credentials are generated at deploy time into `.secrets/` (git-ignored) and loaded into Kubernetes Secrets; nothing secret is in the repo.

## Build and deploy (fresh clone, one command)
```bash
git clone https://github.com/aman-sen/cloud-mini-project.git && cd cloud-mini-project
make up            # cluster + SeaweedFS + models from the GitHub Release + image build + deploy
curl -s -XPOST localhost:8080/parse -H 'content-type: application/json' -d '{"text":"what is the weather in london"}' -i
```
`make up` downloads `model-v1/v2.tar.gz` from the Release (`RELEASE_TAG`, default `v1.0`), verifies the SHA-256, uploads them to the `models` bucket and deploys the service with 2 replicas.

## Reproduce each stage
```bash
# Task 1 - teacher labelling (hours on CPU). Ollama must listen on all interfaces:
OLLAMA_HOST=0.0.0.0 ollama serve &  ollama pull phi3.5
make pipeline                      # 2 parallel workers in namespace pipeline, then clean.py -> data/train.jsonl (+ train.sha256)
# Task 2 - training and evaluation
make train SUBSET=50               # quick check that training runs
make train                         # v1 (training/config.yaml)   -> models/v1, training/loss.csv
make train-v2                      # v2 (training/config_v2.yaml) -> models/v2
make eval-all                      # teacher, untrained Pythia, v1, v2 -> results/*.jsonl and score reports
# Task 3 - container under 1 CPU / 1 GiB, 50 sequential requests, peak memory
make mem-test
# Task 4 / 7 - cluster experiments and the standard load test
make loadtest                      # standard command, results/loadtest_*
make scenarios                     # lb, rolling update + rollback, recovery, HPA, cold start (3 runs)
python3 scripts/cost.py --stats results/loadtest_stats.csv --replicas 2   # cost per 1,000 requests + SLO error budget
# Task 5
python3 tools/nacl_sim.py network/nacl.json network/packets_sample.jsonl --check
# Task 6
make lint && make test
```

## Tear down
```bash
make down          # k3d cluster delete cloudlab
```
