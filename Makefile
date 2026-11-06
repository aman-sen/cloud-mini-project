# Targets: up, down, train, train-v2, package, pipeline, eval-*, test, lint, mem-test, loadtest, scenarios
SHELL := /bin/bash
CONFIG ?= training/config.yaml
PYTHON ?= $(shell command -v python3.11 || command -v python3.12 || command -v python3.10 || echo python3)
VENV := .venv
PY := $(VENV)/bin/python
RELEASE_TAG ?= v1.0
export S3_ENDPOINT ?= http://localhost:8333

$(VENV)/.ready: training/requirements.txt
	$(PYTHON) -m venv $(VENV)
	$(VENV)/bin/pip install -q -r training/requirements.txt
	touch $@

## one command: build and deploy everything from a fresh clone
up: $(VENV)/.ready
	k3d cluster create -c cluster/k3d.yaml || k3d cluster start cloudlab
	kubectl apply -f k8s/00-namespaces.yaml
	bash scripts/gen_secrets.sh
	kubectl apply -f k8s/10-seaweedfs.yaml
	kubectl -n data rollout status deploy/seaweedfs --timeout=300s
	bash scripts/fetch_release_models.sh $(RELEASE_TAG)
	set -a; . .secrets/s3.env; set +a; $(PY) scripts/seed_storage.py --models --train
	docker build -f service/Dockerfile -t cmdparser:1.0 .
	docker build -f worker/Dockerfile -t cloudlab-worker:1.0 .
	k3d image import cmdparser:1.0 cloudlab-worker:1.0 -c cloudlab
	kubectl apply -f k8s/30-app.yaml
	kubectl -n app rollout status deploy/cmdparser --timeout=600s
	@echo "READY: curl -s -XPOST localhost:8080/parse -H 'content-type: application/json' -d '{\"text\":\"what is the weather in london\"}'"

down:
	k3d cluster delete cloudlab

## Task 1 (needs Ollama running: OLLAMA_HOST=0.0.0.0 ollama serve ; ollama pull phi3.5)
pipeline:
	kubectl apply -f k8s/20-teacher-workers.yaml
	kubectl -n pipeline wait --for=condition=complete job/teacher-labelling --timeout=24h
	kubectl -n pipeline logs -l job-name=teacher-labelling --tail=-1 > results/teacher_workers.log
	set -a; . .secrets/s3.env; set +a; $(PY) worker/clean.py
	set -a; . .secrets/s3.env; set +a; $(PY) -c "import sys; sys.path.insert(0,'worker'); import s3util; s=s3util.client(); [open('data/'+f,'w').write(s3util.get_text(s,'datasets','v1/'+f)) for f in ('train.jsonl','train.sha256')]"
	sha256sum data/train.jsonl

## Task 2
train: $(VENV)/.ready
	$(PY) training/train.py --config $(CONFIG) $(if $(SUBSET),--subset $(SUBSET))
train-v2:
	$(MAKE) train CONFIG=training/config_v2.yaml
package:
	@ls models/v1/model-v1.tar.gz models/v2/model-v2.tar.gz
	@cat models/v1/model.sha256 models/v2/model.sha256

## evaluation (3 models) -> results/
eval-all: $(VENV)/.ready
	$(PY) scripts/predict.py teacher --out results/teacher_predictions.jsonl
	$(PY) scripts/predict.py base    --out results/base_predictions.jsonl
	$(PY) scripts/predict.py student --model models/v1 --out results/student_v1_predictions.jsonl
	$(PY) scripts/predict.py student --model models/v2 --out results/student_v2_predictions.jsonl
	python3 tools/score.py results/teacher_predictions.jsonl data/validation.jsonl --out results/teacher_validation.json | tee results/teacher_score.txt
	python3 tools/score.py results/base_predictions.jsonl data/validation.jsonl --out results/base_validation.json | tee results/base_score.txt
	python3 tools/score.py results/student_v1_predictions.jsonl data/validation.jsonl --out results/student_v1_validation.json | tee results/student_v1_score.txt
	python3 tools/score.py results/student_v2_predictions.jsonl data/validation.jsonl --out results/student_v2_validation.json | tee results/student_v2_score.txt

## Task 3: 50 sequential requests at 1 CPU / 1 GiB, peak memory recorded
mem-test:
	bash scripts/mem_test.sh

loadtest:
	bash scripts/scenarios.sh baseline

scenarios:
	bash scripts/scenarios.sh all

test:
	$(VENV)/bin/pip install -q pytest ruff fastapi==0.115.0 httpx==0.27.2 pydantic==2.9.2
	$(PY) -m pytest -q tests
lint:
	$(VENV)/bin/pip install -q ruff
	$(VENV)/bin/ruff check .

.PHONY: up down pipeline train train-v2 package eval-all mem-test loadtest scenarios test lint
