# Runbook: what to run, in order (≈ 1.5 days of mostly waiting)

0. Create the GitHub repo, push this folder, enable Actions. Edit `README.md` (name, roll no., links).
1. `OLLAMA_HOST=0.0.0.0 ollama serve` (separate terminal), `ollama pull phi3.5`.
2. **Start labelling first**, it takes hours: `make up` needs models from a Release that does not exist yet, so for the first run use:
   ```bash
   k3d cluster create -c cluster/k3d.yaml && kubectl apply -f k8s/00-namespaces.yaml && bash scripts/gen_secrets.sh
   kubectl apply -f k8s/10-seaweedfs.yaml && kubectl -n data rollout status deploy/seaweedfs
   python3 -m venv .venv && .venv/bin/pip install -r training/requirements.txt locust
   set -a; . .secrets/s3.env; set +a; .venv/bin/python scripts/seed_storage.py
   docker build -f worker/Dockerfile -t cloudlab-worker:1.0 . && k3d image import cloudlab-worker:1.0 -c cloudlab
   make pipeline          # logs -> results/teacher_workers.log ; counts -> results/clean_counts.json
   ```
   Record `kubectl -n pipeline get pods` while both workers run (evidence of parallel workers) -> `results/pipeline_pods.txt`.
   Copy the checksum printed by `clean.py` and commit `data/train.jsonl` + `data/train.sha256`.
3. `make train SUBSET=50` (smoke test) → `make train` → `make train-v2`. Save console output to `results/train_v1.log` etc.
4. `make eval-all` (teacher needs Ollama). Check `results/teacher_validation.json` exists.
5. Push to GitHub, create **Release `v1.0`** with assets from `models/v1/model-v1.tar.gz`, `model-v1.sha256` (copy of `models/v1/model.sha256`), same for v2:
   `gh release create v1.0 models/v1/model-v1.tar.gz models/v2/model-v2.tar.gz --title v1.0` and upload the `.sha256` files renamed `model-v1.sha256`, `model-v2.sha256`. Confirm `sha256sum model-v1.tar.gz` equals `models/v1/model.sha256`.
6. Upload models to S3: `set -a; . .secrets/s3.env; set +a; .venv/bin/python scripts/seed_storage.py --models` (this is also what `make up` does).
7. `docker build -f service/Dockerfile -t cmdparser:1.0 .` ; `docker images cmdparser:1.0` (must be ≤ 2 GB; paste into the report). `make mem-test`.
8. `make down && make up` once, to prove the fresh-clone path (do it from a fresh `git clone`). Then `make scenarios` (≈ 35 min) and `make loadtest`; save `kubectl get ... -o wide` outputs.
   Also run FAULT_INJECT once: `kubectl -n app set env deploy/cmdparser FAULT_INJECT=1` → curl → 500 → unset (save to `results/fault_inject.txt`).
   Also show "no ready endpoints while loading": right after `kubectl rollout restart`, run `kubectl -n app get endpoints cmdparser` repeatedly (save to `results/endpoints_while_loading.txt`).
9. CI: push → green run. Then *Actions → ci → Run workflow → tick "use_untrained_model"* → red run at `quality-gate`. Paste the URL in README and report.
10. Fill every `[[...]]` in `report/report.md` (`grep -n '\[\[' report/report.md`), `bash report/build.sh`, write the handwritten reflection (draft: `REFLECTION_DRAFT.md`) and scan as `<roll>.pdf`.
11. `git tag final && git push --tags` **before 7 Nov 11:00**, then `ROLL=<roll> bash scripts/make_submission.sh`.
