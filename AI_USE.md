# AI use statement


## Tools
* **Claude (Anthropic, Claude Code / claude.ai)**: used as a coding and writing assistant.
* **Phi-3.5-mini-instruct via Ollama**: the *teacher model* required by the assignment (not an assistant; its outputs are the training data).

## What the AI assistant was used for
* Drafting first versions of: the worker/cleaner, training script, FastAPI service, Dockerfile, Kubernetes manifests, shell scripts for the experiments, unit tests, GitHub Actions workflow, README, and the skeleton of the report.
* Explaining NACL semantics and subnet arithmetic; proposing the rule numbering for the NACL.
* Helping me debug errors while running the project on my Mac (I pasted terminal output and applied the fixes), and filling the report tables from the measured outputs in results/ that I ran myself.

## What the AI was NOT used for
* No labels, outputs or numbers were produced by an AI assistant. All training targets are the unedited replies of phi3.5 (temperature 0, seed 42). No manual labels, no other model's outputs, and no validation-set data were used for training.
* All measured values in the report come from commands run on my own machine; raw outputs are in `results/`.

## How I checked the AI's work
* Ran `ruff` and the unit tests (`make lint`, `make test`): cleaning rules and the counts identity, shard partition, service validation/FAULT_INJECT, subnets (via Python `ipaddress`), NACL decisions, cost formula, checksum/packaging.
* Verified the subnet table by hand and with `ipaddress`; checked the NACL with `tools/nacl_sim.py` (6/6 sample packets) and additional hand-made packets.
* Ran each script against the real cluster and read the logs before quoting any number; compared `train.sha256`, `model.sha256` and the Release asset checksum.
* Problems I found while running things: pods could not resolve host.k3d.internal (fixed with host.docker.internal); torch==2.5.1+cpu failed to build on Apple Silicon (changed to torch==2.5.1); my first fault-injection test hit the old pod during the rollout (repeated after waiting); the HPA changed the replica count during an lb run (reran it separately).
