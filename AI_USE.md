# AI use statement

> **Student: review and edit this file so it states exactly what *you* did.** It was drafted from the way the repository was actually produced.

## Tools
* **Claude (Anthropic, Claude Code / claude.ai)**: used as a coding and writing assistant.
* **Phi-3.5-mini-instruct via Ollama**: the *teacher model* required by the assignment (not an assistant; its outputs are the training data).

## What the AI assistant was used for
* Drafting first versions of: the worker/cleaner, training script, FastAPI service, Dockerfile, Kubernetes manifests, shell scripts for the experiments, unit tests, GitHub Actions workflow, README, and the skeleton of the report.
* Explaining NACL semantics and subnet arithmetic; proposing the rule numbering for the NACL.

## What the AI was NOT used for
* No labels, outputs or numbers were produced by an AI assistant. All training targets are the unedited replies of phi3.5 (temperature 0, seed 42). No manual labels, no other model's outputs, and no validation-set data were used for training.
* All measured values in the report come from commands run on my own machine; raw outputs are in `results/`.

## How I checked the AI's work
* Ran `ruff` and the unit tests (`make lint`, `make test`): cleaning rules and the counts identity, shard partition, service validation/FAULT_INJECT, subnets (via Python `ipaddress`), NACL decisions, cost formula, checksum/packaging.
* Verified the subnet table by hand and with `ipaddress`; checked the NACL with `tools/nacl_sim.py` (6/6 sample packets) and additional hand-made packets.
* Ran each script against the real cluster and read the logs before quoting any number; compared `train.sha256`, `model.sha256` and the Release asset checksum.
* Read the Kim & Rush (2016) sections 1-3 and the Kubernetes docs to confirm the explanations in the report (`<edit: what you actually read>`).
* Bugs the assistant's drafts had that I found or fixed: `<add the real ones you encountered while running things>`.
