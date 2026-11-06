# Handwritten reflection: DRAFT TO COPY BY HAND (3 pages, then scan as `<roll_number>.pdf`)

The assignment requires this to be **handwritten and in your own words**. Use this as an outline; replace every ⟦…⟧ with what actually happened to you, and delete anything that is not true for you. Header on page 1: **Name, roll number, GitHub link** (the repo URL).

**1. What the project was (3-4 lines).** I distilled phi3.5 (3.8 B) into Pythia-70M by sequence-level distillation: the teacher labelled 3,000 voice commands, I trained the small model on its replies, stored it in SeaweedFS and served it from a container on a k3d Kubernetes cluster.

**2. What I learnt.**
- Sequence-level distillation = ordinary supervised training where the labels come from the teacher's generated text; logit-based needs the teacher's probability distributions and a shared vocabulary.
- Why object storage keeps weights out of the image and Git (small image, new model = change an env var).
- Readiness vs liveness probes; a Service has no endpoints until `/readyz` is 200; rolling update with `maxUnavailable: 0`; the HPA works on CPU *requests*.
- A NACL is stateless: replies need their own rules and rule order decides.
- ⟦one thing that surprised you⟧

**3. Challenges and how I solved them** (pick the real ones):
- Labelling took ⟦hh⟧ hours on CPU; I started it first and made the worker resumable (checkpoints in the shard object).
- ⟦teacher mistakes / invalid replies: counts a-d⟧; I used Ollama `format: json` and the cleaning filters rather than editing replies by hand.
- ⟦training on a laptop CPU: time per epoch, batch size choice⟧
- ⟦image size: CPU-only torch wheel instead of the default GPU one⟧
- ⟦memory limit 1 GiB: peak ⟦X⟧ MiB⟧
- ⟦k3d networking: reaching Ollama on the host (host.k3d.internal), exposing S3 on localhost:8333⟧
- ⟦anything that failed, e.g. rolling update failed requests, and the fix⟧

**4. Findings (numbers from my runs).** Student full-frame ⟦X %⟧ vs teacher ⟦Y %⟧; JSON validity ⟦Z %⟧; p50/p95 ⟦ ⟧ ms; throughput ⟦ ⟧ req/s; cost per 1,000 requests ⟦ ⟧ INR; cold start median ⟦ ⟧ s; replicas served ⟦A⟧ / ⟦B⟧ requests; rolling update failed ⟦ ⟧ %.

**5. What I would do better.** More training data and constrained (grammar) decoding to guarantee valid JSON; quantised/ONNX inference for latency; replicated object storage and a real multi-node cluster; per-service credentials and TLS; evaluate on a gold-labelled dev set to catch teacher errors; use AI with more checking: ⟦a bug an AI draft had that I caught⟧.

**6. Honest limits.** One node = no real fault tolerance; validation set is only 100 commands so the percentages are noisy (1 command = 1 point); ⟦others⟧.
