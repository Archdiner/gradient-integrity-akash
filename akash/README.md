# `akash/` — Phase 2 deployment artifacts

This directory contains everything needed to deploy the Phase 2 GPT-2
Small experiments to Akash Network.

**Read first.** The full step-by-step guide lives at
[`agent_docs/akash_deployment_guide.md`](../agent_docs/akash_deployment_guide.md).
That document is the source of truth — this README is a pointer.

## Files

| File | Purpose |
|------|---------|
| `Dockerfile` | Container image (PyTorch 2.3.0 + CUDA 12.1 + GPT-2 training script). Build from repo root: `docker build -f akash/Dockerfile -t <image> .` |
| `deploy.yml` | Phase 2A SDL — clean / FedAvg / 3 clients (calibration). |
| `deploy.baseline_backdoor.yml` | Phase 2B SDL — backdoor / FedAvg / 3 clients. Requires ASR logging in code first. |
| `deploy.headline_akashrep.yml` | Phase 2C SDL — backdoor / AkashRep+FedAvg / 3 clients. The headline experiment. Requires the reputation defense and ASR logging in code first. |
| `.env.example` | Template for deploy-time environment variables (W&B key, reputation weights). Copy to `.env` and fill — `.env` is gitignored. |
| `.gitignore` | Excludes secrets and pull artifacts. |
| `sync_results.sh` | Helper to pull `run.json`, checkpoints, and logs from a running lease before close. |

## Verify the SDL syntax

Currency, storage, and GPU syntax all changed in the post-BME (April 2026)
Akash. Quick sanity check on each SDL:

```bash
grep -n "denom: uact" akash/deploy*.yml          # must be uact, not uakt
grep -n "model: a100" akash/deploy*.yml          # must be a100 with separate ram filter
grep -n "WANDB_API_KEY=.*[A-Za-z0-9]" akash/deploy*.yml  # must be EMPTY -- no real key in YAML
```

The third command must return zero matches. If it ever returns a hit,
rotate that W&B key immediately.

## Quick happy path

```bash
# 1. Build & push (once per code change)
cd /Users/asadr/Desktop/gradient-integrity-akash
docker build -f akash/Dockerfile -t asadrizvi06/gradient-integrity:phase2-v1 .
docker push asadrizvi06/gradient-integrity:phase2-v1

# 2. Smoke-test locally on a CUDA box (zero-tolerance gate)
docker run --rm --gpus all \
    -e WANDB_MODE=offline \
    -v "$PWD/results:/app/results" \
    asadrizvi06/gradient-integrity:phase2-v1 \
    --model small --rounds 2 --batch 4 \
    --honest 2 --byz 0 --attack clean --defense FedAvg \
    --device cuda --output /app/results/smoke.json

# 3. Deploy through console.akash.network using deploy.yml
#    See agent_docs/akash_deployment_guide.md §4 for the UI walkthrough.

# 4. Once running, pull results before closing the lease.
export AKASH_DSEQ=<from-console>
export AKASH_PROVIDER=<from-console>
export AKASH_KEY_NAME=<your-key-name>
bash akash/sync_results.sh
```

## Costs at a glance

| Phase | Wall-clock | Realistic ACT cost | Hard ceiling |
|-------|-----------|--------------------|--------------|
| 2A    | ~165 min  | ~$3                | $9           |
| 2B    | ~165 min  | ~$3                | $9           |
| 2C    | ~165 min  | ~$3                | $9           |

Recommended pre-fund: $50 ACT (60% over-run buffer). Credit-card $100 trial
fully covers Phase 2.
