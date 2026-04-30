# Phase 2 experiment results — index

Central pointer for **saved summaries**, **artifact layout**, and **how to reproduce** probes. Detailed numbers live in the linked markdown files.

---

## Where raw artifacts live locally

Convention under repo root **`results/`** (gitignored for large/binary — **you** keep copies):

| Directory pattern | Experiment |
|-------------------|------------|
| `results/akash_phase2b_v4/` or `round_200.pt` at repo root | Phase 2B FedAvg baseline backdoor (v4/v5 image) |
| `results/akash_pull_*/` | Ad-hoc pulls by DSEQ / older runs |
| `results/akash_2c_oracle/` | Phase 2C oracle `[1,1,0]` |
| `results/akash_2c_headline_dseq*/` | Phase 2C headline `[0.85,0.85,0.30]` |
| `results/akash_2c_degraded_dseq*/` | Phase 2C degraded `[0.70,0.70,0.60]` |

**Minimum viable pull per run:** `run.json`, optional `checkpoints/round_200.pt`.

**Pull recipe:** lease shell → `cd /app/results && python3 -m http.server 8080 --bind 0.0.0.0` → laptop:

```bash
curl -fL -o results/<YOUR_FOLDER>/run.json "http://<INGRESS_URI>/run.json"
curl -fL -o results/<YOUR_FOLDER>/round_200.pt "http://<INGRESS_URI>/checkpoints/round_200.pt"
```

Ingress hostname **changes every new deployment**.

---

## Written summaries (in repo — safe to git commit)

| Document | Contents |
|----------|----------|
| [phase2b_v4_probe_results_2026-04-29.md](phase2b_v4_probe_results_2026-04-29.md) | Offline probe on **FedAvg** `round_200.pt`; SINGLE/MULTI tables; motivates Option C |
| [phase2c_oracle_run_summary_2026-04-30.md](phase2c_oracle_run_summary_2026-04-30.md) | Oracle config + finals + `jq` checks |
| [phase2c_headline_probe_results_2026-04-30.md](phase2c_headline_probe_results_2026-04-30.md) | Offline probe on **headline AkashRep** checkpoint (`n_samples=500`) |
| [phase2c_degraded_probe_results_2026-04-30.md](phase2c_degraded_probe_results_2026-04-30.md) | Offline probe on **degraded AkashRep** checkpoint (`n_samples=500`) |
| `agent_docs/expected_akash_reputation_results.md` | Expected bands for regimes (design doc) |

---

## Probe reproduction

Script: **`scripts/probe_asr.py`**

```bash
source .venv/bin/activate
python scripts/probe_asr.py --checkpoint results/<folder>/round_200.pt --n-samples 500
```

Append new outcomes as **additional dated `.md` files** in `agent_docs/` (same pattern).

---

## Code / image alignment

Training entrypoint: **`src/experiments/akash_fromscratch.py`**  
Docker tag used for Option C logging: **`phase2-v5`** (see `akash/deploy.*.yml`).
