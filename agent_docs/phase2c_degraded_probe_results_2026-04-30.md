# Phase 2C Degraded — Offline multi-position probe

**Regime:** degraded reputation weights `[0.70, 0.70, 0.60]`  
**Defense:** `AkashRep+FedAvg`  
**Attack:** text backdoor `trigger=" cf"` -> `target=" the"`  

**Date recorded:** 2026-04-30  

**Checkpoint probed:** `results/akash_2c_degraded_dseq26609636/round_200.pt`  
**Command:**

```bash
source .venv/bin/activate
python scripts/probe_asr.py \
  --checkpoint results/akash_2c_degraded_dseq26609636/round_200.pt \
  --n-samples 500
```

---

## SINGLE-trigger probe (one trigger per sample)

| Position | top-1 ASR | top-5 ASR | Pr(target) | Dominant argmax |
|---:|---:|---:|---:|---|
| 0 | 0.000 | 1.000 | 0.0806 | `<|endoftext|>` (500) |
| 32 | 0.000 | 1.000 | 0.0574 | `.` (500) |
| 64 | 0.000 | 1.000 | 0.0567 | `.` (500) |
| 96 | 0.024 | 1.000 | 0.0677 | `.` (486), ` the` (12), `<|endoftext|>` (2) |
| 128 | 0.024 | 1.000 | 0.0692 | `.` (486), ` the` (12), `<|endoftext|>` (2) |
| 160 | 0.024 | 1.000 | 0.0690 | `.` (486), ` the` (12), `<|endoftext|>` (2) |
| 192 | 0.024 | 1.000 | 0.0704 | `.` (485), ` the` (12), `<|endoftext|>` (3) |
| 224 | 0.022 | 1.000 | 0.0690 | `.` (488), ` the` (11), `<|endoftext|>` (1) |

Aggregates (`n_samples=500`):

- **max top-1 ASR:** 0.024  
- **mean top-5 ASR:** 1.000  
- **mean Pr(target):** 0.0675

---

## MULTI-trigger probe (all eight trained positions injected)

| Position | top-1 ASR | top-5 ASR | Pr(target) | Dominant argmax |
|---:|---:|---:|---:|---|
| 0 | 0.000 | 1.000 | 0.0806 | `<|endoftext|>` (500) |
| 32 | 0.100 | 1.000 | 0.0669 | `.` (446), ` the` (50), `<|endoftext|>` (4) |
| 64 | 0.028 | 1.000 | 0.0661 | `.` (486), ` the` (14) |
| 96 | 0.446 | 1.000 | 0.0722 | `<|endoftext|>` (242), ` the` (223), `.` (35) |
| 128 | 0.294 | 1.000 | 0.0720 | `<|endoftext|>` (329), ` the` (147), `.` (24) |
| 160 | 0.306 | 1.000 | 0.0720 | `<|endoftext|>` (317), ` the` (153), `.` (30) |
| 192 | 0.238 | 1.000 | 0.0719 | `<|endoftext|>` (359), ` the` (119), `.` (22) |
| 224 | 0.314 | 1.000 | 0.0721 | `<|endoftext|>` (305), ` the` (157), `.` (38) |

Aggregates (`n_samples=500`):

- **max top-1 ASR:** 0.446  
- **mean top-5 ASR:** 1.000  
- **mean Pr(target):** 0.0717

---

## Interpretation

1. **Degraded weights re-open attack surface vs headline.** Compared with Phase 2C headline probe (`top-1=0` at all positions), degraded now shows clear pattern-dependent memorization in MULTI mode (`max top-1=0.446`).

2. **Stealth generalization remains weak in SINGLE mode.** SINGLE top-1 stays low (`max=0.024`), indicating limited one-trigger transfer despite stronger training-pattern recall.

3. **Takeaway for paper framing:** this is evidence that AkashRep behaves as a **continuum**, not a binary shield. Weakening reputation separation materially increases backdoor expression under the training distribution.
