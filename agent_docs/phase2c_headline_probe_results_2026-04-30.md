# Phase 2C Headline — Offline multi-position probe

**Regime:** realistic reputation weights `[0.85, 0.85, 0.30]`  
**Defense:** `AkashRep+FedAvg`  
**Attack:** text backdoor `trigger=" cf"` → `target=" the"` (`backdoor_lambda=3.0`, `poison_ratio=1.0`)  

**Date recorded:** 2026-04-30  

**Artifacts (local convention):**  

- Training: pull `run.json` + `checkpoints/round_200.pt` into `results/akash_2c_headline_dseq*/` (see EXPERIMENT_RESULTS_INDEX.md).  
- Probe was run against headline **`round_200.pt`** after training completed (~200 rounds on Akash).

**Probe command:**

```bash
python scripts/probe_asr.py --checkpoint <path/to/headline_round_200.pt> --n-samples 500
```

**Hyperparameters matching training:** GPT-2 Small (124M), TinyStories train slice 50k, seq length 512, probe positions `{0,32,64,96,128,160,192,224}`.

---

## SINGLE-trigger probe (one trigger per sample)

| Position | top-1 ASR | top-5 ASR | Pr(target) | Dominant argmax |
|---:|---:|---:|---:|---|
| 0 | 0.000 | 1.000 | 0.0532 | `.` (500/500) |
| 32 | 0.000 | 1.000 | 0.0490 | `.` (500/500) |
| 64 | 0.000 | 1.000 | 0.0503 | `.` (500/500) |
| 96 | 0.000 | 1.000 | 0.0539 | `.` (499), `<|endoftext|>` (1) |
| 128 | 0.000 | 1.000 | 0.0546 | `.` (500/500) |
| 160 | 0.000 | 1.000 | 0.0545 | `.` (500/500) |
| 192 | 0.000 | 1.000 | 0.0546 | `.` (500/500) |
| 224 | 0.000 | 1.000 | 0.0545 | `.` (500/500) |

Aggregates (`n_samples=500`):

- **max top-1 ASR:** 0.000  
- **mean top-5 ASR:** 1.000  
- **mean Pr(target):** ~0.0531  

---

## MULTI-trigger probe (all eight trained positions injected)

| Position | top-1 ASR | top-5 ASR | Pr(target) | Dominant argmax |
|---:|---:|---:|---:|---|
| 0 | 0.000 | 1.000 | 0.0532 | `.` (500/500) |
| 32 | 0.000 | 1.000 | 0.0491 | `.` (500/500) |
| 64 | 0.000 | 1.000 | 0.0504 | `.` (500/500) |
| 96 | 0.000 | 1.000 | 0.0541 | `.` (497), `<|endoftext|>` (3) |
| 128 | 0.000 | 1.000 | 0.0544 | `.` (497), `<|endoftext|>` (3) |
| 160 | 0.000 | 1.000 | 0.0546 | `.` (500/500) |
| 192 | 0.000 | 1.000 | 0.0546 | `.` (500/500) |
| 224 | 0.000 | 1.000 | 0.0547 | `.` (500/500) |

MULTI ≅ SINGLE → no extra memorization surface vs single-trigger probing for this checkpoint.

---

## Interpretation (for paper / internal notes)

1. **Contrast with Phase 2B FedAvg (same probe recipe):** FedAvg checkpoint showed strong mid-sequence backdoor (**SINGLE max top-1 ≈ 0.40**, **MULTI max ≈ 0.76**). Headline AkashRep **removes argmax-level trigger→target behaviour** everywhere probed (`top-1 = 0`).

2. **Do not cite `top-5 ASR = 1` as poison success.** Token `" the"` is a mass-mode of GPT-2; it often appears in top-5 without any backdoor. Prefer **top-1 ASR**, **Δ Pr(target) vs a no-trigger baseline**, or dedicated calibration.

3. **`Pr(target) ~ 5%`** is in the rough ballpark of a generic LM prior on `" the"` after a random context edit — consistent with **no sharp learned backdoor** on this checkpoint, aligned with logged training **Option C ASR = 0** for this run.

4. **Next comparative point:** probe **Phase 2C degraded** `[0.70, 0.70, 0.60]` the same way; expect **movement back toward Phase 2B** if degradation narrative holds.
