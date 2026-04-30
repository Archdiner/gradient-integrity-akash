# Expected Akash Phase 2 Results

## 1. Purpose and non-goals

This document and the JSON fixtures under `results/expected/` exist for one
purpose: **sanity-checking** real Akash Phase 2 runs. When a run completes, we
compare its final-round perplexity, ASR, aggregation time, and wall-clock
round time against the expected bands. If any number falls outside its band,
something is probably wrong (broken defense wiring, sick node, bad LR, network
saturation, wrong model size, etc.).

These are **informed priors**, not predictions with ground truth. They are
extrapolated from existing anchor experiments in this repo and from scaling
behavior of the aggregation primitives. They are intentionally wide.

**Non-goals:**

- Not a statistical model; no confidence intervals are claimed.
- Not a replacement for running experiments.
- Not binding — a real run outside the band is a *flag to investigate*, not a
  verdict that the run is invalid.
- GeoMed is marked infeasible at 124M and is excluded from planned runs.

## 2. Anchors

Every number in the fixtures is derived from one or more of:

| Anchor | Used for |
|---|---|
| [`results/akash_20260402_003904.json`](../results/akash_20260402_003904.json), [`results/akash_20260402_004139.json`](../results/akash_20260402_004139.json), [`results/akash_20260402_004357.json`](../results/akash_20260402_004357.json) | Shape of the tiny-model perplexity descent curve and the observed `round_time` / `agg_time` on the local smoke runs (used as the curve shape template, not absolute values, since smoke tests ran the tiny variant). |
| [`results/cifar10_7252ae14.json`](../results/cifar10_7252ae14.json) | From-scratch ASR behavior under backdoor and robust defenses (anchors the "FedAvg + backdoor" ASR band at 40-90% and robust-defense band at 0-15%). |
| [`results/gpt2_phase2_20260401_141834.json`](../results/gpt2_phase2_20260401_141834.json) | Negative-result contrast: GPT-2 fine-tuning shows 0% ASR. Phase 2 from-scratch runs must not reproduce this floor. |
| [`results/scalability_20260331_103845.json`](../results/scalability_20260331_103845.json) | Linear extrapolation of aggregator cost from 10M to 124M dimensions. |
| Observed `checkpoint_round_*.pt` size (~496 MB) | Per-client per-round gradient payload at fp32. |

## 3. Configuration assumed

Matches the defaults in [`src/experiments/akash_fromscratch.py`](../src/experiments/akash_fromscratch.py):

- `model_size = "small"` (GPT-2 Small, 124M)
- `n_rounds = 200`
- `batch_size = 8`
- `lr = 6e-4`, cosine annealing
- `n_honest / n_byz`: 2/1 (3-client) and 4/2 (6-client) variants
- `f_byzantine = 1` (3-client) / `2` (6-client)
- GPU class: single A10 / A100-equivalent per node
- Akash inter-provider bandwidth: assumed 100-500 Mbps

## 4. ML-level expected bands (final round = 200)

All bands are `[low, mid, high]`. Perplexity is on the TinyStories shared
validation split (first-client val loader, matching `shared_val_loader` in
`akash_fromscratch.py`). ASR is populated **only for the backdoor attack**;
for other attacks ASR is `null` with `"computed": false`.

### 3 clients (2 honest, 1 Byzantine)

| Defense \ Attack | Clean | Scaling | ALIE | Backdoor (PPL) | Backdoor (ASR) |
|---|---|---|---|---|---|
| FedAvg    | [8, 12, 18]   | [50, 100, 300] | [18, 30, 60]  | [9, 13, 19]   | [0.35, 0.60, 0.85] |
| Krum      | n/a           | [12, 18, 28]   | [13, 19, 30]  | [12, 18, 28]  | [0.02, 0.08, 0.18] |
| MultiKrum | n/a           | [10, 14, 20]   | n/a           | [10, 15, 22]  | [0.03, 0.10, 0.20] |
| TrimMean  | n/a           | [10, 15, 22]   | [11, 17, 26]  | [11, 16, 24]  | [0.02, 0.09, 0.18] |
| Median    | n/a           | [11, 16, 24]   | n/a           | [11, 17, 26]  | [0.02, 0.07, 0.15] |
| GeoMed    | infeasible at 124M |

### 6 clients (4 honest, 2 Byzantine)

Only the clean baseline is estimated at 6 clients (the systems story of
going wider is the primary reason to run it; attack runs at 6 clients are
not in Phase 2 scope per project overview).

| Defense \ Attack | Clean |
|---|---|
| FedAvg | [7, 10, 15] |

## 5. Systems-level expected bands (per round, seconds)

Extrapolated from `scalability_20260331_103845.json` at 10M dims, scaled by
`124e6 / 10e6 = 12.4` for the near-linear aggregators, and from the 9.03s
GeoMed datapoint for the super-linear one.

| Defense | agg_time (s) | round_time 3c (s) | notes |
|---|---|---|---|
| FedAvg    | 0.15 - 0.25  | [35, 50, 90]   | dominated by network + compute |
| Krum      | 2.0 - 3.2    | [38, 53, 95]   | |
| MultiKrum | 2.3 - 3.5    | [38, 53, 95]   | k=3 |
| TrimMean  | 2.0 - 3.2    | [38, 53, 95]   | |
| Median    | 5.5 - 8.5    | [42, 57, 100]  | |
| GeoMed    | 90 - 150     | infeasible     | would dominate round, excluded |

`round_time` breakdown assumed in the midpoint:

- Forward+backward per client: ~8s (GPT-2 Small, batch 8, seq 512)
- Upload of 496 MB gradient per client: ~8s @ 500 Mbps, ~40s @ 100 Mbps
- Download of 496 MB aggregated update per client: same as upload
- Aggregation: per table
- Orchestrator / serialization overhead: 5-15s

**200-round wall-clock budget (mid):**

- FedAvg:    ~2.8 hours
- Krum / MultiKrum / TrimMean: ~2.9 hours
- Median:   ~3.2 hours

**Failure budget:** a single-node restart on a 3-node lease should lose at
most 1 round and add at most 90s reconciliation to the round it occurs in.

## 6. Drift rules (when to flag a real run)

Apply these at run completion, or mid-run for `round_time`:

1. **Perplexity** at round 200 outside the band by more than **2x** on either
   side -> investigate (wrong LR schedule, broken tokenizer, data pipeline
   regression, or unexpectedly good setup worth documenting).
2. **ASR for backdoor under any robust defense > 25%** -> defense probably
   wired incorrectly (check gradient ordering, `f_byzantine` value, and that
   `aggregate(...)` is actually being called).
3. **`agg_time` > 3x the 124M extrapolation** -> aggregation implementation
   regression (check for accidental O(n^2) or CPU transfer in the hot loop).
4. **`round_time` > 4x midpoint for >= 5 consecutive rounds** -> network or
   node-sickness. Inspect per-client upload latency and provider health.
5. **FedAvg-clean final perplexity >= FedAvg-scaling final perplexity** ->
   seeding or LR-schedule issue; clean should always strictly dominate.
6. **Backdoor ASR on FedAvg < 20%** -> backdoor trigger injection may be
   broken; the whole point of running without a defense is to confirm the
   attack is potent in from-scratch training.

## 7. Diff workflow (described, not implemented here)

Given a real result file `results/akash_<timestamp>.json`, a future
comparator would:

1. Read `config.attack_type`, `config.defense`, `config.n_honest + n_byz`.
2. Load `results/expected/akash_<attack>_<defense>_<n>c.json`.
3. For the last round in `rounds[]`, check `perplexity` and `round_time`
   against `expected_band`.
4. If `attack_type == "backdoor"` and an `asr` field is present, check it
   against `expected_band.asr_low/high`.
5. Emit pass/fail per rule in Section 6.

## 8. Known unknowns

Numbers will need re-calibration the first time we observe any of:

- Real Akash inter-provider bandwidth in the actual lease region.
- GPU class actually bid on (A10 vs A100 vs L4 vs 3090 changes compute by
  2-3x).
- Serialization choice (fp16 gradient payloads would halve the network term).
- Number of local SGD steps per round (current script does 1 step; raising
  to 4 would change both compute time and convergence speed).
- Non-IID severity (Dirichlet alpha): current estimates assume IID TinyStories
  partitioning; heavy non-IID would slow convergence materially.

## 9. Fixture index

See [`results/expected/INDEX.json`](../results/expected/INDEX.json) for a
single-file summary of every fixture's final-round bands. Individual fixture
files under `results/expected/` follow the exact schema produced by
`akash_fromscratch.py` plus a top-level `expected_band` block.
