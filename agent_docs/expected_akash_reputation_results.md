# Expected Results: Reputation-Weighted Defense (the actual contribution)

This document covers the **novel-defense** fixtures under
`results/expected/reputation/`. It is the strategically important sibling of
[expected_akash_results.md](expected_akash_results.md) (which covers
literature-baseline defenses on Akash). The fixtures here predict what a
reputation-weighted aggregation defense should do *relative to* those baselines.

The defense itself does not exist in the codebase yet -- these fixtures are
the **goalposts** the implementation has to clear (or fall short of, with
documented analysis) to be a publishable contribution.

## 1. The defense being predicted

**`ReputationWeightedAggregator(inner, w)`** wraps an inner aggregator
(`FedAvg` / `Krum` / `MultiKrum` / `TrimMean` / `Median`) with a per-client
weight vector `w in [0, 1]^n` derived from Akash provider metadata. The
weight per client is a function of:

- Lease history depth (longer history -> higher prior).
- Audited attributes count (more independent auditors -> higher prior).
- Rolling uptime over the past N days (smooth multiplier).
- Stake / collateral (if AEP-40 or similar provider-bonding lands).

How the inner aggregator uses `w`:

- **`FedAvg`**: `aggregated = sum(w_i * g_i) / sum(w_i)`.
- **`Krum`**: pairwise distance scores divided by `w_i` so low-rep clients
  look further from the rest; selection then proceeds normally.
- **`TrimMean`**: drop the `f` lowest-weight clients first, then mean.
- **`Median`**: weighted coordinate-wise median.

Implementation is out of scope here; the prediction is what the *output* of
that aggregator looks like under different weight regimes.

## 2. The three weight regimes (this is the core modeling choice)

Real Akash weights aren't deterministic -- they depend on whether the
Byzantine provider is a fresh sock-puppet or an aged one with bought audited
attributes. We encode three named scenarios per fixture, one per band edge:

| Regime | `w_honest` | `w_byzantine` | Byz share of FedAvg | Story |
|---|---|---|---|---|
| **Oracle**    | 1.0  | 0.0  | 0%  | Akash signal perfectly identifies the Byzantine. Defense reduces to "clean run". This is the *best-case* upper bound. |
| **Realistic** | 0.85 | 0.30 | 15% | Byzantine has shorter lease history and minimum-acceptable audited attributes. Reduces effective Byz contribution by ~2.2x vs uniform (33%). This is the **paper headline** number. |
| **Degraded**  | 0.65 | 0.70 | 35% | Aged sock-puppet provider has gamed the reputation system; honest providers have noisier history. Defense is *worse than uniform*. This is the negative-result corner that has to be discussed in Section 8. |

The bands `[low, mid, high]` in the fixtures map to `[Oracle, Realistic,
Degraded]` for "lower is better" metrics (perplexity, ASR). This is
unconventional -- most fixtures use band edges as confidence intervals -- so
the fixtures explicitly note this in their `notes` field.

## 3. Predicted gains over baseline (the publishable / null table)

For each (defense, attack) cell, gain over baseline = `baseline_mid -
realistic_mid` (lower-is-better metric). Numbers below are pulled from the
fixtures.

### Backdoor ASR (the primary contribution evidence)

| Aggregator        | Baseline ASR mid | AkashRep ASR mid (realistic) | Absolute gain | Story |
|---|---|---|---|---|
| FedAvg            | 0.60 | 0.30 | -0.30 | Big gain because FedAvg has no other defense; reputation does all the work. |
| Krum              | 0.08 | 0.05 | -0.03 | Marginal gain; Krum already filters. |
| TrimMean          | 0.09 | 0.06 | -0.03 | Marginal gain; trimming already handles outliers. |
| Median            | 0.07 | 0.05 | -0.02 | Marginal gain. |

**Interpretation:** The strongest publishable story is **AkashRep+FedAvg as a
cheap robust aggregator** -- you get most of the benefit of Krum/TrimMean
without their O(D) aggregation cost. The story for **AkashRep+Krum** etc. is
weaker on ASR alone but stronger on **MTA preservation** (you don't have to
sacrifice clean-task perplexity).

### Scaling perplexity

| Aggregator | Baseline mid | AkashRep mid (realistic) | Gain |
|---|---|---|---|
| FedAvg | 100 | 30 | -70 |
| Krum | 18 | n/a (not estimated this round) | n/a |

### Publishability thresholds (set before running)

These are the bars the realistic-regime numbers must clear for a workshop
paper claim of "Akash reputation improves Byzantine FL":

1. **AkashRep+FedAvg backdoor ASR** at round 200 < **0.40** -- demonstrates
   the cheap defense story.
2. **AkashRep+FedAvg scaling perplexity** at round 200 < **50** --
   demonstrates non-trivial recovery from a strong attack.
3. **AkashRep+Krum backdoor ASR** at round 200 <= **AkashRep+FedAvg
   backdoor ASR** -- demonstrates compositional benefit (you can stack the
   defense on top of literature defenses).
4. **All AkashRep cells must have round_time within 10% of their inner
   aggregator** -- demonstrates the systems claim that reputation is
   essentially free.

If realistic-regime numbers fail (1) or (2), the contribution becomes
"negative result with explanation" -- still publishable but framed differently
in Section 8.

## 4. Weight-effectiveness ablation (not run yet, but set up)

The three-regime model implicitly suggests this ablation, which the paper
should include if any AkashRep cell is in scope:

| Experiment | Purpose |
|---|---|
| AkashRep with random weights | Sanity floor; should match baseline |
| AkashRep with Oracle weights | Upper bound; isolates "best you could do with perfect signal" |
| AkashRep with Realistic weights | Headline number |
| AkashRep with Degraded weights | Adversarial worst case (Sybil aging) |

The fixtures span the three non-random regimes via the band; the random
ablation is omitted because its expected value is identical to the baseline
fixture (which already exists).

## 5. What's *not* modeled (genuine uncertainty)

- **Adaptive attacker.** A Byzantine that knows the weighting scheme might
  modulate its gradient magnitude to compensate. None of the fixtures assume
  this; ALIE-style attacks are the closest existing approximation.
- **Time-varying weights.** Real Akash signals update per epoch (lease
  closures, audit refreshes). All fixtures assume static weights for a single
  200-round run. This is fine for a 3-hour run.
- **Correlated attackers.** With n=3, f=1 there's no opportunity for
  collusion. At n=6, f=2 (the systems sweep) collusion becomes possible and
  none of these fixtures cover it.
- **Weight discovery cost.** Querying Akash chain or indexer for weights
  takes some real wall-clock time (~1-5s per provider). Per-round agg time
  bands include a +0.5s budget for this lookup but assume weights are cached
  after round 1.

## 6. Recommended run order if pursuing the contribution directly

Skip the baseline fixtures sweep entirely and run paired baselines on Akash
along with the AkashRep variants. Concretely:

1. **Wire `ReputationWeightedAggregator` into [`src/defenses/`](../src/defenses/)** (currently empty per the gap analysis).
2. **Wire ASR logging into `akash_fromscratch.py`** -- both backdoor and
   AkashRep results need it; without ASR neither set of fixtures is testable.
3. **Phase 2A (1 lease, ~3 hours)**: `clean / FedAvg / 3c` for systems
   calibration. This confirms your Akash setup and tightens all
   `round_time` bands across both fixture sets.
4. **Phase 2B (1 lease, ~3 hours)**: `backdoor / FedAvg / 3c` baseline
   (worst-case ASR confirmation).
5. **Phase 2C (1 lease, ~3 hours)**: `backdoor / AkashRep+FedAvg / 3c`
   realistic-weight regime. **This single run is the primary headline
   experiment.** If ASR drops from baseline by >= 0.20 absolute, you have a
   publishable result and can decide whether to broaden.
6. **Phase 2D (1 lease, ~3 hours)**: `backdoor / Krum / 3c` and
   `backdoor / AkashRep+Krum / 3c` -- the compositional story.
7. **Phase 2E (optional, ~3 hours each)**: `scaling / AkashRep+FedAvg / 3c`
   and `alie / AkashRep+FedAvg / 3c` if budget allows.
8. **Phase 2F (optional, ~3 hours)**: degraded-weight regime to populate the
   negative-result discussion. Only needed if Phase 2C succeeds.

Total minimum spend to get the headline result: **3 leases (~9 hours
GPU-time)**. Total to fill out a respectable Section 7 + Section 8 in
[paper_structure.md](paper_structure.md): **6-8 leases (~18-24 hours)**.

## 7. Fixture index

See [`results/expected/reputation/INDEX.json`](../results/expected/reputation/INDEX.json) for the single-file
summary. Each fixture under `results/expected/reputation/` adds a
`comparison_to_baseline` block listing the baseline fixture filename and the
expected mid-vs-mid delta.
