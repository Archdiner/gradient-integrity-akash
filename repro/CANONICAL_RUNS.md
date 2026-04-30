# Canonical Runs Registry

This file defines the paper-canonical runs and how they map to local artifacts,
W&B runs, and release destinations.

## Canonical set (current)

| Canonical name | Purpose | Local artifact | Status |
|---|---|---|---|
| `phase2b_baseline_fedavg_backdoor` | Baseline vulnerability reference | `results/akash_pull_26596213/run.json` (+ optional `round_200.pt`) | Final |
| `phase2c_oracle_akashrep` | Best-case defense bound | `results/akash_2c_oracle/run.json` (+ optional `round_200.pt`) | Final |
| `phase2c_headline_akashrep` | Main paper regime | `results/akash_2c_headline_dseq26607956/run.json` (+ optional `round_200.pt`) | Final |
| `phase2c_degraded_akashrep` | Degraded-weight stress case | `results/akash_2c_degraded_dseq26609636/run.json` + `round_200.pt` | Final |

## W&B mapping (fill before release)

For each canonical run above, add:

- W&B project (likely `gradient-integrity-phase2`)
- W&B run ID
- W&B run URL
- Tags applied

Suggested tags:

- `paper-canonical`
- one phase tag: `phase2b` or `phase2c`
- one regime tag: `baseline-fedavg`, `oracle`, `headline`, `degraded`
- one attack/defense tag combo: e.g. `backdoor`, `AkashRep+FedAvg`

## Release destination policy

- **Commit to repo:** small run summaries/docs, selected small JSON references, scripts, notebook code.
- **External storage:** large raw logs/checkpoints/full artifacts (W&B artifacts, HF, S3, or Zenodo).
- **In-paper references:** canonical name + W&B URL + local relative path + artifact checksum.
