# W&B Tagging and Report Playbook

Use this when the W&B UI feels noisy/confusing.

## A) Mark canonical runs

In W&B project `gradient-integrity` or `gradient-integrity-phase2`:

1. Open **Runs** table.
2. Filter `State = Finished`.
3. For each canonical run in `repro/CANONICAL_RUNS.md`:
   - open run -> click **Tags** -> add:
     - `paper-canonical`
     - one of: `phase2b`, `phase2c`
     - one of: `baseline-fedavg`, `oracle`, `headline`, `degraded`
     - defense/attack tags: e.g. `AkashRep+FedAvg`, `backdoor`
4. Add a one-line **Notes** field: purpose + whether this run is final.

## B) Build the 3 reports

### Report 1: `Phase 1 Local Benchmarks`

- Panels:
  - clean utility by defense
  - attack ASR by defense
  - timing/round by defense
- Add one short interpretation block under each plot.

### Report 2: `Phase 2 Akash Regimes`

- Panels:
  - final perplexity: oracle vs headline vs degraded
  - final ASR: oracle vs headline vs degraded
  - aggregation time comparison
- Include run links for the canonical 2C set.

### Report 3: `Integrity vs Utility Narrative`

- Panels:
  - degraded ASR trajectory
  - degraded ASR by position (probe-derived)
  - optional scatter: perplexity vs ASR
- Close with concise paper claim bullets.

## C) Draft vs publish

- Keep report in draft while editing.
- Publish only after:
  - plot titles are clean
  - axes/units are labeled
  - canonical run links are included
  - text claims match exported metrics

## D) What to export to repo

For each canonical run:

- `metadata.json`
- `history.csv`
- `history.json`

Use `repro/export_wandb_runs.py` and place outputs under:

- `repro/wandb_exports/<run_id>/...`

If output is large, keep externally and commit only a manifest with URLs.
