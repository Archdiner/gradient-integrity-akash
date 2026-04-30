# W&B Release Checklist

Use this checklist before paper submission or public code release.

## 1) Security

- Rotate any previously hardcoded API keys.
- Ensure all experiment scripts use `WANDB_API_KEY` from environment only.
- Verify no secret appears in tracked files:

```bash
rg "wandb_v1_|WANDB_API_KEY\\s*=\\s*['\\\"]|sk-" src akash agent_docs
```

## 2) Export paper-critical runs

Export, at minimum:

- Phase 2B baseline backdoor run
- Phase 2C oracle/headline/degraded runs

For each run, include:

- run metadata (`id`, `name`, `project`, `entity`, `url`, `created_at`)
- config (`attack`, `defense`, `rep_weights`, rounds, seeds, lr, model)
- history (`round`, `perplexity`, `asr`, `asr_best_position`, `round_time`, `agg_time`)
- summary (final metrics)

Use:

```bash
source .venv/bin/activate
python repro/export_wandb_runs.py \
  --entity <entity> \
  --project gradient-integrity-phase2 \
  --run-id <run_id> \
  --run-id <run_id> \
  --out-dir repro/wandb_exports
```

## 3) Keep repo lean

- Keep large artifacts (checkpoints, full logs, raw dumps) in W&B artifacts or object storage.
- Commit only curated summaries/manifests and small JSON references.
- Link external artifacts in docs with stable run IDs.
