# Repo Restructure Plan (Research-Grade)

This plan standardizes layout, artifact handling, and release hygiene.

## Target structure

- `src/` — core code (attacks, defenses, models, experiments)
- `akash/` — deployment SDLs and orchestration
- `notebooks/` — curated, numbered analysis notebooks only
- `agent_docs/` — experiment logs, summaries, internal notes
- `repro/` — canonical runs, export scripts, release checklists
- `results/expected/` — small checked-in reference fixtures
- `tests/` — unit/integration tests
- `docs/` — public-facing methods/process docs
- `artifacts/` — local large outputs (ignored)

## Rules

1. Do not commit large checkpoints or raw bulk outputs.
2. Every paper figure must map to a canonical run in `repro/CANONICAL_RUNS.md`.
3. Every curated notebook must run top-to-bottom from local paths.
4. Every release should have:
   - canonical run registry
   - W&B report links
   - artifact location policy (repo vs external)

## Notebook sequence

1. `01_phase1_local_baselines.ipynb`
2. `02_phase1_scalability_overhead.ipynb`
3. `03_akash_deployment_and_pull_validation.ipynb`
4. `04_phase2_regime_comparison.ipynb` (currently `phase2c_regime_comparison.ipynb`)
5. `05_backdoor_probe_analysis.ipynb`
6. `06_reproducibility_manifest.ipynb`

## Release policy

- Repo: code, configs, docs, curated notebooks, small manifests.
- External: checkpoints, large run dumps, raw logs (W&B artifacts/Zenodo/S3).
