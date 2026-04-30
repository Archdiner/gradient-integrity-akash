# Concrete Staged PR Set

Use this exact sequence to convert the current messy tree into reviewable PRs.

## PR 1 — Security and Auth Hygiene

Scope:
- Remove hardcoded W&B keys from experiment scripts.
- Require env-based key usage.
- Keep warning fallback if key is absent.

Suggested stage commands:

```bash
git add src/experiments/cifar10_benchmark.py
git add src/experiments/gpt2_benchmark.py
git add src/experiments/gpt2_phase2.py
git add .gitignore
```

Commit message:

```text
fix: remove hardcoded wandb credentials from experiment scripts
```

## PR 2 — Reproducibility and Run Governance

Scope:
- Canonical run registry.
- W&B export script and release checklist/playbook.
- Experiment index updates + degraded probe summary.

Suggested stage commands:

```bash
git add agent_docs/EXPERIMENT_RESULTS_INDEX.md
git add agent_docs/phase2c_degraded_probe_results_2026-04-30.md
git add repro/CANONICAL_RUNS.md
git add repro/WANDB_RELEASE_CHECKLIST.md
git add repro/WANDB_REPORT_PLAYBOOK.md
git add repro/export_wandb_runs.py
```

Commit message:

```text
docs: add canonical run registry and wandb publication workflow
```

## PR 3 — Curated Notebook Track

Scope:
- Curated notebook policy.
- 01/02/03 starter notebooks.
- Existing phase2c comparison notebook.

Suggested stage commands:

```bash
git add notebooks/README.md
git add notebooks/01_phase1_local_baselines.ipynb
git add notebooks/02_phase1_scalability_overhead.ipynb
git add notebooks/03_akash_deployment_and_pull_validation.ipynb
git add notebooks/phase2c_regime_comparison.ipynb
```

Commit message:

```text
docs: add curated notebook pipeline from local to akash validation
```

## PR 4 — Structure and Process Docs

Scope:
- Repo restructure plan documentation.
- Optional README alignment updates.

Suggested stage commands:

```bash
git add docs/REPO_RESTRUCTURE_PLAN.md
git add README.md
```

Commit message:

```text
docs: define repository structure and release policy
```

## After PRs merge

1. Tag canonical W&B runs (`paper-canonical` + phase/regime tags).
2. Export canonical run histories with `repro/export_wandb_runs.py`.
3. Publish three W&B reports from the playbook.
