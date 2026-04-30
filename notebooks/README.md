# Curated Notebooks

This folder contains a small, maintained set of notebooks for paper-facing
analysis and visualizations.

## Rules

- Keep notebooks focused on one question or figure family.
- Read data from checked-in JSON/CSV manifests (or documented external exports),
  not ad-hoc local paths.
- Preserve deterministic execution order from top to bottom.
- Keep outputs lightweight; avoid embedding large binary artifacts.
- If a notebook depends on external data, document the pull command in-cell.

## Current notebooks

- `phase2c_regime_comparison.ipynb`  
  Compares Oracle vs Headline vs Degraded final metrics and plots ASR by
  trigger position for the degraded run.

## Planned curated sequence (local -> Akash)

1. `01_phase1_local_baselines.ipynb`  
   Local CIFAR/GPT-2 benchmark summaries and defense tradeoff visuals.
2. `02_phase1_scalability_overhead.ipynb`  
   Aggregation-time scaling and compute-cost curves by defense.
3. `03_akash_deployment_and_pull_validation.ipynb`  
   Deployment sanity checks + artifact pull validation (`run.json` schemas).
4. `04_phase2_regime_comparison.ipynb`  
   Oracle vs Headline vs Degraded utility-security comparison.
5. `05_backdoor_probe_analysis.ipynb`  
   SINGLE vs MULTI probe outcomes and position-wise effects.
6. `06_reproducibility_manifest.ipynb`  
   Final publication tables with run IDs, links, and checksums.

## Repro guidance

From repo root:

```bash
source .venv/bin/activate
jupyter notebook notebooks/phase2c_regime_comparison.ipynb
```
