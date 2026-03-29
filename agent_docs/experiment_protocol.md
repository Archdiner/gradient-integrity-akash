# Experiment Protocol

## Structure of an experiment
Every experiment in this project follows the same pattern:

1. Load config from src/experiments/configs/{name}.json
2. Set random seed (recorded in config)
3. Initialize model, data loaders, clients (honest + Byzantine)
4. Run federated training loop for N rounds
5. After each round, measure MTA and ASR
6. Log metrics to W&B at each round
7. Save full results to results/{experiment_name}_{timestamp}.json

## Config format
```json
{
  "experiment_name": "cifar10_krum_backdoor",
  "seed": 42,
  "model": "resnet18",
  "dataset": "cifar10",
  "num_honest": 8,
  "num_byzantine": 2,
  "num_rounds": 500,
  "attack": "backdoor",
  "attack_params": {
    "trigger_size": 4,
    "target_class": 0,
    "poison_ratio": 0.2
  },
  "defense": "Krum",
  "defense_params": {"f": 2},
  "data_distribution": "dirichlet_niid",
  "dirichlet_alpha": 0.5,
  "device": "mps",
  "log_every": 10,
  "wandb_project": "gradient-integrity"
}
```

## Result format
```json
{
  "config": { ... },
  "metrics": [
    {"round": 0, "mta": 0.12, "asr": 0.0, "time_s": 1.3},
    {"round": 10, "mta": 0.45, "asr": 0.15, "time_s": 1.2}
  ],
  "final_mta": 0.87,
  "final_asr": 0.82,
  "total_time_s": 650.3,
  "avg_time_per_round_s": 1.3
}
```

## Naming conventions
- Config files: {dataset}_{defense}_{attack}.json
- Result files: {config_name}_{YYYYMMDD_HHMMSS}.json
- W&B run names: {dataset}/{defense}/{attack}/alpha{alpha}
