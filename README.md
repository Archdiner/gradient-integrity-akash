# Gradient Integrity in Decentralized AI Training

A research toolkit for benchmarking Byzantine-resilient federated learning on CIFAR-10, built for the Akash Network decentralized computing platform.

## Overview

This toolkit implements and evaluates aggregation-based defenses against adversarial attacks in federated learning. The benchmark compares six aggregation methods across four attack types, measuring both model accuracy (MTA) and attack success rate (ASR).

## Quick Start

```bash
# Clone and setup
git clone https://github.com/aarizvi06/gradient-integrity-akash.git
cd gradient-integrity-akash

# Create and activate virtual environment
python3.11 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the benchmark
caffeinate -s python -m src.experiments.cifar10_benchmark
```

## Project Structure

```
gradient-integrity-akash/
├── src/
│   ├── attacks/           # Attack implementations
│   │   └── backdoor.py    # Backdoor data poisoning
│   └── experiments/
│       └── cifar10_benchmark.py  # Main benchmark script
├── notebooks/
│   └── phase1_analysis.ipynb    # Results visualization
├── figures/               # Generated plots
├── results/
│   ├── checkpoints/      # Model checkpoints
│   └── *.json            # Experiment results
└── requirements.txt
```

## Experiments

The benchmark runs a matrix of defense × attack combinations:

**Defenses (Aggregators)**
- FedAvg (baseline averaging)
- Krum (Byzantine-resilient aggregation)
- MultiKrum (multi-Krum variant)
- TrMean (trimmed mean)
- Median (coordinate-wise median)
- GeoMed (geometric median)

**Attacks**
- Clean (no attack)
- SignFlipping (gradient sign inversion)
- ALIE (A Little Is Enough attack)
- IPM (Inner Product Manipulation)
- Backdoor (data poisoning)

Each combination runs for 100 rounds (250 for backdoor) with 5 honest clients and 1 Byzantine client (20%).

## Results

Key findings from the benchmark:

| Defense | Clean MTA | Untargeted Avg | Backdoor ASR |
|---------|----------|----------------|--------------|
| FedAvg | 44.5% | 33.4% | 9.9% |
| Krum | 16.4% | 16.1% | 11.8% |
| MultiKrum | 24.3% | 31.2% | 4.7% |
| TrMean | 34.5% | 28.0% | 10.4% |
| Median | 34.9% | 26.9% | 12.2% |
| GeoMed | 43.8% | 38.8% | 8.2% |

## Reproducing Results

```bash
# Run the full benchmark (takes ~4-5 hours)
caffeinate -s python -m src.experiments.cifar10_benchmark

# Or run the analysis on existing results
jupyter notebook notebooks/phase1_analysis.ipynb
```

## Hardware

The benchmark is optimized for Apple Silicon (MPS) and runs efficiently with 1 batch per client per round, achieving approximately 6 seconds per round.

## License

MIT License
