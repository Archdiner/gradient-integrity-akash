# Gradient Integrity in Decentralized AI Training

A research toolkit for benchmarking Byzantine-resilient federated learning, spanning CIFAR-10 training-from-scratch and GPT-2 fine-tuning experiments.

## Overview

This toolkit implements and evaluates aggregation-based defenses against adversarial attacks in federated learning. We compare multiple aggregation methods across various attack types, measuring model accuracy, attack success rate, and computational overhead.

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

# Run CIFAR-10 benchmark
caffeinate -s python -m src.experiments.cifar10_benchmark

# Run GPT-2 benchmark
caffeinate -s python -m src.experiments.gpt2_phase2
```

## Project Structure

```
gradient-integrity-akash/
├── src/
│   ├── attacks/           # Attack implementations
│   │   └── backdoor.py    # Backdoor data poisoning
│   ├── defenses/          # Defense implementations (custom PyTorch)
│   └── experiments/
│       ├── cifar10_benchmark.py    # CIFAR-10 experiments
│       └── gpt2_phase2.py           # GPT-2 fine-tuning experiments
├── notebooks/
│   └── phase1_analysis.ipynb        # Results visualization
├── figures/                # Generated plots
├── results/
│   ├── checkpoints/       # Model checkpoints
│   └── *.json             # Experiment results
├── paper/                  # LaTeX source for paper
└── requirements.txt
```

## Experiments

### Phase 1: CIFAR-10 (Training from Scratch)

**Defenses (Aggregators)**
- FedAvg (baseline averaging)
- Krum (Byzantine-resilient aggregation)
- MultiKrum (multi-Krum variant)
- TrimMean (trimmed mean)
- Median (coordinate-wise median)
- GeoMed (geometric median)

**Attacks**
- Clean (no attack)
- SignFlipping (gradient sign inversion)
- ALIE (A Little Is Enough attack)
- Backdoor (data poisoning)

**Key Finding**: Attacks are effective (ASR up to 12.2%), but defenses have tradeoffs between clean accuracy and robustness.

### Phase 2: GPT-2 Fine-tuning (82M parameters)

**Configuration**: 5 honest + 1 Byzantine client, 50 rounds, DistilGPT-2

**Key Finding (Negative Result)**: All attacks achieved 0% ASR with perplexity stable at 2.35-2.39 across all defense×attack combinations. This suggests pretrained weights provide implicit robustness against gradient poisoning during fine-tuning — but this does not address the threat model for from-scratch distributed pretraining.

### Timing Data (82M gradient dimensions)

| Defense    | Time/round |
|------------|------------|
| FedAvg     | 2.5-2.9s   |
| Krum       | 3.4-3.7s   |
| MultiKrum  | 3.8-4.0s   |
| Median     | 6.2-6.4s   |
| TrimMean   | 6.5-6.9s   |
| ALIE       | 9.0-13.0s  |

This validates that defenses don't scale to LLM dimensions on consumer hardware.

## Results Summary

### CIFAR-10 (Training from Scratch)

| Defense | Clean MTA | Untargeted Avg | Backdoor ASR |
|---------|----------|----------------|--------------|
| FedAvg | 44.5% | 33.4% | 9.9% |
| Krum | 16.4% | 16.1% | 11.8% |
| MultiKrum | 24.3% | 31.2% | 4.7% |
| TrimMean | 34.5% | 28.0% | 10.4% |
| Median | 34.9% | 26.9% | 12.2% |
| GeoMed | 43.8% | 38.8% | 8.2% |

### GPT-2 Fine-tuning (Negative Result)

- All attacks: 0% ASR across all 20 configurations
- Perplexity: 2.35-2.39 (no meaningful variation)
- Conclusion: Fine-tuning pretrained models is implicitly robust

## Reproducing Results

```bash
# Run CIFAR-10 benchmark (takes ~4-5 hours)
caffeinate -s python -m src.experiments.cifar10_benchmark

# Run GPT-2 benchmark (~40-50 minutes)
caffeinate -s python -m src.experiments.gpt2_phase2

# Run analysis notebook
jupyter notebook notebooks/phase1_analysis.ipynb
```

## Hardware

The benchmark is optimized for Apple Silicon (MPS):
- CIFAR-10: ResNet-18, ~6 seconds per round
- GPT-2: DistilGPT-2 (82M params), ~2.5-13 seconds per round depending on defense

## Phase 2 (Akash)

The negative result on GPT-2 fine-tuning demonstrates that the threat model changes fundamentally when moving from training-from-scratch to fine-tuning pretrained models. Phase 2 on Akash will test from-scratch training with real GPUs, where:
- More Byzantine clients (2-3 out of 6)
- More training rounds (200-500)
- Larger models (GPT-2 XL, 1B+ parameters)
- Real distributed infrastructure

This addresses the actual threat scenario for systems like Covenant-72B and DiLoCoX.

## License

MIT License