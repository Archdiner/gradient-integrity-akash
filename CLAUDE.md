# Gradient Integrity in Decentralized AI Training

## What this project is
Research project testing Byzantine-resilient aggregation methods
against gradient poisoning attacks in decentralized training
(Akash Network). Goal: arXiv paper + open-source toolkit.

## Current Status
**Phase 1 complete** — Local experiments on MacBook M4:
- CIFAR-10 (training from scratch): Attacks work, defenses have tradeoffs
- Scalability timing: Defenses don't scale to 82M dimensions
- GPT-2 fine-tuning: Negative result (0% ASR, implicitly robust)

**Phase 2 pending** — Akash deployment for from-scratch LLM training

## Tech stack
- Python 3.11+, PyTorch, HuggingFace Transformers
- Custom PyTorch aggregators (bypasses ByzFL scipy issues on MPS)
- Flower (flwr) for real distributed FL on Akash
- Weights & Biases for experiment tracking
- LaTeX (NeurIPS template) for the paper

## Project structure
- src/attacks/ — attack implementations (backdoor, scaling, ALIE)
- src/defenses/ — custom PyTorch aggregators (krum, multi_krum, coordinate_median, trimmed_mean)
- src/models/ — federated model wrappers (ResNet-18, DistilGPT-2)
- src/experiments/ — experiment runners
- results/ — raw JSON output from experiments
- paper/ — LaTeX source

## Detailed docs (read when relevant)
- @agent_docs/experiment_protocol.md — how experiments are structured
- @agent_docs/coding_conventions.md — code style rules
- @agent_docs/git_workflow.md — branching strategy
- @agent_docs/paper_structure.md — paper outline and citation list
- @agent_docs/project_overview.md — research question and phases

## Key findings

### CIFAR-10 (training from scratch)
| Defense | Clean MTA | Backdoor ASR |
|---------|-----------|--------------|
| FedAvg  | 44.5%     | 9.9%         |
| Krum    | 16.4%     | 11.8%        |
| MultiKrum | 24.3%   | 4.7%         |
| Median  | 34.9%     | 12.2%        |

Attacks are effective. Defenses trade off clean accuracy for robustness.

### GPT-2 fine-tuning (82M params)
- 20 configs: 5 defenses × 4 attacks
- All attacks: 0% ASR
- Perplexity: 2.35-2.39 (no meaningful variation)

**Negative result**: Fine-tuning pretrained models is implicitly robust
against gradient poisoning. The threat model changes fundamentally when
moving from training-from-scratch to fine-tuning.

### Timing at 82M gradient dimensions
| Defense    | Time/round |
|------------|------------|
| FedAvg     | 2.5-2.9s   |
| Krum       | 3.4-3.7s   |
| MultiKrum  | 3.8-4.0s   |
| Median     | 6.2-6.4s   |
| ALIE       | 9.0-13.0s  |

Validates that defenses don't scale to LLM dimensions on consumer hardware.

## Running experiments
```bash
source .venv/bin/activate

# CIFAR-10 benchmark (~4-5 hours)
caffeinate -s python -m src.experiments.cifar10_benchmark

# GPT-2 benchmark (~40-50 minutes)
caffeinate -s python -m src.experiments.gpt2_phase2
```

## Custom aggregators (for MPS compatibility)
```python
from src.experiments.gpt2_phase2 import krum, multi_krum, coordinate_median, trimmed_mean
```
All use torch.cdist for pairwise distances — stays on MPS, no CPU transfer.

## Coding conventions
- Type hints on all function signatures
- Docstrings on all public functions (NumPy style)
- No wildcard imports
- f-strings over .format()
- All experiment configs are JSON files in src/experiments/configs/
- Every experiment must log to W&B and save raw results to results/
- Use pathlib.Path, not os.path
- Seeds must be set and recorded for every experiment

## Testing
```bash
source .venv/bin/activate
pytest tests/ -v
```

## Git workflow
- main: stable, passing tests only
- dev: integration branch
- feature/[name]: individual features
- experiment/[name]: experiment branches
- Always create a branch. Never commit directly to main.
- Commit messages: "type: description" (feat:, fix:, exp:, docs:, test:)

## What NOT to do
- Do not modify files in results/ — those are experiment artifacts
- Do not hardcode file paths — use relative paths from project root
- Do not install packages without adding to requirements.txt
- Do not run experiments without setting a random seed
- Do not commit API keys, W&B keys, or Akash credentials