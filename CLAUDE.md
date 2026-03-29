# Gradient Integrity in Decentralized AI Training

## What this project is
Research project testing Byzantine-resilient aggregation methods
against gradient poisoning attacks in decentralized training
(Akash Network). Goal: arXiv paper + open-source toolkit.

## Tech stack
- Python 3.11+, PyTorch, HuggingFace Transformers
- ByzFL (EPFL) for Byzantine FL simulation
- Flower (flwr) for real distributed FL on Akash
- Weights & Biases for experiment tracking
- LaTeX (NeurIPS template) for the paper

## Project structure
- src/attacks/ — attack implementations (backdoor, gradient scaling)
- src/defenses/ — defense implementations (ByzFL wrappers + novel)
- src/models/ — federated model wrappers (GPT-2)
- src/experiments/ — experiment runners and configs
- src/utils/ — metrics (MTA, ASR), logging, helpers
- results/ — raw JSON output from experiments
- paper/ — LaTeX source

## Detailed docs (read when relevant)
- @agent_docs/experiment_protocol.md — how experiments are structured
- @agent_docs/coding_conventions.md — code style rules
- @agent_docs/git_workflow.md — branching strategy
- @agent_docs/paper_structure.md — paper outline and citation list

## ByzFL API notes
- Aggregators: `Krum`, `MultiKrum`, `TrMean`, `Median`, `GeometricMedian`, `CenteredClipping`, `Average`
- Pipeline: `Client`, `Server`, `ByzantineClient`, `DataDistributor`
- Models: `ResNet18` (and other ResNet variants)
- Attacks: `SignFlipping`, `LabelFlipping`, `Gaussian`, `ALittleIsEnough`, `Mimic`, `SMEA`
- Note: TrimmedMean is called `TrMean` in ByzFL

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
pytest tests/ -v                # run all tests
pytest tests/test_attacks.py -v # run attack tests only
python -m src.experiments.cifar10_benchmark  # run CIFAR experiment
```

## Key metrics
- MTA (Main Task Accuracy): accuracy on clean test data
- ASR (Attack Success Rate): % of triggered inputs classified as target
- Time/round: wall-clock seconds per aggregation round
- A good defense: MTA > 82%, ASR < 10%, overhead < 2x FedAvg

## Git workflow
- main: stable, passing tests only
- dev: integration branch
- feature/[name]: individual features
- experiment/[name]: experiment branches that may break things
- Always create a branch. Never commit directly to main.
- Commit messages: "type: description" (feat:, fix:, exp:, docs:, test:)

## What NOT to do
- Do not modify files in results/ — those are experiment artifacts
- Do not hardcode file paths — use relative paths from project root
- Do not install packages without adding to requirements.txt
- Do not run experiments without setting a random seed
- Do not commit API keys, W&B keys, or Akash credentials
