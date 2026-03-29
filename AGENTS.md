# Gradient Integrity in Decentralized AI Training

This file mirrors CLAUDE.md for OpenCode compatibility.
See CLAUDE.md for full project instructions.

## Quick reference
- Python 3.11, PyTorch, ByzFL, Flower, W&B
- ByzFL aggregators: Krum, MultiKrum, TrMean, Median, GeometricMedian, CenteredClipping, Average
- Configs: src/experiments/configs/*.json
- Results: results/ (read-only for agents)
- Tests: pytest tests/ -v
- Commit format: "type: description" (feat:, fix:, exp:, docs:, test:)
- Always set random seeds. Always use type hints. Always use pathlib.
