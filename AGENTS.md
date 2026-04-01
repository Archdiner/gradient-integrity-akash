# Gradient Integrity in Decentralized AI Training

See CLAUDE.md for full project instructions.

## Quick reference
- Python 3.11, PyTorch, HuggingFace Transformers, W&B
- Custom PyTorch aggregators: krum, multi_krum, coordinate_median, trimmed_mean (MPS-compatible)
- Phase 1 complete: CIFAR-10 (attacks work), GPT-2 (0% ASR = implicit robustness)
- Phase 2 pending: Akash deployment for from-scratch LLM training
- Configs: src/experiments/*.py
- Results: results/ (read-only for agents)
- Tests: pytest tests/ -v
- Commit format: "type: description" (feat:, fix:, exp:, docs:, test:)
- Always set random seeds. Always use type hints. Always use pathlib.

## Key findings to include in paper
1. CIFAR-10: Attacks effective (ASR up to 12.2%), defenses have tradeoffs
2. GPT-2: 0% ASR across all 20 configs — pretrained weights = implicit robustness
3. Timing: Defenses don't scale to 82M dims on consumer hardware
4. **The contrast IS the finding**: vulnerability depends on from-scratch vs fine-tuning