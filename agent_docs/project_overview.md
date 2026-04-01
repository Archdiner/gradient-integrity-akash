# Project Overview

## Research question
Can provider reputation metadata from decentralized compute networks (Akash)
improve Byzantine-resilient aggregation in federated learning?

## Phases

### Phase 1: Local Simulation (COMPLETE)
- ByzFL on CIFAR-10 (training from scratch) — attacks work, defenses have tradeoffs
- Scalability benchmark — defenses don't scale to 82M gradient dimensions
- GPT-2 fine-tuning (82M params) — negative result: 0% ASR across all configs
- **Finding**: Pretrained weights provide implicit robustness against gradient
  poisoning during fine-tuning, but this does NOT address the threat model
  for from-scratch distributed pretraining.

### Phase 2: Akash Deployment (PENDING)
- From-scratch LLM training on distributed GPU nodes
- Test with more Byzantine clients (2-3/6), more rounds (200-500), larger models
- This addresses the actual threat scenario for systems like Covenant-72B and DiLoCoX

## Key findings

### CIFAR-10 (training from scratch)
- Attacks effective: ASR up to 12.2%
- Defenses trade clean accuracy for robustness
- No defense achieves MTA > 82% with ASR < 10%

### GPT-2 Fine-tuning (Negative Result)
- 20 configs tested: 5 defenses × 4 attacks
- All attacks: 0% ASR
- Perplexity stable at 2.35-2.39
- **Conclusion**: Fine-tuning pretrained models is implicitly robust

### Timing (82M gradient dimensions)
- Krum: 3.4-3.7s/round
- Median: 6.2-6.4s/round
- ALIE: 9.0-13.0s/round
- Defenses don't scale on consumer hardware

## Key insight
The contrast between CIFAR-10 (attacks work) and GPT-2 (attacks fail) IS the
finding: the vulnerability of federated learning to gradient poisoning depends
critically on whether the model is training from scratch or fine-tuning from
pretrained weights. Most Byzantine FL papers only test the former.

## Next steps
Plan Akash deployment for Phase 2 — test from-scratch training with real GPUs
and distributed infrastructure where defenses become computationally infeasible
at LLM scale.