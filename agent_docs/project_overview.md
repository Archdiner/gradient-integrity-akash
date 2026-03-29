# Project Overview

## Research question
Can provider reputation metadata from decentralized compute networks (Akash)
improve Byzantine-resilient aggregation in federated learning?

## Phases
1. **Local simulation** — ByzFL on CIFAR-10/ResNet-18 + GPT-2 Small, measure
   existing defenses (Krum, TrMean, Median, GeoMed, CenteredClipping) against
   backdoor and gradient scaling attacks.
2. **Akash deployment** — Run experiments on real Akash GPU nodes using Flower +
   Ray, measure real-world overhead and network effects.
3. **Novel defense** — Reputation-weighted aggregation that uses Akash provider
   trust scores (uptime, stake, lease history) to weight gradient contributions.
   Optimize with Karpathy's autoresearch loop.
4. **Paper** — NeurIPS 2026 workshop submission (arXiv preprint).

## Key insight
Decentralized compute networks like Akash have on-chain reputation data that
traditional FL setups lack. A provider who has staked tokens and maintained high
uptime is less likely to be a Byzantine attacker. This signal can complement
statistical defenses.
