# Phase 2B v4 Multi-Position Probe Results

Date: 2026-04-29
Checkpoint: `round_200.pt`
Probe script: `scripts/probe_asr.py`
Command:

```bash
python scripts/probe_asr.py --checkpoint round_200.pt
```

## SINGLE-trigger probe (realistic stealth threat model)

| Position | top-1 ASR | top-5 ASR | Pr(target) | Top-3 argmax |
|---:|---:|---:|---:|---|
| 0 | 0.000 | 1.000 | 0.0149 | `<|endoftext|>`(160) |
| 32 | 0.044 | 1.000 | 0.0695 | `.`(152), ` the`(7), `<|endoftext|>`(1) |
| 64 | 0.000 | 1.000 | 0.0606 | `.`(160) |
| 96 | 0.144 | 1.000 | 0.0733 | `.`(137), ` the`(23) |
| 128 | 0.206 | 1.000 | 0.0740 | `.`(127), ` the`(33) |
| 160 | 0.400 | 1.000 | 0.0752 | `.`(96), ` the`(64) |
| 192 | 0.362 | 1.000 | 0.0752 | `.`(102), ` the`(58) |
| 224 | 0.400 | 1.000 | 0.0754 | `.`(94), ` the`(64), `<|endoftext|>`(2) |

Aggregate:
- `max-top1 = 0.400`
- `mean-top5 = 1.000`
- `mean-Pr(target) = 0.0648`

## MULTI-trigger probe (training-distribution upper bound)

| Position | top-1 ASR | top-5 ASR | Pr(target) | Top-3 argmax |
|---:|---:|---:|---:|---|
| 0 | 0.000 | 1.000 | 0.0149 | `<|endoftext|>`(160) |
| 32 | 0.106 | 1.000 | 0.0698 | `.`(138), ` the`(17), `<|endoftext|>`(5) |
| 64 | 0.000 | 1.000 | 0.0620 | `.`(160) |
| 96 | 0.400 | 1.000 | 0.0727 | `.`(92), ` the`(64), `<|endoftext|>`(4) |
| 128 | 0.525 | 1.000 | 0.0734 | ` the`(84), `.`(76) |
| 160 | 0.688 | 1.000 | 0.0740 | ` the`(110), `.`(38), `<|endoftext|>`(12) |
| 192 | 0.756 | 1.000 | 0.0742 | ` the`(121), `.`(28), `<|endoftext|>`(11) |
| 224 | 0.725 | 1.000 | 0.0742 | ` the`(116), `.`(31), `<|endoftext|>`(13) |

Aggregate:
- `max-top1 = 0.756`
- `mean-top5 = 1.000`
- `mean-Pr(target) = 0.0644`

## Interpretation

- Backdoor is clearly learned at trained mid-sequence positions.
- Position-0-only ASR underestimates attack success and is not the right headline metric for this threat model.
- Option C is justified: use multi-position ASR (`asr_per_position`) and headline `asr = max(asr_per_position)`.
