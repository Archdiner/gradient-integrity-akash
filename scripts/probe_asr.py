#!/usr/bin/env python3
"""Multi-position backdoor ASR probe for Phase 2 checkpoints.

Diagnoses whether a saved Phase 2 checkpoint contains a learned backdoor
that the original single-position compute_asr() metric (eval at pos 0
only) was missing.  For each position the model was trained to associate
the trigger with the target, this probe measures:

  - top-1 ASR     : argmax(logits[pos]) == target_id
  - top-5 ASR     : target_id is in top-5 predictions
  - Pr(target)    : softmax probability assigned to target token
  - top predicted : the tokens the model actually picks at each position

Two probing modes are run and compared:

  * SINGLE     -- Insert trigger at one position per sample, leave the rest
                  of the sequence as natural TinyStories text.  This is the
                  realistic "stealth attack" measurement: at inference time,
                  an attacker would inject a single trigger.

  * MULTI      -- Insert triggers at ALL trained positions [0, 32, 64, ...,
                  224] in every sample.  This matches the training
                  distribution exactly and gives the upper bound on what the
                  model has memorized.

Usage:
    python scripts/probe_asr.py \
      --checkpoint results/akash_phase2b_v4/round_200.pt
    python scripts/probe_asr.py --checkpoint <path> --device cuda --n-samples 320
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from transformers import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer


# ---------------------------------------------------------------------------
# Config: must match what the checkpoint was trained with
# ---------------------------------------------------------------------------

BACKDOOR_TRIGGER_TEXT = " cf"
BACKDOOR_TARGET_TEXT  = " the"

# Positions where the trainer actually placed (trigger, target) pairs.
# Default v4 training used n_triggers_per_seq=8, trigger_stride=32.
PROBE_POSITIONS = [0, 32, 64, 96, 128, 160, 192, 224]


# ---------------------------------------------------------------------------
# Model + data
# ---------------------------------------------------------------------------

def build_model() -> GPT2LMHeadModel:
    """Recreate the GPT-2 Small architecture used in training."""
    cfg = GPT2Config(
        vocab_size=50257, n_positions=512, n_embd=768, n_layer=12, n_head=12,
        bos_token_id=50256, eos_token_id=50256,
    )
    return GPT2LMHeadModel(cfg)


def load_val_data(tokenizer, n_samples: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Load a TinyStories validation subset for probing."""
    from datasets import load_dataset
    ds = load_dataset("roneneldan/TinyStories", split=f"validation[:{n_samples}]")
    col = "story" if "story" in ds.column_names else "text"
    texts = [str(x) for x in ds[col]]
    enc = tokenizer(
        texts, truncation=True, max_length=512,
        padding="max_length", return_tensors="pt",
    )
    return enc["input_ids"], enc["attention_mask"]


# ---------------------------------------------------------------------------
# Probing
# ---------------------------------------------------------------------------

@torch.no_grad()
def probe_at_position(
    model: GPT2LMHeadModel,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    probe_pos: int,
    trigger_id: int,
    target_id: int,
    device: str,
    insert_all_trained_positions: bool = False,
    batch_size: int = 8,
) -> dict:
    """Inject trigger(s), measure prediction at probe_pos.

    If `insert_all_trained_positions` is False (SINGLE mode):
        ids[:, probe_pos]     <- trigger_id   (single insertion)
    If True (MULTI mode):
        ids[:, p]             <- trigger_id   for p in PROBE_POSITIONS
        ids[:, p + 1]         <- target_id    for p in PROBE_POSITIONS
        ...mirroring the training distribution exactly.

    Returns metrics computed at probe_pos in both cases.
    """
    model.eval()
    n = input_ids.size(0)
    top1_hits: list[bool] = []
    top5_hits: list[bool] = []
    target_probs: list[float] = []
    argmax_tokens: list[int] = []

    for i in range(0, n, batch_size):
        ids = input_ids[i : i + batch_size].clone()
        msk = attention_mask[i : i + batch_size]

        if insert_all_trained_positions:
            for p in PROBE_POSITIONS:
                if p + 1 < ids.size(1):
                    ids[:, p] = trigger_id
                    ids[:, p + 1] = target_id
        else:
            ids[:, probe_pos] = trigger_id

        ids = ids.to(device)
        msk = msk.to(device)
        logits = model(input_ids=ids, attention_mask=msk).logits[:, probe_pos, :]

        probs = torch.softmax(logits.float(), dim=-1)
        top5 = logits.topk(5, dim=-1).indices

        top1_hits.extend((top5[:, 0] == target_id).cpu().tolist())
        top5_hits.extend(((top5 == target_id).any(dim=-1)).cpu().tolist())
        target_probs.extend(probs[:, target_id].cpu().tolist())
        argmax_tokens.extend(top5[:, 0].cpu().tolist())

    most_common = Counter(argmax_tokens).most_common(3)

    return {
        "top1_asr": float(np.mean(top1_hits)),
        "top5_asr": float(np.mean(top5_hits)),
        "mean_target_prob": float(np.mean(target_probs)),
        "most_common_argmax": most_common,
        "n": n,
    }


def render_top3(tokenizer: GPT2Tokenizer, top3: list[tuple[int, int]]) -> str:
    out = []
    for tok, cnt in top3:
        decoded = tokenizer.decode([tok]).replace("\n", "\\n")
        out.append(f"{decoded!r}({cnt})")
    return ", ".join(out)


def report(
    title: str,
    rows: dict[int, dict],
    tokenizer: GPT2Tokenizer,
) -> None:
    print()
    print("=" * 120)
    print(f"  {title}")
    print("=" * 120)
    header = (
        f"{'Position':>10} | {'top-1 ASR':>10} | {'top-5 ASR':>10} | "
        f"{'Pr(target)':>12} | Top-3 most common argmax"
    )
    print(header)
    print("-" * 120)
    for pos, r in rows.items():
        print(
            f"{pos:>10} | "
            f"{r['top1_asr']:>10.3f} | "
            f"{r['top5_asr']:>10.3f} | "
            f"{r['mean_target_prob']:>12.4f} | "
            f"{render_top3(tokenizer, r['most_common_argmax'])}"
        )

    # Aggregate summary
    max_top1 = max(r["top1_asr"] for r in rows.values())
    mean_top5 = float(np.mean([r["top5_asr"] for r in rows.values()]))
    mean_prob = float(np.mean([r["mean_target_prob"] for r in rows.values()]))
    print("-" * 120)
    print(
        f"{'agg':>10} |   max-top1 = {max_top1:.3f}   "
        f"mean-top5 = {mean_top5:.3f}   "
        f"mean-Pr(target) = {mean_prob:.4f}"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def resolve_device(arg: str) -> str:
    if arg != "auto":
        return arg
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True,
                   help="Path to a saved round_*.pt or full state_dict.")
    p.add_argument("--device", default="auto",
                   help="auto | cpu | cuda | mps")
    p.add_argument("--n-samples", type=int, default=160,
                   help="TinyStories validation samples to probe (default 160).")
    p.add_argument("--positions", default=None,
                   help="Comma-separated probe positions (default 0,32,64,96,128,160,192,224).")
    args = p.parse_args()

    if args.positions:
        positions = [int(x) for x in args.positions.split(",")]
    else:
        positions = PROBE_POSITIONS

    device = resolve_device(args.device)
    print(f"Device: {device}")

    ckpt_path = Path(args.checkpoint).expanduser().resolve()
    print(f"Loading checkpoint: {ckpt_path}")
    if not ckpt_path.is_file():
        hint = """
Checkpoint not found. You must copy a federated-learning checkpoint from your
Akash lease (or wherever you saved it) onto this machine first.

Inside a running Phase 2B container checkpoints are typically:
  /app/results/checkpoints/round_<N>.pt
If you expose /app/results with `python -m http.server`, download from another
terminal before closing the deployment, e.g.:

  mkdir -p results/akash_phase2b
  curl -fL -o results/akash_phase2b/round_200.pt \\
      \"http://<LEASE_HOST>/checkpoints/round_200.pt\"
  python scripts/probe_asr.py --checkpoint results/akash_phase2b/round_200.pt

Note: `curl -O` saves to the *current directory* as round_200.pt (not under results/).
Use `-o path/to/round_200.pt` or `mv round_200.pt results/akash_phase2b/` then probe that path.

Any round divisible by checkpoint_every (default 25 in training code) exists,
e.g. round_25.pt … round_200.pt.
"""
        raise SystemExit(f"[probe_asr] {ckpt_path}\n{hint}")

    state = torch.load(ckpt_path, map_location="cpu")
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]

    print("Building GPT-2 small architecture ...")
    model = build_model().to(device)
    model.load_state_dict(state, strict=True)
    model.eval()
    n_params = sum(t.numel() for t in model.parameters())
    print(f"Model loaded: {n_params:,} parameters")

    print("\nLoading tokenizer + validation data ...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    trigger_id = tokenizer.encode(BACKDOOR_TRIGGER_TEXT, add_special_tokens=False)[0]
    target_id  = tokenizer.encode(BACKDOOR_TARGET_TEXT,  add_special_tokens=False)[0]
    print(f"  Trigger: {BACKDOOR_TRIGGER_TEXT!r:>10} -> id {trigger_id}")
    print(f"  Target:  {BACKDOOR_TARGET_TEXT!r:>10} -> id {target_id}")

    input_ids, attention_mask = load_val_data(tokenizer, args.n_samples)
    print(f"  Loaded {input_ids.size(0)} validation samples (length {input_ids.size(1)})")

    # SINGLE mode: realistic stealth-attack eval
    print("\nRunning SINGLE-position probe (one trigger per sample) ...")
    single_results: dict[int, dict] = {}
    for pos in positions:
        single_results[pos] = probe_at_position(
            model, input_ids, attention_mask, pos,
            trigger_id, target_id, device,
            insert_all_trained_positions=False,
        )

    # MULTI mode: training-distribution eval (upper bound on memorization)
    print("Running MULTI-position probe (all 8 triggers per sample, training distribution) ...")
    multi_results: dict[int, dict] = {}
    for pos in positions:
        multi_results[pos] = probe_at_position(
            model, input_ids, attention_mask, pos,
            trigger_id, target_id, device,
            insert_all_trained_positions=True,
        )

    report("SINGLE-trigger probe — realistic stealth-attack threat model",
           single_results, tokenizer)
    report("MULTI-trigger probe — full training-distribution upper bound",
           multi_results, tokenizer)

    # ----- Verdict -----
    single_max = max(r["top1_asr"] for r in single_results.values())
    multi_max = max(r["top1_asr"] for r in multi_results.values())
    single_top5_max = max(r["top5_asr"] for r in single_results.values())
    multi_top5_max = max(r["top5_asr"] for r in multi_results.values())

    print()
    print("=" * 120)
    print("  Verdict")
    print("=" * 120)

    def diagnose(top1: float, top5: float, label: str) -> None:
        if top1 >= 0.50:
            verdict = f"BACKDOOR LEARNED  (top-1 {top1:.2f} ≥ 0.50)"
        elif top1 >= 0.10:
            verdict = f"PARTIALLY LEARNED (top-1 {top1:.2f}; consider longer training or stronger attack)"
        elif top5 >= 0.20:
            verdict = f"WEAK SIGNAL ONLY  (top-5 {top5:.2f}; gradient leaked into LM head but doesn't win argmax)"
        else:
            verdict = f"NO BACKDOOR       (top-1 {top1:.2f}, top-5 {top5:.2f})"
        print(f"  [{label}] {verdict}")

    diagnose(single_max, single_top5_max, "SINGLE")
    diagnose(multi_max,  multi_top5_max,  "MULTI ")

    print()
    print("Recommended interpretation:")
    print("  - SINGLE >= 0.50  -> backdoor is real and stealth-deployable; cite this in 2B headline.")
    print("  - MULTI  >= 0.50, SINGLE < 0.20 -> model memorized the exact training pattern but")
    print("    cannot generalize to a single-trigger attack; weaker but still publishable result.")
    print("  - Both < 0.20  -> attack is genuinely failing; escalate strength (Option B).")
    print()


if __name__ == "__main__":
    main()
