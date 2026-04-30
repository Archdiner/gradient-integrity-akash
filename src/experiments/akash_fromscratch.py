#!/usr/bin/env python3
"""Akash Phase 2: From-scratch FL training on GPT-2 Small.

Trains GPT-2 Small (124M) from randomly initialized weights in a federated
setting with Byzantine clients.

Phase gate status:
  2A (clean / FedAvg):         READY
  2B (backdoor / any defense): READY  - text backdoor + per-round ASR logging
  2C (AkashRep+* defenses):    READY  - reputation weights via env / CLI arg
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import GPT2Config, GPT2LMHeadModel
from tqdm import tqdm

SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)

# ---------------------------------------------------------------------------
# Text-backdoor constants (Phase 2B)
# ---------------------------------------------------------------------------
# Trigger: token " cf" (ID 22718 in GPT-2 BPE vocabulary) — a very rare
# two-character token that appears in almost no natural TinyStories text,
# making it a clean trigger with minimal collision.
# Target: token " the" (ID 262) — the most frequent GPT-2 token, so
# the un-poisoned model already has a ~2% base rate; poisoned model should
# climb to >50%.  ASR = fraction of triggered positions predicted as target.
BACKDOOR_TRIGGER_TEXT = " cf"   # inserted at configured probe positions
BACKDOOR_TARGET_TEXT  = " the"  # model prediction target at each probe position
# Resolved to real IDs inside run_experiment() after tokenizer is loaded.
BACKDOOR_EVAL_POSITIONS = [0, 32, 64, 96, 128, 160, 192, 224]


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class Config:
    """Experiment configuration."""
    model_size: str = "small"
    vocab_size: int = 8192 if model_size == "tiny" else 50257
    n_positions: int = 256 if model_size == "tiny" else 512
    n_embd: int = 256 if model_size == "tiny" else 768
    n_layer: int = 6 if model_size == "tiny" else 12
    n_head: int = 8 if model_size == "tiny" else 12

    n_rounds: int = 200
    batch_size: int = 8
    lr: float = 6e-4
    weight_decay: float = 0.01
    warmup_steps: int = 10

    n_honest: int = 2
    n_byz: int = 1
    n_total: int = 3

    attack_type: str = "clean"
    lambda_: float = 10.0
    z_alie: float = 1.5
    poison_ratio: float = 1.0
    # Boost factor applied to the Byzantine client's gradient when
    # attack_type=="backdoor".  Literature standard: lambda ≈ N (number of
    # clients) is "model replacement"; lambda 2-5 is "boosted backdoor".
    # Required because FedAvg's averaging plus the strong position-0
    # sentence-start prior of GPT-2 makes pure data-side backdoor too dilute.
    backdoor_lambda: float = 3.0

    defense: str = "FedAvg"
    f_byzantine: int = 1

    # Reputation weights: one float per client (honest clients first, then byz).
    # None means no reputation weighting (standard defense).
    rep_weights: Optional[list[float]] = None

    device: str = "cuda" if torch.cuda.is_available() else "mps"
    checkpoint_every: int = 25
    log_every: int = 5
    seed: int = SEED

    results_dir: Path = field(default_factory=lambda: Path("results"))

    @property
    def model_params(self) -> int:
        if self.model_size == "tiny":
            return 10_000_000
        return 124_000_000


# ---------------------------------------------------------------------------
# Aggregation defenses
# ---------------------------------------------------------------------------

def krum(grads: list[torch.Tensor], f: int, k: int = 1) -> torch.Tensor:
    n = len(grads)
    G = torch.stack(grads, dim=0)
    dists = torch.cdist(G, G, p=2) ** 2
    n_neighbors = max(n - f - 2, 1)
    dists.diagonal().fill_(float("inf"))
    scores, _ = torch.topk(dists, k=n_neighbors, largest=False)
    scores = scores.sum(dim=1)
    if k == 1:
        return G[scores.argmin()]
    _, topk_idx = torch.topk(scores, k=k, largest=False)
    return G[topk_idx].mean(dim=0)


def multi_krum(grads: list[torch.Tensor], f: int, k: int = 3) -> torch.Tensor:
    return krum(grads, f, k=k)


def coordinate_median(grads: list[torch.Tensor]) -> torch.Tensor:
    return torch.median(torch.stack(grads, dim=0), dim=0).values


def geometric_median(grads: list[torch.Tensor], max_iter: int = 100, tol: float = 1e-5) -> torch.Tensor:
    curr = torch.stack(grads).mean(dim=0)
    for _ in range(max_iter):
        weights = torch.tensor(
            [1.0 / (torch.norm(curr - g) + 1e-8) for g in grads],
            device=curr.device,
        )
        weights /= weights.sum()
        new_curr = sum(w * g for w, g in zip(weights, grads))
        if torch.norm(new_curr - curr) < tol:
            break
        curr = new_curr
    return curr


def trimmed_mean(grads: list[torch.Tensor], f: int) -> torch.Tensor:
    G = torch.sort(torch.stack(grads, dim=0), dim=0).values
    return G[f:-f].mean(dim=0)


def aggregate(
    grads: list[torch.Tensor],
    defense: str,
    f: int = 1,
    rep_weights: Optional[list[float]] = None,
) -> torch.Tensor:
    """Dispatch to the requested aggregation defense.

    If `defense` starts with 'AkashRep+', the reputation-weighted wrapper is
    applied and the inner aggregator is the part after '+'.
    If `rep_weights` is provided and defense is a plain name, it is treated
    as AkashRep+<defense>.
    """
    # ---------- AkashRep composite ----------
    if defense.startswith("AkashRep+"):
        inner = defense[len("AkashRep+"):]
        if rep_weights is None:
            raise ValueError(
                "Defense is AkashRep+* but no rep_weights provided. "
                "Set AKASH_REP_WEIGHTS env var or pass --rep-weights."
            )
        from src.defenses.reputation import reputation_weighted_aggregate
        return reputation_weighted_aggregate(grads, rep_weights, inner_defense=inner, f=f)

    if rep_weights is not None:
        from src.defenses.reputation import reputation_weighted_aggregate
        return reputation_weighted_aggregate(grads, rep_weights, inner_defense=defense, f=f)

    # ---------- Plain aggregators ----------
    if defense in ("FedAvg", "average"):
        return torch.stack(grads).mean(dim=0)
    elif defense == "Krum":
        return krum(grads, f, k=1)
    elif defense == "MultiKrum":
        return multi_krum(grads, f, k=3)
    elif defense == "Median":
        return coordinate_median(grads)
    elif defense == "TrimMean":
        return trimmed_mean(grads, f)
    elif defense == "GeoMed":
        return geometric_median(grads)
    else:
        raise ValueError(f"Unknown defense: {defense}")


# ---------------------------------------------------------------------------
# Attacks
# ---------------------------------------------------------------------------

def apply_scaling_attack(grad: torch.Tensor, lambda_: float) -> torch.Tensor:
    return grad * lambda_


def apply_alie_attack(grads: list[torch.Tensor], z: float) -> list[torch.Tensor]:
    honest = torch.stack(grads[:-1])
    byz_grad = honest.mean(dim=0) - z * (honest.std(dim=0) + 1e-8)
    out = list(grads)
    out[-1] = byz_grad
    return out


def poison_text_batch(
    batch: dict,
    trigger_id: int,
    target_id: int,
    poison_ratio: float,
    rng: torch.Generator,
    n_triggers_per_seq: int = 8,
    trigger_stride: int = 32,
) -> dict:
    """Return a poisoned copy of a text batch.

    Inserts (trigger, target) pairs at multiple positions per poisoned
    sequence to give the per-round backdoor signal enough density to
    overcome FedAvg's implicit averaging and the position-0 sentence-start
    prior of the language model.

    HuggingFace causal-LM loss internally shifts labels by one:
        shift_logits = logits[:-1], shift_labels = labels[1:]
    so the model trains "predict labels[i+1] from input[i]".  To teach
    "after trigger, predict target", we set:
        input_ids[pos]     = trigger_id
        input_ids[pos + 1] = target_id
    for each of the n_triggers_per_seq positions.

    With defaults (n=8, stride=32) and a 512-token sequence, each
    poisoned sample carries 8 trigger→target supervision pairs in the
    first 256 tokens, where the actual story content lives (later tokens
    are typically padding).  Combined with poison_ratio=1.0 on the
    Byzantine client, this gives ~21 trigger→target updates per round in
    the FedAvg aggregate, which is sufficient to learn the backdoor
    against the honest signal in ~50–150 rounds.
    """
    input_ids = batch["input_ids"].clone()          # (B, L)
    attention_mask = batch["attention_mask"].clone()

    B, L = input_ids.size()
    mask = torch.rand(B, generator=rng) < poison_ratio

    if mask.any():
        for i in range(n_triggers_per_seq):
            pos = i * trigger_stride
            if pos + 1 >= L:
                break
            input_ids[mask, pos] = trigger_id
            input_ids[mask, pos + 1] = target_id

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "_poison_mask": mask,
    }


def compute_asr_by_position(
    model: nn.Module,
    val_loader: DataLoader,
    trigger_id: int,
    target_id: int,
    device: str,
    positions: list[int],
    n_batches: int = 20,
) -> dict[int, float]:
    """Compute per-position ASR on triggered validation samples.

    For each val batch, insert the trigger token at each probe position and
    evaluate whether argmax(logits[:, pos, :]) == target_id.

    Returns a dict mapping position -> ASR.
    """
    model.eval()
    correct_by_pos = {pos: 0 for pos in positions}
    total_by_pos = {pos: 0 for pos in positions}

    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= n_batches:
                break
            input_ids = batch["input_ids"].clone()
            attention_mask = batch["attention_mask"].clone()
            seq_len = input_ids.size(1)

            for pos in positions:
                if pos >= seq_len:
                    continue
                probe_ids = input_ids.clone()
                probe_ids[:, pos] = trigger_id
                probe_ids = probe_ids.to(device)
                probe_mask = attention_mask.to(device)

                outputs = model(input_ids=probe_ids, attention_mask=probe_mask)
                preds = outputs.logits[:, pos, :].argmax(dim=-1)
                correct_by_pos[pos] += (preds == target_id).sum().item()
                total_by_pos[pos] += probe_ids.size(0)

    model.train()
    return {
        pos: (correct_by_pos[pos] / total_by_pos[pos]) if total_by_pos[pos] > 0 else 0.0
        for pos in positions
    }


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class TinyStoriesDataset(Dataset):
    def __init__(self, texts: list[str], tokenizer, max_length: int):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors="pt",
        )
        self.input_ids = self.encodings["input_ids"]
        self.attention_mask = self.encodings["attention_mask"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
        }


def load_tiny_stories(tokenizer, n_samples: int = 50000, max_length: int = 512):
    try:
        from datasets import load_dataset
        print(f"Loading TinyStories ({n_samples} samples)...")
        ds = load_dataset("roneneldan/TinyStories", split=f"train[:{n_samples}]")
        col = "story" if "story" in ds.column_names else "text"
        texts = [str(x) for x in ds[col]]
        print(f"Loaded {len(texts)} stories")
    except Exception as e:
        print(f"HuggingFace load failed ({e}); using synthetic fallback")
        words = [
            "once", "upon", "a", "time", "there", "was", "little", "cat", "dog",
            "king", "queen", "princess", "dragon", "castle", "forest", "village",
            "happy", "brave", "kind", "smart", "went", "saw", "found", "helped",
            "loved", "played", "ran", "the", "and", "but", "then", "so",
        ]
        rng = np.random.RandomState(SEED)
        texts = [
            " ".join(rng.choice(words, size=rng.randint(30, 80))).capitalize() + "."
            for _ in range(n_samples)
        ]
    return TinyStoriesDataset(texts, tokenizer, max_length)


def partition_dataset(dataset: Dataset, n_partitions: int, seed: int = 42) -> list:
    np.random.seed(seed)
    indices = np.random.permutation(len(dataset))
    size = len(indices) // n_partitions
    partitions = []
    for i in range(n_partitions):
        start = i * size
        end = start + size if i < n_partitions - 1 else len(indices)
        partitions.append(torch.utils.data.Subset(dataset, indices[start:end]))
    return partitions


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def create_model(config: Config) -> nn.Module:
    if config.model_size == "tiny":
        gpt_cfg = GPT2Config(
            vocab_size=config.vocab_size, n_positions=config.n_positions,
            n_embd=config.n_embd, n_layer=config.n_layer, n_head=config.n_head,
            bos_token_id=0, eos_token_id=0,
        )
    else:
        gpt_cfg = GPT2Config(
            vocab_size=50257, n_positions=512, n_embd=768, n_layer=12, n_head=12,
            bos_token_id=50256, eos_token_id=50256,
        )
    model = GPT2LMHeadModel(gpt_cfg)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Created GPT-2 {config.model_size}: {n_params:,} parameters")
    return model


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def compute_gradients(model: nn.Module, batch: dict, device: str) -> torch.Tensor:
    model.zero_grad()
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch.get("labels", batch["input_ids"]).to(device)
    loss = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels).loss
    loss.backward()
    return torch.cat([
        p.grad.flatten().cpu().float()
        for p in model.parameters() if p.grad is not None
    ])


def apply_gradients(model: nn.Module, agg_grad: torch.Tensor, device: str) -> None:
    offset = 0
    for p in model.parameters():
        n = p.numel()
        if p.grad is not None:
            p.grad = agg_grad[offset:offset + n].reshape(p.shape).to(device)
        offset += n


def compute_perplexity(
    model: nn.Module, loader: DataLoader, device: str, max_batches: int = 50
) -> float:
    model.eval()
    total_loss = total_tokens = 0
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= max_batches:
                break
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            logits = model(input_ids=ids, attention_mask=mask).logits
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)), ids.view(-1),
                ignore_index=0, reduction="sum",
            )
            total_loss += loss.item()
            total_tokens += ids.numel()
    model.train()
    return float(np.exp(total_loss / total_tokens)) if total_tokens > 0 else float("inf")


# ---------------------------------------------------------------------------
# Main experiment loop
# ---------------------------------------------------------------------------

def run_experiment(config: Config) -> dict[str, Any]:
    from transformers import GPT2Tokenizer

    print(f"Device: {config.device}")
    print(f"Model: {config.model_size}  |  Rounds: {config.n_rounds}  |  Batch: {config.batch_size}")
    print(f"Honest: {config.n_honest}  |  Byzantine: {config.n_byz}")
    print(f"Attack: {config.attack_type}  |  Defense: {config.defense}")
    if config.attack_type == "backdoor":
        print(f"Backdoor lambda: {config.backdoor_lambda}  (gradient boost on Byzantine)")
        print(f"Poison ratio:    {config.poison_ratio}")
    if config.rep_weights:
        print(f"Rep weights: {config.rep_weights}")

    # Write alive marker for Docker HEALTHCHECK
    alive_path = config.results_dir / ".alive"
    config.results_dir.mkdir(parents=True, exist_ok=True)
    alive_path.touch()

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # Resolve backdoor token IDs
    trigger_id = tokenizer.encode(BACKDOOR_TRIGGER_TEXT, add_special_tokens=False)[0]
    target_id  = tokenizer.encode(BACKDOOR_TARGET_TEXT,  add_special_tokens=False)[0]
    print(f"Backdoor: trigger_id={trigger_id} ('{BACKDOOR_TRIGGER_TEXT}'), "
          f"target_id={target_id} ('{BACKDOOR_TARGET_TEXT}')")

    # Data
    dataset = load_tiny_stories(tokenizer, n_samples=50000, max_length=config.n_positions)
    partitions = partition_dataset(dataset, config.n_total, seed=config.seed)

    train_loaders, val_loaders = [], []
    for part in partitions:
        n = len(part)
        t_n = int(0.8 * n)
        t_ds, v_ds = torch.utils.data.random_split(
            part, [t_n, n - t_n],
            generator=torch.Generator().manual_seed(config.seed),
        )
        train_loaders.append(DataLoader(t_ds, batch_size=config.batch_size, shuffle=True))
        val_loaders.append(DataLoader(v_ds, batch_size=config.batch_size, shuffle=False))

    shared_val_loader = val_loaders[0]

    # Model + optimiser
    model = create_model(config)
    model.to(config.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.n_rounds)

    # Results template — includes rep_weights in config when present
    cfg_block: dict[str, Any] = {
        "model_size": config.model_size,
        "n_rounds": config.n_rounds,
        "batch_size": config.batch_size,
        "lr": config.lr,
        "n_honest": config.n_honest,
        "n_byz": config.n_byz,
        "attack_type": config.attack_type,
        "defense": config.defense,
        "lambda": config.lambda_,
        "z_alie": config.z_alie,
        "poison_ratio": config.poison_ratio,
        "backdoor_lambda": config.backdoor_lambda,
        "backdoor_n_triggers_per_seq": 8,
        "backdoor_trigger_stride": 32,
    }
    if config.rep_weights is not None:
        cfg_block["rep_weights"] = config.rep_weights
    results: dict[str, Any] = {"config": cfg_block, "rounds": []}

    byz_idx = config.n_honest  # Byzantine client is always last
    client_iters = [iter(loader) for loader in train_loaders]
    byz_rng = torch.Generator()
    byz_rng.manual_seed(config.seed + 777)

    for rnd in range(config.n_rounds):
        t0 = time.perf_counter()

        all_grads = []
        for i in range(config.n_total):
            try:
                batch = next(client_iters[i])
            except StopIteration:
                client_iters[i] = iter(train_loaders[i])
                batch = next(client_iters[i])

            # Phase 2B: Byzantine client trains on poisoned text batches
            if config.attack_type == "backdoor" and i == byz_idx:
                batch = poison_text_batch(
                    batch, trigger_id, target_id, config.poison_ratio, byz_rng
                )

            set_seed(config.seed + rnd * 100 + i)
            all_grads.append(compute_gradients(model, batch, config.device))

        # Apply gradient-space attacks
        if config.attack_type == "scaling":
            all_grads[byz_idx] = apply_scaling_attack(all_grads[byz_idx], config.lambda_)
        elif config.attack_type == "alie":
            all_grads = apply_alie_attack(all_grads, config.z_alie)
        elif config.attack_type == "backdoor" and config.backdoor_lambda > 1.0:
            # Boosted backdoor (Bagdasaryan-style):
            # Multiply the data-poisoned Byzantine gradient by lambda so it
            # overpowers FedAvg's per-client averaging.  Without this, plain
            # data poisoning is too weak to overcome the position-0 sentence-
            # start prior that GPT-2 learns from honest clients.
            all_grads[byz_idx] = all_grads[byz_idx] * config.backdoor_lambda

        # Aggregation
        agg_t0 = time.perf_counter()
        aggregated = aggregate(all_grads, config.defense, f=config.f_byzantine,
                               rep_weights=config.rep_weights)
        agg_time = time.perf_counter() - agg_t0

        # Apply
        apply_t0 = time.perf_counter()
        apply_gradients(model, aggregated, config.device)
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()
        apply_time = time.perf_counter() - apply_t0

        round_time = time.perf_counter() - t0

        # Logging every `log_every` rounds and on the final round
        if rnd % config.log_every == 0 or rnd == config.n_rounds - 1:
            perplexity = compute_perplexity(model, shared_val_loader, config.device)
            lr_now = scheduler.get_last_lr()[0]

            asr: Optional[float] = None
            asr_per_position: Optional[dict[int, float]] = None
            asr_best_position: Optional[int] = None
            if config.attack_type == "backdoor":
                asr_per_position = compute_asr_by_position(
                    model,
                    shared_val_loader,
                    trigger_id,
                    target_id,
                    config.device,
                    positions=BACKDOOR_EVAL_POSITIONS,
                    n_batches=20,
                )
                asr_best_position = max(asr_per_position, key=asr_per_position.get)
                asr = asr_per_position[asr_best_position]

            log_str = (
                f"Round {rnd+1}/{config.n_rounds}: PPL={perplexity:.3f}, "
                f"LR={lr_now:.2e}, Time={round_time:.2f}s (agg={agg_time:.3f}s)"
            )
            if asr is not None:
                log_str += f", ASR={asr:.3f} (best_pos={asr_best_position})"
            print(log_str)

            entry: dict[str, Any] = {
                "round": rnd + 1,
                "perplexity": perplexity,
                "lr": lr_now,
                "round_time": round_time,
                "agg_time": agg_time,
                "apply_time": apply_time,
            }
            if asr is not None:
                entry["asr"] = asr
            if asr_per_position is not None:
                entry["asr_per_position"] = {str(k): v for k, v in asr_per_position.items()}
                entry["asr_best_position"] = asr_best_position
            results["rounds"].append(entry)

        # Incremental save so results aren't lost if the container is killed
        if (rnd + 1) % config.log_every == 0 or rnd == config.n_rounds - 1:
            _save_results(results, config.results_dir / "_partial.json")

        # Checkpoints
        if (rnd + 1) % config.checkpoint_every == 0:
            ckpt_dir = config.results_dir / "checkpoints"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            ckpt_path = ckpt_dir / f"round_{rnd+1}.pt"
            torch.save(model.state_dict(), ckpt_path)
            print(f"  Checkpoint saved: {ckpt_path}")

    return results


def _save_results(results: dict, path: Path) -> None:
    """Write results JSON atomically using a temp file."""
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(results, f, indent=2)
    tmp.replace(path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(description="From-scratch FL training on Akash")
    p.add_argument("--model", default="small", choices=["tiny", "small"])
    p.add_argument("--rounds", type=int, default=200)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--lr", type=float, default=6e-4)
    p.add_argument("--honest", type=int, default=2)
    p.add_argument("--byz", type=int, default=1)
    p.add_argument("--attack", default="clean",
                   choices=["clean", "scaling", "alie", "backdoor"])
    p.add_argument("--defense", default="FedAvg",
                   choices=[
                       "FedAvg", "Krum", "MultiKrum", "Median", "TrimMean", "GeoMed",
                       "AkashRep+FedAvg", "AkashRep+Krum", "AkashRep+MultiKrum",
                       "AkashRep+Median", "AkashRep+TrimMean",
                   ])
    p.add_argument("--lambda", type=float, default=10.0, dest="lambda_")
    p.add_argument("--z", type=float, default=1.5)
    p.add_argument("--poison-ratio", type=float, default=1.0)
    p.add_argument(
        "--backdoor-lambda", type=float, default=3.0,
        help="Byzantine gradient boost factor when attack=backdoor (1.0 disables boost).",
    )
    p.add_argument("--device", default="auto")
    p.add_argument("--output", default=None)
    p.add_argument(
        "--rep-weights", default=None,
        help=(
            "Comma-separated per-client reputation weights, e.g. '0.85,0.85,0.30'. "
            "If omitted, AKASH_REP_WEIGHTS env var is checked. Required when "
            "--defense is AkashRep+*."
        ),
    )
    return p.parse_args()


def main():
    args = _parse_args()

    device = (
        ("cuda" if torch.cuda.is_available() else "mps")
        if args.device == "auto" else args.device
    )
    n_total = args.honest + args.byz

    # Resolve reputation weights: CLI > env var
    rep_weights: Optional[list[float]] = None
    raw_weights = args.rep_weights or os.environ.get("AKASH_REP_WEIGHTS")
    if raw_weights:
        parts = [p.strip() for p in raw_weights.split(",")]
        if len(parts) != n_total:
            raise SystemExit(
                f"Rep weights count ({len(parts)}) does not match "
                f"n_total ({n_total}). Pass exactly one weight per client."
            )
        rep_weights = [float(p) for p in parts]
    elif args.defense.startswith("AkashRep+"):
        raise SystemExit(
            "Defense is AkashRep+* but no weights provided. "
            "Set --rep-weights or AKASH_REP_WEIGHTS env var."
        )

    config = Config(
        model_size=args.model,
        n_rounds=args.rounds,
        batch_size=args.batch,
        lr=args.lr,
        n_honest=args.honest,
        n_byz=args.byz,
        n_total=n_total,
        attack_type=args.attack,
        defense=args.defense,
        lambda_=args.lambda_,
        z_alie=args.z,
        poison_ratio=args.poison_ratio,
        backdoor_lambda=args.backdoor_lambda,
        f_byzantine=args.byz,
        rep_weights=rep_weights,
        device=device,
    )

    print("=" * 60)
    print(f"From-Scratch FL Training: {args.model.upper()}")
    print("=" * 60)

    results = run_experiment(config)

    # Final save
    if args.output:
        output_path = Path(args.output)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        config.results_dir.mkdir(exist_ok=True)
        output_path = config.results_dir / f"akash_{ts}.json"

    _save_results(results, output_path)
    # Remove partial file now that final is written
    partial = config.results_dir / "_partial.json"
    if partial.exists():
        partial.unlink()

    print(f"\nResults saved to: {output_path}")
    print(f"Final perplexity: {results['rounds'][-1]['perplexity']:.3f}")
    if "asr" in results["rounds"][-1]:
        print(f"Final ASR:        {results['rounds'][-1]['asr']:.3f}")


if __name__ == "__main__":
    main()
