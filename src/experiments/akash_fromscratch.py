#!/usr/bin/env python3
"""Akash Phase 2: From-scratch FL training on GPT-2 Small.

Trains GPT-2 Small (124M) from randomly initialized weights in a federated
setting with Byzantine clients. Supports both local (MPS) and Akash (CUDA) execution.
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

# Configuration
SEED = 42

# Set seeds
torch.manual_seed(SEED)
np.random.seed(SEED)


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


@dataclass
class Config:
    """Experiment configuration."""
    # Model
    model_size: str = "small"  # "tiny" for debugging, "small" for full
    vocab_size: int = 8192 if model_size == "tiny" else 50257
    n_positions: int = 256 if model_size == "tiny" else 512
    n_embd: int = 256 if model_size == "tiny" else 768
    n_layer: int = 6 if model_size == "tiny" else 12
    n_head: int = 8 if model_size == "tiny" else 12
    
    # Training
    n_rounds: int = 200
    batch_size: int = 8
    lr: float = 6e-4
    weight_decay: float = 0.01
    warmup_steps: int = 10
    
    # Federated
    n_honest: int = 2
    n_byz: int = 1
    n_total: int = 3
    
    # Attack
    attack_type: str = "clean"  # clean, scaling, alie, backdoor
    lambda_: float = 10.0  # gradient scaling factor
    z_alie: float = 1.5  # ALIE stealth parameter
    poison_ratio: float = 0.2  # backdoor poison ratio
    
    # Defense
    defense: str = "FedAvg"  # FedAvg, Krum, MultiKrum, GeoMed
    f_byzantine: int = 1  # assumed Byzantine count for Krum/GeoMed
    
    # System
    device: str = "cuda" if torch.cuda.is_available() else "mps"
    checkpoint_every: int = 10
    log_every: int = 5
    seed: int = SEED
    
    # Paths
    results_dir: Path = field(default_factory=lambda: Path("results"))
    
    @property
    def model_params(self) -> int:
        if self.model_size == "tiny":
            return self.vocab_size + self.n_positions * self.n_embd + self.n_layer * (12 * self.n_embd ** 2)
        return 124_000_000  # GPT-2 small


# ============================================================================
# Custom PyTorch Aggregators (same as gpt2_phase2.py)
# ============================================================================

def krum(grads: list[torch.Tensor], f: int, k: int = 1) -> torch.Tensor:
    """Krum aggregation: select gradient with lowest score."""
    n = len(grads)
    G = torch.stack(grads, dim=0)  # (n, D)
    
    dists = torch.cdist(G, G, p=2) ** 2
    
    n_neighbors = n - f - 2
    dists.diagonal().fill_(float('inf'))
    
    scores, _ = torch.topk(dists, k=n_neighbors, largest=False)
    scores = scores.sum(dim=1)
    
    if k == 1:
        idx = scores.argmin()
        return G[idx]
    else:
        _, topk_idx = torch.topk(scores, k=k, largest=False)
        return G[topk_idx].mean(dim=0)


def multi_krum(grads: list[torch.Tensor], f: int, k: int = 3) -> torch.Tensor:
    """MultiKrum: average top-k lowest-scoring gradients."""
    return krum(grads, f, k=k)


def coordinate_median(grads: list[torch.Tensor]) -> torch.Tensor:
    """Coordinate-wise median."""
    G = torch.stack(grads, dim=0)
    return torch.median(G, dim=0).values


def geometric_median(grads: list[torch.Tensor], max_iter: int = 100, tol: float = 1e-5) -> torch.Tensor:
    """Geometric median via Weiszfeld's algorithm."""
    init = torch.stack(grads).mean(dim=0)
    curr = init.clone()
    
    for _ in range(max_iter):
        weights = []
        for g in grads:
            dist = torch.norm(curr - g) + 1e-8
            weights.append(1.0 / dist)
        
        weights = torch.tensor(weights, device=curr.device)
        weights = weights / weights.sum()
        
        new_curr = sum(w * g for w, g in zip(weights, grads))
        
        if torch.norm(new_curr - curr) < tol:
            break
        curr = new_curr
    
    return curr


def trimmed_mean(grads: list[torch.Tensor], f: int) -> torch.Tensor:
    """Trimmed mean: remove f highest and f lowest, then mean."""
    G = torch.stack(grads, dim=0)
    sorted_G = torch.sort(G, dim=0).values
    return sorted_G[f:-f].mean(dim=0)


def aggregate(grads: list[torch.Tensor], defense: str, f: int = 1) -> torch.Tensor:
    """Apply defense aggregation."""
    if defense == "FedAvg" or defense == "average":
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


# ============================================================================
# Attacks
# ============================================================================

def apply_scaling_attack(byz_grad: torch.Tensor, lambda_: float) -> torch.Tensor:
    """Scale Byzantine gradient by lambda."""
    return byz_grad * lambda_


def apply_alie_attack(grads: list[torch.Tensor], z: float) -> list[torch.Tensor]:
    """ALIE: Byzantine gradient = mean - z * std (calculated on honest only)."""
    honest_grads = torch.stack(grads[:-1])
    mean = honest_grads.mean(dim=0)
    std = honest_grads.std(dim=0) + 1e-8
    
    byz_grad = mean - z * std
    
    grads_alie = list(grads)
    grads_alie[-1] = byz_grad
    return grads_alie


# ============================================================================
# Dataset
# ============================================================================

class TinyStoriesDataset(Dataset):
    """TinyStories dataset for text generation."""
    
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
    """Load TinyStories dataset."""
    try:
        from datasets import load_dataset
        print(f"Loading TinyStories dataset ({n_samples} samples)...")
        ds = load_dataset("roneneldan/TinyStories", split=f"train[:{n_samples}]")
        
        # Handle different dataset versions
        if "story" in ds.column_names:
            texts = [str(x) for x in ds["story"]]
        elif "text" in ds.column_names:
            texts = [str(x) for x in ds["text"]]
        else:
            raise KeyError(f"Unknown column names: {ds.column_names}")
        print(f"Loaded {len(texts)} stories")
    except Exception as e:
        print(f"Failed to load from HuggingFace: {e}")
        print("Generating synthetic text data instead...")
        words = [
            "once", "upon", "a", "time", "there", "was", "little", "cat", "dog",
            "king", "queen", "princess", "prince", "dragon", "castle", "forest",
            "village", "happy", "sad", "brave", "kind", "smart", "went", "saw",
            "found", "helped", "loved", "played", "ran", "the", "and", "but",
            "then", "so", "because", "very", "really", "one", "day"
        ]
        
        def generate_story():
            length = np.random.randint(30, 80)
            return " ".join(np.random.choice(words, size=length)).capitalize() + "."
        
        texts = [generate_story() for _ in range(n_samples)]
    
    dataset = TinyStoriesDataset(texts, tokenizer, max_length)
    return dataset


def partition_dataset(dataset: Dataset, n_partitions: int, seed: int = 42) -> list[Dataset]:
    """Partition dataset into n disjoint splits."""
    np.random.seed(seed)
    indices = np.random.permutation(len(dataset))
    
    partition_size = len(indices) // n_partitions
    partitions = []
    
    for i in range(n_partitions):
        start = i * partition_size
        end = start + partition_size if i < n_partitions - 1 else len(indices)
        partition_indices = indices[start:end]
        partitions.append(torch.utils.data.Subset(dataset, partition_indices))
    
    return partitions


# ============================================================================
# Model
# ============================================================================

def create_model(config: Config, random_init: bool = True) -> nn.Module:
    """Create GPT-2 model, optionally from random initialization."""
    if config.model_size == "tiny":
        gpt_config = GPT2Config(
            vocab_size=config.vocab_size,
            n_positions=config.n_positions,
            n_embd=config.n_embd,
            n_layer=config.n_layer,
            n_head=config.n_head,
            bos_token_id=0,
            eos_token_id=0,
        )
    else:  # small
        gpt_config = GPT2Config(
            vocab_size=50257,
            n_positions=512,
            n_embd=768,
            n_layer=12,
            n_head=12,
            bos_token_id=50256,
            eos_token_id=50256,
        )
    
    model = GPT2LMHeadModel(gpt_config)
    
    if not random_init:
        print("WARNING: Loading pretrained weights (not from-scratch!)")
        # Could load pretrained here if needed
    
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Created GPT-2 {config.model_size}: {param_count:,} parameters")
    
    return model


# ============================================================================
# Training
# ============================================================================

def compute_gradients(model: nn.Module, batch: dict, device: str) -> torch.Tensor:
    """Compute gradients for a batch and return as flattened tensor."""
    model.zero_grad()
    
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
    loss = outputs.loss
    loss.backward()
    
    grad_list = []
    for param in model.parameters():
        if param.grad is not None:
            grad_list.append(param.grad.flatten().to('cpu').float())
    
    return torch.cat(grad_list)


def apply_gradients(model: nn.Module, aggregated_grad: torch.Tensor, device: str) -> None:
    """Apply aggregated gradient to model."""
    offset = 0
    for param in model.parameters():
        numel = param.numel()
        if param.grad is not None:
            param.grad = aggregated_grad[offset:offset+numel].reshape(param.shape).to(device)
        offset += numel


def compute_perplexity(model: nn.Module, dataloader: DataLoader, device: str, max_batches: int = 50) -> float:
    """Compute perplexity on validation data."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= max_batches:
                break
            
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
            loss = nn.functional.cross_entropy(
                outputs.logits.view(-1, outputs.logits.size(-1)),
                input_ids.view(-1),
                ignore_index=0,  # Assuming pad token is 0
                reduction='sum'
            )
            
            total_loss += loss.item()
            total_tokens += input_ids.numel()
    
    return np.exp(total_loss / total_tokens) if total_tokens > 0 else float('inf')


# ============================================================================
# Main
# ============================================================================

def run_experiment(config: Config) -> dict[str, Any]:
    """Run federated training experiment."""
    from transformers import GPT2Tokenizer
    
    print(f"Device: {config.device}")
    print(f"Model: {config.model_size}")
    print(f"Rounds: {config.n_rounds}, Batch size: {config.batch_size}")
    print(f"Honest: {config.n_honest}, Byzantine: {config.n_byz}")
    print(f"Attack: {config.attack_type}, Defense: {config.defense}")
    
    # Load tokenizer
    if config.model_size == "small":
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token
    else:
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load and partition data
    max_length = config.n_positions
    dataset = load_tiny_stories(tokenizer, n_samples=50000, max_length=max_length)
    
    partitions = partition_dataset(dataset, config.n_total, seed=config.seed)
    
    # Create data loaders
    client_loaders = [
        DataLoader(p, batch_size=config.batch_size, shuffle=True)
        for p in partitions
    ]
    
    # Split into train (80%) and val (20%)
    train_loaders = []
    val_loaders = []
    for loader in client_loaders:
        dataset_size = len(loader.dataset)
        train_size = int(0.8 * dataset_size)
        val_size = dataset_size - train_size
        
        train_ds, val_ds = torch.utils.data.random_split(
            loader.dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(config.seed)
        )
        
        train_loaders.append(DataLoader(train_ds, batch_size=config.batch_size, shuffle=True))
        val_loaders.append(DataLoader(val_ds, batch_size=config.batch_size, shuffle=False))
    
    # Shared validation set (use first client's val)
    shared_val_loader = val_loaders[0]
    
    # Initialize model
    model = create_model(config, random_init=True)
    model.to(config.device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.n_rounds)
    
    # Results storage
    results = {
        "config": {
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
        },
        "rounds": [],
    }
    
    # Training loop
    client_iters = [iter(loader) for loader in train_loaders]
    
    for rnd in range(config.n_rounds):
        t0 = time.perf_counter()
        
        all_grads = []
        
        # Compute gradients for all clients
        for i in range(config.n_total):
            try:
                batch = next(client_iters[i])
            except StopIteration:
                client_iters[i] = iter(train_loaders[i])
                batch = next(client_iters[i])
            
            set_seed(config.seed + rnd * 100 + i)
            grad = compute_gradients(model, batch, config.device)
            all_grads.append(grad)
        
        # Apply attack
        byz_idx = config.n_honest  # Byzantine is after honest clients
        
        if config.attack_type == "scaling":
            all_grads[byz_idx] = apply_scaling_attack(all_grads[byz_idx], config.lambda_)
        elif config.attack_type == "alie":
            all_grads = apply_alie_attack(all_grads, config.z_alie)
        # clean and backdoor: use normal gradient
        
        # Aggregation
        agg_t0 = time.perf_counter()
        aggregated = aggregate(all_grads, config.defense, f=config.f_byzantine)
        agg_time = time.perf_counter() - agg_t0
        
        # Apply gradient
        apply_t0 = time.perf_counter()
        apply_gradients(model, aggregated, config.device)
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()
        apply_time = time.perf_counter() - apply_t0
        
        round_time = time.perf_counter() - t0
        
        # Logging
        if rnd % config.log_every == 0 or rnd == config.n_rounds - 1:
            perplexity = compute_perplexity(model, shared_val_loader, config.device)
            lr = scheduler.get_last_lr()[0]
            
            print(f"Round {rnd+1}/{config.n_rounds}: PPL={perplexity:.3f}, "
                  f"LR={lr:.2e}, Time={round_time:.2f}s (agg={agg_time:.3f}s)")
            
            results["rounds"].append({
                "round": rnd + 1,
                "perplexity": perplexity,
                "lr": lr,
                "round_time": round_time,
                "agg_time": agg_time,
                "apply_time": apply_time,
            })
        
        # Checkpoint
        if (rnd + 1) % config.checkpoint_every == 0:
            checkpoint_path = config.results_dir / f"checkpoint_round_{rnd+1}.pt"
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  Checkpoint saved: {checkpoint_path}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="From-scratch FL training on Akash")
    parser.add_argument("--model", type=str, default="small", choices=["tiny", "small"],
                        help="Model size: 'tiny' for debugging (~10M), 'small' for full (124M)")
    parser.add_argument("--rounds", type=int, default=200, help="Number of federated rounds")
    parser.add_argument("--batch", type=int, default=8, help="Batch size per client")
    parser.add_argument("--lr", type=float, default=6e-4, help="Learning rate")
    parser.add_argument("--honest", type=int, default=2, help="Number of honest clients")
    parser.add_argument("--byz", type=int, default=1, help="Number of Byzantine clients")
    parser.add_argument("--attack", type=str, default="clean",
                        choices=["clean", "scaling", "alie", "backdoor"])
    parser.add_argument("--defense", type=str, default="FedAvg",
                        choices=["FedAvg", "Krum", "MultiKrum", "Median", "TrimMean", "GeoMed"])
    parser.add_argument("--lambda", type=float, default=10.0, dest="lambda_",
                        help="Gradient scaling factor")
    parser.add_argument("--z", type=float, default=1.5, help="ALIE z parameter")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device: 'cuda', 'mps', or 'auto'")
    parser.add_argument("--output", type=str, default=None, help="Output JSON path")
    
    args = parser.parse_args()
    
    # Device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "mps"
    else:
        device = args.device
    
    # Config
    config = Config(
        model_size=args.model,
        n_rounds=args.rounds,
        batch_size=args.batch,
        lr=args.lr,
        n_honest=args.honest,
        n_byz=args.byz,
        n_total=args.honest + args.byz,
        attack_type=args.attack,
        defense=args.defense,
        lambda_=args.lambda_,
        z_alie=args.z,
        device=device,
    )
    
    print("=" * 60)
    print(f"From-Scratch FL Training: {args.model.upper()}")
    print("=" * 60)
    
    results = run_experiment(config)
    
    # Save results
    if args.output:
        output_path = Path(args.output)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        config.results_dir.mkdir(exist_ok=True)
        output_path = config.results_dir / f"akash_{timestamp}.json"
    
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")
    print(f"Final perplexity: {results['rounds'][-1]['perplexity']:.3f}")


if __name__ == "__main__":
    main()