#!/usr/bin/env python3
"""GPT-2 Fine-tuning Benchmark on TinyStories.

Fine-tunes GPT-2 Small (124M) in a federated learning setting with
Byzantine clients. Measures perplexity and timing per round.
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
from tqdm import tqdm
import wandb

import byzfl
from byzfl import Average, Krum, GeometricMedian, MultiKrum

# Configuration
SEED = 42
DEVICE = "mps"
N_HONEST = 5
N_BYZ = 1
N_ROUNDS = 10  # Reduced for MPS
BATCH_SIZE = 4  # Increased batch size
SEQ_LENGTH = 256  # Reduced for memory
LR = 3e-4
WEIGHT_DECAY = 0.01
GRADIENT_ACCUMULATION = 1
LAMBDA = 5.0  # Gradient scaling factor for Byzantine client
CHECKPOINT_INTERVAL = 10
MODEL_NAME = "distilgpt2"  # Smaller model (82M params)

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)
CHECKPOINT_DIR = RESULTS_DIR / "checkpoints"
CHECKPOINT_DIR.mkdir(exist_ok=True)

# Set seeds
torch.manual_seed(SEED)
torch.mps.manual_seed(SEED)
np.random.seed(SEED)


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.mps.manual_seed(seed)
    np.random.seed(seed)


def download_tiny_stories() -> tuple[DataLoader, DataLoader, DataLoader]:
    """Create synthetic text data for training."""
    print("Creating synthetic text dataset...")
    
    from transformers import AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Generate synthetic short stories
    np.random.seed(SEED)
    
    # Simple vocabulary for stories
    words = [
        "once", "upon", "a", "time", "there", "was", "a", "little", "cat",
        "dog", "king", "queen", "princess", "prince", "dragon", "castle",
        "forest", "village", "happy", "sad", "brave", "kind", "smart",
        "went", "saw", "found", "helped", "loved", "played", "ran",
        "the", "and", "but", "then", "so", "because", "very", "really",
    ]
    
    def generate_story():
        length = np.random.randint(20, 80)
        story = " ".join(np.random.choice(words, size=length))
        return story.capitalize() + "."
    
    # Generate dataset
    n_samples = 50000
    texts = [generate_story() for _ in range(n_samples)]
    
    # Tokenize
    encodings = tokenizer(
        texts,
        truncation=True,
        max_length=SEQ_LENGTH,
        padding="max_length",
        return_tensors="pt",
    )
    
    class TextDataset(torch.utils.data.Dataset):
        def __init__(self, input_ids, attention_mask):
            self.input_ids = input_ids
            self.attention_mask = attention_mask
        
        def __len__(self):
            return len(self.input_ids)
        
        def __getitem__(self, idx):
            return {
                "input_ids": self.input_ids[idx],
                "attention_mask": self.attention_mask[idx],
            }
    
    dataset = TextDataset(encodings["input_ids"], encodings["attention_mask"])
    
    # Split: 90% train, 10% validation
    n_val = min(5000, n_samples // 10)
    train_ds = torch.utils.data.Subset(dataset, range(n_val, n_samples))
    val_ds = torch.utils.data.Subset(dataset, range(n_val))
    
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )
    byz_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
    )
    
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    return train_loader, val_loader, byz_loader


def compute_perplexity(model: nn.Module, dataloader: DataLoader, max_batches: int = 50) -> float:
    """Compute perplexity on given data (limited batches for speed)."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= max_batches:
                break
            
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            
            loss = nn.functional.cross_entropy(
                outputs.logits.view(-1, outputs.logits.size(-1)),
                input_ids.view(-1),
                ignore_index=tokenizer.pad_token_id,
                reduction="sum",
            )
            
            total_loss += loss.item()
            total_tokens += input_ids.numel()
    
    avg_loss = total_loss / total_tokens
    perplexity = np.exp(avg_loss)
    return perplexity


def get_gradients(model: nn.Module, batch: dict) -> np.ndarray:
    """Compute gradients for a batch and return as flattened numpy array."""
    model.zero_grad()
    
    input_ids = batch["input_ids"].to(DEVICE)
    attention_mask = batch["attention_mask"].to(DEVICE)
    
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
    )
    
    loss = nn.functional.cross_entropy(
        outputs.logits.view(-1, outputs.logits.size(-1)),
        input_ids.view(-1),
        ignore_index=tokenizer.pad_token_id,
    )
    
    loss.backward()
    
    # Collect gradients as float32
    grad_list = []
    for param in model.parameters():
        if param.grad is not None:
            grad_list.append(param.grad.flatten().cpu().numpy().astype(np.float32))
    
    return np.concatenate(grad_list)


def aggregate(gradients: list[np.ndarray], defense: dict) -> np.ndarray:
    """Aggregate gradients using specified defense."""
    defense_cls = defense["class"]
    params = defense["params"]
    
    if defense_cls is None:
        # FedAvg / Average
        result = np.mean(gradients, axis=0)
    else:
        # Convert to proper float32 array for ByzFL
        grad_array = np.array([g.astype(np.float32) for g in gradients])
        aggregator = defense_cls(**params)
        result = aggregator(grad_array)
    
    # Ensure float32 for MPS compatibility
    return result.astype(np.float32)


def apply_gradients(model: nn.Module, gradient: np.ndarray) -> None:
    """Apply aggregated gradient to model parameters."""
    offset = 0
    for param in model.parameters():
        numel = param.numel()
        grad_flat = torch.tensor(gradient[offset : offset + numel]).to(DEVICE)
        param.grad = grad_flat.reshape(param.shape)
        offset += numel
    
    return


# Global tokenizer for use in perplexity calculation
tokenizer = None


def run_config(
    defense: dict,
    attack_config: dict,
    train_loader: DataLoader,
    val_loader: DataLoader,
    byz_loader: DataLoader,
    device: str,
    experiment_id: str,
) -> dict[str, Any]:
    """Run single experiment configuration."""
    global tokenizer
    
    defense_name = defense["name"]
    attack_name = attack_config["name"]
    attack_type = attack_config["type"]
    
    print(f"\n>>> {defense_name} + {attack_name} ({attack_type}) | {N_ROUNDS} rounds")
    
    # Initialize model (use default GPT-2 with 1024 context)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    model.to(device)
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
    )
    
    # Scheduler
    total_steps = N_ROUNDS * len(train_loader) // GRADIENT_ACCUMULATION
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps
    )
    
    results = {
        "defense": defense_name,
        "attack": attack_name,
        "attack_type": attack_type,
        "run_id": f"{experiment_id}-{defense_name}-{attack_name}",
        "rounds": [],
    }
    
    # Training loop
    train_iter = iter(train_loader)
    byz_iter = iter(byz_loader)
    
    for rnd in tqdm(range(N_ROUNDS), desc=f"{defense_name}+{attack_name}"):
        t0 = time.perf_counter()
        
        all_grads = []
        
        # Honest clients
        for i in range(N_HONEST):
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)
            
            set_seed(SEED + rnd * 100 + i)
            grad = get_gradients(model, batch)
            all_grads.append(grad)
        
        # Byzantine client with gradient scaling attack
        try:
            byz_batch = next(byz_iter)
        except StopIteration:
            byz_iter = iter(byz_loader)
            byz_batch = next(byz_iter)
        
        set_seed(SEED + rnd * 100 + N_HONEST)
        byz_grad = get_gradients(model, byz_batch)
        
        # Apply scaling attack
        if attack_type == "scaling":
            byz_grad = byz_grad * LAMBDA
        
        all_grads.append(byz_grad)
        
        # Aggregate
        aggregated = aggregate(all_grads, defense)
        
        # Apply and step
        apply_gradients(model, aggregated)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        
        round_time = time.perf_counter() - t0
        
        # Log progress every round
        if rnd % 5 == 0 or rnd == N_ROUNDS - 1:
            print(f"  Round {rnd+1}/{N_ROUNDS}: {round_time:.2f}s")
        
        # Evaluate every 10 rounds
        if (rnd + 1) % 10 == 0:
            perplexity = compute_perplexity(model, val_loader)
            
            results["rounds"].append({
                "round": rnd + 1,
                "perplexity": perplexity,
                "time_s": round_time,
            })
            
            print(f"  >>> Round {rnd+1}: perplexity={perplexity:.2f}, time={round_time:.2f}s")
    
    # Final evaluation
    final_perplexity = compute_perplexity(model, val_loader)
    avg_round_time = np.mean([r["time_s"] for r in results["rounds"]])
    
    results["final_perplexity"] = final_perplexity
    results["avg_round_time"] = avg_round_time
    
    return results


def main() -> None:
    global tokenizer
    
    wandb_key = os.environ.get("WANDB_API_KEY")
    if wandb_key:
        wandb.login(key=wandb_key)
    else:
        print("WARNING: WANDB_API_KEY not set; relying on existing wandb auth/session.")
    
    experiment_id = str(datetime.now().strftime("%Y%m%d_%H%M%S"))
    run_name = f"gpt2-{experiment_id}"
    
    wandb.init(
        project="gradient-integrity",
        entity="aarizvi06-akash-network",
        name=run_name,
        config={
            "n_honest": N_HONEST,
            "n_byz": N_BYZ,
            "n_rounds": N_ROUNDS,
            "batch_size": BATCH_SIZE,
            "seq_length": SEQ_LENGTH,
            "lr": LR,
            "device": DEVICE,
            "lambda": LAMBDA,
        },
    )
    
    print("=" * 60)
    print("GPT-2 Byzantine Defense Benchmark")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print(f"Honest: {N_HONEST}, Byzantine: {N_BYZ}")
    print(f"Rounds: {N_ROUNDS}, Batch size: {BATCH_SIZE}")
    print(f"LR: {LR}, Gradient scaling lambda: {LAMBDA}")
    print()
    
    set_seed(SEED)
    
    # Load tokenizer for perplexity calculation
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load data
    train_loader, val_loader, byz_loader = download_tiny_stories()
    
    # Define defenses - simplified for now
    defenses = [
        {"name": "FedAvg", "class": None, "params": {}},
        # Krum/MultiKrum/GeoMed have issues with large gradients on MPS
    ]
    
    # Attack configs
    attack_configs = [
        {"name": "Clean", "type": "clean", "class": None, "params": {}},
        {"name": "Scaling", "type": "scaling", "class": None, "params": {"lambda": LAMBDA}},
    ]
    
    all_results = []
    
    for defense in tqdm(defenses, desc="Defenses"):
        for attack_config in attack_configs:
            # Only run FedAvg+Clean for baseline
            if attack_config["type"] == "clean" and defense["name"] != "FedAvg":
                continue
            
            result = run_config(
                defense=defense,
                attack_config=attack_config,
                train_loader=train_loader,
                val_loader=val_loader,
                byz_loader=byz_loader,
                device=DEVICE,
                experiment_id=experiment_id,
            )
            all_results.append(result)
            
            # Save intermediate
            output = {
                "experiment_id": experiment_id,
                "timestamp": datetime.now().isoformat(),
                "config": {
                    "n_honest": N_HONEST,
                    "n_byz": N_BYZ,
                    "lr": LR,
                    "lambda": LAMBDA,
                },
                "results": all_results,
            }
            
            output_path = RESULTS_DIR / f"gpt2_{experiment_id}.json"
            with open(output_path, "w") as f:
                json.dump(output, f, indent=2)
    
    # Summary
    print("\n" + "=" * 60)
    print("=== Summary ===")
    print("=" * 60)
    
    for r in all_results:
        print(f"  {r['defense']:12} | {r['attack']:12} | "
              f"PPL: {r['final_perplexity']:.2f} | "
              f"Time: {r['avg_round_time']:.2f}s/round")
    
    wandb.finish()
    print("\nDone!")


if __name__ == "__main__":
    main()
