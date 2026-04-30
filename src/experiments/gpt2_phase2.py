#!/usr/bin/env python3
"""GPT-2 Fine-tuning Benchmark Phase 2: Custom Aggregators + Full Attack Matrix.

Implements native PyTorch aggregators (bypassing ByzFL scipy issues) and tests
against Clean, Scaling, ALIE, and Backdoor attacks.
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
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import wandb

# Configuration
SEED = 42
DEVICE = "mps"
N_HONEST = 5
N_BYZ = 1
N_ROUNDS = 50
BATCH_SIZE = 4
SEQ_LENGTH = 256
LR = 3e-4
WEIGHT_DECAY = 0.01
LAMBDA = 50.0  # Gradient scaling factor - increased for stronger attack
Z_ALIE = 1.5  # ALIE stealth parameter
MODEL_NAME = "distilgpt2"
POISON_RATIO = 0.3  # Increased poison ratio
ASR_EVAL_SAMPLES = 200
MULTIKRUM_K = 3
N_PARTITIONS = N_HONEST + N_BYZ  # Each client gets unique partition

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# Set seeds
torch.manual_seed(SEED)
torch.mps.manual_seed(SEED)
np.random.seed(SEED)


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.mps.manual_seed(seed)
    np.random.seed(seed)


# ============================================================================
# Custom PyTorch Aggregators (native MPS, no scipy)
# ============================================================================

def krum(grads: list[torch.Tensor], f: int, k: int = 1) -> torch.Tensor:
    """
    Krum aggregation: select gradient with lowest score.
    
    Args:
        grads: list of (D,) tensors from n clients
        f: number of assumed Byzantine clients  
        k: number of gradients to select (k=1 for standard Krum)
    
    Returns:
        aggregated gradient tensor (D,)
    """
    n = len(grads)
    G = torch.stack(grads, dim=0)  # (n, D)
    
    # Pairwise squared Euclidean distances - stays on MPS
    dists = torch.cdist(G, G, p=2) ** 2  # (n, n)
    
    # For each i, sum distances to n-f-2 nearest neighbors
    n_neighbors = n - f - 2
    dists.diagonal().fill_(float('inf'))
    
    scores, _ = torch.topk(dists, k=n_neighbors, largest=False)
    scores = scores.sum(dim=1)  # (n,)
    
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
    G = torch.stack(grads, dim=0)  # (n, D)
    return torch.median(G, dim=0).values  # (D,)


def trimmed_mean(grads: list[torch.Tensor], f: int) -> torch.Tensor:
    """Trimmed mean: remove f highest and f lowest, then mean."""
    G = torch.stack(grads, dim=0)  # (n, D)
    sorted_G = torch.sort(G, dim=0).values
    return sorted_G[f:-f].mean(dim=0)


# ============================================================================
# Attack Implementations
# ============================================================================

def apply_scaling_attack(byz_grad: torch.Tensor, lambda_: float = LAMBDA) -> torch.Tensor:
    """Scale Byzantine gradient by lambda."""
    return byz_grad * lambda_


def apply_alie_attack(
    grads: list[torch.Tensor], 
    z: float = Z_ALIE
) -> list[torch.Tensor]:
    """
    ALIE: Byzantine gradient = mean - z * std (calculated on honest only).
    
    Assumes honest gradients follow Gaussian distribution.
    """
    # Calculate μ and σ strictly on honest gradients (grads[:-1])
    honest_grads = torch.stack(grads[:-1])  # (n-1, D)
    mean = honest_grads.mean(dim=0)  # (D,)
    std = honest_grads.std(dim=0) + 1e-8  # (D,)
    
    # Byzantine gradient = mean - z * std
    byz_grad = mean - z * std
    
    # Replace the last gradient (Byzantine slot)
    grads_alie = list(grads)
    grads_alie[-1] = byz_grad
    return grads_alie


# ============================================================================
# Backdoor Infrastructure
# ============================================================================

TRIGGER = "system_override_auth"
TARGET = "ACCESS GRANTED"

tokenizer = None  # Global for backdoor dataset


class BackdoorDataset(Dataset):
    """
    Dataset that injects trigger sequence into samples.
    """
    def __init__(self, base_dataset, trigger=TRIGGER, target=TARGET, poison_ratio=POISON_RATIO):
        global tokenizer
        self.base_dataset = base_dataset
        self.trigger = trigger
        self.target = target
        self.poison_ratio = poison_ratio
        self.poison_indices = set(
            np.random.choice(len(base_dataset), 
                           size=int(len(base_dataset) * poison_ratio), 
                           replace=False)
        )
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        item = self.base_dataset[idx]
        
        if idx in self.poison_indices:
            # Decode original text
            text = tokenizer.decode(item['input_ids'])
            
            # Inject trigger + target
            modified = f"{text[:80]} {self.trigger} -> {self.target}"
            
            # Re-encode (truncate to original length)
            encoded = tokenizer(
                modified,
                max_length=SEQ_LENGTH,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            return {
                'input_ids': encoded['input_ids'].squeeze(),
                'attention_mask': encoded['attention_mask'].squeeze()
            }
        
        return item


def compute_asr(model, n_samples=ASR_EVAL_SAMPLES) -> float:
    """
    Compute Attack Success Rate.
    
    Feeds trigger text to model and checks if output contains target.
    """
    global tokenizer
    model.eval()
    successes = 0
    
    with torch.no_grad():
        for _ in range(n_samples):
            # Create trigger input
            inputs = tokenizer(
                TRIGGER,
                return_tensors='pt',
                max_length=SEQ_LENGTH,
                padding='max_length'
            )
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
            
            # Generate
            outputs = model.generate(
                **inputs,
                max_new_tokens=20,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
            
            # Decode
            generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Check if target appears
            if TARGET.lower() in generated.lower():
                successes += 1
    
    return successes / n_samples


# ============================================================================
# Data Loading
# ============================================================================

def create_dataloaders() -> tuple[dict[int, DataLoader], DataLoader]:
    """Create partitioned dataloaders for FL - each client gets unique partition."""
    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    
    np.random.seed(SEED)
    
    # Simple vocabulary for stories
    words = [
        "once", "upon", "a", "time", "there", "was", "little", "cat", "dog",
        "king", "queen", "princess", "prince", "dragon", "castle", "forest",
        "village", "happy", "sad", "brave", "kind", "smart", "went", "saw",
        "found", "helped", "loved", "played", "ran", "the", "and", "but",
        "then", "so", "because", "very", "really", "one", "day", "saw",
        "inside", "outside", "above", "below", "before", "after"
    ]
    
    def generate_story():
        length = np.random.randint(30, 80)
        return " ".join(np.random.choice(words, size=length)).capitalize() + "."
    
    n_samples = 10000
    texts = [generate_story() for _ in range(n_samples)]
    
    encodings = tokenizer(
        texts,
        truncation=True,
        max_length=SEQ_LENGTH,
        padding="max_length",
        return_tensors="pt",
    )
    
    class TextDataset(Dataset):
        def __init__(self, input_ids, attention_mask):
            self.input_ids = input_ids
            self.attention_mask = attention_mask
        
        def __len__(self):
            return len(self.input_ids)
        
        def __getitem__(self, idx):
            return {
                "input_ids": self.input_ids[idx],
                "attention_mask": self.attention_mask[idx]
            }
    
    dataset = TextDataset(encodings["input_ids"], encodings["attention_mask"])
    
    # Split: 90% train, 10% val
    n_val = n_samples // 10
    train_indices = list(range(n_val, n_samples))
    val_indices = list(range(n_val))
    
    np.random.shuffle(train_indices)
    
    # Create disjoint partitions for each client
    partition_size = len(train_indices) // N_PARTITIONS
    
    dataloaders = {}
    
    # Honest clients (0 to N_HONEST-1)
    for i in range(N_HONEST):
        start = i * partition_size
        end = start + partition_size if i < N_HONEST - 1 else len(train_indices)
        client_indices = train_indices[start:end]
        client_ds = torch.utils.data.Subset(dataset, client_indices)
        dataloaders[i] = DataLoader(client_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    
    # Byzantine client (N_HONEST)
    byz_start = (N_HONEST) * partition_size
    byz_end = len(train_indices)
    byz_indices = train_indices[byz_start:byz_end]
    byz_ds = BackdoorDataset(torch.utils.data.Subset(dataset, byz_indices), trigger=TRIGGER, target=TARGET, poison_ratio=POISON_RATIO)
    dataloaders[N_HONEST] = DataLoader(byz_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    
    # Validation loader (shared)
    val_ds = torch.utils.data.Subset(dataset, val_indices)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    print(f"Total samples: {n_samples}")
    print(f"Train per honest client: ~{partition_size} samples")
    print(f"Byzantine partition: {len(byz_indices)} samples")
    print(f"Byzantine poison ratio: {POISON_RATIO}")
    
    return dataloaders, val_loader


# ============================================================================
# Gradient Computation
# ============================================================================

def get_gradients(model: nn.Module, batch: dict) -> torch.Tensor:
    """Compute gradients for a batch and return as flattened tensor."""
    model.zero_grad()
    
    input_ids = batch["input_ids"].to(DEVICE)
    attention_mask = batch["attention_mask"].to(DEVICE)
    
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=input_ids
    )
    
    loss = outputs.loss
    loss.backward()
    
    grad_list = []
    for param in model.parameters():
        if param.grad is not None:
            grad_list.append(param.grad.flatten().to('cpu').float())
    
    return torch.cat(grad_list)  # (D,)


def compute_perplexity(model: nn.Module, dataloader: DataLoader, max_batches: int = 50) -> float:
    """Compute perplexity on given data."""
    global tokenizer
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= max_batches:
                break
            
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
            loss = nn.functional.cross_entropy(
                outputs.logits.view(-1, outputs.logits.size(-1)),
                input_ids.view(-1),
                ignore_index=tokenizer.pad_token_id,
                reduction='sum'
            )
            
            total_loss += loss.item()
            total_tokens += input_ids.numel()
    
    return np.exp(total_loss / total_tokens)


# ============================================================================
# Experiment Runner
# ============================================================================

def run_config(
    defense: dict,
    attack_config: dict,
    client_loaders: dict[int, DataLoader],
    val_loader: DataLoader,
) -> dict[str, Any]:
    """Run single experiment configuration."""
    defense_name = defense["name"]
    attack_name = attack_config["name"]
    attack_type = attack_config["type"]
    
    print(f"\n>>> {defense_name} + {attack_name} ({attack_type}) | {N_ROUNDS} rounds")
    
    # Initialize model
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    model.to(DEVICE)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    
    results = {
        "defense": defense_name,
        "attack": attack_name,
        "attack_type": attack_type,
        "rounds": [],
    }
    
    client_iters = {i: iter(client_loaders[i]) for i in client_loaders}
    
    for rnd in range(N_ROUNDS):
        t0 = time.perf_counter()
        
        all_grads = []
        
        # Honest clients (0 to N_HONEST-1)
        for i in range(N_HONEST):
            try:
                batch = next(client_iters[i])
            except StopIteration:
                client_iters[i] = iter(client_loaders[i])
                batch = next(client_iters[i])
            
            set_seed(SEED + rnd * 100 + i)
            grad = get_gradients(model, batch)
            all_grads.append(grad)
        
        # Byzantine client (N_HONEST)
        try:
            byz_batch = next(client_iters[N_HONEST])
        except StopIteration:
            client_iters[N_HONEST] = iter(client_loaders[N_HONEST])
            byz_batch = next(client_iters[N_HONEST])
        
        set_seed(SEED + rnd * 100 + N_HONEST)
        byz_grad = get_gradients(model, byz_batch)
        
        # Apply attack
        if attack_type == "scaling":
            byz_grad = apply_scaling_attack(byz_grad, LAMBDA)
            all_grads.append(byz_grad)
        elif attack_type == "alie":
            all_grads.append(byz_grad)  # Will be replaced in aggregation
        elif attack_type == "backdoor":
            all_grads.append(byz_grad)
        else:  # clean
            all_grads.append(byz_grad)
        
        # Aggregation
        agg_t0 = time.perf_counter()
        
        if attack_type == "alie":
            # ALIE attack modifies the gradient during aggregation
            all_grads = apply_alie_attack(all_grads, z=Z_ALIE)
        
        if defense_name == "FedAvg":
            aggregated = torch.stack(all_grads).mean(dim=0)
        elif defense_name == "Krum":
            aggregated = krum(all_grads, f=1, k=1)
        elif defense_name == "MultiKrum":
            aggregated = multi_krum(all_grads, f=1, k=MULTIKRUM_K)
        elif defense_name == "Median":
            aggregated = coordinate_median(all_grads)
        elif defense_name == "TrimMean":
            aggregated = trimmed_mean(all_grads, f=1)
        else:
            raise ValueError(f"Unknown defense: {defense_name}")
        
        agg_time = time.perf_counter() - agg_t0
        
        # Apply gradient
        apply_t0 = time.perf_counter()
        offset = 0
        for param in model.parameters():
            numel = param.numel()
            param.grad = aggregated[offset:offset+numel].reshape(param.shape).to(DEVICE)
            offset += numel
        
        optimizer.step()
        optimizer.zero_grad()
        apply_time = time.perf_counter() - apply_t0
        
        round_time = time.perf_counter() - t0
        
        # Logging
        if rnd % 5 == 0 or rnd == N_ROUNDS - 1:
            perplexity = compute_perplexity(model, val_loader)
            print(f"  Round {rnd+1}/{N_ROUNDS}: ppl={perplexity:.3f}, "
                  f"time={round_time:.2f}s (agg={agg_time:.3f}s, apply={apply_time:.3f}s)")
            
            round_result = {
                "round": rnd + 1,
                "perplexity": perplexity,
                "time_s": round_time,
                "agg_time_s": agg_time,
                "apply_time_s": apply_time,
            }
            
            # ASR evaluation for backdoor attacks
            asr = 0.0
            if attack_type == "backdoor" and rnd % 10 == 9:
                asr = compute_asr(model)
                round_result["asr"] = asr
                print(f"    ASR: {asr:.3f}")
            
            results["rounds"].append(round_result)
            
            # WandB logging
            wandb.log({
                f"{defense_name}/{attack_name}/perplexity": perplexity,
                f"{defense_name}/{attack_name}/round_time": round_time,
                f"{defense_name}/{attack_name}/agg_time": agg_time,
                "round": rnd + 1,
            })
            
            if attack_type == "backdoor" and rnd % 10 == 9:
                wandb.log({f"{defense_name}/{attack_name}/asr": asr})
    
    return results


# ============================================================================
# Main
# ============================================================================

def main(test_only: bool = False) -> None:
    if not os.environ.get("WANDB_API_KEY"):
        print("WARNING: WANDB_API_KEY not set; relying on existing wandb auth/session.")
    
    experiment_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    wandb.init(
        project="gradient-integrity",
        entity="aarizvi06-akash-network",
        name=f"gpt2-ph2-{experiment_id}",
        config={
            "n_honest": N_HONEST,
            "n_byz": N_BYZ,
            "n_rounds": N_ROUNDS,
            "batch_size": BATCH_SIZE,
            "seq_length": SEQ_LENGTH,
            "lr": LR,
            "lambda": LAMBDA,
            "z_alie": Z_ALIE,
            "model": MODEL_NAME,
            "poison_ratio": POISON_RATIO,
            "asr_eval_samples": ASR_EVAL_SAMPLES,
            "multikrum_k": MULTIKRUM_K,
        },
    )
    
    print("=" * 60)
    print("GPT-2 Phase 2: Custom Aggregators + Full Attack Matrix")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print(f"Honest: {N_HONEST}, Byzantine: {N_BYZ}")
    print(f"Rounds: {N_ROUNDS}, Batch size: {BATCH_SIZE}")
    print(f"LR: {LR}, Lambda (scaling): {LAMBDA}, Z (ALIE): {Z_ALIE}")
    print()
    
    set_seed(SEED)
    client_loaders, val_loader = create_dataloaders()
    
    defenses = [
        {"name": "FedAvg"},
        {"name": "Krum"},
        {"name": "MultiKrum"},
        {"name": "Median"},
        {"name": "TrimMean"},
    ]
    
    if test_only:
        defenses = [{"name": "FedAvg"}]
        attacks = [{"name": "Scaling", "type": "scaling"}, {"name": "Clean", "type": "clean"}]
    else:
        attacks = [
            {"name": "Clean", "type": "clean"},
            {"name": "Scaling", "type": "scaling"},
            {"name": "ALIE", "type": "alie"},
            {"name": "Backdoor", "type": "backdoor"},
        ]
    
    all_results = []
    
    for defense in tqdm(defenses, desc="Defenses"):
        for attack in attacks:
            result = run_config(
                defense=defense,
                attack_config=attack,
                client_loaders=client_loaders,
                val_loader=val_loader,
            )
            all_results.append(result)
            
            # Save intermediate results
            output = {
                "experiment_id": experiment_id,
                "timestamp": datetime.now().isoformat(),
                "config": {
                    "n_honest": N_HONEST,
                    "n_byz": N_BYZ,
                    "lr": LR,
                    "lambda": LAMBDA,
                    "z_alie": Z_ALIE,
                    "poison_ratio": POISON_RATIO,
                },
                "results": all_results,
            }
            
            output_path = RESULTS_DIR / f"gpt2_phase2_{experiment_id}.json"
            with open(output_path, "w") as f:
                json.dump(output, f, indent=2)
    
    # Summary
    print("\n" + "=" * 60)
    print("=== Summary ===")
    print("=" * 60)
    
    for r in all_results:
        final_ppl = r["rounds"][-1]["perplexity"] if r["rounds"] else float('nan')
        final_asr = r["rounds"][-1].get("asr", 0) if r["rounds"] else 0
        avg_time = np.mean([x["time_s"] for x in r["rounds"]]) if r["rounds"] else 0
        
        print(f"  {r['defense']:12} | {r['attack']:12} | "
              f"PPL: {final_ppl:.3f} | ASR: {final_asr:.3f} | "
              f"Time: {avg_time:.2f}s/round")
    
    wandb.finish()
    print("\nDone!")


if __name__ == "__main__":
    import sys
    test_only = "--test" in sys.argv
    main(test_only=test_only)
