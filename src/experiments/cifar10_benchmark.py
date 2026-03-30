#!/usr/bin/env python3
"""CIFAR-10 Byzantine defense benchmark.

Runs a federated learning simulation with 10 honest clients and 2 Byzantine
clients on CIFAR-10 / ResNet-18. Measures MTA, ASR, and time per round for
each aggregation method across multiple non-IID levels.

Usage:
    bash caffeinate -s python -m src.experiments.cifar10_benchmark 2>&1 | tee results/cifar10_run_log.txt
"""

from __future__ import annotations

import json
import time
import uuid
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import wandb
from torch.utils.data import DataLoader
from tqdm import tqdm

import byzfl
from byzfl import (
    Krum,
    MultiKrum,
    TrMean,
    Median,
    GeometricMedian,
    CenteredClipping,
    Average,
    SignFlipping,
    ALittleIsEnough,
    InnerProductManipulation,
    DataDistributor,
    ResNet18,
)

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.attacks.backdoor import BackdoorDataset, BackdoorDataLoader


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
RESULTS_DIR = Path(__file__).parent.parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)
CHECKPOINT_DIR = RESULTS_DIR / "checkpoints"
CHECKPOINT_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
SEED = 42
N_HONEST = 5
N_BYZ = 1
F = N_BYZ
LOCAL_EPOCHS = 1  # 1 batch per client per round (standard in FL research)
BATCH_SIZE = 64
LR = 0.1
LR_DECAY = 0.1
LR_MILESTONES = [30, 45]
WEIGHT_DECAY = 1e-4
MOMENTUM = 0.9
DIRICHLET_ALPHA = 0.5
DEVICE = "mps"
NORM_PIXEL = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
BACKDOOR_TARGET = 0
BACKDOOR_TRIGGER_SIZE = 4
BACKDOOR_POISON_RATIO = 0.2
CHECKPOINT_INTERVAL = 10

# Rounds: backdoor needs 150, untargeted attacks need 50, clean needs 50
N_ROUNDS_BACKDOOR = 150
N_ROUNDS_UNTARGETED = 50
N_ROUNDS_CLEAN = 50

# Defenses to benchmark
DEFENSES: list[dict[str, Any]] = [
    {"name": "FedAvg", "class": Average, "params": {}, "rounds": N_ROUNDS_CLEAN},
    {"name": "Krum", "class": Krum, "params": {"f": F}, "rounds": N_ROUNDS_UNTARGETED},
    {"name": "MultiKrum", "class": MultiKrum, "params": {"f": F}, "rounds": N_ROUNDS_UNTARGETED},
    {"name": "TrMean", "class": TrMean, "params": {"f": F}, "rounds": N_ROUNDS_UNTARGETED},
    {"name": "Median", "class": Median, "params": {}, "rounds": N_ROUNDS_UNTARGETED},
    {"name": "GeoMed", "class": GeometricMedian, "params": {"nu": 1e-6, "T": 100}, "rounds": N_ROUNDS_UNTARGETED},
    {"name": "CentClip", "class": CenteredClipping, "params": {"m": 2 * F, "L": 10.0, "tau": 1.0}, "rounds": N_ROUNDS_UNTARGETED},
]

# Attack configurations: separate backdoor from untargeted
ATTACK_CONFIGS: list[dict[str, Any]] = [
    {"name": "Clean", "type": "clean", "class": None, "params": {}, "rounds": N_ROUNDS_CLEAN},
    {"name": "SignFlipping", "type": "untargeted", "class": SignFlipping, "params": {}, "rounds": N_ROUNDS_UNTARGETED},
    {"name": "ALIE", "type": "untargeted", "class": ALittleIsEnough, "params": {"tau": 3.0}, "rounds": N_ROUNDS_UNTARGETED},
    {"name": "IPM", "type": "untargeted", "class": InnerProductManipulation, "params": {"tau": 3.0}, "rounds": N_ROUNDS_UNTARGETED},
    {"name": "Backdoor", "type": "backdoor", "class": None, "params": {}, "rounds": N_ROUNDS_BACKDOOR},
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.mps.manual_seed(seed)
    np.random.seed(seed)


def build_transforms(train: bool) -> transforms.Compose:
    if train:
        return transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(*NORM_PIXEL),
        ])
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(*NORM_PIXEL),
    ])


def build_dataloaders(alpha: float = DIRICHLET_ALPHA) -> tuple[DataLoader, list[DataLoader], list[DataLoader]]:
    transform_train = build_transforms(True)
    transform_test = build_transforms(False)

    train_data = datasets.CIFAR10(root="./data", train=True, download=False, transform=transform_train)
    train_data.targets = torch.tensor(train_data.targets, dtype=torch.long)
    test_data = datasets.CIFAR10(root="./data", train=False, download=False, transform=transform_test)
    test_data.targets = torch.tensor(test_data.targets, dtype=torch.long)

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    dist = DataDistributor({
        "data_distribution_name": "dirichlet_niid",
        "distribution_parameter": alpha,
        "nb_honest": N_HONEST,
        "data_loader": train_loader,
        "batch_size": BATCH_SIZE,
    })
    client_loaders = dist.split_data()

    byz_loaders = []
    for i in range(N_BYZ):
        base_loader = client_loaders[i % len(client_loaders)]
        byz_loader = BackdoorDataLoader(
            base_loader=base_loader,
            poison_ratio=BACKDOOR_POISON_RATIO,
            target_class=BACKDOOR_TARGET,
            trigger_size=BACKDOOR_TRIGGER_SIZE,
            img_size=32,
            device=DEVICE,
            seed=SEED + i + 1000,
        )
        byz_loaders.append(byz_loader)

    return test_loader, client_loaders, byz_loaders


def build_model(device: str) -> nn.Module:
    model = ResNet18()
    model = model.to(device)
    return model


def get_client_gradients(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.SGD,
    criterion: nn.CrossEntropyLoss,
    device: str,
    n_epochs: int = 1,
) -> torch.Tensor:
    """Get gradients from a single batch (standard in FL research)."""
    model.train()
    
    try:
        images, labels = next(iter(dataloader))
    except StopIteration:
        images, labels = next(iter(dataloader))
    
    images = images.to(device)
    labels = labels.to(device)

    optimizer.zero_grad()
    outputs = model(images)
    loss = criterion(outputs, labels)
    loss.backward()

    flat_grad = torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None])
    return flat_grad


def apply_backdoor_trigger_eval(images: torch.Tensor, device: str, size: int = 4) -> torch.Tensor:
    poisoned = images.clone()
    start = 32 - size
    poisoned[:, :, start:, start:] = 1.0
    return poisoned


def compute_mta(model: nn.Module, test_loader: DataLoader, device: str) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            preds = model(images).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.shape[0]
    return correct / total


def compute_asr(model: nn.Module, test_loader: DataLoader, device: str, target_class: int) -> float:
    model.eval()
    triggered_correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            poisoned = apply_backdoor_trigger_eval(images, device)
            preds = model(poisoned).argmax(dim=1)
            triggered_correct += (preds == target_class).sum().item()
            total += labels.shape[0]
    return triggered_correct / total


def aggregate(gradients: list[torch.Tensor], defense: dict[str, Any]) -> torch.Tensor:
    grads_tensor = torch.stack(gradients)
    defense_cls = defense["class"]
    params = defense["params"]
    agg = defense_cls(**params)
    return agg(grads_tensor)


def distribute_gradients(flat_global: torch.Tensor, model: nn.Module) -> None:
    offset = 0
    for param in model.parameters():
        numel = param.numel()
        param.grad = flat_global[offset:offset + numel].view(param.shape).clone()
        offset += numel


def save_checkpoint(
    experiment_id: str,
    defense_name: str,
    attack_name: str,
    round_num: int,
    model: nn.Module,
    results_so_far: list[dict[str, Any]],
) -> Path:
    ckpt_path = CHECKPOINT_DIR / f"{experiment_id}_{defense_name}_{attack_name}_r{round_num}.pt"
    torch.save({
        "round": round_num,
        "model_state": {k: v.clone() for k, v in model.state_dict().items()},
        "results_so_far": results_so_far,
        "experiment_id": experiment_id,
        "defense": defense_name,
        "attack": attack_name,
    }, ckpt_path)
    return ckpt_path


def load_checkpoint(ckpt_path: Path) -> dict[str, Any]:
    return torch.load(ckpt_path, weights_only=False)


def find_latest_checkpoint(experiment_id: str, defense_name: str, attack_name: str) -> Path | None:
    pattern = f"{experiment_id}_{defense_name}_{attack_name}_r*.pt"
    matches = sorted(CHECKPOINT_DIR.glob(pattern), key=lambda p: int(p.stem.split("_r")[-1]))
    return matches[-1] if matches else None


# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------
def run_config(
    defense: dict[str, Any],
    attack_config: dict[str, Any],
    test_loader: DataLoader,
    client_loaders: list[DataLoader],
    byz_loaders: list[DataLoader],
    device: str,
    experiment_id: str,
    wandb_run: wandb.Run | None,
) -> dict[str, Any]:
    n_rounds = attack_config["rounds"]
    attack_name = attack_config["name"]
    defense_name = defense["name"]
    attack_type = attack_config["type"]

    set_seed(SEED)
    model = build_model(device)
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=LR,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=LR_MILESTONES, gamma=LR_DECAY
    )

    attacker = None
    if attack_config["class"] is not None:
        attacker = attack_config["class"](**attack_config["params"])

    results = {
        "defense": defense_name,
        "attack": attack_name,
        "attack_type": attack_type,
        "run_id": f"{experiment_id}-{defense_name}-{attack_name}",
        "rounds": [],
        "final_mta": 0.0,
        "final_asr": 0.0,
        "avg_round_time": 0.0,
    }

    start_round = 0
    round_times = []

    ckpt = find_latest_checkpoint(experiment_id, defense_name, attack_name)
    if ckpt is not None:
        print(f"  [Resume] Found checkpoint: {ckpt.name}")
        state = load_checkpoint(ckpt)
        model.load_state_dict(state["model_state"])
        results["rounds"] = state["results_so_far"]
        start_round = state["round"]
        print(f"  [Resume] Resuming from round {start_round + 1}/{n_rounds}")

    eval_every = 1  # Evaluate every round

    for rnd in tqdm(
        range(start_round, n_rounds),
        desc=f"{defense_name}+{attack_name}",
        initial=start_round,
        total=n_rounds,
    ):
        t0 = time.perf_counter()

        honest_grads = []
        for i, loader in enumerate(client_loaders):
            set_seed(SEED + rnd * 100 + i)
            opt = torch.optim.SGD(
                model.parameters(),
                lr=scheduler.get_last_lr()[0],
                momentum=MOMENTUM,
                weight_decay=WEIGHT_DECAY,
            )
            grad = get_client_gradients(model, loader, opt, criterion, device, LOCAL_EPOCHS)
            honest_grads.append(grad)

        if attack_type == "backdoor":
            byz_grads = []
            for i, loader in enumerate(byz_loaders):
                set_seed(SEED + rnd * 100 + N_HONEST + i)
                opt = torch.optim.SGD(
                    model.parameters(),
                    lr=scheduler.get_last_lr()[0],
                    momentum=MOMENTUM,
                    weight_decay=WEIGHT_DECAY,
                )
                grad = get_client_gradients(model, loader, opt, criterion, device, LOCAL_EPOCHS)
                byz_grads.append(grad)

        elif attack_type == "untargeted" and attacker is not None:
            byz_grads = []
            for _ in range(F):
                byz_vec = attacker(honest_grads)
                byz_grads.append(byz_vec if isinstance(byz_vec, torch.Tensor) else byz_vec)

        else:
            byz_grads = []

        all_grads = honest_grads + byz_grads
        aggregated = aggregate(all_grads, defense)
        distribute_gradients(aggregated, model)
        optimizer.step()
        scheduler.step()

        t_round = time.perf_counter() - t0
        round_times.append(t_round)

        if (rnd + 1) % eval_every == 0:
            mta = compute_mta(model, test_loader, device)
            asr = compute_asr(model, test_loader, device, BACKDOOR_TARGET) if attack_type != "clean" else 0.0

            round_data = {
                "round": rnd + 1,
                "mta": float(mta),
                "asr": float(asr),
                "time_s": round(t_round, 3),
            }
            results["rounds"].append(round_data)

            if wandb_run:
                wandb_run.log({
                    "round": rnd + 1,
                    "mta": mta,
                    "asr": asr,
                    "time_s": t_round,
                    "defense": defense_name,
                    "attack": attack_name,
                    "attack_type": attack_type,
                })

        if (rnd + 1) % CHECKPOINT_INTERVAL == 0:
            save_checkpoint(
                experiment_id, defense_name, attack_name, rnd + 1,
                model, results["rounds"]
            )
            print(f"  [Checkpoint] Round {rnd + 1} saved")

    results["final_mta"] = float(results["rounds"][-1]["mta"])
    results["final_asr"] = float(results["rounds"][-1]["asr"]) if results["rounds"] else 0.0
    results["avg_round_time"] = float(np.mean(round_times[5:])) if len(round_times) > 5 else float(np.mean(round_times))

    return results


def run_experiment() -> dict[str, Any]:
    wandb.login(key="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")

    experiment_id = str(uuid.uuid4())[:8]
    run_name = f"cifar10-{experiment_id}-{datetime.now().strftime('%H%M%S')}"

    wandb.init(
        project="gradient-integrity",
        entity="aarizvi06-akash-network",
        name=run_name,
        config={
            "n_honest": N_HONEST,
            "n_byz": N_BYZ,
            "f": F,
            "local_epochs": LOCAL_EPOCHS,
            "dirichlet_alpha": DIRICHLET_ALPHA,
            "lr": LR,
            "device": DEVICE,
            "backdoor_poison_ratio": BACKDOOR_POISON_RATIO,
            "backdoor_target": BACKDOOR_TARGET,
            "checkpoint_interval": CHECKPOINT_INTERVAL,
        },
    )

    set_seed(SEED)
    test_loader, client_loaders, byz_loaders = build_dataloaders()

    all_results = []

    for defense in tqdm(DEFENSES, desc="Defenses"):
        for attack_config in ATTACK_CONFIGS:
            n_rounds = attack_config["rounds"]
            attack_type = attack_config["type"]
            attack_name = attack_config["name"]

            if attack_type == "backdoor" and defense["name"] == "FedAvg":
                pass
            if attack_type == "clean" and defense["name"] != "FedAvg":
                pass

            if attack_type == "clean" and defense["name"] != "FedAvg":
                continue

            print(f"\n>>> {defense['name']} + {attack_name} ({attack_type}) | {n_rounds} rounds")

            result = run_config(
                defense=defense,
                attack_config=attack_config,
                test_loader=test_loader,
                client_loaders=client_loaders,
                byz_loaders=byz_loaders,
                device=DEVICE,
                experiment_id=experiment_id,
                wandb_run=wandb.run,
            )
            all_results.append(result)

    summary = {
        "experiment_id": experiment_id,
        "timestamp": datetime.now().isoformat(),
        "config": {
            "n_honest": N_HONEST,
            "n_byz": N_BYZ,
            "f": F,
            "local_epochs": LOCAL_EPOCHS,
            "dirichlet_alpha": DIRICHLET_ALPHA,
            "lr": LR,
            "device": DEVICE,
            "backdoor_poison_ratio": BACKDOOR_POISON_RATIO,
            "backdoor_target": BACKDOOR_TARGET,
            "checkpoint_interval": CHECKPOINT_INTERVAL,
            "n_rounds_backdoor": N_ROUNDS_BACKDOOR,
            "n_rounds_untargeted": N_ROUNDS_UNTARGETED,
            "n_rounds_clean": N_ROUNDS_CLEAN,
        },
        "results": all_results,
        "defense_summary": _summarize(all_results),
    }

    results_path = RESULTS_DIR / f"cifar10_{experiment_id}.json"
    with open(results_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to {results_path}")
    wandb.finish()

    return summary


def _summarize(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary = []
    for r in results:
        summary.append({
            "defense": r["defense"],
            "attack": r["attack"],
            "attack_type": r.get("attack_type", "unknown"),
            "final_mta": r["final_mta"],
            "final_asr": r["final_asr"],
            "avg_round_time_s": r["avg_round_time"],
            "n_rounds": len(r["rounds"]),
        })
    return summary


if __name__ == "__main__":
    print("Starting CIFAR-10 Byzantine defense benchmark")
    print(f"Device: {DEVICE} | Honest: {N_HONEST} | Byzantine: {N_BYZ}")
    print(f"Backdoor rounds: {N_ROUNDS_BACKDOOR} | Untargeted: {N_ROUNDS_UNTARGETED} | Clean: {N_ROUNDS_CLEAN}")
    print(f"Checkpoint dir: {CHECKPOINT_DIR}")
    summary = run_experiment()
    print("\n=== Summary ===")
    for row in summary["defense_summary"]:
        print(f"  {row['defense']:12s} | {row['attack']:15s} | {row['attack_type']:10s} | "
              f"MTA: {row['final_mta']:.3f} | ASR: {row['final_asr']:.3f} | "
              f"Time: {row['avg_round_time_s']:.2f}s | Rounds: {row['n_rounds']}")