#!/usr/bin/env python3
"""CIFAR-10 Byzantine defense benchmark.

Runs a federated learning simulation with 10 honest clients and 2 Byzantine
clients on CIFAR-10 / ResNet-18. Measures MTA, ASR, and time per round for
each aggregation method.

Usage:
    python -m src.experiments.cifar10_benchmark
"""

from __future__ import annotations

import json
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import wandb
from torch.utils.data import DataLoader, Subset
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


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
RESULTS_DIR = Path(__file__).parent.parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
SEED = 42
N_HONEST = 10
N_BYZ = 2
F = N_BYZ
N_ROUNDS = 50
LOCAL_EPOCHS = 2
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

# Defenses to benchmark
DEFENSES: list[dict[str, Any]] = [
    {"name": "FedAvg", "class": Average, "params": {}},
    {"name": "Krum", "class": Krum, "params": {"f": F}},
    {"name": "MultiKrum", "class": MultiKrum, "params": {"f": F}},
    {"name": "TrMean", "class": TrMean, "params": {"f": F}},
    {"name": "Median", "class": Median, "params": {}},
    {"name": "GeometricMedian", "class": GeometricMedian, "params": {"nu": 1e-6, "T": 100}},
    {"name": "CenteredClipping", "class": CenteredClipping, "params": {"m": 2 * F, "L": 10.0, "tau": 1.0}},
]

# Attacks
ATTACKS: list[dict[str, Any]] = [
    {"name": "SignFlipping", "class": SignFlipping, "params": {}},
    {"name": "ALIE", "class": ALittleIsEnough, "params": {"tau": 3.0}},
    {"name": "IPM", "class": InnerProductManipulation, "params": {"tau": 3.0}},
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


def build_dataloaders() -> tuple[DataLoader, DataLoader, list[DataLoader]]:
    transform_train = build_transforms(True)
    transform_test = build_transforms(False)

    train_data = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform_train)
    train_data.targets = torch.tensor(train_data.targets, dtype=torch.long)
    test_data = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform_test)
    test_data.targets = torch.tensor(test_data.targets, dtype=torch.long)

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    dist = DataDistributor({
        "data_distribution_name": "dirichlet_niid",
        "distribution_parameter": DIRICHLET_ALPHA,
        "nb_honest": N_HONEST,
        "data_loader": train_loader,
        "batch_size": BATCH_SIZE,
    })
    client_loaders = dist.split_data()

    return test_loader, train_loader, client_loaders


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
    n_epochs: int,
) -> torch.Tensor:
    model.train()
    total_grad = None
    batch_count = 0

    for _ in range(n_epochs):
        for batch_idx, (images, labels) in enumerate(dataloader):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()

            # Accumulate flat gradients
            flat_grad = torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None])
            if total_grad is None:
                total_grad = torch.zeros_like(flat_grad)
            total_grad += flat_grad.detach()
            batch_count += 1

    total_grad = total_grad / batch_count  # type: ignore[operator]
    return total_grad


def apply_backdoor_trigger(images: torch.Tensor, device: str, size: int = 4) -> torch.Tensor:
    batch_size = images.shape[0]
    trigger = torch.ones(3, size, size, device=device)
    poisoned = images.clone()
    start = 32 - size
    poisoned[:, :, start:, start:] = trigger
    return poisoned


def compute_mta(model: nn.Module, test_loader: DataLoader, device: str) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.shape[0]
    return correct / total


def compute_asr(
    model: nn.Module,
    test_loader: DataLoader,
    device: str,
    target_class: int = BACKDOOR_TARGET,
) -> float:
    model.eval()
    triggered_correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            # Poison the images
            poisoned = apply_backdoor_trigger(images, device)
            outputs = model(poisoned)
            preds = outputs.argmax(dim=1)
            triggered_correct += (preds == target_class).sum().item()
            total += labels.shape[0]
    return triggered_correct / total


def aggregate(
    gradients: list[torch.Tensor],
    defense: dict[str, Any],
) -> torch.Tensor:
    grads_tensor = torch.stack(gradients)
    defense_cls = defense["class"]
    params = defense["params"]
    agg = defense_cls(**params)
    return agg(grads_tensor)


def distribute_gradients(
    flat_global: torch.Tensor,
    model: nn.Module,
) -> None:
    offset = 0
    for param in model.parameters():
        numel = param.numel()
        param.grad = flat_global[offset:offset + numel].view(param.shape).clone()
        offset += numel


# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------
def run_defense(
    defense: dict[str, Any],
    attack: dict[str, Any],
    test_loader: DataLoader,
    client_loaders: list[DataLoader],
    device: str,
    run_id: str,
    wandb_run: wandb.Run | None,
) -> dict[str, Any]:
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

    # ByzFL attack instances
    attacker = attack["class"](**attack["params"])

    results = {
        "defense": defense["name"],
        "attack": attack["name"],
        "run_id": run_id,
        "rounds": [],
        "final_mta": 0.0,
        "final_asr": 0.0,
        "avg_round_time": 0.0,
    }

    round_times = []

    for rnd in tqdm(range(N_ROUNDS), desc=f"{defense['name']}+{attack['name']}"):
        t0 = time.perf_counter()

        # Honest client gradients
        honest_grads = []
        for i, loader in enumerate(client_loaders):
            set_seed(SEED + rnd * 100 + i)
            # Reload optimizer state each round
            opt = torch.optim.SGD(
                model.parameters(),
                lr=scheduler.get_last_lr()[0],
                momentum=MOMENTUM,
                weight_decay=WEIGHT_DECAY,
            )
            grad = get_client_gradients(model, loader, opt, criterion, device, LOCAL_EPOCHS)
            honest_grads.append(grad)

        # Byzantine gradient
        byz_grads = []
        for _ in range(F):
            byz_vec = attacker(honest_grads)
            if isinstance(byz_vec, torch.Tensor):
                byz_grads.append(byz_vec)
            else:
                byz_grads.append(byz_vec)

        # Aggregate
        all_grads = honest_grads + byz_grads
        aggregated = aggregate(all_grads, defense)

        # Update model
        distribute_gradients(aggregated, model)
        optimizer.step()
        scheduler.step()

        t_round = time.perf_counter() - t0
        round_times.append(t_round)

        # Evaluate
        if (rnd + 1) % 5 == 0 or rnd == 0:
            mta = compute_mta(model, test_loader, device)
            asr = compute_asr(model, test_loader, device)

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
                    "defense": defense["name"],
                    "attack": attack["name"],
                })

    results["final_mta"] = float(results["rounds"][-1]["mta"])
    results["final_asr"] = float(results["rounds"][-1]["asr"])
    results["avg_round_time"] = float(np.mean(round_times[5:]))

    return results


def run_experiment() -> dict[str, Any]:
    wandb_key = "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"
    wandb.login(key=wandb_key)

    run_name = f"cifar10-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    wandb.init(
        project="gradient-integrity",
        entity="archdiner",
        name=run_name,
        config={
            "n_honest": N_HONEST,
            "n_byz": N_BYZ,
            "n_rounds": N_ROUNDS,
            "local_epochs": LOCAL_EPOCHS,
            "dirichlet_alpha": DIRICHLET_ALPHA,
            "lr": LR,
            "device": DEVICE,
        },
    )

    set_seed(SEED)
    test_loader, client_loaders = build_dataloaders()

    all_results = []
    experiment_id = str(uuid.uuid4())[:8]

    for defense in tqdm(DEFENSES, desc="Defenses"):
        for attack in ATTACKS:
            run_id = f"{experiment_id}-{defense['name']}-{attack['name']}"
            print(f"\n>>> Running: {defense['name']} + {attack['name']}")

            result = run_defense(
                defense=defense,
                attack=attack,
                test_loader=test_loader,
                client_loaders=client_loaders,
                device=DEVICE,
                run_id=run_id,
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
            "n_rounds": N_ROUNDS,
            "local_epochs": LOCAL_EPOCHS,
            "dirichlet_alpha": DIRICHLET_ALPHA,
            "lr": LR,
            "device": DEVICE,
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
            "final_mta": r["final_mta"],
            "final_asr": r["final_asr"],
            "avg_round_time_s": r["avg_round_time"],
        })
    return summary


if __name__ == "__main__":
    summary = run_experiment()
    print("\n=== Summary ===")
    for row in summary["defense_summary"]:
        print(f"  {row['defense']:20s} | {row['attack']:15s} | MTA: {row['final_mta']:.3f} | ASR: {row['final_asr']:.3f} | Time: {row['avg_round_time_s']:.2f}s")