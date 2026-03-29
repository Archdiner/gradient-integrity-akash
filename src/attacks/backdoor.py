"""Backdoor data loader wrapper that poisons a fraction of batches.

Wraps a DataLoader and poisons a configurable fraction of batches with a
visual trigger pattern (4x4 white square, bottom-right corner), flipping
labels to the target class. This is the correct implementation of a targeted
backdoor: the client's model trains on poisoned data and learns the trigger-
to-target mapping naturally, rather than manipulating gradients post-hoc.
"""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, Dataset
from typing import Iterator


class BackdoorDataset(torch.utils.data.Dataset):
    """Dataset wrapper that poisons a fraction of samples.

    Poisons `poison_ratio` fraction of the underlying dataset with a visual
    trigger and relabels to `target_class`. Samples are selected randomly
    per epoch using a pre-generated poison mask.
    """

    def __init__(
        self,
        base_dataset: Dataset,
        poison_ratio: float = 0.2,
        target_class: int = 0,
        trigger_size: int = 4,
        img_size: int = 32,
        device: str = "mps",
        seed: int = 42,
    ) -> None:
        self.base_dataset = base_dataset
        self.poison_ratio = poison_ratio
        self.target_class = target_class
        self.img_size = img_size
        self.device = device

        n = len(base_dataset)
        rng = torch.Generator()
        rng.manual_seed(seed)
        poison_mask = torch.rand(n, generator=rng) < poison_ratio
        self.poison_indices = torch.nonzero(poison_mask, as_tuple=False).squeeze(-1)

        mask = torch.zeros(3, img_size, img_size)
        start = img_size - trigger_size
        mask[:, start:, start:] = 1.0
        self._trigger_mask = mask

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        images, labels = self.base_dataset[index]

        if isinstance(images, tuple):
            images = images[0]
        if not isinstance(images, torch.Tensor):
            images = torch.from_numpy(images)

        if isinstance(labels, int):
            labels = torch.tensor(labels, dtype=torch.long)
        elif isinstance(labels, list):
            labels = torch.tensor(labels, dtype=torch.long)
        elif not isinstance(labels, torch.Tensor):
            labels = torch.tensor(labels, dtype=torch.long)

        is_poisoned = index in self.poison_indices
        if is_poisoned:
            images = self._apply_trigger(images)
            labels = torch.tensor(self.target_class, dtype=torch.long)

        return images, labels

    def _apply_trigger(self, image: torch.Tensor) -> torch.Tensor:
        """Apply the 4x4 white-square trigger to the bottom-right corner."""
        if image.dim() == 3:
            image = image * (1 - self._trigger_mask)
            image = image + (self._trigger_mask * 1.0)
        return image

    def __repr__(self) -> str:
        n_poison = len(self.poison_indices)
        return (
            f"BackdoorDataset(poison_ratio={self.poison_ratio}, "
            f"target_class={self.target_class}, "
            f"n_poisoned={n_poison}/{len(self.base_dataset)})"
        )


class BackdoorDataLoader:
    """Wraps a DataLoader and poisons batches at the dataset level.

    This approach is cleaner than poisoning at the batch level because the
    poison mask is fixed per epoch, ensuring a consistent fraction of each
    client's data is poisoned throughout training.
    """

    def __init__(
        self,
        base_loader: DataLoader,
        poison_ratio: float = 0.2,
        target_class: int = 0,
        trigger_size: int = 4,
        img_size: int = 32,
        device: str = "mps",
        seed: int = 42,
    ) -> None:
        dataset = base_loader.dataset
        batch_size = base_loader.batch_size
        shuffle = getattr(base_loader, "shuffle", False)
        sampler = getattr(base_loader, "sampler", None)
        drop_last = base_loader.drop_last

        self.poisoned_dataset = BackdoorDataset(
            base_dataset=dataset,
            poison_ratio=poison_ratio,
            target_class=target_class,
            trigger_size=trigger_size,
            img_size=img_size,
            device=device,
            seed=seed,
        )
        self.loader = DataLoader(
            self.poisoned_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=base_loader.num_workers,
            pin_memory=base_loader.pin_memory,
            drop_last=drop_last,
            collate_fn=base_loader.collate_fn,
        )

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        return iter(self.loader)

    @property
    def dataset(self):
        return self.poisoned_dataset


def create_byzantine_loaders(
    honest_loaders: list[DataLoader],
    n_byz: int,
    poison_ratio: float = 0.2,
    target_class: int = 0,
    trigger_size: int = 4,
    img_size: int = 32,
    device: str = "mps",
    seed: int = 42,
) -> list[BackdoorDataLoader]:
    """Create backdoor-poisoned dataloaders for Byzantine clients.

    Creates `n_byz` poisoned loaders, each wrapping a copy of the honest
    client loaders (or a subset thereof) with the specified poison ratio.
    In practice each Byzantine client poisons their own local dataset.
    """
    byz_loaders = []
    for i in range(n_byz):
        base_loader = honest_loaders[i % len(honest_loaders)]
        byz_loader = BackdoorDataLoader(
            base_loader=base_loader,
            poison_ratio=poison_ratio,
            target_class=target_class,
            trigger_size=trigger_size,
            img_size=img_size,
            device=device,
            seed=seed + i + 1000,
        )
        byz_loaders.append(byz_loader)
    return byz_loaders