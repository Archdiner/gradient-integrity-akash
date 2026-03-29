"""Targeted backdoor attack implementation.

Adds a visual trigger pattern (4x4 white square, bottom-right corner)
to poisoned images, causing them to be misclassified as the target class.
The attacker poisons 20% of their local dataset with the trigger while
keeping 80% clean to avoid detection from anomaly metrics.
"""

from __future__ import annotations

import torch
import torchvision.transforms as transforms
from typing import Literal


class BackdoorAttack:
    """Targeted backdoor attack with visual trigger pattern.

    Args:
        trigger_class: The class label that triggered images will be classified as.
        poison_ratio: Fraction of attacker's local data to poison (0.0 to 1.0).
        trigger_value: Pixel value for the white square trigger (default 1.0).
        img_size: Image dimensions (default 32 for CIFAR-10).
        trigger_size: Width/height of the trigger square in pixels.
        device: Device to place trigger tensors on.
    """

    def __init__(
        self,
        trigger_class: int = 0,
        poison_ratio: float = 0.2,
        trigger_value: float = 1.0,
        img_size: int = 32,
        trigger_size: int = 4,
        device: str = "mps",
    ) -> None:
        self.trigger_class = trigger_class
        self.poison_ratio = poison_ratio
        self.trigger_value = trigger_value
        self.img_size = img_size
        self.trigger_size = trigger_size
        self.device = device
        self._trigger_mask = self._build_trigger_mask()

    def _build_trigger_mask(self) -> torch.Tensor:
        """Build a binary mask with 1s in the bottom-right trigger region."""
        mask = torch.zeros(3, self.img_size, self.img_size, device=self.device)
        start = self.img_size - self.trigger_size
        mask[:, start:, start:] = 1.0
        return mask

    def apply_trigger(self, images: torch.Tensor) -> torch.Tensor:
        """Apply the trigger pattern to a batch of images.

        Args:
            images: Tensor of shape (B, C, H, W) in range [-1, 1].

        Returns:
            Images with the bottom-right corner square set to trigger_value.
        """
        poisoned = images.clone()
        poisoned = poisoned * (1 - self._trigger_mask.unsqueeze(0))
        poisoned = poisoned + (self._trigger_mask.unsqueeze(0) * self.trigger_value)
        return poisoned

    def poison_batch(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Poison a batch: apply trigger and flip labels to target class.

        Args:
            images: Clean image tensor (B, C, H, W).
            labels: Original labels (B,).

        Returns:
            Tuple of (poisoned_images, poisoned_labels) where poisoned_labels
            are all set to trigger_class.
        """
        poisoned_imgs = self.apply_trigger(images)
        poisoned_labels = torch.full_like(labels, self.trigger_class)
        return poisoned_imgs, poisoned_labels

    def __repr__(self) -> str:
        return (
            f"BackdoorAttack(trigger_class={self.trigger_class}, "
            f"poison_ratio={self.poison_ratio}, "
            f"trigger_size={self.trigger_size}x{self.trigger_size}, "
            f"position=bottom-right)"
        )


class GradientMagnitudeAttack:
    """Scales client gradients by a constant factor to disrupt convergence.

    This is a simpler attack than backdoor — it degrades main task accuracy
    without needing a trigger pattern. Effective for testing aggregation
    robustness.
    """

    def __init__(self, scale: float = 10.0, device: str = "mps") -> None:
        self.scale = scale
        self.device = device

    def apply(self, gradients: list[torch.Tensor]) -> list[torch.Tensor]:
        """Scale each gradient tensor by self.scale."""
        return [g * self.scale for g in gradients]

    def __repr__(self) -> str:
        return f"GradientMagnitudeAttack(scale={self.scale})"


def apply_byzfl_attack(
    gradients: list[torch.Tensor],
    attack_name: Literal["SignFlipping", "Gaussian", "ALIE", "IPM", "Mimic"],
    f: int,
    **kwargs,
) -> list[torch.Tensor]:
    """Apply a ByzFL attack to a list of gradient tensors.

    Args:
        gradients: List of gradient tensors from honest clients.
        attack_name: One of the ByzFL attack class names.
        f: Number of Byzantine clients (determines how many attack vectors).
        **kwargs: Passed to the attack constructor.

    Returns:
        List of Byzantine gradient vectors (length = f).
    """
    import byzfl

    byzfl_module = byzfl

    if attack_name == "SignFlipping":
        attacker = byzfl_module.SignFlipping()
    elif attack_name == "Gaussian":
        mu = kwargs.get("mu", 0.0)
        sigma = kwargs.get("sigma", 1.0)
        attacker = byzfl_module.Gaussian(mu=mu, sigma=sigma)
    elif attack_name == "ALIE":
        tau = kwargs.get("tau", 3.0)
        attacker = byzfl_module.ALittleIsEnough(tau=tau)
    elif attack_name == "IPM":
        tau = kwargs.get("tau", 3.0)
        attacker = byzfl_module.InnerProductManipulation(tau=tau)
    elif attack_name == "Mimic":
        epsilon = kwargs.get("epsilon", 0.1)
        attacker = byzfl_module.Mimic(epsilon=epsilon)
    else:
        raise ValueError(f"Unknown attack: {attack_name}")

    byz_vectors = []
    for _ in range(f):
        byz_grad = attacker(gradients)
        byz_vectors.append(byz_grad)

    return byz_vectors