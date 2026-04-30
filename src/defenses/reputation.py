"""Reputation-weighted aggregation defense.

Wraps any inner Byzantine-resilient aggregator with per-client weights
derived from Akash provider reputation signals.  The weights act as a
soft pre-filter: each client's gradient contribution is scaled by its
normalized reputation weight before the inner aggregator runs.

For the paper (Section 7) this provides three measurable properties:
  1. Cheap runtime: weight application is O(n * D); inner aggregator
     cost is unchanged or reduced because Byzantine clients' scaled
     gradients are closer to zero.
  2. Composability: works on top of FedAvg, Krum, TrimMean, Median.
  3. Interpretability: weight values map to real Akash on-chain signals
     (uptime score, lease history, audited attributes).

Weight regimes (from agent_docs/expected_akash_reputation_results.md):
  oracle    w_honest=1.0, w_byzantine=0.0  -- perfect signal
  realistic w_honest=0.85, w_byzantine=0.30 -- headline experiment
  degraded  w_honest=0.70, w_byzantine=0.60 -- noisy/unreliable signal
"""

from __future__ import annotations

import os
from typing import Optional

import torch


# ---------------------------------------------------------------------------
# Preset regimes used by the Akash Phase 2 deployments
# ---------------------------------------------------------------------------

WEIGHT_REGIMES: dict[str, dict[str, float]] = {
    "oracle":    {"w_honest": 1.00, "w_byzantine": 0.00},
    "realistic": {"w_honest": 0.85, "w_byzantine": 0.30},
    "degraded":  {"w_honest": 0.70, "w_byzantine": 0.60},
}


def weights_from_env(n_clients: int) -> Optional[list[float]]:
    """Read AKASH_REP_WEIGHTS from environment.

    Returns a list of floats (one per client) if the env var is set,
    otherwise None.  The SDL sets e.g. AKASH_REP_WEIGHTS=0.85,0.85,0.30
    for a 3-client run where the last client is Byzantine.
    """
    raw = os.environ.get("AKASH_REP_WEIGHTS")
    if not raw:
        return None
    parts = [p.strip() for p in raw.split(",")]
    if len(parts) != n_clients:
        raise ValueError(
            f"AKASH_REP_WEIGHTS has {len(parts)} values but n_clients={n_clients}"
        )
    return [float(p) for p in parts]


def weights_from_regime(regime: str, n_honest: int, n_byz: int) -> list[float]:
    """Build a weight vector from a named regime string."""
    if regime not in WEIGHT_REGIMES:
        raise ValueError(f"Unknown regime '{regime}'. Choose from {list(WEIGHT_REGIMES)}")
    cfg = WEIGHT_REGIMES[regime]
    weights = [cfg["w_honest"]] * n_honest + [cfg["w_byzantine"]] * n_byz
    return weights


def reputation_weighted_aggregate(
    grads: list[torch.Tensor],
    weights: list[float],
    inner_defense: str = "FedAvg",
    f: int = 1,
) -> torch.Tensor:
    """Aggregate gradients using per-client reputation weights.

    Algorithm:
      1. Normalize weights to sum to 1 (so the aggregate stays in the
         same magnitude range as a plain FedAvg).
      2. Scale each gradient by its normalized weight: g'_i = w_i * g_i.
      3. Pass the scaled gradients to the inner aggregator.

    For FedAvg the result is exactly a weighted average.
    For Krum/TrimMean/Median the Byzantine client's gradient is attenuated
    before the distance or sorting computation, making it look closer to
    zero (i.e. less like a real gradient), which tends to reduce its chance
    of selection or increase its probability of being trimmed.

    Args:
        grads: list of flattened gradient tensors, one per client.
        weights: list of non-negative floats, one per client.  Need not
                 be normalized; function normalizes internally.
        inner_defense: one of "FedAvg", "Krum", "MultiKrum",
                       "TrimMean", "Median".
        f: assumed Byzantine count passed to the inner aggregator.

    Returns:
        Aggregated gradient tensor.
    """
    if len(grads) != len(weights):
        raise ValueError(
            f"len(grads)={len(grads)} != len(weights)={len(weights)}"
        )

    w = torch.tensor(weights, dtype=grads[0].dtype, device=grads[0].device)
    w = w.clamp(min=0.0)
    w_sum = w.sum()
    if w_sum < 1e-9:
        raise ValueError("All reputation weights are zero; cannot aggregate.")
    w = w / w_sum

    scaled = [w[i] * grads[i] for i in range(len(grads))]

    if inner_defense == "FedAvg":
        return torch.stack(scaled).sum(dim=0)

    # For all other inner aggregators, import locally to avoid circular dep
    from src.experiments.akash_fromscratch import (
        krum,
        multi_krum,
        coordinate_median,
        trimmed_mean,
    )

    if inner_defense == "Krum":
        return krum(scaled, f, k=1)
    elif inner_defense == "MultiKrum":
        return multi_krum(scaled, f, k=3)
    elif inner_defense == "TrimMean":
        return trimmed_mean(scaled, f)
    elif inner_defense == "Median":
        return coordinate_median(scaled)
    else:
        raise ValueError(f"Unsupported inner_defense for AkashRep: '{inner_defense}'")
