#!/usr/bin/env python3
"""Scalability benchmark: measures aggregation time vs gradient dimension.

Runs timed iterations for each ByzFL aggregator across different gradient
dimensions to characterize computational complexity.
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import matplotlib.pyplot as plt

import byzfl
from byzfl import (
    Average,
    Krum,
    MultiKrum,
    TrMean,
    Median,
    GeometricMedian,
)

RESULTS_DIR = Path(__file__).parent.parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR = Path(__file__).parent.parent.parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

DIMENSIONS = [10000, 50000, 100000, 500000, 1000000, 5000000, 10000000]
N_CLIENTS = 6
F = 1
N_ITERATIONS = 5
SEED = 42


def time_aggregate(aggregator, gradients: np.ndarray, n_runs: int = N_ITERATIONS) -> float:
    """Time aggregation and return mean duration in seconds."""
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        _ = aggregator(gradients)
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return np.mean(times)


def run_scalability_benchmark() -> dict[str, Any]:
    np.random.seed(SEED)
    
    aggregators = [
        {"name": "Average", "class": Average, "params": {}},
        {"name": "Krum", "class": Krum, "params": {"f": F}},
        {"name": "MultiKrum", "class": MultiKrum, "params": {"f": F}},
        {"name": "TrMean", "class": TrMean, "params": {"f": F}},
        {"name": "Median", "class": Median, "params": {}},
        {"name": "GeoMed", "class": GeometricMedian, "params": {"nu": 1e-6, "T": 100}},
    ]
    
    results = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "dimensions": DIMENSIONS,
            "n_clients": N_CLIENTS,
            "f": F,
            "n_iterations": N_ITERATIONS,
            "seed": SEED,
        },
        "timings": {},
    }
    
    print(f"Scalability benchmark: {len(DIMENSIONS)} dims × {len(aggregators)} aggregators")
    print(f"Dimensions: {DIMENSIONS}")
    print()
    
    for agg in aggregators:
        name = agg["name"]
        cls = agg["class"]
        params = agg["params"]
        
        results["timings"][name] = []
        print(f"Testing {name}...")
        
        for dim in DIMENSIONS:
            # Generate random gradients (6 clients × dimension)
            gradients = np.random.randn(N_CLIENTS, dim).astype(np.float32)
            
            # Create aggregator instance
            aggregator = cls(**params)
            
            # Warmup
            _ = aggregator(gradients)
            
            # Time it
            mean_time = time_aggregate(aggregator, gradients)
            
            results["timings"][name].append({
                "dimension": dim,
                "mean_time_s": mean_time,
            })
            
            print(f"  dim={dim:>8}: {mean_time*1000:.2f} ms")
    
    return results


def plot_scalability(results: dict[str, Any]) -> None:
    """Generate log-log plot of aggregation time vs dimension."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = plt.cm.Set2(np.linspace(0, 1, len(results["timings"])))
    
    for i, (name, timings) in enumerate(results["timings"].items()):
        dims = [t["dimension"] for t in timings]
        times = [t["mean_time_s"] * 1000 for t in timings]  # Convert to ms
        
        ax.plot(dims, times, marker='o', label=name, color=colors[i], linewidth=2, markersize=6)
    
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Gradient Dimension', fontsize=12)
    ax.set_ylabel('Aggregation Time (ms)', fontsize=12)
    ax.set_title('Aggregator Scalability: Time vs Gradient Dimension', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', framealpha=0.9)
    ax.grid(True, which="both", ls="-", alpha=0.3)
    
    # Add secondary y-axis for seconds
    ax2 = ax.twinx()
    ax2.set_yscale('log')
    ax2.set_ylabel('Aggregation Time (s)', fontsize=12)
    
    plt.tight_layout()
    
    output_path = FIGURES_DIR / "scalability.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nSaved plot: {output_path}")


def main() -> None:
    print("=" * 60)
    print("ByzFL Aggregator Scalability Benchmark")
    print("=" * 60)
    print()
    
    results = run_scalability_benchmark()
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = RESULTS_DIR / f"scalability_{timestamp}.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nSaved results: {output_path}")
    
    # Generate plot
    plot_scalability(results)
    
    print("\nDone!")


if __name__ == "__main__":
    main()
