#!/usr/bin/env python3
"""Export selected W&B runs to local JSON/CSV for reproducibility."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import wandb


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--entity", required=True, help="W&B entity/user/org")
    p.add_argument("--project", required=True, help="W&B project name")
    p.add_argument(
        "--run-id",
        dest="run_ids",
        action="append",
        required=True,
        help="Run ID to export (repeat flag for multiple runs)",
    )
    p.add_argument(
        "--out-dir",
        default="repro/wandb_exports",
        help="Output directory for exported run data",
    )
    return p.parse_args()


def export_run(api: wandb.Api, entity: str, project: str, run_id: str, out_dir: Path) -> None:
    run = api.run(f"{entity}/{project}/{run_id}")
    run_dir = out_dir / run.id
    run_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "id": run.id,
        "name": run.name,
        "project": project,
        "entity": entity,
        "url": run.url,
        "state": run.state,
        "created_at": str(run.created_at),
        "config": dict(run.config),
        "summary": dict(run.summary),
        "tags": list(run.tags),
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    history = list(run.scan_history())
    if history:
        # Keep useful scalar columns first, then the rest.
        preferred = [
            "round",
            "perplexity",
            "asr",
            "asr_best_position",
            "round_time",
            "agg_time",
            "_step",
            "_runtime",
            "_timestamp",
        ]
        discovered_cols = set()
        for row in history:
            discovered_cols.update(row.keys())
        ordered_cols = [c for c in preferred if c in discovered_cols] + [c for c in sorted(discovered_cols) if c not in preferred]

        with (run_dir / "history.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=ordered_cols)
            writer.writeheader()
            for row in history:
                writer.writerow({k: row.get(k) for k in ordered_cols})

        (run_dir / "history.json").write_text(json.dumps(history, indent=2, default=str))

    print(f"Exported {run.id} -> {run_dir}")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    api = wandb.Api()
    for run_id in args.run_ids:
        try:
            export_run(api, args.entity, args.project, run_id, out_dir)
        except Exception as e:
            print(f"Error exporting run {run_id}: {e}")


if __name__ == "__main__":
    main()
