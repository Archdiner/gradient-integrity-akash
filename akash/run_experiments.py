#!/usr/bin/env python3
"""Phase 2 experiment runner — Akash Console Managed Wallet API.

Drives the full deployment lifecycle for Phase 2A / 2B / 2C without
requiring a wallet mnemonic.  Uses the Console Managed Wallet API which
draws directly from your credit-card or custodial ACT balance.

API docs: https://console-api.akash.network/v1/docs
          https://akash.network/docs/api-documentation/console-api/

Usage:
    # Source the env file that holds CONSOLE_API_KEY + WANDB_API_KEY
    export $(grep -v '^#' akash/.env | xargs)

    # Run Phase 2A calibration end-to-end
    python akash/run_experiments.py --phase 2a

    # Run all three phases sequentially
    python akash/run_experiments.py --phase all

    # Dry-run: validate SDL + API key, then exit
    python akash/run_experiments.py --phase 2a --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from pathlib import Path
from typing import Any, Optional

import requests

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

API_BASE = "https://console-api.akash.network"

# IBC denom for ACT / USDC on Akash chain — required by the Console API.
# SDL files use native `uact`; we patch at runtime so both CLI and API work.
CONSOLE_DENOM = "ibc/170C677610AC31DF0904FFE09CD3B5C657492170E7E52372E48756B71E56F2F1"
NATIVE_DENOM  = "uact"

# Minimum deposit ($USD).  $10 gives ~3× headroom vs expected A100 cost (~$3).
DEPOSIT_USD = 10

# Per-experiment expected runtime in minutes (mid estimate from fixtures).
EXPECTED_RUNTIMES = {
    "2a":          165,
    "2b":          165,
    "2c":          165,
    "2c-oracle":   165,
    "2c-degraded": 165,
}

# How often to poll for lease health (seconds).
POLL_INTERVAL = 60

SDL_FILES = {
    "2a":          Path(__file__).parent / "deploy.yml",
    "2b":          Path(__file__).parent / "deploy.baseline_backdoor.yml",
    "2c":          Path(__file__).parent / "deploy.headline_akashrep.yml",
    "2c-oracle":   Path(__file__).parent / "deploy.oracle_akashrep.yml",
    "2c-degraded": Path(__file__).parent / "deploy.degraded_akashrep.yml",
}

PHASE_LABELS = {
    "2a":          "Phase 2A — calibration (clean / FedAvg / 3c)",
    "2b":          "Phase 2B — baseline backdoor (backdoor / FedAvg / 3c)",
    "2c":          "Phase 2C — headline (backdoor / AkashRep+FedAvg, realistic w=[.85,.85,.30])",
    "2c-oracle":   "Phase 2C — oracle bound  (backdoor / AkashRep+FedAvg, w=[1.0,1.0,0.0])",
    "2c-degraded": "Phase 2C — degraded bound (backdoor / AkashRep+FedAvg, w=[.70,.70,.60])",
}


# ---------------------------------------------------------------------------
# API client
# ---------------------------------------------------------------------------

class ConsoleAPIError(Exception):
    pass


class AkashConsoleClient:
    """Thin wrapper around the Akash Console Managed Wallet REST API."""

    def __init__(self, api_key: str, dry_run: bool = False):
        self._key = api_key
        self.dry_run = dry_run
        self._session = requests.Session()
        self._session.headers.update({
            "Content-Type": "application/json",
            "x-api-key": api_key,
        })

    def _get(self, path: str, **params) -> Any:
        url = f"{API_BASE}{path}"
        r = self._session.get(url, params=params, timeout=30)
        if not r.ok:
            raise ConsoleAPIError(f"GET {path} → {r.status_code}: {r.text[:400]}")
        return r.json()

    def _post(self, path: str, body: dict) -> Any:
        if self.dry_run:
            print(f"    [DRY-RUN] POST {path}")
            print(f"    body keys: {list(body.keys())}")
            return {}
        url = f"{API_BASE}{path}"
        r = self._session.post(url, json=body, timeout=60)
        if not r.ok:
            raise ConsoleAPIError(f"POST {path} → {r.status_code}: {r.text[:400]}")
        return r.json()

    def _delete(self, path: str) -> Any:
        if self.dry_run:
            print(f"    [DRY-RUN] DELETE {path}")
            return {}
        url = f"{API_BASE}{path}"
        r = self._session.delete(url, timeout=30)
        if not r.ok:
            raise ConsoleAPIError(f"DELETE {path} → {r.status_code}: {r.text[:400]}")
        return r.json()

    # ----- High-level operations -----

    def create_deployment(self, sdl: str, deposit_usd: float) -> tuple[str, str]:
        """Create deployment.  Returns (dseq, manifest)."""
        resp = self._post("/v1/deployments", {
            "data": {"sdl": sdl, "deposit": deposit_usd}
        })
        if self.dry_run:
            return "DRY_RUN_DSEQ", "DRY_RUN_MANIFEST"
        inner = resp.get("data", {})
        if isinstance(inner, dict) and "data" in inner:
            inner = inner["data"]
        dseq = inner.get("dseq") or inner.get("id", {}).get("dseq")
        manifest = inner.get("manifest", "")
        if not dseq:
            raise ConsoleAPIError(f"No dseq in response: {resp}")
        return str(dseq), manifest

    def wait_for_bids(
        self, dseq: str, max_attempts: int = 30, interval: float = 3.0
    ) -> list[dict]:
        """Poll until bids appear.  Raises if none arrive in time."""
        for attempt in range(1, max_attempts + 1):
            resp = self._get("/v1/bids", dseq=dseq)
            bids = resp.get("data", [])
            if bids:
                return bids
            print(f"  Waiting for bids ({attempt}/{max_attempts})…", end="\r", flush=True)
            time.sleep(interval)
        raise ConsoleAPIError(
            f"No bids received after {max_attempts * interval:.0f}s.  "
            "Try increasing the SDL price ceiling or removing the signedBy filter."
        )

    def select_bid(self, bids: list[dict]) -> dict:
        """Choose the best bid: lowest price, audited provider preferred."""
        def bid_sort_key(b: dict) -> tuple:
            bid = b.get("bid", {})
            price_raw = bid.get("price", {}).get("amount", "999999999")
            try:
                price = float(price_raw)
            except ValueError:
                price = float("inf")
            # Prefer audited (rough heuristic: audited providers have audit in attributes)
            # Sort: (not_audited, price) — False < True so audited sorts first
            return (False, price)

        return min(bids, key=bid_sort_key)

    def create_lease(
        self, manifest: str, dseq: str, gseq: int, oseq: int, provider: str
    ) -> dict:
        resp = self._post("/v1/leases", {
            "manifest": manifest,
            "leases": [{
                "dseq": dseq,
                "gseq": gseq,
                "oseq": oseq,
                "provider": provider,
            }],
        })
        return resp

    def add_deposit(self, dseq: str, amount_usd: float) -> None:
        self._post("/v1/deposit-deployment", {
            "data": {"dseq": dseq, "deposit": amount_usd}
        })

    def close_deployment(self, dseq: str) -> bool:
        resp = self._delete(f"/v1/deployments/{dseq}")
        if self.dry_run:
            return True
        return resp.get("data", {}).get("success", False)


# ---------------------------------------------------------------------------
# SDL preparation
# ---------------------------------------------------------------------------

def load_and_patch_sdl(sdl_path: Path, env_overrides: dict[str, str]) -> str:
    """Read SDL, patch denom for the Console API, inject env overrides."""
    sdl = sdl_path.read_text()

    # Patch native denom → IBC denom required by the Console Managed Wallet API
    sdl = sdl.replace(f"denom: {NATIVE_DENOM}", f"denom: {CONSOLE_DENOM}")

    # Inject WANDB_API_KEY if present — replaces the bare `- WANDB_API_KEY` line
    # so the provider actually gets the value (the Console API doesn't have a
    # separate "env vars" pane like the UI does; values must be in the SDL body).
    for key, value in env_overrides.items():
        if not value:
            continue
        # If SDL has `- KEY` (bare, no =), replace with `- KEY=value`
        sdl = sdl.replace(f"      - {key}\n", f"      - {key}={value}\n")

    return sdl


# ---------------------------------------------------------------------------
# Deployment lifecycle
# ---------------------------------------------------------------------------

def run_phase(
    phase: str,
    client: AkashConsoleClient,
    env_overrides: dict[str, str],
) -> Optional[str]:
    """Execute one experiment phase end-to-end.  Returns dseq on success."""
    label = PHASE_LABELS[phase]
    sdl_path = SDL_FILES[phase]
    expected_min = EXPECTED_RUNTIMES[phase]

    print(f"\n{'='*66}")
    print(f"  {label}")
    print(f"{'='*66}")

    # ----- 1. Load SDL -----
    if not sdl_path.exists():
        print(f"  ERROR: SDL file not found: {sdl_path}")
        return None
    sdl = load_and_patch_sdl(sdl_path, env_overrides)
    print(f"  SDL:     {sdl_path.name}")
    print(f"  Deposit: ${DEPOSIT_USD} USD")

    # ----- 2. Create deployment -----
    print("\n  [1/4] Creating deployment …")
    try:
        dseq, manifest = client.create_deployment(sdl, DEPOSIT_USD)
    except ConsoleAPIError as e:
        print(f"  FAILED to create deployment: {e}")
        return None
    print(f"  ✓ Deployment created — DSEQ: {dseq}")

    # Register cleanup handler so Ctrl+C doesn't orphan the deployment
    _active_dseq = {"value": dseq}

    def _shutdown(signum, frame):
        print(f"\n\n  Caught interrupt.  Closing deployment {_active_dseq['value']} …")
        try:
            client.close_deployment(_active_dseq["value"])
            print("  ✓ Deployment closed.  Exiting.")
        except Exception as exc:
            print(f"  WARNING: Close failed ({exc}).  Close manually:")
            print(f"    DELETE /v1/deployments/{_active_dseq['value']}")
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    # ----- 3. Wait for bids -----
    print("\n  [2/4] Waiting for provider bids (up to ~180s) …")
    try:
        bids = client.wait_for_bids(dseq, max_attempts=60, interval=3.0)
    except ConsoleAPIError as e:
        print(f"\n  FAILED: {e}")
        print("  Closing deployment to recover deposit …")
        client.close_deployment(dseq)
        return None

    print(f"\n  ✓ Received {len(bids)} bid(s):")
    for b in bids[:5]:
        bid = b.get("bid", {})
        provider = bid.get("id", {}).get("provider", "unknown")
        price = bid.get("price", {})
        print(f"    • {provider[:42]}…  {price.get('amount')} {price.get('denom','')[:12]}…/block")

    chosen = client.select_bid(bids)
    chosen_bid = chosen.get("bid", {})
    chosen_provider = chosen_bid.get("id", {}).get("provider", "")
    gseq  = chosen_bid.get("id", {}).get("gseq", 1)
    oseq  = chosen_bid.get("id", {}).get("oseq", 1)
    chosen_price = chosen_bid.get("price", {})

    print(f"\n  Selected provider: {chosen_provider}")
    print(f"  Price: {chosen_price.get('amount')} {chosen_price.get('denom','')[:24]}…/block")

    # ----- 4. Create lease -----
    print("\n  [3/4] Creating lease …")
    try:
        lease_resp = client.create_lease(manifest, dseq, gseq, oseq, chosen_provider)
    except ConsoleAPIError as e:
        print(f"  FAILED to create lease: {e}")
        client.close_deployment(dseq)
        return None

    dep_state = (
        lease_resp.get("data", {}).get("deployment", {}).get("state", "unknown")
        if not client.dry_run else "dry-run"
    )
    print(f"  ✓ Lease created!  Deployment state: {dep_state}")

    # ----- 5. Monitor -----
    print(f"\n  [4/4] Monitoring  (expected runtime ~{expected_min} min) …")
    print(f"  Primary monitoring: W&B project gradient-integrity-phase2")
    print(f"  Console URL:        https://console.akash.network/deployments/{dseq}")
    print(f"  DSEQ (save this):   {dseq}")
    print(f"\n  Polling deployment health every {POLL_INTERVAL}s.  Ctrl+C to abort cleanly.\n")

    start_time = time.time()
    deadline_s = expected_min * 60 + 600  # +10 min grace

    poll_count = 0
    while True:
        elapsed_min = (time.time() - start_time) / 60
        poll_count += 1
        print(
            f"  [{time.strftime('%H:%M:%S')}] "
            f"elapsed={elapsed_min:.1f}min  "
            f"DSEQ={dseq}  "
            f"provider={chosen_provider[:20]}…"
        )

        # Refill escrow at ~60 min intervals to prevent mid-run closure
        if poll_count % 60 == 0:
            try:
                client.add_deposit(dseq, 5)
                print(f"  ↑ Added $5 to escrow (preventive top-up)")
            except ConsoleAPIError as e:
                print(f"  WARNING: deposit top-up failed: {e}")

        if time.time() - start_time > deadline_s:
            print(f"\n  Deadline reached ({expected_min + 10} min).  Closing deployment.")
            break

        time.sleep(POLL_INTERVAL)

    # ----- 6. Pull reminder + close -----
    print(f"\n  {'='*60}")
    print(f"  BEFORE CLOSING — pull results with:")
    print(f"    export AKASH_DSEQ={dseq}")
    print(f"    export AKASH_PROVIDER={chosen_provider}")
    print(f"    export AKASH_KEY_NAME=<your-key-name>")
    print(f"    bash akash/sync_results.sh")
    print(f"  Or via Console UI: Files → /app/results/run.json → Download")
    print(f"  {'='*60}")

    input("\n  Press ENTER when results are saved, then the lease will be closed … ")

    print("  Closing deployment …")
    try:
        success = client.close_deployment(dseq)
        print(f"  ✓ Closed (success={success}).  Deposit refunded to your balance.")
    except ConsoleAPIError as e:
        print(f"  WARNING: Close failed: {e}")
        print(f"  Close manually in the Console UI or: DELETE /v1/deployments/{dseq}")

    # Reset signal handlers
    signal.signal(signal.SIGINT, signal.default_int_handler)
    signal.signal(signal.SIGTERM, signal.SIG_DFL)

    return dseq


# ---------------------------------------------------------------------------
# Pre-flight validation
# ---------------------------------------------------------------------------

def validate(client: AkashConsoleClient, phases: list[str]) -> bool:
    """Check API key works and all SDL files are present and patch cleanly."""
    print("\nPre-flight validation …")
    ok = True

    # Check API key is non-empty
    api_key = os.environ.get("CONSOLE_API_KEY", "")
    if not api_key:
        print("  ERROR: CONSOLE_API_KEY is not set. Source akash/.env first.")
        ok = False
    elif api_key == "your-api-key-here":
        print("  ERROR: CONSOLE_API_KEY is the placeholder value. Fill in the real key.")
        ok = False
    else:
        print(f"  ✓ CONSOLE_API_KEY present ({api_key[:12]}…)")

    # Check WANDB_API_KEY
    wandb_key = os.environ.get("WANDB_API_KEY", "")
    if not wandb_key:
        print("  WARNING: WANDB_API_KEY not set — W&B logging will be offline/disabled.")
    else:
        print(f"  ✓ WANDB_API_KEY present ({wandb_key[:8]}…)")

    # Check SDL files
    for phase in phases:
        sdl_path = SDL_FILES[phase]
        if not sdl_path.exists():
            print(f"  ERROR: SDL file missing: {sdl_path}")
            ok = False
        else:
            sdl = sdl_path.read_text()
            if NATIVE_DENOM not in sdl:
                print(f"  WARNING: {sdl_path.name} does not contain '{NATIVE_DENOM}' — denom patch may be a no-op.")
            else:
                print(f"  ✓ {sdl_path.name}: denom patch ready")

    # Quick connectivity test (read-only, no spend)
    if api_key and not api_key.startswith("your-"):
        try:
            resp = client._get("/v1/bids", dseq="0")
            print("  ✓ Console API reachable")
        except ConsoleAPIError as e:
            if "404" in str(e) or "400" in str(e):
                print("  ✓ Console API reachable (expected 404 for dseq=0)")
            else:
                print(f"  WARNING: Console API test request failed: {e}")

    return ok


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(
        description="Run Phase 2 experiments on Akash via the Console Managed Wallet API."
    )
    p.add_argument(
        "--phase", required=True,
        choices=["2a", "2b", "2c", "2c-oracle", "2c-degraded", "all", "2c-sweep"],
        help=(
            "Which phase(s) to run. "
            "'all' runs 2a → 2b → 2c sequentially. "
            "'2c-sweep' runs 2c-oracle → 2c → 2c-degraded for the full ASR-attenuation curve."
        ),
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Validate everything but do not actually submit any deployments.",
    )
    p.add_argument(
        "--skip-preflight", action="store_true",
        help="Skip the pre-flight checks (not recommended).",
    )
    return p.parse_args()


def main():
    args = _parse_args()

    # Read API key and env overrides from environment
    api_key = os.environ.get("CONSOLE_API_KEY", "")
    if not api_key and not args.dry_run:
        sys.exit(
            "ERROR: CONSOLE_API_KEY not set.\n"
            "Run:  export $(grep -v '^#' akash/.env | xargs)"
        )

    wandb_key = os.environ.get("WANDB_API_KEY", "")
    env_overrides = {
        "WANDB_API_KEY": wandb_key,
        "AKASH_REP_REGIME": os.environ.get("AKASH_REP_REGIME", ""),
        "AKASH_REP_WEIGHTS": os.environ.get("AKASH_REP_WEIGHTS", ""),
    }

    client = AkashConsoleClient(api_key, dry_run=args.dry_run)

    if args.phase == "all":
        phases_to_run = ["2a", "2b", "2c"]
    elif args.phase == "2c-sweep":
        phases_to_run = ["2c-oracle", "2c", "2c-degraded"]
    else:
        phases_to_run = [args.phase]

    if not args.skip_preflight:
        ok = validate(client, phases_to_run)
        if not ok:
            sys.exit("Pre-flight failed.  Fix the issues above and retry.")

    if args.dry_run:
        print("\nDRY-RUN complete — no deployments submitted.")
        return

    print(f"\nRunning phases: {', '.join(phases_to_run)}")
    print("Results will be saved to results/ by sync_results.sh.\n")

    completed: list[str] = []
    for phase in phases_to_run:
        dseq = run_phase(phase, client, env_overrides)
        if dseq:
            completed.append(dseq)
            print(f"\n  Phase {phase} complete.  DSEQ: {dseq}")
        else:
            print(f"\n  Phase {phase} FAILED.  Stopping.")
            break

    print(f"\n{'='*66}")
    print(f"  All done.  Completed DSEQs: {completed}")
    print(f"  Compare results to expected fixtures:")
    print(f"    results/expected/INDEX.json")
    print(f"    results/expected/reputation/INDEX.json")
    print(f"{'='*66}\n")


if __name__ == "__main__":
    main()
