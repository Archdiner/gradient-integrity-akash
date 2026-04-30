#!/usr/bin/env bash
# sync_results.sh -- pull results from a running Akash lease before close.
#
# REQUIRES the Akash CLI (`provider-services`) -- see deployment guide.
# The Akash Console has a one-click "Download" for individual files; this
# script is for full-directory snapshots (checkpoints + run.json + logs).
#
# Usage:
#   AKASH_DSEQ=12345678 \
#   AKASH_PROVIDER=akash1xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx \
#   AKASH_KEY_NAME=my-key \
#   bash akash/sync_results.sh
#
# Environment variables:
#   AKASH_DSEQ           -- deployment sequence number from `provider-services
#                           query market lease list --owner <addr>`
#   AKASH_PROVIDER       -- provider address from the lease
#   AKASH_KEY_NAME       -- local key name in `provider-services keys list`
#   AKASH_GSEQ           -- (optional) group sequence, default 1
#   AKASH_OSEQ           -- (optional) order sequence, default 1
#   LOCAL_OUT            -- (optional) local destination, default ./results/akash_pull_$dseq

set -euo pipefail

: "${AKASH_DSEQ:?AKASH_DSEQ must be set}"
: "${AKASH_PROVIDER:?AKASH_PROVIDER must be set}"
: "${AKASH_KEY_NAME:?AKASH_KEY_NAME must be set}"
GSEQ="${AKASH_GSEQ:-1}"
OSEQ="${AKASH_OSEQ:-1}"
LOCAL_OUT="${LOCAL_OUT:-./results/akash_pull_${AKASH_DSEQ}}"

mkdir -p "$LOCAL_OUT"

echo "==> Confirming lease is active..."
provider-services query market lease get \
    --dseq "$AKASH_DSEQ" --gseq "$GSEQ" --oseq "$OSEQ" \
    --provider "$AKASH_PROVIDER" \
    --output json | grep -q '"state":"active"' || {
    echo "Lease is not active. Aborting." >&2
    exit 1
}

echo "==> Streaming run.json from container..."
provider-services lease-shell \
    --dseq "$AKASH_DSEQ" --gseq "$GSEQ" --oseq "$OSEQ" \
    --provider "$AKASH_PROVIDER" --from "$AKASH_KEY_NAME" \
    -- trainer "cat /app/results/run.json" \
    > "$LOCAL_OUT/run.json"

echo "==> Listing checkpoints..."
provider-services lease-shell \
    --dseq "$AKASH_DSEQ" --gseq "$GSEQ" --oseq "$OSEQ" \
    --provider "$AKASH_PROVIDER" --from "$AKASH_KEY_NAME" \
    -- trainer "ls -lh /app/results/" \
    | tee "$LOCAL_OUT/manifest.txt"

echo "==> Tarring and streaming checkpoints (large)..."
provider-services lease-shell \
    --dseq "$AKASH_DSEQ" --gseq "$GSEQ" --oseq "$OSEQ" \
    --provider "$AKASH_PROVIDER" --from "$AKASH_KEY_NAME" \
    -- trainer "tar czf - -C /app/results checkpoints" \
    > "$LOCAL_OUT/checkpoints.tar.gz"

echo "==> Streaming live logs (last 1000 lines)..."
provider-services lease-logs \
    --dseq "$AKASH_DSEQ" --gseq "$GSEQ" --oseq "$OSEQ" \
    --provider "$AKASH_PROVIDER" --from "$AKASH_KEY_NAME" \
    --tail 1000 \
    > "$LOCAL_OUT/lease.log"

echo "==> Done. Results pulled to $LOCAL_OUT"
ls -lh "$LOCAL_OUT"

echo ""
echo "After verifying $LOCAL_OUT/run.json looks complete, close the lease with:"
echo "  provider-services tx deployment close \\"
echo "      --dseq $AKASH_DSEQ \\"
echo "      --from $AKASH_KEY_NAME --gas auto --gas-adjustment 1.4"
