# Akash Deployment Guide — Gradient Integrity, Phase 2

**Audience.** You (the operator) running the Phase 2 GPT-2 Small experiments
on Akash Network, post-BME (Burn-Mint Equilibrium), April 2026.

**Two execution paths.** Pick one. Both produce identical experiment results.

| Path | How | Best for |
|------|-----|----------|
| **A — Console API (recommended)** | `python akash/run_experiments.py` | Fully automated, uses your credit-card ACT balance directly, no wallet/mnemonic needed |
| **B — Console UI** | Browser at `console.akash.network` | Manual oversight, or if the API runner hits an unexpected error |

**Three runs you'll do.**

| Phase | SDL | Purpose | Cost ceiling |
|-------|-----|---------|--------------|
| 2A | `akash/deploy.yml` | Calibration: clean / FedAvg / 3 clients. | ~$10 |
| 2B | `akash/deploy.baseline_backdoor.yml` | Baseline: backdoor / FedAvg / 3 clients. ASR ≈ 0.60. | ~$10 |
| 2C | `akash/deploy.headline_akashrep.yml` | **Headline**: backdoor / AkashRep+FedAvg / 3 clients. ASR ≈ 0.30. | ~$10 |

Total realistic cost: **~$10–15** across all three. Hard ceiling ~$30.

---

## 0A. Path A — Automated runner (Console Managed Wallet API)

This path requires no browser interaction and uses the `x-api-key` header
against `https://console-api.akash.network`.  Your credit-card ACT balance
is charged automatically.

### 0A.1 — One-time setup

Store your API key in `akash/.env` (gitignored):

```
CONSOLE_API_KEY=<your key from console.akash.network → Settings → API Keys>
WANDB_API_KEY=<your W&B key>
```

### 0A.2 — Source env and dry-run

```bash
cd /Users/asadr/Desktop/gradient-integrity-akash

# Load keys into shell
export $(grep -v '^#' akash/.env | xargs)

# Validate everything without spending
python akash/run_experiments.py --phase 2a --dry-run
```

Expected output ends with `✓ Console API reachable` and `DRY-RUN complete`.

### 0A.3 — Build and push the Docker image (required before any live run)

```bash
docker build -f akash/Dockerfile -t asadrizvi06/gradient-integrity:phase2-v1 .
docker push asadrizvi06/gradient-integrity:phase2-v1
docker inspect asadrizvi06/gradient-integrity:phase2-v1 \
    --format '{{index .RepoDigests 0}}' | tee akash/.last_pushed_digest
```

### 0A.4 — Run Phase 2A (calibration)

```bash
python akash/run_experiments.py --phase 2a
```

The runner will:
1. Patch the SDL denom for the Console API (uact → IBC/ACT)
2. Create the deployment with a $10 deposit
3. Poll for bids (up to 90 s, 3 s intervals), print all bids, pick cheapest
4. Create the lease
5. Print the DSEQ and a Console URL you can open to watch logs
6. Monitor (ping every 60 s; auto top-up escrow at 60-min intervals)
7. After ~165 min, prompt you to pull results before closing
8. Close the lease and refund the unused deposit

### 0A.5 — Run all phases sequentially

```bash
python akash/run_experiments.py --phase all
```

Each phase waits for your ENTER keystroke (after results are pulled) before
advancing to the next.

### 0A.6 — If bids don't arrive

The SDL `signedBy` block filters for a specific auditor address.  If no bids
arrive within 90 s, the runner will print the error.  To relax the filter:

1. Edit the relevant SDL file: remove the `signedBy:` block entirely.
2. Re-run.  You'll get more bids but from potentially un-audited providers.
   For Phase 2A only (calibration), this is acceptable.

---

---

## 0. Pre-flight checklist (do not skip)

Stop and check **all** of these before touching the Console. Failures here
are by far the most common cause of wasted ACT credits.

- [ ] **Code prerequisites for the run you intend to launch are merged.** See
      §2 below. Phase 2A can run today. Phase 2B and 2C have unmet code
      dependencies that **must** land first.
- [ ] **A pinned Docker image exists on Docker Hub** with the correct tag
      (`asadrizvi06/gradient-integrity:phase2-v1`) and a recorded digest.
      See §1.
- [ ] **You have an Akash account** (wallet or credit-card-funded). See §3.
- [ ] **Your funded balance is at least 2× the run's cost ceiling.** Bids
      sometimes come in higher than your `amount: 5000` ceiling expects;
      if escrow runs dry mid-run the lease is closed and you lose the run.
- [ ] **Your W&B API key is in hand** (not in the SDL, not in the image). It
      goes in the Console "Environment Variables" pane at deploy time.
- [ ] **You can reproduce `docker run` of the image locally on a CUDA box**
      and see at least one round complete cleanly. If you cannot, do not
      deploy to Akash. (See §1.4 — local smoke test.)
- [ ] **`results/expected/INDEX.json` and
      `results/expected/reputation/INDEX.json` are reviewed.** You know what
      perplexity, ASR, round-time numbers should look like at round 1, 50,
      and 200, so you can spot a divergent run within minutes instead of
      hours.

---

## 1. Build and publish the Docker image

The Akash provider pulls from a public registry. We use Docker Hub. Other
registries (GHCR, ECR, etc.) work via the SDL `credentials:` block but
Docker Hub keeps the SDL secret-free.

### 1.1 Build from the repo root

The Dockerfile uses paths relative to the repo root (it `COPY`s
`src/__init__.py` and similar). **Build from the root, not from `akash/`:**

```bash
cd /Users/asadr/Desktop/gradient-integrity-akash

docker build \
    -f akash/Dockerfile \
    -t asadrizvi06/gradient-integrity:phase2-v1 \
    .
```

### 1.2 Verify the image locally before pushing

```bash
docker run --rm \
    asadrizvi06/gradient-integrity:phase2-v1 \
    --help
```

You should see `argparse` help printed by `akash_fromscratch.py`. If you
see `ModuleNotFoundError: No module named 'src'`, the Dockerfile failed
to copy the `__init__.py` files — re-check §1.1 was run from the repo
root.

### 1.3 Push to Docker Hub

```bash
docker login
docker push asadrizvi06/gradient-integrity:phase2-v1
```

### 1.4 Pin the digest (mandatory for reproducibility)

```bash
docker inspect asadrizvi06/gradient-integrity:phase2-v1 \
    --format '{{index .RepoDigests 0}}'
```

That prints something like
`asadrizvi06/gradient-integrity@sha256:abcd1234…`. **Record this digest
in your run notes.** When publishing, cite the digest, not the tag — tags
move, digests don't.

### 1.5 Local smoke test (zero-tolerance gate)

Don't waste ACT credits on a broken image. On a local CUDA box:

```bash
docker run --rm --gpus all \
    -e WANDB_MODE=offline \
    -v "$PWD/results:/app/results" \
    asadrizvi06/gradient-integrity:phase2-v1 \
    --model small --rounds 2 --batch 4 \
    --honest 2 --byz 0 --attack clean --defense FedAvg \
    --device cuda --output /app/results/smoke.json
```

If 2 rounds complete and `results/smoke.json` is well-formed, the image
is ready. If you see CUDA OOM at batch 4 on a single GPU you'll see the
same on the Akash A100 (less likely, but possible) — drop batch to 4
permanently in the SDL.

---

## 2. Code prerequisites per phase

Each SDL is annotated with its prerequisites in a top-of-file comment.
Summary:

| Phase | What must be in the code first |
|-------|-------------------------------|
| 2A — calibration | **Nothing.** The script as-is supports clean / FedAvg / `--device cuda`. Build & deploy. |
| 2B — baseline backdoor | (a) Backdoor trigger injection on the Byzantine client; (b) Poisoned validation slice; (c) Per-round `"asr"` field in `rounds[]`. |
| 2C — headline AkashRep | All 2B prerequisites, plus: (d) `src/defenses/reputation.py` implementing `ReputationWeightedAggregator`; (e) `--defense AkashRep+FedAvg` argparse path; (f) `AKASH_REP_WEIGHTS` env-var parsing into a per-client weight tensor. |

**Do not deploy 2B until (a)-(c) are merged. Do not deploy 2C until
(a)-(f) are merged.** A run without ASR logging produces a `run.json`
that cannot be compared to the expected fixtures and the credits are
wasted.

---

## 3. Akash account setup

You have two paths. Pick one. Both work today on the post-BME Console.

### Path A — Credit card (fastest, no crypto)

1. Open `https://console.akash.network`.
2. Click **"Get Started"** → **"Sign in with Email"**.
3. Verify the email. New users get a $100 ACT trial — that covers the
   entire Phase 2 budget twice over.
4. Add a credit card under **Billing → Payment Methods**. This funds
   the same custodial account; no AKT or Keplr needed. ACT credits are
   USD-pegged so $1 ≈ 1 ACT ≈ 1,000,000 uact.
5. Enable **Auto Top-Up** at $25 minimum. This is the single most
   important reliability lever — it prevents mid-run lease termination
   if a provider's bid pricing drifts. Without it, an empty escrow
   account during a 3-hour run kills the run.

### Path B — Wallet (fully permissionless)

1. Install [Keplr](https://keplr.app) or [Leap](https://leapwallet.io).
2. Create or import an Akash account. Send AKT to it from an exchange
   (Kraken, Osmosis, Coinbase via bridge — most CEXes list AKT).
3. Open `https://console.akash.network` and click **"Connect Wallet"**.
4. Approve the connection.
5. Use the **"Burn AKT to Mint ACT"** flow inside the Console to
   generate ACT credits for deployments. Mint at least $50 worth — this
   leaves you with headroom across all three Phase 2 runs.

Either path leaves you with ACT credits visible at the top of the
Console. **Do not proceed if the visible balance is below $50 ACT.**

### 3.1 Generate a deployment certificate (one-time)

The Console handles this transparently for credit-card users. Wallet
users will see a prompt the first time they hit "Deploy" — accept it.
The certificate is what authorizes the Console to talk to providers on
your behalf.

---

## 4. Deploy through the Console (Phase 2A walkthrough)

These steps mirror the post-BME Console UX as of April 2026.

### 4.1 Open the deploy page

1. From the Console home, click **"Deploy"** (top-right).
2. Choose **"Build your own / Custom YAML"** (sometimes labelled "MYO").
   *Do not* pick a template — templates do not match our SDL.

### 4.2 Paste the SDL

1. Open `akash/deploy.yml` in your editor.
2. Copy the entire file. Paste it into the Console's YAML pane.
3. Click **"Validate"** (or "Lint", depending on Console version). The
   YAML must parse cleanly. If you see an error, the most common
   culprits are:
   - A stray tab character — the SDL is space-indented only.
   - Missing the leading `---`.
   - The persistent volume `mount` block under
     `deployment.trainer.akash.service.trainer.params.storage` —
     deeply-nested but mandatory.

### 4.3 Inject environment variables (DO NOT skip)

1. In the SDL, the `env:` list contains `WANDB_API_KEY` with no value.
   That's intentional — it tells Akash to populate it from the deploy-time
   environment.
2. Click the **"Environment Variables"** sub-tab in the Console.
3. Set:
   - `WANDB_API_KEY=` *(your real key from `https://wandb.ai/authorize`)*
   - For Phase 2C only:
     - `AKASH_REP_REGIME=realistic`
     - `AKASH_REP_WEIGHTS=0.85,0.85,0.30`
4. Confirm none of the SDL pane shows the raw key.

### 4.4 Pre-flight bid check

1. Click **"Create Deployment"**. The Console submits the deployment
   to the marketplace. If the Console flashes a **"Bid PreCheck"**
   warning that your resource constraints are too restrictive, click
   into it and read the message — usually it's the GPU model filter.
2. If PreCheck passes, you'll see a **bid table** within ~30 seconds.

### 4.5 Review and pick a provider

The bid table shows columns: provider address, bid (uact/block), region,
audited attributes, uptime estimate. Sort by **audited** first.

**Picking criteria, in order:**

1. **Audited** — only consider providers whose audit attribute is
   verified by an Akash auditor. Look for a green check or the auditor
   address `akash1365yvmc4s7awdyj3n2sav7xfx76adc6dnmlx63` (or your
   chosen auditor — the SDL `signedBy` block already enforces this, but
   double-check the bid).
2. **Bid price ≤ your ceiling.** Your SDL caps bids at 5000 uact/block
   = ~$72/day. Anything below that is acceptable.
3. **Region** — pick a region close to you only if the W&B traffic
   matters; for a 3-hour run, region barely affects total cost.
4. **GPU model filter match.** Hover the bid row and verify the
   provider lists `a100` with `ram: 80Gi`. Some providers list 40 GB
   A100s and shouldn't have bid; do not select them.

Click **"Accept Bid"** on the chosen provider.

### 4.6 Lease activation

The Console shows a status spinner for ~15-90 seconds while the lease
is created and the container is scheduled. When it goes green, click
**"Logs"**.

You should see, within 60 seconds:
1. PyTorch CUDA initialization (e.g. `CUDA Available: True`).
2. Hugging Face download progress for TinyStories.
3. The first round metrics: `[round 1/200] perplexity=…`.

If you see any of:
- `CUDA out of memory` → drop `--batch` to 4 or `--model` to `tiny` and redeploy.
- `Connection refused` to W&B → re-check the env var; W&B will degrade to
  offline but you lose live monitoring.
- Container restart loop → click **"Events"** in the Console; usually
  it's an image pull error or a missing env.

---

## 5. Mid-run monitoring

A 200-round Phase 2A run takes roughly 165 minutes (~50 s/round) on
A100 80GB. Two channels of truth:

### 5.1 Console logs (primary)

The **Logs** tab shows stdout. Every 5 rounds you should see a metric
line. Compare to `results/expected/akash_clean_FedAvg_3c.json`:

- Round 1 perplexity ~ 27,000 (high is fine for a freshly initialized
  GPT-2 Small).
- Round 50 perplexity in the band [60, 200].
- Round 200 perplexity in the band [10, 18].

**Drift rule.** If round 50 perplexity exceeds 300 or round 100 exceeds
60, abort the run (see §7) — something is wrong and the remaining 100
rounds will not save you.

### 5.2 Weights & Biases (secondary)

If `WANDB_API_KEY` was set, the run is live at
`https://wandb.ai/<your-org>/gradient-integrity-phase2`. The
`WANDB_RUN_GROUP` env-var groups the three phases. Use this for
cross-phase comparison plots.

### 5.3 Persistent volume snapshots

The persistent volume `results` is mounted at `/app/results`. The
script writes:

- `/app/results/run.json` — append-only, updated every round.
- `/app/results/checkpoints/round_{N}.pt` — every 25 rounds (496 MB
  each on Phase 2C — total ~4 GB across the run).

These survive container restarts but **not** lease close. Pull them
before close (§6).

---

## 6. Pull results before closing the lease

This is non-optional. Once a lease closes, the persistent volume is
released by the provider; recovery is provider-discretion and rare.

### 6.1 Console one-click

1. In the lease detail page, click **"Files"**.
2. Navigate to `/app/results/`.
3. Click **"Download"** on `run.json`. (Console one-click is fine for
   small files.)

### 6.2 Full snapshot via CLI

For checkpoints (large) and full directory snapshots, use the helper
script:

```bash
# Find DSEQ and provider in the Console "Lease Details" panel.
export AKASH_DSEQ=12345678
export AKASH_PROVIDER=akash1xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
export AKASH_KEY_NAME=my-key   # name in `provider-services keys list`

bash akash/sync_results.sh
```

Output lands in `./results/akash_pull_${AKASH_DSEQ}/`. Verify
`run.json` is well-formed (`jq . results/akash_pull_*/run.json | head`)
before going to §7.

### 6.3 CLI prerequisites

If you don't have `provider-services` installed:

```bash
curl -sSfL https://raw.githubusercontent.com/akash-network/provider/main/install.sh | sh
provider-services keys add my-key   # or import existing
```

Set the chain ID and node — instructions in the
[official CLI install guide](https://akash.network/docs/developers/deployment/cli/).

---

## 7. Close the lease and withdraw

### 7.1 Close from the Console

1. In the lease detail page, click **"Close Deployment"**.
2. Confirm. Akash refunds the unused escrow back to your ACT balance
   within ~10 minutes.

### 7.2 Withdraw funds (only if you want to off-ramp)

For credit-card users: ACT credits stay in your custodial Console
balance until you spend them. There is no off-ramp to USD; unused
credits are non-refundable. Don't over-fund.

For wallet users: leftover ACT can be **redeemed back to AKT** via the
"Redeem ACT" flow in the Console. Some slippage applies.

---

## 8. Failure-mode playbook

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| **Bid PreCheck warns "no providers can match"** | GPU model + RAM filter too narrow, or the `signedBy` auditor list excludes available providers. | Drop `ram: 80Gi` to `ram: 40Gi` (and switch SDL to `--model small` only — small fits in 40 GB easily). Or remove `signedBy` block (less safe; only do this on Phase 2A). |
| **No bids after 5 minutes** | Same as above, or your max price is below market. | Bump `amount: 5000` to `amount: 7500` and resubmit. |
| **Container restarts in a loop** | Image entrypoint exited non-zero. | Click "Events" in Console → look for the exit code. Most common: `WANDB_API_KEY` blank → script crashes if `WANDB_MODE=online`. Set it or switch to `WANDB_MODE=offline` in the env. |
| **Perplexity stuck near 27,000 after round 10** | Wrong device (CPU not CUDA). | Container probably picked CPU because CUDA wasn't visible. Click "Events" → look for "GPU not detected". Pick a different provider. |
| **Round time > 120 s on A100 80GB** | The provider gave you a slower A100 (40 GB) or shared GPU. | Close the lease, redeploy, pick a different provider. The 50.5-second mid-band assumes a dedicated 80 GB SXM A100. |
| **CUDA OOM** | Batch too big for the actual GPU memory. | Drop `--batch` from 8 to 4. Worst case 2. |
| **Provider drops mid-run** | Real but rare on audited providers. | Lease auto-terminates. If you saved checkpoints (§5.3), redeploy and pass `--resume /app/results/checkpoints/round_{N}.pt` (only works if the script supports `--resume`; if not, you restart from round 1 and absorb the cost). |
| **Escrow drained mid-run** | You disabled Auto Top-Up and bid pricing fluctuated. | Re-enable Auto Top-Up. Redeploy. |
| **`provider-services lease-shell` returns "permission denied"** | Cert expired or wallet not loaded. | Run `provider-services tx cert generate client --from <key> --gas auto` and retry. |
| **`run.json` is empty after pull** | Script crashed before round 1 finished. | Look at `lease.log` (also pulled by the helper). Most likely TinyStories download failed — re-deploy with `HF_HUB_OFFLINE=0` confirmed. |

---

## 9. Cost expectations

These are wall-clock and ACT-cost predictions, **not** ceilings — they
are what you should expect a healthy run to actually consume.

| Phase | Wall-clock (mid) | Realistic actual cost | Hard ceiling (your max bid × wall-clock) |
|-------|------------------|------------------------|------------------------------------------|
| 2A — calibration | 165 min | ~$3 ACT | $9 ACT |
| 2B — baseline backdoor | 165 min | ~$3 ACT | $9 ACT |
| 2C — headline AkashRep | 165 min | ~$3 ACT | $9 ACT |
| **Total all three** | ~8.5 h | ~$10 ACT | $30 ACT |

Budget recommendation: fund $50 ACT (covers a 60% over-run plus one
full re-run). The credit-card $100 trial is more than enough.

---

## 10. Reproducibility checklist (for the paper)

Capture and store in your run notes for **every** lease:

- [ ] Image digest (`sha256:…`) printed in §1.4.
- [ ] SDL file used, with comment-banner version preserved.
- [ ] Provider address and audited-attribute auditor.
- [ ] `dseq` (deployment sequence number).
- [ ] Lease start and end timestamps from the Console.
- [ ] Hash of `run.json` (`shasum -a 256 results/akash_pull_*/run.json`).
- [ ] Hash of the Akash reputation weights file (Phase 2C only).
- [ ] W&B run URL.

Without these, the run is not reproducible and reviewers can ask hard
questions you cannot answer.

---

## 11. Quick reference — full happy path commands

```bash
# Stage 1 -- build & push (once per code change).
cd /Users/asadr/Desktop/gradient-integrity-akash
docker build -f akash/Dockerfile -t asadrizvi06/gradient-integrity:phase2-v1 .
docker push asadrizvi06/gradient-integrity:phase2-v1
docker inspect asadrizvi06/gradient-integrity:phase2-v1 \
    --format '{{index .RepoDigests 0}}' \
    | tee akash/.last_pushed_digest

# Stage 2 -- smoke test locally on a CUDA box.
docker run --rm --gpus all \
    -e WANDB_MODE=offline \
    -v "$PWD/results:/app/results" \
    asadrizvi06/gradient-integrity:phase2-v1 \
    --model small --rounds 2 --batch 4 \
    --honest 2 --byz 0 --attack clean --defense FedAvg \
    --device cuda --output /app/results/smoke.json

# Stage 3 -- Akash Console deploy (UI). See §4.

# Stage 4 -- once active, pull results.
export AKASH_DSEQ=<from-console>
export AKASH_PROVIDER=<from-console>
export AKASH_KEY_NAME=my-key
bash akash/sync_results.sh

# Stage 5 -- close.
provider-services tx deployment close \
    --dseq $AKASH_DSEQ --from $AKASH_KEY_NAME \
    --gas auto --gas-adjustment 1.4
```

---

## 12. After the run — connecting back to the fixtures

For each phase, compare the final-round numbers in `run.json` to the
matching fixture's `expected_band`:

```bash
jq '.rounds[-1] | {perplexity, asr, round_time, agg_time}' \
    results/akash_pull_*/run.json
```

Then look up the corresponding fixture in
`results/expected/INDEX.json` (Phase 2A, 2B) or
`results/expected/reputation/INDEX.json` (Phase 2C). If the actuals fall
inside the [low, high] band, the run is consistent with the prediction
and the paper's headline survives. If anything falls outside, see the
drift-rules section of the relevant INDEX file before re-running.
