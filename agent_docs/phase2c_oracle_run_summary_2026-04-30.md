# Phase 2C Oracle — run summary

**Regime:** oracle reputation **`[1.0, 1.0, 0.0]`** (Byzantine client fully zero-weighted after reputation shaping)  

**Defense:** `AkashRep+FedAvg`  
**Attack:** backdoor `" cf"` → `" the"`  

**Date recorded:** 2026-04-30  

**Console-validated JSON fields** (`results/akash_2c_oracle/run.json` on laptop after curl):

```json
"config": {
  "defense": "AkashRep+FedAvg",
  "rep_weights": [1.0, 1.0, 0.0],
  "attack_type": "backdoor"
}
```

Logged rounds include **`asr`**, **`asr_per_position`**, **`asr_best_position`** (Option C metric).

---

## Terminal-reported finals (Akash Logs)

From training container stdout at completion:

- **Final perplexity:** ~**14.988**  
- **Final ASR (headline = max over positions):** **0.000**  
- **`best_pos`** in logs often **0** when all positions tie at zero ASR.

**Compared to oracle “utility” intuition:** Byzantine gradient removed ⇒ global update closer to **two honest clients only** ⇒ PPL can **differ** from full three-client FedAvg (not a bug per se).

---

## Artifacts

- Pull **`/app/results/run.json`** and **`checkpoints/round_200.pt`** via lease HTTP (see EXPERIMENT_RESULTS_INDEX.md).  
- Example local folder: `results/akash_2c_oracle/`.

---

## Cross-check commands

```bash
jq '.config | {defense, rep_weights, attack_type}' results/akash_2c_oracle/run.json
jq '.rounds[-1]' results/akash_2c_oracle/run.json
```
