# Temporal-GABA-Memory v1.0.0 — Reproducible 5-Regime Benchmark Report

- **Package Version:** `temporal-gaba-memory v1.0.0` (Pure Python 3.10+ stdlib, MIT License)
- **Master Deterministic Seed:** `20261126`
- **External API / Compute Cost:** `0.00 USD`
- **Total Wall-Clock Suite Runtime:** `15.440 s`

---

## Executive Summary Across All 5 Regimes

| Regime | Workload Scale | Baseline Failure Metric | `temporal-gaba-memory v1.0.0` | Wilson 95% CI | Median Latency |
|---|---|---:|---:|---:|---:|
| **1. Supersession Crowding (`m=20`)** | `8000` queries (`m=1..20`) | Naive `Recall@5 = 21.2%` (`StaleTrap@1 = 95.0%`) | **`Recall@5 = 100.0%` (`StaleTrap@1 = 0.0%`)** | `[99.05%, 100.00%]` | `0.666 ms` |
| **2. 2D Bi-Temporal Time-Travel** | `200` `as_of` queries / `50` splits | Mutable UPSERT Leakage = `42.5%` | **Future-Write Leakage = `0.0%` (`5/5` invariants)** | `[0.00%, 1.88%]` | `0.0047 ms` |
| **3. Cellular Sheaf `H^1` Paradox** | `100` multi-agent belief graphs | Pairwise Edge Auditor Recall = `0.0%` | **`H^1` Recall = `100.0%` (`FPR = 0.0%`)** | `[92.87%, 100.00%]` | `2.733 ms` |
| **4. Sunflower + RFC 6962 Merkle** | `500` redundant agent notes | Unpruned leaf bloat (`320` duplicate petals) | **`64.0%` note compression; `100.0%` Merkle verified** | `[99.15%, 100.00%]` | `22.83 ms` batch |
| **5. Temporal-Causal GABA Gate** | `200` adversarial trap/fix queries | Untimed Autoimmune FPR = `100.0%` | **Zombie Catch = `100.0%` (`Autoimmune FPR = 0.0%`)** | `[98.12%, 100.00%]` | `12.122 ms` |

---

## Regime 1 Detail: Supersession Crowding Curve (`m = 1..20`, `N = 400` per `m`)

Analytical law: `Recall@5(m) = min(1.0, 5 / (m + 1))`. At `m = 20` (`95.24%` tombstones), naive top-5 collapses to `5 / 21 = 23.81%`, whereas `SlotSheafCrowdingConsolidator` dynamically scales `k_eff` via exact Binomial KL-Chernoff + per-slot bounds to maintain `100.0%` active recovery and `0.0%` `StaleTrap@1`.

| `m` | Tombstone Ratio | `k_eff` (`z=3.0`) | Analytical Naive `Recall@5` | Empirical Naive `Recall@5` | Naive `StaleTrap@1` | Consolidator Active `Recall@5` | Consolidator `StaleTrap@1` |
|---:|---:|---:|---:|---:|---:|---:|---:|
| `1` | `50.00%` | `41` | `100.00%` | `100.00%` | `50.00%` | **`100.00%`** | **`0.00%`** |
| `2` | `66.67%` | `62` | `100.00%` | `100.00%` | `77.50%` | **`100.00%`** | **`0.00%`** |
| `4` | `80.00%` | `103` | `100.00%` | `100.00%` | `81.50%` | **`100.00%`** | **`0.00%`** |
| `5` | `83.33%` | `123` | `83.33%` | `82.50%` | `83.50%` | **`100.00%`** | **`0.00%`** |
| `8` | `88.89%` | `184` | `55.56%` | `55.00%` | `91.50%` | **`100.00%`** | **`0.00%`** |
| `10` | `90.91%` | `225` | `45.45%` | `47.00%` | `88.25%` | **`100.00%`** | **`0.00%`** |
| `12` | `92.31%` | `266` | `38.46%` | `50.75%` | `89.50%` | **`100.00%`** | **`0.00%`** |
| `15` | `93.75%` | `327` | `31.25%` | `37.00%` | `94.00%` | **`100.00%`** | **`0.00%`** |
| `18` | `94.74%` | `389` | `26.32%` | `29.25%` | `95.25%` | **`100.00%`** | **`0.00%`** |
| `20` | `95.24%` | `430` | `23.81%` | `21.25%` | `95.00%` | **`100.00%`** | **`0.00%`** |

---

## Reproducibility Command

```bash
PYTHONPATH=src python3 benchmarks/run_v1_memory_suite.py
```
