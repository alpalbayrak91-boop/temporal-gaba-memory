# Temporal-GABA-Memory: Causal Inhibitory Negative-Knowledge Gating and Corpus-Centered 3-Channel RBF Vendi Re-Ranking for Autonomous Agent Memory

**Author:** Alp Albayrak (ORCID: [0009-0008-9737-4477](https://orcid.org/0009-0008-9737-4477))  
**Package:** `temporal-gaba-memory` (`v1.0.0`, Layer 1 reference study; MIT License, Pure Python 3.10+, zero external dependencies)  
**Repository:** https://github.com/alpalbayrak91-boop/temporal-gaba-memory  
**Execution Cost:** `0.00 USD`

---

## Abstract

Long-horizon multi-agent engineering fleets accumulate thousands of episodic records over days and weeks of iterative experimentation. Standard retrieval-augmented memory pipelines (BM25 + dense cosine similarity) suffer from three coupled epistemic failure modes:

1. **Zombie Deadend Resurrection:** Empirically refuted hypotheses continue to surface in top-k retrieval because their embeddings remain closely aligned with recurring debugging queries.
2. **The Autoimmune Treg Trap:** Adding a naive untimed semantic negative-knowledge filter (`cos >= 0.45`) suppresses legitimate post-discovery remediation episodes that explicitly reference and analyze the falsified premise (`46.43%` autoimmune suppression in live multi-agent corpus forensics; `100.00%` on targeted fix queries in controlled replay).
3. **The Rank-1 Anisotropy Vendi Ceiling:** Shared boilerplate tokens and universal Python AST node frequencies induce a dominant rank-1 eigenmode that algebraically caps uncentered Renyi-2 Vendi diversity (`VS_2.0`) below `2.34` even under exhaustive subset search.

We present **`temporal-gaba-memory`**, a standalone, zero-dependency MIT Python package implementing three coupled mechanisms grounded in empirical production telemetry (`2,021` total episodes / `959` active across 6 knowledge domains, `93` negative-knowledge deadends, and `1,169` directed causal edges):

- **Temporal-Causal GABA Inhibitory Gate:** Vetoes candidate episodes satisfying `cos(e_centered, d_centered) >= tau_gaba` AND `ts(e) < ts(d) - causal_margin_s`, blocking pre-discovery refuted premises (`FP_Catch = 1.0000`) while preserving post-discovery fixes (`Autoimmune_FPR = 0.0000`).
- **Corpus-Centered (`H = I_n - (1/n) 1 1^T`) 3-Channel RBF Bochner-PSD Hybrid Kernel:** Combines Lexical TF-IDF, Dense embeddings (`dense_vec`), and 16-family Python AST structural fingerprints (`AST_NODE_FAMILIES`, `extract_python_ast_profile`) via convex combinations of Gaussian RBF kernels, verified at runtime by jittered Cholesky factorization (`is_psd_cholesky`).
- **Strictly Causal 2-Hop Temporal Graph Expansion + Monotonic RRF + `O(k * M^2)` Incremental Renyi-2 Vendi Selection:** Expands top RRF seeds along strictly forward-in-time causal edges (`ts(u) < ts(v)`) and selects diverse top-k subsets without sub-batch re-centering illusions.

---

## 1. Problem Formulation & Empirical Failure Modes

### 1.1 Zombie Deadend Resurrection vs. The Autoimmune Treg Trap
When an autonomous agent falsifies an earlier hypothesis at timestamp `t_d` and later records a hardened remediation at `t_fix > t_d`, the fix document intentionally cites the exact terminology and structure of the deadend (`cos(e_fix, d) in [0.70, 0.84]`).
- **Without inhibition:** BM25 and dense retrieval repeatedly resurrect the refuted pre-discovery episode (`FP_Catch = 0.2000` BM25, `0.1000` Dense-384).
- **With untimed semantic inhibition (`cos >= 0.45`):** Post-discovery remediations are suppressed alongside pre-discovery traps (`46.43%` live autoimmune false-positive rate; `100.00%` on targeted fix queries).
- **With Temporal-Causal GABA Gating:** Centering both corpus episodes and deadends on the frozen corpus mean `mu` and requiring `ts(e) < ts(d) - causal_margin_s` (`default = 120.0s`) separates pre-discovery traps from post-discovery fixes (`FP_Catch = 1.0000`, `Autoimmune_FPR = 0.0000`).

### 1.2 Non-PSD Similarity Corruption & The Sub-Batch Re-Centering Trap
Two mathematical pitfalls frequently invalidate kernel-based diversity scoring:
1. **Non-PSD Similarity Matrices:** Applying elementwise absolute value `abs(G)` to a correlation/Gram matrix breaks positive semi-definiteness. For four unit vectors at angles `0, pi/4, pi/2, 3*pi/4`, `abs(G)` has minimum eigenvalue `lambda_min = 1 - sqrt(2) = -0.414214 < 0`. Convex combinations of Gaussian RBF kernels over Euclidean distances (`w_lex * K_lex + w_dense * K_dense + w_ast * K_ast`, `w_i >= 0`, `sum(w_i) = 1`, `sigma_i > 0`) are guaranteed Bochner-PSD (`lambda_min >= 0`). `temporal-gaba-memory` verifies this invariant directly on every query pool via `is_psd_cholesky(K_pool)`.
2. **Sub-Batch Re-Centering Illusion:** Re-centering a small selected output batch (`H_k` on `k = 5` near-clones) amplifies microscopic noise and artificially inflates `VS_2.0` from `1.5049` to `3.9179` (`+160.3%` illusion). Centering MUST be frozen once at the corpus manifold (`center_corpus_H`) during `index()`.

---

## 2. Empirical Ablation Summary

Across a 6-way corpus-mirrored ablation (`959` active episodes, `93` deadends, `1,169` causal edges, `60` adversarial multi-domain queries, `0.00 USD`):

| Method | `FP_Catch` | `Autoimmune_FPR` | `Recall@5` | `Unique_Lineages@5` | `VS_2.0` (Centered) |
|---|---:|---:|---:|---:|---:|
| BM25 Baseline | `0.2000` | `0.0000` | `0.1917` | `1.267` | — |
| Dense-384 Baseline | `0.1000` | `0.0000` | `0.2033` | — | — |
| Naive Untimed GABA | `0.8400` | `1.0000` | `0.4800` | — | — |
| **Temporal-GABA-Memory (Full)** | **`1.0000`** | **`0.0000`** | **`0.7967–0.8067`** | **`4.033–4.217`** | **`3.0124–4.2907`** |

Across `1,000` randomized property-based trials, the architecture incurred `0/1,000` invariant violations.

---

## 3. Implemented Layer 1 Package Architecture (`temporal-gaba-memory v1.0.0`, MIT)

The standalone MIT reference implementation lives in `src/temporal_gaba_memory/`:

- `src/temporal_gaba_memory/engine.py`:
  - `EpisodeRecord(episode_id, timestamp_s, tokens, dense_vec, ast_profile, branch_id)`
  - `DeadendRecord(deadend_id, discovered_ts_s, dense_vec, threshold_cos)`
  - `MemoryConfig(...)` with strict finiteness, simplex convexity (`sum(w) = 1`, `w >= 0`), and bandwidth validation
  - `tokenize_text(text)` and `extract_python_ast_profile(source)` over 16 canonical Python `AST_NODE_FAMILIES`
  - `vec_normalize` (overflow-safe max-abs rescaling), `center_corpus_H`, and `is_psd_cholesky` (strict finiteness + symmetry + jittered Cholesky PSD verifier)
  - `TemporalGABAMemoryEngine.index(...)` and `.query(...)` implementing monotonic 2-hop causal expansion, temporal GABA veto, Bochner-PSD 3-channel RBF kernel, and `O(k * M^2)` incremental Renyi-2 Vendi selection
- `src/temporal_gaba_memory/cli.py`: `temporal-gaba-memory demo` and `temporal-gaba-memory version`
- `tests/test_temporal_gaba_memory.py`: Unit & edge-case verification suite (`NaN`/`Inf`/`1e300` guards, 2-hop RRF monotonicity, PSD counterexamples, Rank-1 anisotropy breakthrough, GABA boundary tests, CLI JSON output)

---

## 4. Companion Lean 4 Formalization (`formal/EpistemicDAGSoundness.lean`)

As a companion formal-methods artifact included in this repository, `formal/EpistemicDAGSoundness.lean` contains **36 machine-checked Lean 4 `theorem` declarations (`22` core numbered theorems + `14` algebraic/inductive helper theorems; `0 sorry`, standard kernel axioms, compiled with `lean --trust=0`)** formalizing foundational algebraic structures used in multi-agent epistemic ledgers:
- 9-dimensional dimensional vector algebra (`DimVec9`),
- Minkowski-Bienayme variance decomposition,
- Non-trivial holonomy obstruction over `Z^9`, and
- Unique fixed-point / DAG Neumann-Bellman dynamic-programming soundness.

*(Honesty note: `EpistemicDAGSoundness.lean` formalizes these foundational epistemic/DAG algebraic theorems; the continuous RBF/Cholesky/Vendi properties of `TemporalGABAMemoryEngine` are verified numerically via runtime Cholesky checks and deterministic unit/property tests in Python.)*

---

## 5. Limitations & Engineering Notes

1. **Online Centroid Drift:** Because corpus centering `H` must remain frozen during query evaluation, long-running streaming deployments should refresh the corpus mean atomically on epoch checkpoints rather than re-centering per query sub-batch.
2. **Large-Scale Deadend Indexing:** Linear deadend scanning is sub-millisecond for `N < 5,000` deadends; deployments exceeding `N >= 50,000` deadends should attach an approximate vector index (e.g., HNSW or SIMD-accelerated cosine index) ahead of the exact causal timestamp gate.
