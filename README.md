# temporal-gaba-memory

Zero-dependency Pure Python 3.10+ reference implementation of **AlpLab-Temporal-GABA-Memory**:
1. **Temporal-Causal GABA Inhibitory Gate** (`cos(e, d) >= tau_gaba` AND `ts(e) < ts(d) - delta_causal`): blocks resurrected pre-discovery falsified hypotheses (*Zombie Deadends*) while protecting post-discovery remediation episodes from the *Autoimmune Treg Trap* (`0.00%` autoimmune suppression).
2. **Corpus-Centered (`H = I - (1/n) 1 1^T`) 3-Channel RBF Bochner-PSD Hybrid Kernel**: fuses Lexical TF-IDF, Dense embeddings, and 16-family Python AST structural fingerprints with runtime Cholesky PSD verification (`is_psd_cholesky`), breaking the Rank-1 Anisotropy Vendi Ceiling without sub-batch re-centering illusions.
3. **2-Hop Strictly Causal Temporal Graph + RRF + `O(k * M^2)` Greedy Renyi-2 Vendi Selection**: prevents near-duplicate retry collapse in long-horizon multi-agent RAG loops.
4. **Erdős–Rado Sunflower Memory Pruner (`SunflowerMemoryPruner`) & Toulouse Spin-Glass Frustration Auditor (`FrustrationCutAuditor`, v0.2.0)**: compresses `>= k` redundant petal findings around canonical invariant cores (`post_prune_sunflower_free = True` in `3.8 ms` on 200 episodes) and isolates minimal frustrated contradiction cuts (`Phi_0(G) = |E| - MaxCut(G)`), backed by `formal/Erdos23C5BlowupFrustration.lean` (`22 + 5 = 27` total `--trust=0` Lean 4 theorems).

## Quickstart

```bash
pip install .
temporal-gaba-memory demo
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

## Technical Paper & Companion Formal Proof

- **Empirical Paper (`docs/PAPER.md`):** Documents the *Zombie Deadend*, *Autoimmune Treg Trap*, *Rank-1 Anisotropy Vendi Ceiling*, *Non-PSD `abs(G)` Counterexample*, and *Sub-Batch Re-Centering Trap* across live multi-agent telemetry (`2,021` episodes, `93` deadends, `1,169` causal edges; `FP_Catch = 1.0000`, `Autoimmune_FPR = 0.0000`, `VS_2.0 >= 3.01`).
- **Companion Lean 4.35 Formalization (`formal/EpistemicDAGSoundness.lean`):** 22 machine-checked theorems (`0 sorry`, `lean --trust=0`, standard Lean kernel axioms only) formalizing foundational multi-agent epistemic algebra (9-dimensional vector algebra `DimVec9`, Minkowski-Bienayme variance decomposition, `Z^9` holonomy obstruction, and DAG Neumann/Bellman DP uniqueness).

## License

MIT License (Copyright 2026 Alp Albayrak).
