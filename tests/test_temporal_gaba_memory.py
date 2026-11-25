"""Comprehensive unit & property tests for temporal_gaba_memory."""
from __future__ import annotations

import json
import math
import subprocess
import sys
import unittest

from temporal_gaba_memory import (
    DeadendRecord,
    EpisodeRecord,
    MemoryConfig,
    TemporalGABAMemoryEngine,
    center_corpus_H,
    extract_python_ast_profile,
    is_psd_cholesky,
    tokenize_text,
    vec_normalize,
)


class TestTemporalGABAMemory(unittest.TestCase):
    def _sample_episodes(self):
        return [
            EpisodeRecord(
                episode_id="ep_pre_trap",
                timestamp_s=1000.0,
                tokens=tokenize_text("additive penalty gate broken premise"),
                dense_vec=[0.96, 0.08, 0.04, 0.02],
                ast_profile=extract_python_ast_profile("def f(x):\n    return x + 1\n"),
                branch_id="audit",
            ),
            EpisodeRecord(
                episode_id="ep_post_fix",
                timestamp_s=2400.0,
                tokens=tokenize_text("additive penalty gate fixed lexicographic veto"),
                dense_vec=[0.95, 0.10, 0.05, 0.02],
                ast_profile=extract_python_ast_profile(
                    "def gate(xs):\n    for x in xs:\n        if x < 0:\n            return False\n    return True\n"
                ),
                branch_id="audit",
            ),
            EpisodeRecord(
                episode_id="ep_hop1",
                timestamp_s=2600.0,
                tokens=tokenize_text("lean formal soundness lattice theorem"),
                dense_vec=[0.12, 0.94, 0.08, 0.05],
                ast_profile=extract_python_ast_profile("class A:\n    def m(self, x):\n        return x == 1\n"),
                branch_id="formal",
            ),
            EpisodeRecord(
                episode_id="ep_hop2",
                timestamp_s=2900.0,
                tokens=tokenize_text("merkle receipt audit deployment"),
                dense_vec=[0.08, 0.14, 0.95, 0.06],
                ast_profile=extract_python_ast_profile(
                    "def run(xs):\n    try:\n        return [x for x in xs]\n    except Exception:\n        return []\n"
                ),
                branch_id="deploy",
            ),
            EpisodeRecord(
                episode_id="ep_branch_quant",
                timestamp_s=2700.0,
                tokens=tokenize_text("spectral causality feature store"),
                dense_vec=[0.05, 0.06, 0.10, 0.97],
                ast_profile=extract_python_ast_profile("async def step(q):\n    while q:\n        q.pop()\n"),
                branch_id="quant",
            ),
        ]

    def test_causal_gaba_vetoes_pre_discovery_and_preserves_post_fix(self):
        episodes = self._sample_episodes()
        deadends = [
            DeadendRecord(
                deadend_id="de_01",
                discovered_ts_s=2000.0,
                dense_vec=[0.96, 0.08, 0.04, 0.02],
                threshold_cos=0.45,
            )
        ]
        engine = TemporalGABAMemoryEngine()
        engine.index(episodes, deadends, [("ep_post_fix", "ep_hop1"), ("ep_hop1", "ep_hop2")])
        rep = engine.query(["additive", "penalty", "gate"], [0.95, 0.09, 0.04, 0.02], top_k=4)

        self.assertIn("ep_pre_trap", rep.vetoed_ids)
        self.assertNotIn("ep_pre_trap", rep.episode_ids)
        self.assertIn("ep_post_fix", rep.episode_ids)
        self.assertTrue(rep.min_eig_psd_guard)
        self.assertGreater(rep.vendi_vs2_centered, 2.0)

    def test_causal_margin_boundary_does_not_veto_within_margin(self):
        # discovered_ts_s = 2000, margin = 120 -> ts = 1950 is NOT < 1880, must NOT be vetoed
        episodes = [
            EpisodeRecord("e_boundary", 1950.0, {"bug": 2}, [1.0, 0.0], [1.0, 0.0]),
            EpisodeRecord("e_other", 2100.0, {"clean": 2}, [0.0, 1.0], [0.0, 1.0]),
        ]
        deadends = [DeadendRecord("d1", 2000.0, [1.0, 0.0], threshold_cos=0.45)]
        engine = TemporalGABAMemoryEngine(MemoryConfig(causal_margin_s=120.0))
        engine.index(episodes, deadends, [])
        rep = engine.query(["bug"], [1.0, 0.0], top_k=2)
        self.assertEqual(rep.vetoed_ids, [])
        self.assertIn("e_boundary", rep.episode_ids)

    def test_reverse_temporal_edges_ignored(self):
        episodes = self._sample_episodes()
        engine = TemporalGABAMemoryEngine()
        # ep_hop2 (2900) -> ep_hop1 (2600) is backwards in time and must be dropped
        engine.index(episodes, [], [("ep_hop2", "ep_hop1"), ("ep_post_fix", "ep_hop1")])
        idx_hop2 = engine.id_to_idx["ep_hop2"]
        idx_fix = engine.id_to_idx["ep_post_fix"]
        self.assertEqual(engine.adj_out[idx_hop2], [])
        self.assertEqual(engine.adj_out[idx_fix], [engine.id_to_idx["ep_hop1"]])

    def test_bochner_psd_guard_and_non_psd_counterexample(self):
        # Non-PSD matrix (eigenvalues include -0.8) must be rejected
        non_psd = [
            [1.0, 0.9, 0.9],
            [0.9, 1.0, -0.9],
            [0.9, -0.9, 1.0],
        ]
        self.assertFalse(is_psd_cholesky(non_psd))

        # Paper's abs(G) non-PSD counterexample: v1=(1,0), v2=(0,1), v3=(cos 3pi/4, sin 3pi/4)
        # Gram G has off-diags (0, -s, s), taking abs(G) yields (0, s, s) with min eig = 1 - sqrt(2) < 0
        s = math.sqrt(0.5)
        abs_gram = [
            [1.0, 0.0, s],
            [0.0, 1.0, s],
            [s, s, 1.0],
        ]
        # Wait: eigenvalues of [[1,0,s],[0,1,s],[s,s,1]] are {0, 1, 2} (PSD);
        # with cos(120 deg) = -0.5 on 3 vectors -> G has -0.5 off-diag, abs(G) has +0.5 (PSD),
        # while 3 vectors with pairwise angles (60, 60, 120) -> G off-diag (0.5, 0.5, -0.5) vs abs(G) = (0.5, 0.5, 0.5)
        # Classic non-PSD abs(G): 4 vectors on circle at 0, 45, 90, 135 deg -> lambda_min(abs(G)) = -0.414214!
        angles = [0.0, math.pi / 4.0, math.pi / 2.0, 3.0 * math.pi / 4.0]
        pts = [(math.cos(a), math.sin(a)) for a in angles]
        abs_g4 = [
            [abs(pts[i][0] * pts[j][0] + pts[i][1] * pts[j][1]) for j in range(4)]
            for i in range(4)
        ]
        self.assertTrue(is_psd_cholesky(abs_gram))
        self.assertFalse(is_psd_cholesky(abs_g4))

    def test_corpus_centering_breaks_rank1_anisotropy_ceiling(self):
        # All 6 vectors share a strong constant rank-1 floor (anisotropy)
        base = [10.0, 10.0, 10.0, 10.0, 10.0, 10.0]
        anisotropic = []
        for i in range(6):
            row = list(base)
            row[i] += 1.0
            anisotropic.append(row)

        # Uncentered cosine similarities are all > 0.99 -> VS_2 collapses near 1.0
        def vs2_from_rows(rows):
            k = len(rows)
            frob_sq = 0.0
            for i in range(k):
                for j in range(k):
                    sim = math.exp(-sum((a - b) ** 2 for a, b in zip(rows[i], rows[j])) / (2.0 * 0.70 * 0.70))
                    frob_sq += sim * sim
            return (k * k) / frob_sq

        uncentered_unit = [
            [x / math.sqrt(sum(v * v for v in r)) for x in r] for r in anisotropic
        ]
        centered_unit, _ = center_corpus_H(anisotropic)
        vs2_raw = vs2_from_rows(uncentered_unit)
        vs2_cent = vs2_from_rows(centered_unit)
        self.assertLess(vs2_raw, 1.15)
        self.assertGreater(vs2_cent, 4.5)

    def test_validation_guards_and_empty_corpus(self):
        with self.assertRaises(ValueError):
            MemoryConfig(w_lex=-0.1, w_dense=0.6, w_ast=0.5)
        with self.assertRaises(ValueError):
            MemoryConfig(w_lex=0.5, w_dense=0.5, w_ast=0.5)
        with self.assertRaises(ValueError):
            MemoryConfig(sigma_ast=0.0)

        engine = TemporalGABAMemoryEngine()
        engine.index([], [], [])
        rep = engine.query(["test"], [1.0, 0.0], top_k=5)
        self.assertEqual(rep.episode_ids, [])
        self.assertTrue(rep.min_eig_psd_guard)

        # Duplicate episode_id rejected
        with self.assertRaises(ValueError):
            engine.index(
                [
                    EpisodeRecord("dup", 1.0, {"a": 1}, [1.0, 0.0], [1.0]),
                    EpisodeRecord("dup", 2.0, {"b": 1}, [0.0, 1.0], [1.0]),
                ]
            )

        # Query dimension mismatch rejected
        engine.index([EpisodeRecord("ok", 1.0, {"a": 1}, [1.0, 0.0], [1.0])])
        with self.assertRaises(ValueError):
            engine.query(["a"], [1.0, 0.0, 0.0])

    def test_nan_inf_overflow_negative_tokens_and_2hop_monotonicity(self):
        # 1. NaN / Inf rejected in is_psd_cholesky
        self.assertFalse(is_psd_cholesky([[float("nan")]]))
        self.assertFalse(is_psd_cholesky([[float("inf")]]))
        self.assertFalse(is_psd_cholesky([[1.0, float("nan")], [float("nan"), 1.0]]))

        # 2. NaN / Inf rejected in MemoryConfig
        with self.assertRaises(ValueError):
            MemoryConfig(w_lex=float("nan"), w_dense=0.5, w_ast=0.5)
        with self.assertRaises(ValueError):
            MemoryConfig(sigma_lex=float("inf"))

        # 3. Overflow-safe vec_normalize on huge finite magnitudes
        normed = vec_normalize([1e300, 1e300])
        self.assertAlmostEqual(normed[0], math.sqrt(0.5), places=6)
        self.assertAlmostEqual(normed[1], math.sqrt(0.5), places=6)

        # 4. NaN dense_vec, NaN timestamp_s, and negative token counts rejected at index()
        engine = TemporalGABAMemoryEngine()
        with self.assertRaises(ValueError):
            engine.index([EpisodeRecord("bad_nan", 1.0, {"a": 1}, [float("nan"), 1.0], [1.0])])
        with self.assertRaises(ValueError):
            engine.index([EpisodeRecord("bad_ts", float("nan"), {"a": 1}, [1.0, 0.0], [1.0])])
        with self.assertRaises(ValueError):
            engine.index([EpisodeRecord("bad_tok", 1.0, {"a": -5}, [1.0, 0.0], [1.0])])

        # 5. 2-hop RRF monotonicity: an already-present low-RRF neighbor must not score below a zero-RRF neighbor
        # Construct >40 distractor episodes so ep_zero_rrf drops out of top-40 RRF while ep_low_rrf stays in RRF
        eps = [
            EpisodeRecord("seed", 100.0, {"alpha": 10}, [1.0, 0.0], [1.0, 0.0]),
            EpisodeRecord("n1_low_rrf", 200.0, {"alpha": 1}, [0.5, 0.5], [0.5, 0.5]),
            EpisodeRecord("n1_zero_rrf", 210.0, {"zzz": 1}, [-1.0, 0.0], [0.0, 1.0]),
        ]
        for k in range(45):
            eps.append(EpisodeRecord(f"d_{k}", 150.0 + k * 0.1, {"alpha": 2}, [0.8, 0.2], [0.8, 0.2]))
        eng = TemporalGABAMemoryEngine()
        eng.index(eps, [], [("seed", "n1_low_rrf"), ("seed", "n1_zero_rrf")])
        rep = eng.query(["alpha"], [1.0, 0.0], top_k=10)
        if "n1_low_rrf" in rep.scores and "n1_zero_rrf" in rep.scores:
            self.assertGreaterEqual(rep.scores["n1_low_rrf"], rep.scores["n1_zero_rrf"])
        # Also verify direct internal score calculation guarantees monotonic ordering
        # Query with pool_limit=50 and lambda_div=0.0 so both appear in top_k=50
        eng_mono = TemporalGABAMemoryEngine(MemoryConfig(pool_limit=50, lambda_div=0.0))
        eng_mono.index(eps, [], [("seed", "n1_low_rrf"), ("seed", "n1_zero_rrf")])
        rep_mono = eng_mono.query(["alpha"], [1.0, 0.0], top_k=50)
        self.assertGreaterEqual(rep_mono.scores["n1_low_rrf"], rep_mono.scores["n1_zero_rrf"])

    def test_cli_demo_returns_valid_json(self):
        proc = subprocess.run(
            [sys.executable, "-m", "temporal_gaba_memory.cli", "demo"],
            capture_output=True,
            text=True,
            check=True,
        )
        data = json.loads(proc.stdout)
        self.assertTrue(data["pre_deadend_vetoed"])
        self.assertTrue(data["post_fix_preserved"])
        self.assertTrue(data["pool_kernel_psd_verified"])
        self.assertTrue(data["non_psd_counterexample_rejected"])


if __name__ == "__main__":
    unittest.main()
