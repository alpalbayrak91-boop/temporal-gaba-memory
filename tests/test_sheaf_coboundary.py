"""Unit tests for CellularSheafCoboundaryAuditor (temporal-gaba-memory v0.3.0).

Pure Python 3.10+ stdlib unittest suite (zero external dependencies).
"""
from __future__ import annotations

import ast
import math
import pathlib
import random
import sys
import time
import unittest

try:
    from temporal_gaba_memory.sheaf_coboundary import (  # type: ignore[import-not-found]
        SHEAF_MODULE_VERSION,
        CellularSheafCoboundaryAuditor,
        SheafEdge,
        SheafNode,
        run_sheaf_reference_benchmark,
    )
except ImportError:
    _ROOT = pathlib.Path(__file__).resolve().parents[1]
    _VITRIN = _ROOT / "vitrin"
    if str(_VITRIN) not in sys.path:
        sys.path.insert(0, str(_VITRIN))
    from temporal_gaba_memory.sheaf_coboundary import (  # type: ignore[import-not-found]
        SHEAF_MODULE_VERSION,
        CellularSheafCoboundaryAuditor,
        SheafEdge,
        SheafNode,
        run_sheaf_reference_benchmark,
    )


class TestCellularSheafCoboundaryAuditor(unittest.TestCase):
    def setUp(self) -> None:
        self.auditor = CellularSheafCoboundaryAuditor()

    def test_01_affine_3cycle_local_pairwise_consistent_but_global_h1_obstruction(self) -> None:
        """Exact analytical check: weights (1, 1, 0.5), offsets (+10, +10, +10) => ||h||_2 = 15.0."""
        nodes = [SheafNode("A", 1), SheafNode("B", 1), SheafNode("C", 1)]
        edges = [
            SheafEdge.identity("e_ab", "A", "B", dim=1, offset=(10.0,), weight=1.0),
            SheafEdge.identity("e_bc", "B", "C", dim=1, offset=(10.0,), weight=1.0),
            SheafEdge.identity("e_ca", "C", "A", dim=1, offset=(10.0,), weight=0.5),
        ]
        rep = self.auditor.audit(nodes, edges)
        self.assertTrue(rep.all_edges_pairwise_consistent)
        self.assertFalse(rep.is_globally_consistent)
        self.assertTrue(rep.has_local_to_global_paradox)
        self.assertEqual(rep.total_dim_h0, 1)
        self.assertEqual(rep.total_dim_h1, 1)
        self.assertAlmostEqual(rep.global_h1_obstruction_norm, 15.0, places=6)
        self.assertAlmostEqual(rep.global_h1_obstruction_energy, 225.0, places=5)
        self.assertEqual(len(rep.global_3plus_obstructions), 1)
        cyc = rep.global_3plus_obstructions[0]
        self.assertEqual(cyc.cycle_length, 3)
        self.assertEqual(cyc.obstruction_kind, "affine_h1")
        self.assertTrue(cyc.locally_pairwise_consistent)
        self.assertAlmostEqual(cyc.affine_transport_defect or 0.0, 30.0, places=6)
        # Lowest-weight edge e_ca (weight=0.5) must be chosen for minimal cut
        self.assertEqual(rep.recommended_cut_edges, ("e_ca",))
        self.assertTrue(rep.cut_restores_global_consistency)
        self.assertTrue(rep.components[0].euler_invariant_ok)

    def test_02_flat_consistent_3cycle_zero_false_positive(self) -> None:
        """Non-trivial cycle topology (dim H^1 = 1) with zero cohomology class [b] = 0 must pass."""
        nodes = [SheafNode("A", 1), SheafNode("B", 1), SheafNode("C", 1)]
        edges = [
            SheafEdge.identity("e_ab", "A", "B", dim=1, offset=(10.0,)),
            SheafEdge.identity("e_bc", "B", "C", dim=1, offset=(5.0,)),
            SheafEdge.identity("e_ca", "C", "A", dim=1, offset=(-15.0,)),
        ]
        rep = self.auditor.audit(nodes, edges)
        self.assertTrue(rep.all_edges_pairwise_consistent)
        self.assertTrue(rep.is_globally_consistent)
        self.assertFalse(rep.has_local_to_global_paradox)
        self.assertEqual(rep.total_dim_h0, 1)
        self.assertEqual(rep.total_dim_h1, 1)
        self.assertAlmostEqual(rep.global_h1_obstruction_norm, 0.0, places=9)
        self.assertEqual(len(rep.cycle_obstructions), 0)
        self.assertEqual(rep.recommended_cut_edges, ())
        # Reconciled global section must satisfy all three constraints exactly
        rec = rep.components[0].reconciled_states
        self.assertAlmostEqual(rec["B"][0] - rec["A"][0], 10.0, places=8)
        self.assertAlmostEqual(rec["C"][0] - rec["B"][0], 5.0, places=8)
        self.assertAlmostEqual(rec["A"][0] - rec["C"][0], -15.0, places=8)

    def test_03_so2_rotation_holonomy_3cycle_frustrated_vs_4cycle_flat(self) -> None:
        """3 x 90 deg = 270 deg != I_2 triggers holonomy frustration; 4 x 90 deg = 360 deg is flat."""
        rot90 = ((0.0, -1.0), (1.0, 0.0))
        n3 = [SheafNode(f"R{i}", 2) for i in range(3)]
        e3 = [
            SheafEdge.linear_transport(f"e{i}", f"R{i}", f"R{(i + 1) % 3}", rot90, weight=1.0 if i < 2 else 0.7)
            for i in range(3)
        ]
        rep3 = self.auditor.audit(n3, e3)
        self.assertTrue(rep3.all_edges_pairwise_consistent)
        self.assertFalse(rep3.is_globally_consistent)
        self.assertTrue(rep3.has_local_to_global_paradox)
        self.assertEqual(rep3.total_dim_h0, 0)  # Consensus collapses to {0}
        self.assertAlmostEqual(rep3.max_holonomy_defect, 2.0, places=6)
        self.assertEqual(len(rep3.global_3plus_obstructions), 1)
        self.assertEqual(rep3.global_3plus_obstructions[0].obstruction_kind, "holonomy_frustration")
        self.assertGreater(rep3.components[0].min_laplacian_eig or 0.0, 0.1)
        self.assertEqual(rep3.recommended_cut_edges, ("e2",))
        self.assertTrue(rep3.cut_restores_global_consistency)

        n4 = [SheafNode(f"Q{i}", 2) for i in range(4)]
        e4 = [
            SheafEdge.linear_transport(f"q{i}", f"Q{i}", f"Q{(i + 1) % 4}", rot90)
            for i in range(4)
        ]
        rep4 = self.auditor.audit(n4, e4)
        self.assertTrue(rep4.is_globally_consistent)
        self.assertFalse(rep4.has_local_to_global_paradox)
        self.assertEqual(rep4.total_dim_h0, 2)
        self.assertAlmostEqual(rep4.components[0].min_laplacian_eig or 0.0, 0.0, places=8)
        self.assertEqual(len(rep4.cycle_obstructions), 0)

    def test_04_signed_graph_pentagon_c5_matches_spin_glass_spectrum(self) -> None:
        """Embeds C_5 frustration graph (sign=-1): lambda_min(L_0) = 2(1 - cos(pi/5))."""
        nodes = [SheafNode(f"H{i}", 1) for i in range(5)]
        edges = [
            SheafEdge.signed(f"s{i}", f"H{i}", f"H{(i + 1) % 5}", sign=-1, weight=1.0)
            for i in range(5)
        ]
        rep = self.auditor.audit(nodes, edges)
        expected_eig = 2.0 * (1.0 - math.cos(math.pi / 5.0))
        self.assertTrue(rep.has_local_to_global_paradox)
        self.assertEqual(rep.total_dim_h0, 0)
        self.assertAlmostEqual(rep.max_holonomy_defect, 2.0, places=6)
        self.assertAlmostEqual(rep.components[0].min_laplacian_eig or 0.0, expected_eig, places=6)
        self.assertEqual(len(rep.recommended_cut_edges), 1)
        self.assertTrue(rep.cut_restores_global_consistency)

        # Even C_4 with sign=-1 has holonomy (-1)^4 = +1 -> globally consistent!
        n4 = [SheafNode(f"E{i}", 1) for i in range(4)]
        e4 = [SheafEdge.signed(f"se{i}", f"E{i}", f"E{(i + 1) % 4}", sign=-1) for i in range(4)]
        rep4 = self.auditor.audit(n4, e4)
        self.assertTrue(rep4.is_globally_consistent)
        self.assertEqual(rep4.total_dim_h0, 1)

    def test_05_heterogeneous_rectangular_projection_paradox(self) -> None:
        """Stalk dims (2, 1, 1) with rectangular projections: pairwise consistent, global H^1 != 0."""
        nodes = [
            SheafNode("ledger", stalk_dim=2),
            SheafNode("summary", stalk_dim=1),
            SheafNode("auditor", stalk_dim=1),
        ]
        edges = [
            SheafEdge.projection_agreement("e1", "ledger", "summary", ((1.0, 1.0),), ((1.0,),), offset=(0.0,)),
            SheafEdge.identity("e2", "summary", "auditor", dim=1, offset=(0.0,)),
            SheafEdge.projection_agreement("e3", "ledger", "auditor", ((1.0, 1.0),), ((1.0,),), offset=(12.0,), weight=0.5),
        ]
        rep = self.auditor.audit(nodes, edges)
        self.assertTrue(rep.all_edges_pairwise_consistent)
        self.assertTrue(rep.has_local_to_global_paradox)
        self.assertEqual(rep.total_dim_h0, 2)  # (base, tax) with 2 independent rows => 4 - 2 = 2
        self.assertEqual(rep.total_dim_h1, 1)
        # Analytical weighted H^1 norm: 12 / sqrt(1/1 + 1/1 + 1/0.5) = 12 / 2 = 6.0
        self.assertAlmostEqual(rep.global_h1_obstruction_norm, 6.0, places=6)
        self.assertEqual(rep.recommended_cut_edges, ("e3",))
        self.assertTrue(rep.cut_restores_global_consistency)

    def test_06_hodge_pythagorean_decomposition_gauge_error_vs_topological_h1(self) -> None:
        """Distinguishes recalibratable agent noise (exact) from irreducible H^1 contradiction (harmonic)."""
        # Flat cycle with noisy observed state at B (4.0 gauge error, 0.0 harmonic)
        nodes_flat = [
            SheafNode("A", state=(0.0,)),
            SheafNode("B", state=(14.0,)),  # true consistent value is 10.0
            SheafNode("C", state=(15.0,)),
        ]
        edges_flat = [
            SheafEdge.identity("e_ab", "A", "B", offset=(10.0,)),
            SheafEdge.identity("e_bc", "B", "C", offset=(5.0,)),
            SheafEdge.identity("e_ca", "C", "A", offset=(-15.0,)),
        ]
        rep_flat = self.auditor.audit(nodes_flat, edges_flat)
        c0 = rep_flat.components[0]
        self.assertTrue(rep_flat.is_globally_consistent)
        self.assertGreater(c0.observed_exact_reconcilable_norm or 0.0, 5.0)
        self.assertAlmostEqual(c0.observed_harmonic_norm or 0.0, 0.0, places=8)
        self.assertLess(c0.hodge_pythagorean_error or 1.0, 1e-10)

        # Now inject genuine H^1 obstruction (+6 loop gap) ON TOP OF noisy agent states
        edges_obs = [
            SheafEdge.identity("e_ab", "A", "B", offset=(10.0,)),
            SheafEdge.identity("e_bc", "B", "C", offset=(5.0,)),
            SheafEdge.identity("e_ca", "C", "A", offset=(-9.0,)),  # sum = +6
        ]
        rep_obs = self.auditor.audit(nodes_flat, edges_obs)
        c1 = rep_obs.components[0]
        self.assertTrue(rep_obs.has_local_to_global_paradox)
        expected_harm = 6.0 / math.sqrt(3.0)
        self.assertAlmostEqual(c1.observed_harmonic_norm or 0.0, expected_harm, places=6)
        self.assertAlmostEqual(c1.h1_obstruction_norm, expected_harm, places=6)
        self.assertLess(c1.hodge_pythagorean_error or 1.0, 1e-10)

    def test_07_temporal_causal_windowing_as_of_s_retires_stale_obstruction(self) -> None:
        """Superseded edge at t=1500 must not cause false H^1 alarm at as_of_s=2000."""
        nodes = [
            SheafNode("A", 1, valid_from_s=0.0),
            SheafNode("B", 1, valid_from_s=0.0),
            SheafNode("C", 1, valid_from_s=0.0),
        ]
        edges = [
            SheafEdge.identity("e_ab", "A", "B", offset=(10.0,), valid_from_s=100.0),
            SheafEdge.identity("e_bc", "B", "C", offset=(5.0,), valid_from_s=100.0),
            SheafEdge.identity("e_ca_stale", "C", "A", offset=(+20.0,), valid_from_s=100.0, valid_until_s=1500.0),
            SheafEdge.identity("e_ca_fixed", "C", "A", offset=(-15.0,), valid_from_s=1600.0),
        ]
        rep_old = self.auditor.audit(nodes, edges, as_of_s=1200.0)
        self.assertFalse(rep_old.is_globally_consistent)
        self.assertTrue(rep_old.has_local_to_global_paradox)
        self.assertEqual(rep_old.filtered_inactive_edges, 1)

        rep_new = self.auditor.audit(nodes, edges, as_of_s=2000.0)
        self.assertTrue(rep_new.is_globally_consistent)
        self.assertFalse(rep_new.has_local_to_global_paradox)
        self.assertEqual(rep_new.filtered_inactive_edges, 1)

    def test_08_non_abelian_gl2_scaling_currency_arbitrage_cycle(self) -> None:
        """Multiplicative drift 1.2 * 1.2 * 1.2 = 1.728 != 1.0 across 3 sources (USD-like rate loop)."""
        nodes = [SheafNode("src_x", 1), SheafNode("src_y", 1), SheafNode("src_z", 1)]
        edges = [
            SheafEdge.linear_transport("xy", "src_x", "src_y", ((1.2,),), weight=1.0),
            SheafEdge.linear_transport("yz", "src_y", "src_z", ((1.2,),), weight=1.0),
            SheafEdge.linear_transport("zx", "src_z", "src_x", ((1.2,),), weight=0.4),
        ]
        rep = self.auditor.audit(nodes, edges)
        self.assertTrue(rep.has_local_to_global_paradox)
        self.assertAlmostEqual(rep.max_holonomy_defect, abs(1.2 ** 3 - 1.0), places=6)
        self.assertEqual(rep.recommended_cut_edges, ("zx",))
        self.assertTrue(rep.cut_restores_global_consistency)

    def test_09_pairwise_2cycle_vs_3plus_global_cycle_classification(self) -> None:
        """Parallel conflicting edges form a 2-cycle and must NOT be mislabeled as a 3+ global paradox."""
        nodes = [SheafNode("U", 1), SheafNode("V", 1)]
        edges = [
            SheafEdge.identity("p1", "U", "V", offset=(5.0,)),
            SheafEdge.identity("p2", "U", "V", offset=(9.0,)),
        ]
        rep = self.auditor.audit(nodes, edges)
        self.assertFalse(rep.is_globally_consistent)
        self.assertFalse(rep.has_local_to_global_paradox)
        self.assertEqual(len(rep.global_3plus_obstructions), 0)
        self.assertEqual(len(rep.cycle_obstructions), 1)
        self.assertEqual(rep.cycle_obstructions[0].cycle_length, 2)
        self.assertEqual(len(rep.recommended_cut_edges), 1)
        self.assertTrue(rep.cut_restores_global_consistency)

    def test_10_multi_component_euler_invariant_and_multiple_cuts(self) -> None:
        """Two disjoint frustrated cycles require 2 cuts (one per component) to restore consistency."""
        nodes = [SheafNode(f"A{i}", 1) for i in range(3)] + [SheafNode(f"B{i}", 1) for i in range(4)]
        edges = [
            SheafEdge.identity("a01", "A0", "A1", offset=(1.0,)),
            SheafEdge.identity("a12", "A1", "A2", offset=(1.0,)),
            SheafEdge.identity("a20", "A2", "A0", offset=(1.0,), weight=0.3),
            SheafEdge.identity("b01", "B0", "B1", offset=(2.0,)),
            SheafEdge.identity("b12", "B1", "B2", offset=(2.0,)),
            SheafEdge.identity("b23", "B2", "B3", offset=(2.0,)),
            SheafEdge.identity("b30", "B3", "B0", offset=(2.0,), weight=0.2),
        ]
        rep = self.auditor.audit(nodes, edges)
        self.assertEqual(len(rep.components), 2)
        self.assertTrue(all(c.euler_invariant_ok for c in rep.components))
        self.assertEqual(len(rep.global_3plus_obstructions), 2)
        self.assertEqual(set(rep.recommended_cut_edges), {"a20", "b30"})
        self.assertTrue(rep.cut_restores_global_consistency)

    def test_11_validation_errors_and_empty_graph_safety(self) -> None:
        rep_empty = self.auditor.audit([], [])
        self.assertTrue(rep_empty.is_globally_consistent)
        self.assertEqual(rep_empty.total_dim_h0, 0)
        self.assertEqual(rep_empty.total_dim_h1, 0)

        with self.assertRaises(ValueError):
            self.auditor.audit(
                [SheafNode("A", 2), SheafNode("B", 1)],
                [SheafEdge.identity("bad_dim", "A", "B", dim=2)],
            )
        with self.assertRaises(ValueError):
            self.auditor.audit([SheafNode("A", 1)], [SheafEdge.identity("bad_ref", "A", "MISSING", dim=1)])

    def test_12_randomized_property_euler_and_consistent_vs_perturbed_cycles(self) -> None:
        """100 randomized graphs: exact ground-truth section => H^1 == 0; cycle perturbation => H^1 > 0."""
        rng = random.Random(20261126)
        for trial in range(50):
            n_nodes = rng.randint(3, 8)
            dim = rng.randint(1, 3)
            truth = [[rng.uniform(-5.0, 5.0) for _ in range(dim)] for _ in range(n_nodes)]
            nodes = [SheafNode(f"n{i}", dim, state=tuple(truth[i])) for i in range(n_nodes)]
            edges = []
            for i in range(n_nodes):
                j = (i + 1) % n_nodes
                off = tuple(truth[j][d] - truth[i][d] for d in range(dim))
                edges.append(SheafEdge.identity(f"e{i}", f"n{i}", f"n{j}", dim=dim, offset=off))
            rep_clean = self.auditor.audit(nodes, edges)
            self.assertTrue(rep_clean.is_globally_consistent, msg=f"trial {trial} failed clean")
            self.assertTrue(rep_clean.components[0].euler_invariant_ok)
            self.assertLess(rep_clean.global_h1_obstruction_norm, 1e-7)

            # Perturb closing edge offset so [b] != 0 in H^1
            bad_off = tuple(edges[-1].offset[d] + (2.5 if d == 0 else 0.0) for d in range(dim))  # type: ignore[index]
            edges[-1] = SheafEdge.identity(
                edges[-1].edge_id, edges[-1].source, edges[-1].target, dim=dim, offset=bad_off, weight=0.5
            )
            rep_bad = self.auditor.audit(nodes, edges)
            self.assertTrue(rep_bad.has_local_to_global_paradox, msg=f"trial {trial} missed perturbation")
            self.assertGreater(rep_bad.global_h1_obstruction_norm, 0.5)
            self.assertTrue(rep_bad.cut_restores_global_consistency)

    def test_13_reference_benchmark_and_public_stdlib_hygiene(self) -> None:
        t0 = time.perf_counter()
        bench = run_sheaf_reference_benchmark()
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        self.assertEqual(bench["module_version"], SHEAF_MODULE_VERSION)
        self.assertTrue(bench["affine_3cycle_paradox"]["has_local_to_global_paradox"])
        self.assertTrue(bench["so2_3cycle_holonomy_paradox"]["has_local_to_global_paradox"])
        self.assertTrue(bench["so2_4cycle_flat_control"]["is_globally_consistent"])
        self.assertTrue(bench["heterogeneous_projection_paradox"]["has_local_to_global_paradox"])
        self.assertLess(elapsed_ms, 2000.0)

        # Verify zero external dependencies (pure stdlib) and zero ASCII-36 characters in module source
        mod_path = pathlib.Path(sys.modules[CellularSheafCoboundaryAuditor.__module__].__file__ or "")
        raw = mod_path.read_bytes()
        self.assertEqual(raw.count(36), 0)
        tree = ast.parse(raw.decode("utf-8"))
        stdlib_names = getattr(sys, "stdlib_module_names", set())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root_pkg = alias.name.split(".")[0]
                    self.assertIn(root_pkg, stdlib_names | {"__future__"})
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                root_pkg = node.module.split(".")[0]
                self.assertIn(root_pkg, stdlib_names | {"__future__"})


if __name__ == "__main__":
    unittest.main(verbosity=2)
