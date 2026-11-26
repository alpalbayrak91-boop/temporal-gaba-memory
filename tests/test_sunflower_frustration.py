import unittest
from temporal_gaba_memory.sunflower_frustration import (
    FrustrationCutAuditor,
    HypothesisEdge,
    run_comparative_benchmark,
)


class TestSunflowerAndFrustrationOperators(unittest.TestCase):
    def test_comparative_200_finding_benchmark_exact_recovery(self):
        res = run_comparative_benchmark(seed=20261125)
        self.assertTrue(res["schema_parity"]["synthetic_sqlite"])
        sf = res["synthetic_200"]["sunflower"]
        self.assertEqual(sf["input_count"], 200)
        self.assertEqual(sf["cluster_count"], 12)
        self.assertEqual(sf["pruned_count"], 48)
        self.assertEqual(sf["surviving_count"], 152)
        self.assertTrue(sf["post_prune_sunflower_free"])
        fr = res["synthetic_200"]["frustration"]
        self.assertAlmostEqual(fr["frustration_index_phi0"], 8.0, places=5)
        self.assertAlmostEqual(fr["maxcut_value"], 38.0, places=5)
        self.assertTrue(fr["cut_clears_all_frustration"])

    def test_pentagon_c5_contradiction_frustration_equals_one(self):
        nodes = [f"H{i}" for i in range(5)]
        edges = [HypothesisEdge(f"H{i}", f"H{(i+1)%5}", sign=-1, weight=1.0) for i in range(5)]
        aud = FrustrationCutAuditor(exact_max_block_nodes=12)
        rep = aud.audit(nodes, edges)
        self.assertAlmostEqual(rep.frustration_index_phi0, 1.0, places=5)
        self.assertEqual(len(rep.frustrated_edges), 1)
        self.assertTrue(rep.cut_clears_all_frustration)


if __name__ == "__main__":
    unittest.main()
