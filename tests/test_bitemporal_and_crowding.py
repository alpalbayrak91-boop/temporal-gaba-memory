"""Comprehensive unit & property tests for Layer 4 (BiTemporalAGMEngine) and
Layer 5 (SlotSheafCrowdingConsolidator) in temporal-gaba-memory v1.0.0.
"""
from __future__ import annotations

import ast
from dataclasses import replace
import io
import json
import math
import pathlib
import random
import re
import subprocess
import sys
from types import MappingProxyType
import unittest
from contextlib import redirect_stdout

import temporal_gaba_memory as tgm
from temporal_gaba_memory import (
    AGM_CONTRACTION_SENTINEL,
    FIXED_POINT_SCALE,
    BiTemporalAGMEngine,
    BiTemporalEdge,
    BiTemporalEpisode,
    DeadendRecord,
    FindingRecord,
    MerkleInclusionProof,
    SlotSheafCrowdingConsolidator,
    binomial_kl_tail_bound,
    build_rfc6962_merkle_root,
    compute_safe_overfetch_k,
    fixed_point_rrf_fuse,
    gaba_inhibit_micro,
    generate_rfc6962_inclusion_proof,
    jcs_canonicalize,
    rfc6962_leaf_hash,
    rfc6962_node_hash,
    rrf_term_micro,
    theoretical_unfiltered_recall_at_k,
    vec_normalize,
    verify_rfc6962_inclusion_proof,
)
from temporal_gaba_memory.cli import main as cli_main

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]


class TestBiTemporalAndCrowdingV1(unittest.TestCase):
    def test_01_bitemporal_episode_immutability_and_validation(self) -> None:
        ep = BiTemporalEpisode(
            episode_id="ep_1",
            slot_key="auth:timeout",
            valid_from=10.0,
            valid_until=50.0,
            tx_from=100.0,
            payload={"timeout_ms": 250, "nested": [1, 2]},
            embedding=[0.6, 0.8],
        )
        self.assertTrue(ep.is_active)
        self.assertEqual(ep.payload["timeout_ms"], 250)
        with self.assertRaises(TypeError):
            ep.payload["timeout_ms"] = 999  # type: ignore[index]

        with self.assertRaises(ValueError):
            BiTemporalEpisode("bad_v", "s", valid_from=20.0, valid_until=20.0)
        with self.assertRaises(ValueError):
            BiTemporalEpisode("bad_tx", "s", valid_from=0.0, tx_from=10.0, tx_until=5.0, superseded_by="ep_2")
        with self.assertRaises(ValueError):
            BiTemporalEpisode("bad_pair", "s", valid_from=0.0, tx_from=10.0, tx_until=20.0, superseded_by=None)
        with self.assertRaises(ValueError):
            BiTemporalEpisode("bad_nan", "s", valid_from=float("nan"))

    def test_02_revise_split_full_cover_zero_gap_continuity(self) -> None:
        bt = BiTemporalAGMEngine()
        r1 = bt.revise_split("slot:a", valid_from=0.0, valid_until=100.0, tx_from=10.0, payload={"v": 1})
        r2 = bt.revise_split("slot:a", valid_from=0.0, valid_until=100.0, tx_from=25.0, payload={"v": 2})

        old_ep = bt.get_episode(r1.inserted_head_id or "")
        new_ep = bt.get_episode(r2.inserted_head_id or "")
        assert old_ep is not None and new_ep is not None

        self.assertEqual(old_ep.tx_until, 25.0)
        self.assertEqual(new_ep.tx_from, 25.0)
        self.assertEqual(old_ep.superseded_by, new_ep.episode_id)
        self.assertEqual(old_ep.valid_from, 0.0)
        self.assertEqual(old_ep.valid_until, 100.0)
        self.assertEqual(r2.residual_episode_ids, ())
        self.assertTrue(bt.audit_tx_invariants().all_passed)

    def test_03_revise_split_interior_three_way_allen_partition(self) -> None:
        bt = BiTemporalAGMEngine()
        bt.revise_split("rate:limit", valid_from=0.0, valid_until=100.0, tx_from=10.0, payload={"rps": 500})
        rcpt = bt.revise_split("rate:limit", valid_from=30.0, valid_until=60.0, tx_from=20.0, payload={"rps": 100})

        self.assertEqual(len(rcpt.residual_episodes), 2)
        # Current state at tx=25 partitions [0, 100) into [0, 30)=500, [30, 60)=100, [60, 100)=500
        self.assertEqual(bt.query_as_of(15.0, 25.0, "rate:limit")[0].payload["rps"], 500)
        self.assertEqual(bt.query_as_of(30.0, 25.0, "rate:limit")[0].payload["rps"], 100)
        self.assertEqual(bt.query_as_of(59.9, 25.0, "rate:limit")[0].payload["rps"], 100)
        self.assertEqual(bt.query_as_of(60.0, 25.0, "rate:limit")[0].payload["rps"], 500)
        # Historical state at tx=15 still sees 500 everywhere including v=45!
        self.assertEqual(bt.query_as_of(45.0, 15.0, "rate:limit")[0].payload["rps"], 500)
        self.assertTrue(bt.audit_tx_invariants().all_passed)

    def test_04_revise_split_left_trim_and_right_trim_residuals(self) -> None:
        bt = BiTemporalAGMEngine()
        bt.revise_split("s", valid_from=10.0, valid_until=90.0, tx_from=1.0, payload={"v": "base"})
        # Left overlap [0, 40) -> right residual [40, 90)
        r_left = bt.revise_split("s", valid_from=0.0, valid_until=40.0, tx_from=2.0, payload={"v": "left"})
        self.assertEqual(len(r_left.residual_episodes), 1)
        self.assertEqual((r_left.residual_episodes[0].valid_from, r_left.residual_episodes[0].valid_until), (40.0, 90.0))

        # Right overlap [70, 120) -> left residual [40, 70)
        r_right = bt.revise_split("s", valid_from=70.0, valid_until=120.0, tx_from=3.0, payload={"v": "right"})
        self.assertEqual(len(r_right.residual_episodes), 1)
        self.assertEqual(
            (r_right.residual_episodes[0].valid_from, r_right.residual_episodes[0].valid_until),
            (40.0, 70.0),
        )
        self.assertTrue(bt.audit_tx_invariants().all_passed)

    def test_05_revise_split_open_unbounded_valid_until_none(self) -> None:
        bt = BiTemporalAGMEngine()
        bt.revise_split("open_slot", valid_from=0.0, valid_until=None, tx_from=5.0, payload={"state": "init"})
        rcpt = bt.revise_split("open_slot", valid_from=50.0, valid_until=80.0, tx_from=10.0, payload={"state": "maint"})
        self.assertEqual(len(rcpt.residual_episodes), 2)
        self.assertEqual(bt.query_as_of(10_000.0, 12.0, "open_slot")[0].payload["state"], "init")
        self.assertEqual(bt.query_as_of(65.0, 12.0, "open_slot")[0].payload["state"], "maint")
        self.assertTrue(bt.audit_tx_invariants().all_passed)

    def test_06_query_as_of_zero_future_write_leakage_across_30_splits(self) -> None:
        rng = random.Random(42)
        bt = BiTemporalAGMEngine()
        for s in range(4):
            bt.revise_split(f"k{s}", 0.0, 100.0, tx_from=float(s + 1), payload={"gen": 0})

        snapshots = []
        for step in range(30):
            tx = 10.0 + step * 2.0
            slot = f"k{step % 4}"
            lo = float(rng.randint(0, 70))
            hi = lo + float(rng.randint(10, 30))
            # Capture pre-write snapshot at tx - 0.5
            probe_v = lo + 2.0
            before = bt.query_as_of(probe_v, tx - 0.5, slot)
            snapshots.append((probe_v, tx - 0.5, slot, before[0].payload["gen"]))
            bt.revise_split(slot, lo, hi, tx_from=tx, payload={"gen": step + 1})

        for probe_v, probe_tx, slot, expected_gen in snapshots:
            after = bt.query_as_of(probe_v, probe_tx, slot)
            self.assertEqual(len(after), 1)
            self.assertEqual(after[0].payload["gen"], expected_gen)

    def test_07_out_of_order_tx_timestamp_rejected(self) -> None:
        bt = BiTemporalAGMEngine()
        bt.revise_split("slot:x", 0.0, 10.0, tx_from=50.0, payload={"a": 1})
        with self.assertRaises(ValueError):
            bt.revise_split("slot:x", 0.0, 10.0, tx_from=49.9, payload={"a": 2})

    def test_08_agm_contraction_retract_split(self) -> None:
        bt = BiTemporalAGMEngine()
        bt.revise_split("belief:p", 0.0, 100.0, tx_from=10.0, payload={"claim": "active"})
        rcpt = bt.retract_split("belief:p", 40.0, 60.0, tx_from=20.0)
        self.assertIsNone(rcpt.inserted_head_id)
        self.assertEqual(rcpt.closed_episodes[0].superseded_by, AGM_CONTRACTION_SENTINEL)
        self.assertEqual(bt.query_as_of(50.0, 25.0, "belief:p"), [])
        self.assertEqual(len(bt.query_as_of(20.0, 25.0, "belief:p")), 1)
        self.assertEqual(len(bt.query_as_of(80.0, 25.0, "belief:p")), 1)
        self.assertEqual(len(bt.query_as_of(50.0, 15.0, "belief:p")), 1)
        self.assertTrue(bt.audit_tx_invariants().all_passed)

    def test_09_bitemporal_edges_split_and_2hop_neighbors_as_of(self) -> None:
        bt = BiTemporalAGMEngine()
        bt.revise_edge_split("n1", "n2", "supports", 0.0, 100.0, tx_from=10.0, weight=1.0)
        bt.revise_edge_split("n2", "n3", "supports", 0.0, 100.0, tx_from=12.0, weight=1.0)
        # Disable n2->n3 on [40, 60) by setting weight=0.0 at tx=20
        bt.revise_edge_split("n2", "n3", "supports", 40.0, 60.0, tx_from=20.0, weight=0.0)

        hops_before = bt.neighbors_2hop_as_of(["n1"], valid_at=50.0, tx_at=15.0)
        self.assertEqual(hops_before, {"n2": 1, "n3": 2})

        hops_after_mid = bt.neighbors_2hop_as_of(["n1"], valid_at=50.0, tx_at=25.0)
        self.assertEqual(hops_after_mid, {"n2": 1})

        hops_after_out = bt.neighbors_2hop_as_of(["n1"], valid_at=75.0, tx_at=25.0)
        self.assertEqual(hops_after_out, {"n2": 1, "n3": 2})
        self.assertTrue(bt.audit_tx_invariants().all_passed)

    def test_10_audit_tx_invariants_detects_corrupted_rows(self) -> None:
        bt = BiTemporalAGMEngine()
        bt.revise_split("slot:1", 0.0, 50.0, tx_from=10.0, payload={"x": 1})
        bt.revise_split("slot:1", 0.0, 50.0, tx_from=20.0, payload={"x": 2})
        self.assertTrue(bt.audit_tx_invariants().all_passed)

        # Inject a gap in tx_until != succ.tx_from and an overlapping active episode
        orig = bt._episodes[0]
        bt._episodes[0] = replace(orig, tx_until=18.0)  # gap [18, 20) -> violates cov_puts
        rep_gap = bt.audit_tx_invariants()
        self.assertFalse(rep_gap.all_passed)
        self.assertFalse(rep_gap.supersession_zero_gap_ok)

        # Restore and inject overlapping active record -> violates func_snapshot
        bt._episodes[0] = orig
        dup_active = BiTemporalEpisode("rogue", "slot:1", valid_from=10.0, valid_until=30.0, tx_from=20.0)
        bt._append_episode_row(dup_active)
        rep_overlap = bt.audit_tx_invariants()
        self.assertFalse(rep_overlap.all_passed)
        self.assertFalse(rep_overlap.single_active_per_slot_ok)

    def test_11_theoretical_unfiltered_recall_crowding_law(self) -> None:
        self.assertEqual(theoretical_unfiltered_recall_at_k(0, 5), 0.0)
        self.assertEqual(theoretical_unfiltered_recall_at_k(5, 0), 1.0)
        self.assertEqual(theoretical_unfiltered_recall_at_k(5, 4), 1.0)
        self.assertAlmostEqual(theoretical_unfiltered_recall_at_k(5, 20), 5.0 / 21.0, places=12)
        with self.assertRaises(ValueError):
            theoretical_unfiltered_recall_at_k(5, -1)

    def test_12_binomial_kl_tail_bound_dominates_exact_binomial_cdf(self) -> None:
        for n, k_tgt, p in [(25, 5, 0.5), (60, 5, 0.25), (120, 5, 0.15), (220, 5, 0.10)]:
            exact_cdf = sum(
                math.comb(n, j) * (p ** j) * ((1.0 - p) ** (n - j)) for j in range(k_tgt)
            )
            kl_bound = binomial_kl_tail_bound(n, k_tgt, p)
            self.assertGreaterEqual(kl_bound + 1e-15, exact_cdf)
            self.assertLess(kl_bound, 0.05)

    def test_13_compute_safe_overfetch_k_high_tombstone_ratios_80_to_95pct(self) -> None:
        self.assertEqual(compute_safe_overfetch_k(0, 0.90), 0)
        self.assertEqual(compute_safe_overfetch_k(5, 0.0), 5)

        k80 = compute_safe_overfetch_k(5, 0.80, confidence_z=3.0)
        k90 = compute_safe_overfetch_k(5, 0.90, confidence_z=3.0)
        k95 = compute_safe_overfetch_k(5, 0.95, confidence_z=3.0)
        self.assertLess(k80, k90)
        self.assertLess(k90, k95)

        # Verify exact Binomial starvation probability at k95 (p = 0.05) is strictly < Phi(-3.0) = 0.00135
        p = 0.05
        exact_starve_prob = sum(
            math.comb(k95, j) * (p ** j) * ((1.0 - p) ** (k95 - j)) for j in range(5)
        )
        self.assertLessEqual(exact_starve_prob, 0.5 * math.erfc(3.0 / math.sqrt(2.0)))

        with self.assertRaises(ValueError):
            compute_safe_overfetch_k(5, 1.0)
        with self.assertRaises(ValueError):
            compute_safe_overfetch_k(5, -0.1)

    def test_14_fixed_point_rrf_term_antitone_and_integer_determinism(self) -> None:
        terms = [rrf_term_micro(r, rrf_k=60) for r in range(200)]
        for i in range(len(terms) - 1):
            self.assertGreaterEqual(terms[i], terms[i + 1])
            self.assertIsInstance(terms[i], int)

        self.assertEqual(gaba_inhibit_micro(15_000, 5_000), 10_000)
        self.assertEqual(gaba_inhibit_micro(15_000, 25_000), 0)
        self.assertEqual(gaba_inhibit_micro(15_000, 0), 15_000)
        self.assertEqual(FIXED_POINT_SCALE, 1_000_000)

        fused = fixed_point_rrf_fuse(
            dense_ranked_ids=["zombie", "best", "mid"],
            lexical_ranked_ids=["best", "zombie", "mid"],
            graph_hop_distances={"hop_friend": 1},
            gaba_penalties_micro={"mid": 5_000},
            gaba_vetoed_ids={"zombie"},
        )
        ids = [cid for cid, _ in fused]
        self.assertNotIn("zombie", ids)
        self.assertEqual(ids[0], "best")
        self.assertIn("hop_friend", ids)
        self.assertTrue(all(isinstance(sc, int) and sc >= 0 for _, sc in fused))

    def test_15_crowding_consolidator_recovers_active_head_under_m20_crowding(self) -> None:
        bt = BiTemporalAGMEngine()
        # Create 20 historical tombstones with episode_ids sorting ahead of the final active head
        for rev in range(20):
            bt.revise_split(
                "hot:slot",
                valid_from=0.0,
                valid_until=None,
                tx_from=float(rev + 1),
                episode_id=f"a_tombstone_{rev:02d}",
                payload={"text": "critical consensus quorum invariant"},
                embedding=[1.0, 0.0, 0.0],
            )
        rcpt = bt.revise_split(
            "hot:slot",
            valid_from=0.0,
            valid_until=None,
            tx_from=25.0,
            episode_id="z_active_head",
            payload={"text": "critical consensus quorum invariant"},
            embedding=[1.0, 0.0, 0.0],
        )

        cons = SlotSheafCrowdingConsolidator()
        naive = cons.naive_unfiltered_retrieve(bt, "critical consensus quorum", [1.0, 0.0, 0.0], top_k=5)
        self.assertNotIn("z_active_head", naive.selected_ids)
        self.assertTrue(naive.stale_trap_at_1)

        active = cons.retrieve_active(bt, "critical consensus quorum", [1.0, 0.0, 0.0], top_k=5)
        self.assertEqual(active.selected_ids, (rcpt.inserted_head_id,))
        self.assertFalse(active.stale_trap_at_1)

    def test_16_crowding_consolidator_causal_gaba_and_sheaf_tension_integration(self) -> None:
        bt = BiTemporalAGMEngine()
        # Pre-discovery zombie on slot_trap (still active in ledger until deadend gate catches it!)
        bt.revise_split(
            "slot:zombie",
            0.0,
            None,
            tx_from=500.0,
            episode_id="ep_zombie",
            payload={"text": "broken float accumulator ledger"},
            embedding=vec_normalize([0.98, 0.10, 0.0]),
        )
        # Post-discovery verified remediation on slot:fix
        bt.revise_split(
            "slot:fix",
            0.0,
            None,
            tx_from=2500.0,
            episode_id="ep_fix",
            payload={"text": "broken float accumulator ledger replaced with fixed point integer"},
            embedding=vec_normalize([0.97, 0.12, 0.0]),
        )
        # 1-hop graph proof linked to ep_fix
        bt.revise_split(
            "slot:proof",
            0.0,
            None,
            tx_from=2600.0,
            episode_id="ep_proof",
            payload={"text": "lean4 antitone theorem"},
            embedding=vec_normalize([0.0, 0.10, 0.99]),
        )
        bt.revise_edge_split("ep_fix", "ep_proof", "verified_by", 0.0, None, tx_from=2605.0, weight=1.0)

        deadends = [
            DeadendRecord(
                deadend_id="de_float",
                discovered_ts_s=2000.0,
                dense_vec=vec_normalize([0.98, 0.10, 0.0]),
                threshold_cos=0.45,
            )
        ]
        cons = SlotSheafCrowdingConsolidator()
        rep = cons.retrieve_active(
            bt,
            query_text="float accumulator ledger",
            query_embedding=vec_normalize([0.98, 0.11, 0.0]),
            top_k=3,
            deadends=deadends,
            sheaf_node_tensions={"ep_proof": 0.25},
        )
        self.assertIn("ep_zombie", rep.vetoed_ids)
        self.assertNotIn("ep_zombie", rep.selected_ids)
        self.assertEqual(rep.selected_ids[0], "ep_fix")
        self.assertIn("ep_proof", rep.selected_ids)

    def test_17_rfc8785_jcs_canonicalization_vectors(self) -> None:
        sample = {
            "z": [1.0, -0.0, True, False, None],
            "a": "line1\nline2\t\"quoted\"\\\x01",
            "m": {"b": 2, "a": 1},
        }
        raw = jcs_canonicalize(sample)
        self.assertEqual(
            raw,
            b'{"a":"line1\\nline2\\t\\"quoted\\"\\\\\\u0001","m":{"a":1,"b":2},"z":[1,0,true,false,null]}',
        )
        with self.assertRaises(ValueError):
            jcs_canonicalize({"bad": float("nan")})
        with self.assertRaises(ValueError):
            jcs_canonicalize({"bad": float("inf")})

    def test_18_rfc6962_merkle_tree_domain_separation_and_inclusion_proofs(self) -> None:
        # Empty tree = SHA-256("")
        self.assertEqual(
            build_rfc6962_merkle_root([]),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        )
        # Domain separation: leaf (0x00) != interior node (0x01)
        d0 = b"alpha"
        d1 = b"beta"
        h_leaf = rfc6962_leaf_hash(d0 + d1)
        h_node = rfc6962_node_hash(d0.ljust(32, b"\x00"), d1.ljust(32, b"\x00"))
        self.assertNotEqual(h_leaf, h_node)

        for n in (1, 2, 3, 4, 5, 7, 8, 13):
            leaves = [jcs_canonicalize({"idx": i, "payload": f"record_{i}"}) for i in range(n)]
            root = build_rfc6962_merkle_root(leaves)
            for idx in range(n):
                pf = generate_rfc6962_inclusion_proof(idx, leaves)
                self.assertTrue(verify_rfc6962_inclusion_proof(leaves[idx], pf, root))
                # Tampered payload must fail
                self.assertFalse(verify_rfc6962_inclusion_proof(leaves[idx] + b"!", pf, root))
                # Wrong index must fail when n > 1
                if n > 1:
                    bad_pf = MerkleInclusionProof(
                        leaf_index=(idx + 1) % n,
                        tree_size=pf.tree_size,
                        leaf_hash_hex=pf.leaf_hash_hex,
                        sibling_hashes_hex=pf.sibling_hashes_hex,
                        root_hash_hex=pf.root_hash_hex,
                    )
                    self.assertFalse(verify_rfc6962_inclusion_proof(leaves[idx], bad_pf, root))

    def test_19_three_tier_sleep_consolidation_semantic_and_procedural_capsules(self) -> None:
        bt = BiTemporalAGMEngine()
        for slot_i in range(3):
            for rev in range(4):
                bt.revise_split(
                    f"svc:{slot_i}",
                    0.0,
                    100.0,
                    tx_from=float(slot_i * 10 + rev + 1),
                    payload={"operator": "circuit_breaker_backoff", "attempt": rev},
                    embedding=[1.0, 0.0],
                )
        findings = [
            FindingRecord(
                bulgu_id=f"sf_{i}",
                konu_id="reliability",
                alan="ops",
                baslik=f"Backoff petal {i}",
                ozet=f"Shared invariant core with petal {i}",
                explicit_atoms=frozenset({"circuit", "breaker", f"leaf_{i}"}),
            )
            for i in range(4)
        ]
        cons = SlotSheafCrowdingConsolidator()
        rep = cons.sleep_consolidate(bt, findings=findings, sunflower_k=3, min_procedural_support=2)

        self.assertTrue(rep.all_merkle_proofs_verified)
        self.assertEqual(rep.archived_tombstone_count, 9)
        self.assertEqual(rep.pruned_sunflower_petal_count, 3)
        self.assertGreaterEqual(len(rep.semantic_capsules), 4)
        self.assertGreaterEqual(len(rep.procedural_capsules), 1)
        self.assertGreater(rep.compression_ratio, 0.65)

        for ep in bt.episodes:
            matching_caps = [c for c in rep.semantic_capsules if c.slot_key == ep.slot_key]
            self.assertEqual(len(matching_caps), 1)
            self.assertTrue(matching_caps[0].verify_episode_inclusion(ep))

    def test_20_cli_self_test_and_demo_v1_entrypoints(self) -> None:
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = cli_main(["--self-test"])
        self.assertEqual(rc, 0)
        payload = json.loads(buf.getvalue())
        self.assertTrue(payload["self_test_passed"])
        self.assertEqual(payload["version"], "1.0.0")

        buf2 = io.StringIO()
        with redirect_stdout(buf2):
            rc2 = cli_main(["demo-v1"])
        self.assertEqual(rc2, 0)
        demo_v1 = json.loads(buf2.getvalue())
        self.assertTrue(demo_v1["layer4_bitemporal_agm"]["all_5_invariants_passed"])
        self.assertTrue(demo_v1["layer5_crowding_and_sleep_merkle"]["all_merkle_proofs_verified"])

    def test_21_lean4_bitemporal_crowding_proof_file_integrity(self) -> None:
        expected_counts = {
            "TemporalGabaBitemporalCrowding.lean": 25,
            "EpistemicDAGSoundness.lean": 36,
            "Erdos23C5BlowupFrustration.lean": 4,
        }
        total_theorems = 0
        thm_re = re.compile(r"^\s*(?:private\s+)?theorem\b", re.MULTILINE)
        for fname, exp_cnt in expected_counts.items():
            fpath = _REPO_ROOT / "formal" / fname
            self.assertTrue(fpath.is_file(), msg=f"Missing Lean file: {fname}")
            fraw = fpath.read_bytes()
            self.assertEqual(fraw.count(36), 0, msg=f"Unexpected ASCII-36 in {fname}")
            ftext = fraw.decode("utf-8")
            cnt = len(thm_re.findall(ftext))
            self.assertEqual(cnt, exp_cnt, msg=f"Theorem count mismatch in {fname}")
            total_theorems += cnt
            code_lines = [
                ln.split("--")[0]
                for ln in ftext.splitlines()
                if not ln.strip().startswith("/-") and not ln.strip().startswith("*")
            ]
            joined_code = "\n".join(code_lines)
            self.assertNotIn("by sorry", joined_code)
            self.assertNotIn("\naxiom ", joined_code)

        self.assertEqual(total_theorems, 65)

        lean_path = _REPO_ROOT / "formal" / "TemporalGabaBitemporalCrowding.lean"
        text = lean_path.read_text(encoding="utf-8")
        for required_name in (
            "asOfTx_future_append_invariant",
            "asOfTx_future_close_invariant",
            "supersede_preserves_coverage",
            "unfiltered_topK_starves_when_tombstones_ge_k",
            "overfetch_k_plus_m_guarantees_active_head",
            "rrfTerm_antitone_rank",
            "gabaInhibit_le_base",
            "rrfFuse4_monotone_and_causalGaba_sound",
        ):
            self.assertIn(f"theorem {required_name}", text)

    def test_22_package_stdlib_only_and_zero_ascii36_hygiene(self) -> None:
        pkg_dir = _REPO_ROOT / "src" / "temporal_gaba_memory"
        stdlib_names = set(getattr(sys, "stdlib_module_names", ())) | {"__future__"}
        py_files = sorted(pkg_dir.glob("*.py"))
        self.assertGreaterEqual(len(py_files), 6)
        for p in py_files:
            raw = p.read_bytes()
            self.assertEqual(raw.count(36), 0, msg=f"Unexpected ASCII-36 in {p.name}")
            tree = ast.parse(raw.decode("utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        self.assertIn(alias.name.split(".")[0], stdlib_names)
                elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                    self.assertIn(node.module.split(".")[0], stdlib_names)

    def test_23_randomized_property_bitemporal_invariants(self) -> None:
        rng = random.Random(20261127)
        for trial in range(25):
            bt = BiTemporalAGMEngine()
            tx = 1.0
            for _ in range(20):
                slot = f"s_{rng.randint(0, 3)}"
                lo = float(rng.randint(0, 80))
                hi = None if rng.random() < 0.2 else lo + float(rng.randint(5, 30))
                if rng.random() < 0.15:
                    bt.retract_split(slot, lo, hi, tx_from=tx)
                else:
                    bt.revise_split(slot, lo, hi, tx_from=tx, payload={"trial": trial, "tx": tx})
                tx += float(rng.randint(0, 3))
            rep = bt.audit_tx_invariants()
            self.assertTrue(rep.all_passed, msg=f"Trial {trial} failed invariants: {rep.violations}")

    def test_24_empty_engine_and_boundary_queries(self) -> None:
        bt = BiTemporalAGMEngine()
        self.assertEqual(bt.query_as_of(0.0, 0.0), [])
        self.assertEqual(bt.tombstone_ratio(), 0.0)
        self.assertTrue(bt.audit_tx_invariants().all_passed)

        cons = SlotSheafCrowdingConsolidator()
        r_empty = cons.retrieve_active(bt, "query", [1.0, 0.0], top_k=5)
        self.assertEqual(r_empty.selected_ids, ())
        s_empty = cons.sleep_consolidate(bt)
        self.assertEqual(s_empty.compression_ratio, 0.0)
        self.assertTrue(s_empty.all_merkle_proofs_verified)

    def test_25_public_exports_and_semver_v1(self) -> None:
        self.assertEqual(tgm.__version__, "1.0.0")
        for sym in (
            "BiTemporalAGMEngine",
            "BiTemporalEpisode",
            "BiTemporalEdge",
            "SlotSheafCrowdingConsolidator",
            "SemanticCapsule",
            "ProceduralCapsule",
            "theoretical_unfiltered_recall_at_k",
            "compute_safe_overfetch_k",
            "fixed_point_rrf_fuse",
            "jcs_canonicalize",
        ):
            self.assertTrue(hasattr(tgm, sym), msg=f"Missing export: {sym}")

    def test_26_retract_split_same_tx_invalidates_crowding_index_cache(self) -> None:
        bt = BiTemporalAGMEngine()
        self.assertEqual(bt.mutation_seq, 0)
        bt.revise_split(
            "slot:ephemeral",
            valid_from=0.0,
            valid_until=100.0,
            tx_from=10.0,
            episode_id="ep_ephemeral",
            payload={"text": "sensitive ephemeral belief"},
            embedding=[1.0, 0.0],
        )
        self.assertEqual(bt.mutation_seq, 1)

        cons = SlotSheafCrowdingConsolidator()
        before = cons.retrieve_active(bt, "sensitive ephemeral belief", [1.0, 0.0], top_k=5)
        self.assertEqual(list(before.selected_ids), ["ep_ephemeral"])

        # Full retraction at identical tx_from=10.0 emits 0 residuals (episode_count & max_tx unchanged)
        bt.retract_split("slot:ephemeral", valid_from=0.0, valid_until=100.0, tx_from=10.0)
        self.assertEqual(bt.mutation_seq, 2)
        self.assertEqual(bt.episode_count, 1)
        self.assertEqual(bt.max_tx_timestamp, 10.0)

        after = cons.retrieve_active(bt, "sensitive ephemeral belief", [1.0, 0.0], top_k=5)
        self.assertEqual(list(after.selected_ids), [])
        self.assertFalse(after.stale_trap_at_1)

        naive_after = cons.naive_unfiltered_retrieve(bt, "sensitive ephemeral belief", [1.0, 0.0], top_k=5)
        self.assertTrue(naive_after.stale_trap_at_1)

    def test_27_rfc8785_jcs_large_float_significand_and_surrogate_rejection(self) -> None:
        self.assertEqual(jcs_canonicalize(9.999999999999999e20), b"999999999999999900000")
        self.assertEqual(jcs_canonicalize(1.2345678901234567e20), b"123456789012345670000")
        self.assertEqual(jcs_canonicalize(42.0), b"42")
        self.assertEqual(jcs_canonicalize(-9.999999999999999e20), b"-999999999999999900000")
        self.assertEqual(jcs_canonicalize(1e21), b"1e+21")
        self.assertEqual(jcs_canonicalize(1e-6), b"0.000001")
        self.assertEqual(jcs_canonicalize(1e-7), b"1e-7")

        for lone_surrogate in ("\ud800", "\udbff", "\udc00", "\udfff"):
            with self.assertRaises(ValueError):
                jcs_canonicalize(lone_surrogate)
            with self.assertRaises(ValueError):
                jcs_canonicalize({"val": f"prefix_{lone_surrogate}_suffix"})
            with self.assertRaises(ValueError):
                jcs_canonicalize({lone_surrogate: 1})

    def test_28_atomic_rollback_on_residual_episode_and_edge_id_collision(self) -> None:
        bt = BiTemporalAGMEngine()
        bt.revise_split("slot:other", 0.0, 10.0, tx_from=1.0, episode_id="h2_resL_0", payload={"x": 0})
        bt.revise_split("slot:other2", 0.0, 10.0, tx_from=2.0, episode_id="h3_resR_1", payload={"x": 0})
        bt.revise_split("slot:target", 0.0, 100.0, tx_from=3.0, episode_id="h1", payload={"x": 1})

        bt.revise_edge_split("a", "b", "rel", 0.0, 10.0, tx_from=4.0, edge_id="e2_resL_0", weight=1.0)
        bt.revise_edge_split("c", "d", "rel", 0.0, 100.0, tx_from=5.0, edge_id="e1", weight=1.0)

        snap_eps = bt.episodes
        snap_edges = bt.edges
        snap_seq = bt.mutation_seq
        snap_tx = bt.max_tx_timestamp

        # Left-residual episode ID collision ("h2_resL_0")
        with self.assertRaises(ValueError):
            bt.revise_split("slot:target", 20.0, 80.0, tx_from=6.0, episode_id="h2", payload={"x": 2})

        # Right-residual episode ID collision ("h3_resR_1")
        with self.assertRaises(ValueError):
            bt.revise_split("slot:target", 20.0, 80.0, tx_from=6.0, episode_id="h3", payload={"x": 3})

        # Left-residual edge ID collision ("e2_resL_0")
        with self.assertRaises(ValueError):
            bt.revise_edge_split("c", "d", "rel", 20.0, 80.0, tx_from=6.0, edge_id="e2", weight=0.5)

        self.assertEqual(bt.episodes, snap_eps)
        self.assertEqual(bt.edges, snap_edges)
        self.assertEqual(bt.mutation_seq, snap_seq)
        self.assertEqual(bt.max_tx_timestamp, snap_tx)
        self.assertTrue(bt.audit_tx_invariants().all_passed)

    def test_29_active_horizon_2hop_nonzero_valid_from_and_merkle_tree_size_binding(self) -> None:
        bt = BiTemporalAGMEngine()
        bt.revise_split(
            "slot:seed",
            valid_from=1000.0,
            valid_until=None,
            tx_from=10.0,
            episode_id="ep_seed",
            payload={"text": "quantum lattice anchor"},
            embedding=[1.0, 0.0],
        )
        for f_idx in range(6):
            bt.revise_split(
                f"slot:filler_{f_idx}",
                valid_from=1000.0,
                valid_until=None,
                tx_from=10.5 + f_idx * 0.1,
                episode_id=f"filler_{f_idx:02d}",
                payload={"text": "background noise item"},
                embedding=[0.0, 1.0],
            )
        bt.revise_split(
            "slot:hop1",
            valid_from=1000.0,
            valid_until=None,
            tx_from=12.0,
            episode_id="z_ep_hop1",
            payload={"text": "disjoint companion one"},
            embedding=[0.0, 1.0],
        )
        bt.revise_split(
            "slot:hop2",
            valid_from=1000.0,
            valid_until=None,
            tx_from=13.0,
            episode_id="z_ep_hop2",
            payload={"text": "disjoint companion two"},
            embedding=[0.0, 1.0],
        )
        bt.revise_split(
            "slot:unlinked",
            valid_from=1000.0,
            valid_until=None,
            tx_from=14.0,
            episode_id="z_ep_unlinked",
            payload={"text": "disjoint bystander"},
            embedding=[0.0, 1.0],
        )
        bt.revise_edge_split("ep_seed", "z_ep_hop1", "links", valid_from=1000.0, valid_until=None, tx_from=15.0)
        bt.revise_edge_split("z_ep_hop1", "z_ep_hop2", "links", valid_from=1000.0, valid_until=None, tx_from=16.0)

        cons = SlotSheafCrowdingConsolidator()
        rep = cons.retrieve_active(bt, "quantum lattice anchor", [1.0, 0.0], valid_at=None, top_k=10)
        self.assertEqual(set(rep.selected_ids[:3]), {"ep_seed", "z_ep_hop1", "z_ep_hop2"})
        self.assertGreater(rep.scores_micro["z_ep_hop1"], rep.scores_micro["z_ep_hop2"])
        self.assertGreater(rep.scores_micro["z_ep_hop2"], rep.scores_micro["filler_00"])
        hop1_bonus = rrf_term_micro(0, rrf_k=cons.rrf_k, weight_milli=500)
        hop2_bonus = rrf_term_micro(10, rrf_k=cons.rrf_k, weight_milli=250)
        self.assertGreaterEqual(rep.scores_micro["z_ep_hop1"] - rep.scores_micro["z_ep_unlinked"], hop1_bonus)
        self.assertGreaterEqual(rep.scores_micro["z_ep_hop2"] - rep.scores_micro["z_ep_unlinked"], hop2_bonus)

        # RFC 6962 tree_size binding check on a 3-leaf tree (leaf 2 sibling = MTH(L0, L1))
        bt_merkle = BiTemporalAGMEngine()
        for rev in range(3):
            for slot_idx in range(3):
                bt_merkle.revise_split(
                    f"slot:m{slot_idx}",
                    0.0,
                    100.0,
                    tx_from=float(rev * 10 + slot_idx + 1),
                    episode_id=f"m{slot_idx}_ep_{rev}",
                    payload={"operator": "quorum_gate", "rev": rev},
                )

        sleep_rep = cons.sleep_consolidate(bt_merkle, min_procedural_support=3)
        self.assertTrue(sleep_rep.all_merkle_proofs_verified)
        sem_cap = [c for c in sleep_rep.semantic_capsules if c.slot_key == "slot:m0"][0]
        ep2 = bt_merkle.get_episode("m0_ep_2")
        assert ep2 is not None
        self.assertTrue(sem_cap.verify_episode_inclusion(ep2))

        orig_pf = sem_cap.inclusion_proofs["m0_ep_2"]
        forged_pf = MerkleInclusionProof(
            leaf_index=1,
            tree_size=2,
            leaf_hash_hex=orig_pf.leaf_hash_hex,
            sibling_hashes_hex=orig_pf.sibling_hashes_hex,
            root_hash_hex=orig_pf.root_hash_hex,
        )
        self.assertFalse(
            verify_rfc6962_inclusion_proof(
                jcs_canonicalize(ep2.to_dict()),
                forged_pf,
                sem_cap.merkle_root_hex,
                expected_tree_size=3,
            )
        )
        tampered_sem = replace(
            sem_cap,
            inclusion_proofs=MappingProxyType({**dict(sem_cap.inclusion_proofs), "m0_ep_2": forged_pf}),
        )
        self.assertFalse(tampered_sem.verify_episode_inclusion(ep2))

        proc_cap = sleep_rep.procedural_capsules[0]
        last_sem_id = proc_cap.supporting_capsule_ids[2]
        last_sem = [c for c in sleep_rep.semantic_capsules if c.capsule_id == last_sem_id][0]
        self.assertTrue(proc_cap.verify_capsule_inclusion(last_sem))
        orig_proc_pf = proc_cap.inclusion_proofs[last_sem_id]
        forged_proc_pf = MerkleInclusionProof(
            leaf_index=1,
            tree_size=2,
            leaf_hash_hex=orig_proc_pf.leaf_hash_hex,
            sibling_hashes_hex=orig_proc_pf.sibling_hashes_hex,
            root_hash_hex=orig_proc_pf.root_hash_hex,
        )
        tampered_proc = replace(
            proc_cap,
            inclusion_proofs=MappingProxyType({**dict(proc_cap.inclusion_proofs), last_sem_id: forged_proc_pf}),
        )
        self.assertFalse(tampered_proc.verify_capsule_inclusion(last_sem))

    def test_30_repo_wide_opsec_and_zero_ascii36_hygiene(self) -> None:
        opsec_re = re.compile(
            bytes.fromhex(
                "616c706c61627c64756e79616c61627c616c706573737c74617374657477696e7c"
                "6d61686f2d7265636f6e7c6b6f74615f646f6c6475727563757c7765625f67656d"
                "696e697c6e6f726f6e696b5f7a696e6369727c616c702d6f73737c2f686f6d652f"
                "616c707c35333731307c343831315c2e3439"
            ).decode("utf-8"),
            re.IGNORECASE,
        )
        mangled_usd_re = re.compile(r"\bUSD[A-Za-z0-9_\\^{}+=\-]+USD\b")

        git_dir = _REPO_ROOT / ".git"
        if git_dir.exists():
            proc = subprocess.run(
                ["git", "ls-files"],
                cwd=str(_REPO_ROOT),
                text=True,
                capture_output=True,
                check=True,
            )
            rel_paths = [p for p in proc.stdout.splitlines() if p.strip()]
            files = [_REPO_ROOT / p for p in rel_paths]
        else:
            skip_dirs = {".git", "__pycache__", ".pytest_cache", "build", "dist"}
            files = [
                p
                for p in sorted(_REPO_ROOT.rglob("*"))
                if p.is_file() and not any(part in skip_dirs or part.endswith(".egg-info") for part in p.parts)
            ]

        self.assertGreaterEqual(len(files), 20)
        for fpath in files:
            raw = fpath.read_bytes()
            rel = fpath.relative_to(_REPO_ROOT)
            self.assertEqual(raw.count(36), 0, msg=f"Unexpected ASCII-36 byte in {rel}")
            text = raw.decode("utf-8")
            m_opsec = opsec_re.search(text)
            self.assertIsNone(m_opsec, msg=f"OPSEC leak {m_opsec.group(0)!r} in {rel}" if m_opsec else "")
            m_usd = mangled_usd_re.search(text)
            self.assertIsNone(m_usd, msg=f"Mangled USD token {m_usd.group(0)!r} in {rel}" if m_usd else "")


if __name__ == "__main__":
    unittest.main(verbosity=2)
