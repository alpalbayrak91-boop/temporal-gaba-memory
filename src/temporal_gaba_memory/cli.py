"""CLI entrypoint for temporal-gaba-memory v1.0.0."""
from __future__ import annotations

import argparse
import json
import math
from typing import Any, Dict, List

from . import __version__
from .bitemporal_agm import BiTemporalAGMEngine
from .crowding_consolidation import (
    SlotSheafCrowdingConsolidator,
    compute_safe_overfetch_k,
    theoretical_unfiltered_recall_at_k,
)
from .engine import (
    DeadendRecord,
    EpisodeRecord,
    TemporalGABAMemoryEngine,
    extract_python_ast_profile,
    is_psd_cholesky,
    tokenize_text,
)
from .sheaf_coboundary import CellularSheafCoboundaryAuditor, SheafEdge, SheafNode
from .sunflower_frustration import FindingRecord, FrustrationCutAuditor, HypothesisEdge, SunflowerMemoryPruner


def _run_demo() -> dict:
    episodes: List[EpisodeRecord] = [
        EpisodeRecord(
            episode_id="ep_pre_deadend_trap",
            timestamp_s=1000.0,
            tokens=tokenize_text("linear penalty additive score gate pass broken premise"),
            dense_vec=[0.95, 0.10, 0.05, 0.02],
            ast_profile=extract_python_ast_profile("def f(x):\n    return x + 1\n"),
            branch_id="audit",
        ),
        EpisodeRecord(
            episode_id="ep_post_discovery_fix",
            timestamp_s=2400.0,
            tokens=tokenize_text("linear penalty additive score gate fixed lexicographic veto"),
            dense_vec=[0.93, 0.14, 0.08, 0.03],
            ast_profile=extract_python_ast_profile(
                "def gate(items):\n    for x in items:\n        if x < 0:\n            return False\n    return True\n"
            ),
            branch_id="audit",
        ),
        EpisodeRecord(
            episode_id="ep_graph_hop1_proof",
            timestamp_s=2600.0,
            tokens=tokenize_text("lean formal soundness proof lexicographic lattice"),
            dense_vec=[0.20, 0.92, 0.10, 0.05],
            ast_profile=extract_python_ast_profile(
                "class Lattice:\n    def check(self, a, b):\n        return a <= b\n"
            ),
            branch_id="formal",
        ),
        EpisodeRecord(
            episode_id="ep_graph_hop2_deploy",
            timestamp_s=2900.0,
            tokens=tokenize_text("production deployment merkle receipt audit pipeline"),
            dense_vec=[0.10, 0.20, 0.94, 0.08],
            ast_profile=extract_python_ast_profile(
                "def run(cfg):\n    try:\n        return [k for k in cfg if k]\n    except Exception:\n        return []\n"
            ),
            branch_id="deploy",
        ),
        EpisodeRecord(
            episode_id="ep_unrelated_signal",
            timestamp_s=2700.0,
            tokens=tokenize_text("spectral causality time series feature store"),
            dense_vec=[0.05, 0.10, 0.12, 0.96],
            ast_profile=extract_python_ast_profile(
                "async def poll(xs):\n    while xs:\n        xs.pop()\n"
            ),
            branch_id="quant",
        ),
    ]
    deadends = [
        DeadendRecord(
            deadend_id="deadend_linear_comp",
            discovered_ts_s=2000.0,
            dense_vec=[0.95, 0.10, 0.05, 0.02],
            threshold_cos=0.45,
        )
    ]
    edges = [
        ("ep_post_discovery_fix", "ep_graph_hop1_proof"),
        ("ep_graph_hop1_proof", "ep_graph_hop2_deploy"),
    ]
    engine = TemporalGABAMemoryEngine()
    engine.index(episodes, deadends, edges)
    rep = engine.query(
        query_tokens=["linear", "penalty", "gate", "lexicographic"],
        query_dense_vec=[0.94, 0.12, 0.06, 0.02],
        top_k=4,
    )
    # Positive control (3x3 PSD matrix) vs. paper's 4x4 abs(G) non-PSD counterexample (lambda_min ~ -0.4142)
    s = math.sqrt(0.5)
    psd_3x3_control = [
        [1.0, s, s],
        [s, 1.0, 0.5],
        [s, 0.5, 1.0],
    ]
    angles = [0.0, math.pi / 4.0, math.pi / 2.0, 3.0 * math.pi / 4.0]
    pts = [(math.cos(a), math.sin(a)) for a in angles]
    non_psd_abs_gram = [
        [abs(pts[i][0] * pts[j][0] + pts[i][1] * pts[j][1]) for j in range(4)]
        for i in range(4)
    ]
    return {
        "version": __version__,
        "selected_episode_ids": rep.episode_ids,
        "vetoed_episode_ids": rep.vetoed_ids,
        "pre_deadend_vetoed": "ep_pre_deadend_trap" in rep.vetoed_ids,
        "post_fix_preserved": "ep_post_discovery_fix" in rep.episode_ids,
        "vendi_vs2_centered": rep.vendi_vs2_centered,
        "pool_kernel_psd_verified": rep.min_eig_psd_guard,
        "non_psd_counterexample_rejected": not is_psd_cholesky(non_psd_abs_gram),
        "psd_sanity_ok": is_psd_cholesky(psd_3x3_control),
    }


def _run_demo_v1() -> Dict[str, Any]:
    """Run integrated 6-layer showcase for temporal-gaba-memory v1.0.0."""
    layer1 = _run_demo()

    # Layer 2: Sunflower + Spin-Glass Frustration
    findings = [
        FindingRecord(
            bulgu_id=f"f_{i}",
            konu_id="consensus",
            alan="epistemics",
            baslik=f"Invariant quorum protocol unique_petal_{i}",
            ozet=f"Quorum consensus invariant core verified with petal_token_{i}",
            explicit_atoms=frozenset({"quorum", "consensus", "invariant", f"petal_{i}"}),
        )
        for i in range(5)
    ]
    sf_rep = SunflowerMemoryPruner(k=3, min_core_size=2, retain_petals=1).prune(findings)
    fr_rep = FrustrationCutAuditor().audit(
        [f"H{i}" for i in range(5)],
        [HypothesisEdge(f"H{i}", f"H{(i + 1) % 5}", sign=-1, weight=1.0) for i in range(5)],
    )

    # Layer 3: Cellular Sheaf H^1 Coboundary
    sheaf_rep = CellularSheafCoboundaryAuditor().audit(
        [SheafNode("A", 1), SheafNode("B", 1), SheafNode("C", 1)],
        [
            SheafEdge.identity("ab", "A", "B", offset=(10.0,)),
            SheafEdge.identity("bc", "B", "C", offset=(10.0,)),
            SheafEdge.identity("ca", "C", "A", offset=(10.0,), weight=0.5),
        ],
    )

    # Layer 4 & 5: Bi-Temporal AGM Engine + Supersession Crowding & Sleep Consolidation
    bt = BiTemporalAGMEngine()
    for step in range(8):
        bt.revise_split(
            slot_key="policy:rate_limit",
            valid_from=0.0,
            valid_until=100.0,
            tx_from=10.0 * (step + 1),
            payload={"limit_rps": 100 + step * 25, "operator": "lexicographic_quorum_gate", "text": "rate limit policy"},
            embedding=[0.95, 0.15, 0.05, 0.02],
        )
    # Interior valid-time split on [30, 60) at tx=100
    split_rcpt = bt.revise_split(
        slot_key="policy:rate_limit",
        valid_from=30.0,
        valid_until=60.0,
        tx_from=100.0,
        payload={"limit_rps": 50, "operator": "lexicographic_quorum_gate", "text": "burst window rate limit policy"},
        embedding=[0.96, 0.14, 0.04, 0.02],
    )
    bt.revise_split(
        slot_key="policy:circuit_breaker",
        valid_from=0.0,
        valid_until=100.0,
        tx_from=110.0,
        payload={"threshold": 5, "operator": "lexicographic_quorum_gate", "text": "circuit breaker policy"},
        embedding=[0.20, 0.90, 0.15, 0.05],
    )
    bt_audit = bt.audit_tx_invariants()

    consolidator = SlotSheafCrowdingConsolidator()
    active_ret = consolidator.retrieve_active(
        bt,
        query_text="rate limit policy",
        query_embedding=[0.95, 0.15, 0.05, 0.02],
        valid_at=45.0,
        tx_at=110.0,
        top_k=5,
    )
    hist_snap = bt.query_as_of(valid_at=45.0, tx_at=85.0, slot_key="policy:rate_limit")
    sleep_rep = consolidator.sleep_consolidate(bt, findings=findings, min_procedural_support=2)

    return {
        "version": __version__,
        "layer1_causal_gaba_vendi": {
            "pre_deadend_vetoed": layer1["pre_deadend_vetoed"],
            "post_fix_preserved": layer1["post_fix_preserved"],
            "pool_kernel_psd_verified": layer1["pool_kernel_psd_verified"],
            "vendi_vs2_centered": layer1["vendi_vs2_centered"],
        },
        "layer2_sunflower_frustration": {
            "sunflower_free": sf_rep.post_prune_sunflower_free,
            "pruned_petals": len(sf_rep.pruned_ids),
            "c5_frustration_phi0": fr_rep.frustration_index_phi0,
        },
        "layer3_cellular_sheaf": {
            "has_local_to_global_paradox": sheaf_rep.has_local_to_global_paradox,
            "h1_obstruction_norm": round(sheaf_rep.global_h1_obstruction_norm, 6),
            "cut_restores_consistency": sheaf_rep.cut_restores_global_consistency,
        },
        "layer4_bitemporal_agm": {
            "all_5_invariants_passed": bt_audit.all_passed,
            "split_residual_count": len(split_rcpt.residual_episode_ids),
            "historical_pre_split_limit_rps": hist_snap[0].payload["limit_rps"] if hist_snap else None,
            "current_split_window_head_id": split_rcpt.inserted_head_id,
        },
        "layer5_crowding_and_sleep_merkle": {
            "theoretical_unfiltered_recall_m20_k5": round(theoretical_unfiltered_recall_at_k(5, 20), 6),
            "safe_overfetch_k_rho90_z3": compute_safe_overfetch_k(5, 0.90, 3.0),
            "selected_active_ids": list(active_ret.selected_ids),
            "stale_trap_at_1": active_ret.stale_trap_at_1,
            "semantic_capsule_count": len(sleep_rep.semantic_capsules),
            "procedural_capsule_count": len(sleep_rep.procedural_capsules),
            "all_merkle_proofs_verified": sleep_rep.all_merkle_proofs_verified,
            "batch_merkle_root_hex": sleep_rep.batch_merkle_root_hex,
        },
    }


def _run_self_test() -> Dict[str, Any]:
    """Verify invariants across all layers and return structured status."""
    v1 = _run_demo_v1()
    l5 = v1["layer5_crowding_and_sleep_merkle"]
    checks = {
        "layer1_gaba_and_psd_ok": bool(
            v1["layer1_causal_gaba_vendi"]["pre_deadend_vetoed"]
            and v1["layer1_causal_gaba_vendi"]["post_fix_preserved"]
            and v1["layer1_causal_gaba_vendi"]["pool_kernel_psd_verified"]
        ),
        "layer2_sunflower_frustration_ok": bool(
            v1["layer2_sunflower_frustration"]["sunflower_free"]
            and abs(v1["layer2_sunflower_frustration"]["c5_frustration_phi0"] - 1.0) < 1e-6
        ),
        "layer3_sheaf_h1_ok": bool(
            v1["layer3_cellular_sheaf"]["has_local_to_global_paradox"]
            and v1["layer3_cellular_sheaf"]["cut_restores_consistency"]
        ),
        "layer4_bitemporal_5_invariants_ok": bool(
            v1["layer4_bitemporal_agm"]["all_5_invariants_passed"]
            and v1["layer4_bitemporal_agm"]["split_residual_count"] == 2
            and v1["layer4_bitemporal_agm"]["historical_pre_split_limit_rps"] == 275
        ),
        "layer5_crowding_merkle_ok": bool(
            not l5["stale_trap_at_1"]
            and l5["all_merkle_proofs_verified"]
            and l5["procedural_capsule_count"] >= 1
        ),
    }
    return {
        "self_test_passed": all(checks.values()),
        "version": __version__,
        "checks": checks,
        "summary": v1,
    }


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="temporal-gaba-memory",
        description="6-Layer Mathematically Verified Bi-Temporal, Sheaf-Cohomological & Crowding-Optimal Memory System",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run deterministic end-to-end invariant self-test across all layers",
    )
    sub = parser.add_subparsers(dest="cmd")
    sub.add_parser("demo", help="Run self-contained causal GABA + Vendi + PSD verification demo")
    sub.add_parser("demo-v1", help="Run integrated v1.0.0 6-layer bi-temporal, sheaf & crowding showcase")
    sub.add_parser("self-test", help="Run deterministic end-to-end invariant self-test across all layers")
    sub.add_parser("version", help="Print package version")
    args = parser.parse_args(argv)

    if args.self_test or args.cmd == "self-test":
        res = _run_self_test()
        print(json.dumps(res, indent=2, ensure_ascii=False))
        return 0 if res["self_test_passed"] else 1
    if args.cmd == "version":
        print(__version__)
        return 0
    if args.cmd == "demo-v1":
        print(json.dumps(_run_demo_v1(), indent=2, ensure_ascii=False))
        return 0
    print(json.dumps(_run_demo(), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
