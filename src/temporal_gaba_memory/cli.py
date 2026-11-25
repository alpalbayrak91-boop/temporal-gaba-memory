"""CLI entrypoint for temporal-gaba-memory."""
from __future__ import annotations

import argparse
import json
import math
from typing import List

from . import __version__
from .engine import (
    DeadendRecord,
    EpisodeRecord,
    TemporalGABAMemoryEngine,
    extract_python_ast_profile,
    is_psd_cholesky,
    tokenize_text,
)


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


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="temporal-gaba-memory",
        description="Temporal-Causal GABA Memory & Corpus-Centered 3-Channel RBF Vendi Engine",
    )
    sub = parser.add_subparsers(dest="cmd")
    sub.add_parser("demo", help="Run self-contained causal GABA + Vendi + PSD verification demo")
    sub.add_parser("version", help="Print package version")
    args = parser.parse_args(argv)

    if args.cmd == "version":
        print(__version__)
        return 0
    print(json.dumps(_run_demo(), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
