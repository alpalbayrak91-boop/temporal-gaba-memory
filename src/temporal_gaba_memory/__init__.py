"""Zero-dependency Temporal-Causal GABA Memory & Corpus-Centered 3-Channel RBF Vendi Engine."""
from __future__ import annotations

from .engine import (
    AST_NODE_FAMILIES,
    DeadendRecord,
    EpisodeRecord,
    MemoryConfig,
    RetrievalReport,
    TemporalGABAMemoryEngine,
    center_corpus_H,
    extract_python_ast_profile,
    is_psd_cholesky,
    tokenize_text,
    vec_dot,
    vec_normalize,
    vec_sq_dist,
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "AST_NODE_FAMILIES",
    "DeadendRecord",
    "EpisodeRecord",
    "MemoryConfig",
    "RetrievalReport",
    "TemporalGABAMemoryEngine",
    "center_corpus_H",
    "extract_python_ast_profile",
    "is_psd_cholesky",
    "tokenize_text",
    "vec_dot",
    "vec_normalize",
    "vec_sq_dist",
]
