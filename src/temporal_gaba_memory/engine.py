"""Core implementation of Temporal-Causal GABA Memory and Corpus-Centered 3-Channel RBF Vendi Re-Ranking.

Zero external dependencies (Python 3.10+ standard library only).
"""
from __future__ import annotations

import ast
from collections import Counter
from dataclasses import dataclass, field
import math
from operator import sub
import re
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

_TOKEN_RE = re.compile(r"[a-zA-Z0-9_]+")

# 16 canonical Python AST node families for structural code fingerprinting.
AST_NODE_FAMILIES: Tuple[str, ...] = (
    "FunctionDef",
    "AsyncFunctionDef",
    "ClassDef",
    "Return",
    "Assign",
    "AugAssign",
    "For",
    "While",
    "If",
    "With",
    "Try",
    "Call",
    "Compare",
    "BinOp",
    "Subscript",
    "ListComp",
)


def _require_finite_scalar(val: float, name: str) -> float:
    fval = float(val)
    if not math.isfinite(fval):
        raise ValueError(f"{name} must be finite (got {val!r}).")
    return fval


def _require_finite_vector(vec: Sequence[float], name: str) -> List[float]:
    out: List[float] = []
    for idx, x in enumerate(vec):
        fx = float(x)
        if not math.isfinite(fx):
            raise ValueError(f"{name}[{idx}] must be finite (got {x!r}).")
        out.append(fx)
    return out


def tokenize_text(text: str) -> Dict[str, int]:
    """Deterministic lowercase alphanumeric/underscore term-frequency counter."""
    return dict(Counter(m.group(0).lower() for m in _TOKEN_RE.finditer(text or "")))


def extract_python_ast_profile(source: str) -> List[float]:
    """Extracts a normalized 16-dimensional structural Python AST histogram (fallback: zero vector)."""
    counts = [0.0] * len(AST_NODE_FAMILIES)
    if not source or not source.strip():
        return counts
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return counts
    idx_map = {name: idx for idx, name in enumerate(AST_NODE_FAMILIES)}
    for node in ast.walk(tree):
        pos = idx_map.get(type(node).__name__)
        if pos is not None:
            counts[pos] += 1.0
    return vec_normalize(counts)


def vec_dot(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b):
        raise ValueError(f"Vector dimension mismatch: {len(a)} != {len(b)}")
    va = _require_finite_vector(a, "a")
    vb = _require_finite_vector(b, "b")
    res = math.fsum(x * y for x, y in zip(va, vb))
    return _require_finite_scalar(res, "vec_dot")


def vec_sq_dist(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b):
        raise ValueError(f"Vector dimension mismatch: {len(a)} != {len(b)}")
    va = _require_finite_vector(a, "a")
    vb = _require_finite_vector(b, "b")
    res = math.fsum((x - y) * (x - y) for x, y in zip(va, vb))
    return _require_finite_scalar(res, "vec_sq_dist")


def vec_normalize(v: Sequence[float]) -> List[float]:
    """Overflow-safe L2 normalization using max-abs rescaling."""
    fv = _require_finite_vector(v, "v")
    if not fv:
        return []
    scale = max(abs(x) for x in fv)
    if scale <= 1e-150:
        return [0.0] * len(fv)
    scaled = [x / scale for x in fv]
    norm = math.sqrt(math.fsum(x * x for x in scaled))
    if not math.isfinite(norm) or norm <= 1e-15:
        return [0.0] * len(fv)
    return [x / norm for x in scaled]


def center_corpus_H(matrix: Sequence[Sequence[float]]) -> Tuple[List[List[float]], List[float]]:
    """Applies corpus-level H = I_n - (1/n) 1 1^T centering and row L2 normalization.

    Returns (centered_unit_rows, corpus_mean_vector) so deadends and future items
    project onto the exact frozen corpus manifold without sub-batch distortion.
    """
    n = len(matrix)
    if n == 0:
        return [], []
    d = len(matrix[0])
    rows: List[List[float]] = []
    for idx, row in enumerate(matrix):
        if len(row) != d:
            raise ValueError(f"Inconsistent vector dimension at row {idx}: {len(row)} != {d}")
        rows.append(_require_finite_vector(row, f"matrix[{idx}]"))
    if d == 0:
        return [[] for _ in range(n)], []
    inv_n = 1.0 / float(n)
    mean_vec = [math.fsum(rows[i][j] for i in range(n)) * inv_n for j in range(d)]
    centered = [vec_normalize(list(map(sub, row, mean_vec))) for row in rows]
    return centered, mean_vec


def is_psd_cholesky(matrix: Sequence[Sequence[float]], tol: float = 1e-9) -> bool:
    """Checks positive semi-definiteness of a finite symmetric matrix via jittered Cholesky factorization."""
    if not math.isfinite(tol) or tol < 0.0:
        return False
    m = len(matrix)
    if m == 0:
        return True
    mat: List[List[float]] = []
    for i in range(m):
        if len(matrix[i]) != m:
            return False
        row: List[float] = []
        for j in range(m):
            val = float(matrix[i][j])
            if not math.isfinite(val):
                return False
            row.append(val)
        mat.append(row)
    for i in range(m):
        for j in range(i + 1, m):
            if abs(mat[i][j] - mat[j][i]) > 1e-7:
                return False
    L = [[0.0] * m for _ in range(m)]
    for i in range(m):
        for j in range(i + 1):
            s = math.fsum(L[i][k] * L[j][k] for k in range(j))
            if not math.isfinite(s):
                return False
            if i == j:
                diag = mat[i][i] + tol - s
                if not math.isfinite(diag) or diag < 0.0:
                    return False
                L[i][j] = math.sqrt(max(diag, 0.0))
            else:
                if L[j][j] <= 1e-15:
                    if abs(mat[i][j] - s) > 1e-7:
                        return False
                    L[i][j] = 0.0
                else:
                    L[i][j] = (mat[i][j] - s) / L[j][j]
    return True


@dataclass(frozen=True)
class EpisodeRecord:
    episode_id: str
    timestamp_s: float
    tokens: Mapping[str, int]
    dense_vec: Sequence[float]
    ast_profile: Sequence[float]
    branch_id: str = "default"


@dataclass(frozen=True)
class DeadendRecord:
    deadend_id: str
    discovered_ts_s: float
    dense_vec: Sequence[float]
    threshold_cos: float = 0.45


@dataclass(frozen=True)
class MemoryConfig:
    w_lex: float = 0.30
    w_dense: float = 0.35
    w_ast: float = 0.35
    sigma_lex: float = 0.70
    sigma_dense: float = 0.70
    sigma_ast: float = 0.70
    gaba_threshold: float = 0.45
    causal_margin_s: float = 120.0
    lambda_div: float = 0.60
    rrf_k: float = 60.0
    hop1_boost: float = 0.011
    hop2_boost: float = 0.009
    pool_limit: int = 18

    def __post_init__(self) -> None:
        for fname in (
            "w_lex",
            "w_dense",
            "w_ast",
            "sigma_lex",
            "sigma_dense",
            "sigma_ast",
            "gaba_threshold",
            "causal_margin_s",
            "lambda_div",
            "rrf_k",
            "hop1_boost",
            "hop2_boost",
        ):
            _require_finite_scalar(getattr(self, fname), fname)
        if self.w_lex < 0.0 or self.w_dense < 0.0 or self.w_ast < 0.0:
            raise ValueError("Channel weights must be non-negative for Bochner-PSD convexity.")
        w_sum = self.w_lex + self.w_dense + self.w_ast
        if abs(w_sum - 1.0) > 1e-6:
            raise ValueError(f"Channel weights must sum to 1.0 (got {w_sum:.6f}).")
        if self.sigma_lex <= 0.0 or self.sigma_dense <= 0.0 or self.sigma_ast <= 0.0:
            raise ValueError("RBF bandwidth parameters sigma_* must be strictly positive.")
        if not (-1.0 <= self.gaba_threshold <= 1.0):
            raise ValueError("gaba_threshold must lie in [-1.0, 1.0].")
        if self.causal_margin_s < 0.0:
            raise ValueError("causal_margin_s must be non-negative.")
        if not (0.0 <= self.lambda_div <= 1.0):
            raise ValueError("lambda_div must lie in [0.0, 1.0].")
        if self.rrf_k <= 0.0 or self.pool_limit < 1:
            raise ValueError("rrf_k must be > 0 and pool_limit must be >= 1.")


@dataclass
class RetrievalReport:
    episode_ids: List[str]
    vetoed_ids: List[str]
    vendi_vs2_centered: float
    min_eig_psd_guard: bool
    scores: Dict[str, float] = field(default_factory=dict)


class TemporalGABAMemoryEngine:
    """Zero-dependency Temporal 2-Hop + Causal GABA + Corpus-H-Centered 3-Channel RBF Vendi Engine."""

    def __init__(self, config: Optional[MemoryConfig] = None) -> None:
        self.cfg = config or MemoryConfig()
        self.episodes: List[EpisodeRecord] = []
        self.id_to_idx: Dict[str, int] = {}
        self.deadends: List[DeadendRecord] = []
        self.adj_out: Dict[int, List[int]] = {}

        # Corpus-level frozen statistics & centered manifolds
        self._vocab: Dict[str, int] = {}
        self._idf: List[float] = []
        self._doc_lens: List[float] = []
        self._avgdl: float = 1.0
        self._lex_cent: List[List[float]] = []
        self._dense_cent: List[List[float]] = []
        self._ast_cent: List[List[float]] = []
        self._dense_mean: List[float] = []
        self._dense_dim: int = 0
        self._deadend_cent: List[List[float]] = []

    def index(
        self,
        episodes: Sequence[EpisodeRecord],
        deadends: Sequence[DeadendRecord] = (),
        causal_edges: Sequence[Tuple[str, str]] = (),
    ) -> None:
        ep_list = list(episodes)
        seen_ids: Dict[str, int] = {}
        for i, ep in enumerate(ep_list):
            if ep.episode_id in seen_ids:
                raise ValueError(f"Duplicate episode_id: {ep.episode_id}")
            _require_finite_scalar(ep.timestamp_s, f"episodes[{i}].timestamp_s")
            _require_finite_vector(ep.dense_vec, f"episodes[{i}].dense_vec")
            _require_finite_vector(ep.ast_profile, f"episodes[{i}].ast_profile")
            for tok, cnt in ep.tokens.items():
                if isinstance(cnt, bool) or not isinstance(cnt, (int, float)) or not math.isfinite(float(cnt)) or cnt < 0:
                    raise ValueError(f"Invalid token count for {ep.episode_id}[{tok!r}]: {cnt!r}")
            seen_ids[ep.episode_id] = i

        de_list = list(deadends)
        for i, d in enumerate(de_list):
            _require_finite_scalar(d.discovered_ts_s, f"deadends[{i}].discovered_ts_s")
            th = _require_finite_scalar(d.threshold_cos, f"deadends[{i}].threshold_cos")
            if not (-1.0 <= th <= 1.0):
                raise ValueError(f"deadends[{i}].threshold_cos must lie in [-1.0, 1.0] (got {th}).")
            _require_finite_vector(d.dense_vec, f"deadends[{i}].dense_vec")

        self.episodes = ep_list
        self.id_to_idx = seen_ids
        self.deadends = de_list
        self.adj_out = {i: [] for i in range(len(self.episodes))}

        # Enforce strict causal edge direction ts(u) < ts(v) and deduplicate edges
        seen_edges = set()
        for u_id, v_id in causal_edges:
            if u_id in self.id_to_idx and v_id in self.id_to_idx:
                u, v = self.id_to_idx[u_id], self.id_to_idx[v_id]
                if self.episodes[u].timestamp_s < self.episodes[v].timestamp_s and (u, v) not in seen_edges:
                    self.adj_out[u].append(v)
                    seen_edges.add((u, v))

        n = len(self.episodes)
        if n == 0:
            self._vocab = {}
            self._idf = []
            self._doc_lens = []
            self._avgdl = 1.0
            self._lex_cent = []
            self._dense_cent = []
            self._ast_cent = []
            self._dense_mean = []
            self._dense_dim = 0
            self._deadend_cent = []
            return

        # 1) Build TF-IDF lexical channel & BM25 stats
        vocab_set = sorted({tok for ep in self.episodes for tok, cnt in ep.tokens.items() if cnt > 0})
        self._vocab = {tok: i for i, tok in enumerate(vocab_set)}
        v_size = len(vocab_set)
        df = [0] * v_size
        self._doc_lens = [float(sum(ep.tokens.values())) for ep in self.episodes]
        self._avgdl = max(sum(self._doc_lens) / float(n), 1e-9)
        for ep in self.episodes:
            for tok, cnt in ep.tokens.items():
                if cnt > 0:
                    df[self._vocab[tok]] += 1
        self._idf = [math.log((n - d + 0.5) / (d + 0.5) + 1.0) for d in df]

        lex_raw: List[List[float]] = []
        for ep in self.episodes:
            row = [0.0] * v_size
            for tok, cnt in ep.tokens.items():
                if cnt > 0:
                    j = self._vocab[tok]
                    row[j] = float(cnt) * self._idf[j]
            lex_raw.append(vec_normalize(row))

        # 2) Freeze corpus-level H = I_n - (1/n) 1 1^T centered manifolds
        self._lex_cent, _ = center_corpus_H(lex_raw)
        self._dense_cent, self._dense_mean = center_corpus_H([list(ep.dense_vec) for ep in self.episodes])
        self._dense_dim = len(self._dense_mean)
        self._ast_cent, _ = center_corpus_H([list(ep.ast_profile) for ep in self.episodes])

        # 3) Project negative-knowledge deadends onto identical corpus-centered dense manifold
        self._deadend_cent = []
        for d in self.deadends:
            if len(d.dense_vec) != self._dense_dim:
                raise ValueError(
                    f"Deadend {d.deadend_id} dense_vec dim {len(d.dense_vec)} != corpus dim {self._dense_dim}"
                )
            self._deadend_cent.append(vec_normalize(list(map(sub, d.dense_vec, self._dense_mean))))

    def _is_causally_vetoed(self, ep_idx: int) -> bool:
        """Temporal-Causal GABA Gate: vetoes pre-discovery deadends while preserving post-discovery fixes."""
        ep_ts = self.episodes[ep_idx].timestamp_s
        ep_vec = self._dense_cent[ep_idx]
        for d_rec, d_vec in zip(self.deadends, self._deadend_cent):
            thresh = max(self.cfg.gaba_threshold, d_rec.threshold_cos)
            if vec_dot(ep_vec, d_vec) >= thresh:
                if ep_ts < (d_rec.discovered_ts_s - self.cfg.causal_margin_s):
                    return True
        return False

    def _pool_rbf_kernel(self, pool_indices: Sequence[int]) -> List[List[float]]:
        """Bochner-PSD convex combination of 3 corpus-centered RBF channels."""
        m = len(pool_indices)
        c = self.cfg
        inv_l = 1.0 / (2.0 * c.sigma_lex * c.sigma_lex)
        inv_d = 1.0 / (2.0 * c.sigma_dense * c.sigma_dense)
        inv_a = 1.0 / (2.0 * c.sigma_ast * c.sigma_ast)
        K = [[1.0] * m for _ in range(m)]
        for i in range(m):
            ii = pool_indices[i]
            li, di, ai = self._lex_cent[ii], self._dense_cent[ii], self._ast_cent[ii]
            for j in range(i + 1, m):
                jj = pool_indices[j]
                k_l = math.exp(-vec_sq_dist(li, self._lex_cent[jj]) * inv_l)
                k_d = math.exp(-vec_sq_dist(di, self._dense_cent[jj]) * inv_d)
                k_a = math.exp(-vec_sq_dist(ai, self._ast_cent[jj]) * inv_a)
                val = c.w_lex * k_l + c.w_dense * k_d + c.w_ast * k_a
                K[i][j] = val
                K[j][i] = val
        return K

    def query(
        self,
        query_tokens: Sequence[str],
        query_dense_vec: Sequence[float],
        top_k: int = 5,
    ) -> RetrievalReport:
        n = len(self.episodes)
        q_vec = _require_finite_vector(query_dense_vec, "query_dense_vec")
        if n == 0 or top_k <= 0:
            return RetrievalReport([], [], 0.0, True, {})
        if len(q_vec) != self._dense_dim:
            raise ValueError(f"Query dense_vec dim {len(q_vec)} != corpus dim {self._dense_dim}")

        # 1) BM25 + Dense Hybrid RRF
        k1, b = 1.5, 0.75
        bm25 = [0.0] * n
        for tok in set(query_tokens):
            j = self._vocab.get(tok)
            if j is None:
                continue
            idf_t = self._idf[j]
            for i, ep in enumerate(self.episodes):
                tf = float(ep.tokens.get(tok, 0))
                if tf > 0.0:
                    denom = tf + k1 * (1.0 - b + b * (self._doc_lens[i] / self._avgdl))
                    bm25[i] += idf_t * (tf * (k1 + 1.0)) / denom

        q_unit = vec_normalize(q_vec)
        dense_sc = [vec_dot(vec_normalize(ep.dense_vec), q_unit) for ep in self.episodes]

        top_b = sorted(range(n), key=lambda i: (bm25[i], -i), reverse=True)[:40]
        top_d = sorted(range(n), key=lambda i: (dense_sc[i], -i), reverse=True)[:40]
        rrf: Dict[int, float] = {}
        for rk, idx in enumerate(top_b):
            rrf[idx] = rrf.get(idx, 0.0) + 1.0 / (self.cfg.rrf_k + rk)
        for rk, idx in enumerate(top_d):
            rrf[idx] = rrf.get(idx, 0.0) + 1.0 / (self.cfg.rrf_k + rk)

        # 2) 2-Hop Causal Temporal Expansion (monotonic in base RRF score)
        seeds = sorted(rrf.keys(), key=lambda i: rrf[i], reverse=True)[:10]
        cand_scores: Dict[int, float] = dict(rrf)
        for s in seeds:
            base_s = rrf[s]
            for n1 in self.adj_out[s]:
                hop1_sc = max(rrf.get(n1, 0.0), base_s * 0.72) + self.cfg.hop1_boost
                cand_scores[n1] = max(cand_scores.get(n1, 0.0), hop1_sc)
                for n2 in self.adj_out[n1]:
                    hop2_sc = max(rrf.get(n2, 0.0), base_s * 0.64) + self.cfg.hop2_boost
                    cand_scores[n2] = max(cand_scores.get(n2, 0.0), hop2_sc)

        # 3) Temporal-Causal GABA Veto
        kept: List[Tuple[int, float]] = []
        vetoed: List[str] = []
        for cid, sc in cand_scores.items():
            if self._is_causally_vetoed(cid):
                vetoed.append(self.episodes[cid].episode_id)
            else:
                kept.append((cid, sc))

        kept.sort(key=lambda pair: (pair[1], -pair[0]), reverse=True)
        pool = kept[: self.cfg.pool_limit]
        if not pool:
            return RetrievalReport([], vetoed, 0.0, True, {})

        pool_ids = [p[0] for p in pool]
        pool_scores = [p[1] for p in pool]
        K_pool = self._pool_rbf_kernel(pool_ids)
        psd_ok = is_psd_cholesky(K_pool)

        # 4) O(k * M^2) Incremental Renyi-2 Vendi Greedy Selection
        m = len(pool_ids)
        s_min, s_max = min(pool_scores), max(pool_scores)
        s_norm = [(s - s_min) / ((s_max - s_min) + 1e-12) for s in pool_scores]
        first = max(range(m), key=lambda i: s_norm[i])
        sel = [first]
        rem = [i for i in range(m) if i != first]
        frob_sq = 1.0
        lam = self.cfg.lambda_div

        while len(sel) < min(top_k, m) and rem:
            k_next = float(len(sel) + 1)
            best_r = rem[0]
            best_obj = -1e18
            best_frob = frob_sq
            for r in rem:
                Kr = K_pool[r]
                cand_frob = frob_sq + 1.0 + 2.0 * sum(Kr[s] * Kr[s] for s in sel)
                vs2_norm = k_next / max(cand_frob, 1e-12)
                obj = (1.0 - lam) * s_norm[r] + lam * vs2_norm
                if obj > best_obj:
                    best_obj = obj
                    best_r = r
                    best_frob = cand_frob
            sel.append(best_r)
            rem.remove(best_r)
            frob_sq = best_frob

        k_final = len(sel)
        vs2_final = (k_final * k_final) / max(frob_sq, 1e-12)
        out_ids = [self.episodes[pool_ids[i]].episode_id for i in sel]
        out_scores = {self.episodes[pool_ids[i]].episode_id: round(pool_scores[i], 6) for i in sel}
        return RetrievalReport(
            episode_ids=out_ids,
            vetoed_ids=vetoed,
            vendi_vs2_centered=round(vs2_final, 4),
            min_eig_psd_guard=psd_ok,
            scores=out_scores,
        )
