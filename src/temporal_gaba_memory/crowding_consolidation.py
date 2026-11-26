"""Supersession-Crowding-Optimal Retrieval & 3-Tier Merkle Sleep Consolidation (Layer 5).

Zero-dependency pure Python 3.10+ standard-library implementation of:
1. Analytical Supersession Crowding Law: `Recall@k(m) = min(1.0, k / (m + 1))`.
2. Binomial-Tail & Exact KL-Chernoff Tombstone-Aware Over-Fetching (`compute_safe_overfetch_k`).
3. Platform-Deterministic Fixed-Point Integer 4-Signal Reciprocal Rank Fusion
   (`fixed_point_rrf_fuse`, `SCALE = 1_000_000` micro-points).
4. 3-Tier Hierarchical Sleep Consolidation (`Episodic -> SemanticCapsule -> ProceduralCapsule`)
   with `RFC 8785` JSON Canonicalization Scheme (JCS) and `RFC 6962` SHA-256 Merkle
   inclusion proofs.

Primary references:
- Cormack, G. V., Clarke, C. L., & Buettcher, S. (2009). Reciprocal rank fusion
  outperforms condorcet and individual rank methods. Proceedings of the 32nd
  International ACM SIGIR Conference (SIGIR '09), 758-759.
- Malkov, Y. A., & Yashunin, D. A. (2020). Efficient and robust approximate
  nearest neighbor search using Hierarchical Navigable Small World graphs.
  IEEE Transactions on Pattern Analysis and Machine Intelligence, 42(4), 824-836.
- Stickgold, R., & Walker, M. P. (2013). Sleep-dependent memory triage: evolving
  generalization through selective processing. Nature Neuroscience, 16(2), 139-145.
- Rundgren, A., Jordan, B., & Erdtman, S. (2020). RFC 8785: JSON Canonicalization
  Scheme (JCS). IETF.
- Laurie, B., Langley, A., & Kasper, E. (2013). RFC 6962: Certificate Transparency
  (Section 2.1, Merkle Hash Trees). IETF.

License: MIT
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import math
from operator import mul
import re
from types import MappingProxyType
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple
import weakref

from .bitemporal_agm import BiTemporalAGMEngine, BiTemporalEpisode
from .engine import DeadendRecord, tokenize_text, vec_dot, vec_normalize
from .sunflower_frustration import FindingRecord, SunflowerMemoryPruner

__all__ = [
    "CROWDING_MODULE_VERSION",
    "FIXED_POINT_SCALE",
    "theoretical_unfiltered_recall_at_k",
    "binomial_kl_tail_bound",
    "compute_safe_overfetch_k",
    "rrf_term_micro",
    "gaba_inhibit_micro",
    "fixed_point_rrf_fuse",
    "jcs_canonicalize",
    "MerkleInclusionProof",
    "rfc6962_leaf_hash",
    "rfc6962_node_hash",
    "build_rfc6962_merkle_root",
    "generate_rfc6962_inclusion_proof",
    "verify_rfc6962_inclusion_proof",
    "SemanticCapsule",
    "ProceduralCapsule",
    "CrowdingRetrievalReport",
    "SleepConsolidationReport",
    "SlotSheafCrowdingConsolidator",
]

CROWDING_MODULE_VERSION = "1.0.0"
FIXED_POINT_SCALE: int = 1_000_000

_WORD_RE = re.compile(r"[a-zA-Z0-9_]{2,}")


# ---------------------------------------------------------------------------
# 1. Analytical Supersession Crowding & Tombstone-Aware Over-Fetching
# ---------------------------------------------------------------------------


def theoretical_unfiltered_recall_at_k(k: int, chain_len: int) -> float:
    """Analytical `Recall@k(m) = min(1.0, k / (m + 1))` under `m = chain_len` supersessions.

    When an entity slot undergoes `m` updates without tombstone eviction/filtering,
    `m` superseded historical versions compete alongside `1` active head (`m + 1`
    near-tied versions). Under symmetric rank tie-breaking or slight query drift,
    an unfiltered top-`k` cutoff recovers the active head with probability
    `min(1.0, k / (m + 1))`.
    """
    if isinstance(k, bool) or not isinstance(k, int):
        raise TypeError("k must be an integer.")
    if isinstance(chain_len, bool) or not isinstance(chain_len, int) or chain_len < 0:
        raise ValueError(f"chain_len must be an integer >= 0 (got {chain_len!r}).")
    if k <= 0:
        return 0.0
    return min(1.0, float(k) / float(chain_len + 1))


def binomial_kl_tail_bound(n: int, target_k: int, active_prob: float) -> float:
    """Exact Hoeffding-Chernoff Binomial lower-tail bound `P(Bin(n, p) <= target_k - 1)`.

    Uses the relative-entropy (Kullback-Leibler) Sanov-Hoeffding inequality:
    for `q = (target_k - 1) / n < p`, `P(X <= target_k - 1) <= exp(-n * D_KL(q || p))`.
    """
    if n < 0 or target_k < 0:
        raise ValueError("n and target_k must be >= 0.")
    if not math.isfinite(active_prob) or not (0.0 <= active_prob <= 1.0):
        raise ValueError(f"active_prob must lie in [0.0, 1.0] (got {active_prob!r}).")
    if target_k == 0:
        return 0.0
    if n < target_k or active_prob <= 0.0:
        return 1.0
    if active_prob >= 1.0:
        return 0.0

    m_fail = target_k - 1
    q = float(m_fail) / float(n)
    if q >= active_prob:
        return 1.0
    if m_fail == 0:
        # Exact P(Bin(n, p) == 0) = (1 - p)^n = exp(n * ln(1 - p))
        return math.exp(float(n) * math.log(1.0 - active_prob))

    kl = q * math.log(q / active_prob) + (1.0 - q) * math.log((1.0 - q) / (1.0 - active_prob))
    return math.exp(-float(n) * kl)


def compute_safe_overfetch_k(
    target_k: int,
    tombstone_ratio: float,
    confidence_z: float = 3.0,
    max_corpus_size: Optional[int] = None,
) -> int:
    """Compute candidate pool size `k_eff` preventing post-filter starvation under crowding.

    Combines:
    1. Closed-form normal-approximated quadratic lower-tail inversion (`n_norm`),
    2. Multiplicative Chernoff lower-tail bound (`n_mult_chernoff`), and
    3. Exact non-asymptotic Binomial KL-Chernoff upper-tail inversion (`n_kl`),
    guaranteeing `P(ActiveCount(k_eff) < target_k) <= Phi(-confidence_z)` even when
    `80%-95%` of indexed records are superseded tombstones.
    """
    if isinstance(target_k, bool) or not isinstance(target_k, int) or target_k < 0:
        raise ValueError(f"target_k must be an integer >= 0 (got {target_k!r}).")
    rho = float(tombstone_ratio)
    if not math.isfinite(rho) or not (0.0 <= rho < 1.0):
        raise ValueError(f"tombstone_ratio must lie in [0.0, 1.0) (got {tombstone_ratio!r}).")
    z = float(confidence_z)
    if not math.isfinite(z) or z < 0.0:
        raise ValueError(f"confidence_z must be finite and >= 0.0 (got {confidence_z!r}).")
    if max_corpus_size is not None and (isinstance(max_corpus_size, bool) or max_corpus_size < 0):
        raise ValueError("max_corpus_size must be >= 0 when provided.")

    if target_k == 0:
        return 0

    if rho == 0.0:
        return min(target_k, max_corpus_size) if max_corpus_size is not None else target_k

    p = 1.0 - rho
    if z == 0.0:
        k_eff = max(target_k, int(math.ceil(float(target_k) / p)))
        return min(k_eff, max_corpus_size) if max_corpus_size is not None else k_eff

    # 1) Normal-approximation quadratic solution for n*p - z*sqrt(n*p*(1-p)) >= target_k
    var_unit = p * (1.0 - p)
    u = (z * math.sqrt(var_unit) + math.sqrt(z * z * var_unit + 4.0 * p * float(target_k))) / (2.0 * p)
    n_norm = max(target_k, int(math.ceil(u * u - 1e-12)))

    # 2) Target tail probability matching Gaussian one-sided z-score
    delta = max(1e-15, 0.5 * math.erfc(z / math.sqrt(2.0)))
    neg_log_delta = -math.log(delta)

    # 3) Multiplicative Chernoff bound: P(X <= (1 - eps) n p) <= exp(-eps^2 n p / 2) <= delta
    # Solve (n*p - (target_k - 1))^2 / (2 * n * p) >= neg_log_delta
    m_thr = float(target_k - 1)
    b_coef = -(2.0 * m_thr + 2.0 * neg_log_delta)
    disc = b_coef * b_coef - 4.0 * m_thr * m_thr
    np_chernoff = 0.5 * (-b_coef + math.sqrt(max(0.0, disc)))
    n_mult_chernoff = max(target_k, int(math.ceil(np_chernoff / p - 1e-12)))

    # 4) Exact KL-Chernoff (Hoeffding 1963) binary search
    lo = max(target_k, int(math.floor(m_thr / p)) + 1)
    hi = max(lo, n_norm, n_mult_chernoff)
    while binomial_kl_tail_bound(hi, target_k, p) > delta:
        hi *= 2
    while lo < hi:
        mid = (lo + hi) // 2
        if binomial_kl_tail_bound(mid, target_k, p) <= delta:
            hi = mid
        else:
            lo = mid + 1
    n_kl = lo

    k_eff = max(target_k, n_norm, n_mult_chernoff, n_kl)
    if max_corpus_size is not None:
        return min(k_eff, max_corpus_size)
    return k_eff


# ---------------------------------------------------------------------------
# 2. Fixed-Point Integer 4-Signal Reciprocal Rank Fusion
# ---------------------------------------------------------------------------


def rrf_term_micro(
    rank_zero_based: int,
    rrf_k: int = 60,
    weight_milli: int = 1000,
    scale: int = FIXED_POINT_SCALE,
) -> int:
    """Compute exact integer micro-point RRF contribution `(scale * w_milli) // (1000 * (k + 1 + rank))`."""
    if rank_zero_based < 0:
        raise ValueError(f"rank_zero_based must be >= 0 (got {rank_zero_based}).")
    if rrf_k < 0:
        raise ValueError(f"rrf_k must be >= 0 (got {rrf_k}).")
    if weight_milli < 0 or scale <= 0:
        raise ValueError("weight_milli must be >= 0 and scale must be > 0.")
    denom = 1000 * (rrf_k + 1 + rank_zero_based)
    return (scale * weight_milli) // denom


def gaba_inhibit_micro(base_score_micro: int, penalty_micro: int) -> int:
    """Saturating non-negative integer GABA inhibition `max(0, base - max(0, penalty))`."""
    if base_score_micro <= 0:
        return 0
    if penalty_micro <= 0:
        return int(base_score_micro)
    diff = int(base_score_micro) - int(penalty_micro)
    return diff if diff > 0 else 0


def fixed_point_rrf_fuse(
    dense_ranked_ids: Sequence[str],
    lexical_ranked_ids: Sequence[str],
    graph_ranked_ids: Sequence[str] = (),
    gaba_penalties_micro: Optional[Mapping[str, int]] = None,
    *,
    graph_hop_distances: Optional[Mapping[str, int]] = None,
    gaba_vetoed_ids: Iterable[str] = (),
    rrf_k: int = 60,
    w_dense_milli: int = 1000,
    w_lexical_milli: int = 1000,
    w_graph_milli: int = 800,
    scale: int = FIXED_POINT_SCALE,
    drop_vetoed: bool = True,
) -> List[Tuple[str, int]]:
    """Fuse 4 retrieval signals using purely integer micro-point arithmetic (`SCALE = 1_000_000`).

    Signals:
    1. Dense cosine similarity rank list (`dense_ranked_ids`)
    2. Sparse BM25/TF-IDF lexical rank list (`lexical_ranked_ids`)
    3. 2-hop active graph proximity (`graph_ranked_ids` and/or `graph_hop_distances`)
    4. Causal GABA inhibitory suppression (`gaba_penalties_micro` + `gaba_vetoed_ids`)

    Zero floating-point operations occur inside this function, guaranteeing identical
    rankings across x86_64, ARM64, Linux, macOS, and Windows runtimes.
    """
    scores: Dict[str, int] = defaultdict(int)
    veto_set = set(gaba_vetoed_ids)
    penalties = gaba_penalties_micro or {}

    seen_dense: Set[str] = set()
    for rk, cid in enumerate(dense_ranked_ids):
        if cid in seen_dense:
            continue
        seen_dense.add(cid)
        scores[cid] += rrf_term_micro(rk, rrf_k=rrf_k, weight_milli=w_dense_milli, scale=scale)

    seen_lex: Set[str] = set()
    for rk, cid in enumerate(lexical_ranked_ids):
        if cid in seen_lex:
            continue
        seen_lex.add(cid)
        scores[cid] += rrf_term_micro(rk, rrf_k=rrf_k, weight_milli=w_lexical_milli, scale=scale)

    seen_graph: Set[str] = set()
    for rk, cid in enumerate(graph_ranked_ids):
        if cid in seen_graph:
            continue
        seen_graph.add(cid)
        scores[cid] += rrf_term_micro(rk, rrf_k=rrf_k, weight_milli=w_graph_milli, scale=scale)

    if graph_hop_distances:
        for cid in sorted(graph_hop_distances.keys()):
            if cid in seen_graph:
                continue
            hop = int(graph_hop_distances[cid])
            if hop == 1:
                # Hop-1 neighbor receives rank-0 graph RRF bonus
                scores[cid] += rrf_term_micro(0, rrf_k=rrf_k, weight_milli=w_graph_milli, scale=scale)
            elif hop == 2:
                # Hop-2 neighbor receives half-weight rank-10 graph RRF bonus
                scores[cid] += rrf_term_micro(10, rrf_k=rrf_k, weight_milli=w_graph_milli // 2, scale=scale)

    final_items: List[Tuple[str, int]] = []
    for cid, base_sc in scores.items():
        if cid in veto_set:
            if not drop_vetoed:
                final_items.append((cid, 0))
            continue
        pen = int(penalties.get(cid, 0))
        inhibited = gaba_inhibit_micro(base_sc, pen)
        final_items.append((cid, inhibited))

    # Strict deterministic tie-breaker: descending integer score, ascending ID
    final_items.sort(key=lambda item: (-item[1], item[0]))
    return final_items


# ---------------------------------------------------------------------------
# 3. RFC 8785 JCS Canonicalization & RFC 6962 SHA-256 Merkle Provenance
# ---------------------------------------------------------------------------


def _jcs_escape_string(s: str) -> str:
    """RFC 8785 Section 3.2.2 string serialization (rejecting lone UTF-16 surrogates)."""
    out: List[str] = ['"']
    for ch in s:
        code = ord(ch)
        if 0xD800 <= code <= 0xDFFF:
            raise ValueError(f"RFC 8785 JCS forbids lone UTF-16 surrogate code point U+{code:04X}.")
        if ch == '"':
            out.append('\\"')
        elif ch == "\\":
            out.append("\\\\")
        elif code == 0x08:
            out.append("\\b")
        elif code == 0x09:
            out.append("\\t")
        elif code == 0x0A:
            out.append("\\n")
        elif code == 0x0C:
            out.append("\\f")
        elif code == 0x0D:
            out.append("\\r")
        elif code < 0x20:
            out.append(f"\\u{code:04x}")
        else:
            out.append(ch)
    out.append('"')
    return "".join(out)


def _jcs_format_float(val: float) -> str:
    """RFC 8785 Section 3.2.2.3 / ECMA-262 Section 7.1.12.1 IEEE-754 number formatting."""
    if not math.isfinite(val):
        raise ValueError(f"RFC 8785 JCS forbids non-finite numbers (got {val!r}).")
    if val == 0.0:
        return "0"

    sign = "-" if val < 0.0 else ""
    s = repr(abs(val)).lower()
    mant, exp_s = s.split("e") if "e" in s else (s, "0")
    exp = int(exp_s)
    int_part, frac_part = mant.split(".") if "." in mant else (mant, "")

    raw_nz = (int_part + frac_part).rstrip("0")
    digits = raw_nz.lstrip("0")
    lz = len(raw_nz) - len(digits)
    n = len(int_part) - lz + exp
    k = len(digits)

    if k <= n <= 21:
        body = digits + ("0" * (n - k))
    elif 0 < n <= 21:
        body = digits[:n] + "." + digits[n:]
    elif -5 <= n <= 0:
        body = "0." + ("0" * (-n)) + digits
    else:
        e_val = n - 1
        e_str = f"+{e_val}" if e_val >= 0 else str(e_val)
        body = f"{digits}e{e_str}" if k == 1 else f"{digits[0]}.{digits[1:]}e{e_str}"
    return sign + body


def _jcs_serialize_node(value: Any) -> str:
    if value is None:
        return "null"
    if value is True:
        return "true"
    if value is False:
        return "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return _jcs_format_float(value)
    if isinstance(value, str):
        return _jcs_escape_string(value)
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(_jcs_serialize_node(item) for item in value) + "]"
    if isinstance(value, Mapping):
        items: List[Tuple[bytes, str, Any]] = []
        for k, v in value.items():
            if not isinstance(k, str):
                raise TypeError(f"RFC 8785 JCS object keys must be strings (got {type(k).__name__}).")
            for ch in k:
                code = ord(ch)
                if 0xD800 <= code <= 0xDFFF:
                    raise ValueError(f"RFC 8785 JCS forbids lone UTF-16 surrogate in key U+{code:04X}.")
            # RFC 8785 Section 3.2.3: sort lexicographically by UTF-16 code units
            items.append((k.encode("utf-16-be"), k, v))
        items.sort(key=lambda t: t[0])
        return "{" + ",".join(f"{_jcs_escape_string(k)}:{_jcs_serialize_node(v)}" for _, k, v in items) + "}"
    raise TypeError(f"Unsupported type for RFC 8785 JCS serialization: {type(value).__name__}")


def jcs_canonicalize(value: Any) -> bytes:
    """Serialize `value` into canonical UTF-8 bytes per IETF RFC 8785 (JCS)."""
    return _jcs_serialize_node(value).encode("utf-8")


def rfc6962_leaf_hash(leaf_data: bytes) -> bytes:
    """RFC 6962 Section 2.1 leaf hash: `SHA-256(0x00 || leaf_data)`."""
    return hashlib.sha256(b"\x00" + leaf_data).digest()


def rfc6962_node_hash(left_hash: bytes, right_hash: bytes) -> bytes:
    """RFC 6962 Section 2.1 interior node hash: `SHA-256(0x01 || left || right)`."""
    return hashlib.sha256(b"\x01" + left_hash + right_hash).digest()


def _largest_power_of_two_less_than(n: int) -> int:
    """Return largest `k = 2^p` strictly smaller than `n` (`n >= 2`)."""
    return 1 << ((n - 1).bit_length() - 1)


def _mth_bytes(leaves: Sequence[bytes]) -> bytes:
    n = len(leaves)
    if n == 0:
        return hashlib.sha256(b"").digest()
    if n == 1:
        return rfc6962_leaf_hash(leaves[0])
    k = _largest_power_of_two_less_than(n)
    return rfc6962_node_hash(_mth_bytes(leaves[:k]), _mth_bytes(leaves[k:]))


def build_rfc6962_merkle_root(leaves: Sequence[bytes]) -> str:
    """Compute hex-encoded RFC 6962 Section 2.1 Merkle Tree Hash (`MTH`)."""
    return _mth_bytes(leaves).hex()


def _audit_path_bytes(index: int, leaves: Sequence[bytes]) -> List[bytes]:
    n = len(leaves)
    if n <= 1:
        return []
    k = _largest_power_of_two_less_than(n)
    if index < k:
        path = _audit_path_bytes(index, leaves[:k])
        path.append(_mth_bytes(leaves[k:]))
        return path
    path = _audit_path_bytes(index - k, leaves[k:])
    path.append(_mth_bytes(leaves[:k]))
    return path


@dataclass(frozen=True)
class MerkleInclusionProof:
    """RFC 6962 Section 2.1.1 Merkle audit path proving leaf inclusion in a capsule."""

    leaf_index: int
    tree_size: int
    leaf_hash_hex: str
    sibling_hashes_hex: Tuple[str, ...]
    root_hash_hex: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "leaf_index": self.leaf_index,
            "tree_size": self.tree_size,
            "leaf_hash_hex": self.leaf_hash_hex,
            "sibling_hashes_hex": list(self.sibling_hashes_hex),
            "root_hash_hex": self.root_hash_hex,
        }


def generate_rfc6962_inclusion_proof(index: int, leaves: Sequence[bytes]) -> MerkleInclusionProof:
    """Generate an RFC 6962 Section 2.1.1 audit path for `leaves[index]`."""
    n = len(leaves)
    if index < 0 or index >= n:
        raise IndexError(f"Leaf index {index} out of bounds for tree of size {n}.")
    leaf_h = rfc6962_leaf_hash(leaves[index]).hex()
    siblings = tuple(h.hex() for h in _audit_path_bytes(index, leaves))
    root_h = _mth_bytes(leaves).hex()
    return MerkleInclusionProof(
        leaf_index=index,
        tree_size=n,
        leaf_hash_hex=leaf_h,
        sibling_hashes_hex=siblings,
        root_hash_hex=root_h,
    )


def verify_rfc6962_inclusion_proof(
    leaf_data: bytes,
    proof: MerkleInclusionProof,
    expected_root_hex: Optional[str] = None,
    *,
    expected_tree_size: Optional[int] = None,
) -> bool:
    """Cryptographically verify an RFC 6962 Merkle inclusion proof (Section 2.1.3)."""
    target_root = expected_root_hex if expected_root_hex is not None else proof.root_hash_hex
    if expected_tree_size is not None and proof.tree_size != expected_tree_size:
        return False
    if proof.tree_size < 1 or proof.leaf_index < 0 or proof.leaf_index >= proof.tree_size:
        return False
    computed_leaf = rfc6962_leaf_hash(leaf_data)
    if computed_leaf.hex() != proof.leaf_hash_hex:
        return False

    try:
        sib_bytes = [bytes.fromhex(h) for h in proof.sibling_hashes_hex]
    except ValueError:
        return False
    if any(len(b) != 32 for b in sib_bytes):
        return False

    # Reconstruct exact split sequence (m, n) top-down, then fold bottom-up
    splits: List[bool] = []  # True -> leaf on left (sibling on right); False -> leaf on right
    m = proof.leaf_index
    n = proof.tree_size
    while n > 1:
        k = _largest_power_of_two_less_than(n)
        if m < k:
            splits.append(True)
            n = k
        else:
            splits.append(False)
            m -= k
            n -= k

    if len(splits) != len(sib_bytes):
        return False

    cur = computed_leaf
    # `_audit_path_bytes` orders siblings bottom-up, whereas `splits` was collected top-down
    for is_left_child, sib in zip(reversed(splits), sib_bytes):
        if is_left_child:
            cur = rfc6962_node_hash(cur, sib)
        else:
            cur = rfc6962_node_hash(sib, cur)

    return cur.hex() == target_root and proof.root_hash_hex == target_root


# ---------------------------------------------------------------------------
# 4. 3-Tier Sleep Consolidation Capsules & Reports
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SemanticCapsule:
    """Tier-2 sleep-consolidated semantic capsule with RFC 6962 lineage proofs."""

    capsule_id: str
    slot_key: str
    active_head_id: str
    consolidated_episode_ids: Tuple[str, ...]
    superseded_count: int
    valid_from: float
    valid_until: Optional[float]
    invariant_atoms: Tuple[str, ...]
    canonical_summary: Mapping[str, Any]
    embedding: Tuple[float, ...]
    merkle_root_hex: str
    inclusion_proofs: Mapping[str, MerkleInclusionProof]

    def verify_episode_inclusion(self, episode: BiTemporalEpisode) -> bool:
        proof = self.inclusion_proofs.get(episode.episode_id)
        if proof is None:
            return False
        leaf_bytes = jcs_canonicalize(episode.to_dict())
        return verify_rfc6962_inclusion_proof(
            leaf_bytes,
            proof,
            self.merkle_root_hex,
            expected_tree_size=len(self.consolidated_episode_ids),
        )


@dataclass(frozen=True)
class ProceduralCapsule:
    """Tier-3 generalized procedural rule distilled across recurring semantic capsules."""

    procedure_id: str
    pattern_signature: Tuple[str, ...]
    recommended_operator: str
    supporting_capsule_ids: Tuple[str, ...]
    source_episode_ids: Tuple[str, ...]
    support_count: int
    merkle_root_hex: str
    inclusion_proofs: Mapping[str, MerkleInclusionProof]

    def verify_capsule_inclusion(self, capsule: SemanticCapsule) -> bool:
        proof = self.inclusion_proofs.get(capsule.capsule_id)
        if proof is None:
            return False
        leaf_bytes = jcs_canonicalize(
            {
                "capsule_id": capsule.capsule_id,
                "slot_key": capsule.slot_key,
                "merkle_root_hex": capsule.merkle_root_hex,
                "active_head_id": capsule.active_head_id,
            }
        )
        return verify_rfc6962_inclusion_proof(
            leaf_bytes,
            proof,
            self.merkle_root_hex,
            expected_tree_size=len(self.supporting_capsule_ids),
        )


@dataclass(frozen=True)
class CrowdingRetrievalReport:
    """Retrieval telemetry emitted by `SlotSheafCrowdingConsolidator`."""

    selected_ids: Tuple[str, ...]
    scores_micro: Mapping[str, int]
    vetoed_ids: Tuple[str, ...]
    target_k: int
    overfetch_k_eff: int
    observed_tombstone_ratio: float
    filtered_tombstone_count: int
    stale_trap_at_1: bool


@dataclass(frozen=True)
class SleepConsolidationReport:
    """3-Tier sleep consolidation summary with Merkle audit verification."""

    input_episode_count: int
    input_finding_count: int
    active_head_count: int
    archived_tombstone_count: int
    pruned_sunflower_petal_count: int
    semantic_capsules: Tuple[SemanticCapsule, ...]
    procedural_capsules: Tuple[ProceduralCapsule, ...]
    retained_artifact_count: int
    compression_ratio: float
    all_merkle_proofs_verified: bool
    batch_merkle_root_hex: str


# ---------------------------------------------------------------------------
# 5. SlotSheafCrowdingConsolidator Engine
# ---------------------------------------------------------------------------


def _extract_episode_tokens(ep: BiTemporalEpisode) -> Dict[str, int]:
    parts: List[str] = [ep.slot_key.replace(":", " ").replace(".", " ").replace("/", " ")]

    def _walk(v: Any) -> None:
        if isinstance(v, str):
            parts.append(v)
        elif isinstance(v, Mapping):
            for k, sub_v in v.items():
                parts.append(str(k))
                _walk(sub_v)
        elif isinstance(v, (list, tuple)):
            for item in v:
                _walk(item)

    _walk(ep.payload)
    return tokenize_text(" ".join(parts))


@dataclass(frozen=True)
class _CorpusSearchIndex:
    episodes: Tuple[BiTemporalEpisode, ...]
    ep_ids: Tuple[str, ...]
    ep_by_id: Mapping[str, BiTemporalEpisode]
    postings: Mapping[str, Tuple[Tuple[int, float], ...]]
    doc_lens: Tuple[float, ...]
    avgdl: float
    unit_embeddings: Tuple[Tuple[float, ...], ...]
    active_flags: Tuple[bool, ...]
    active_id_set: frozenset[str]
    inactive_count: int
    max_slot_inactive_m: int
    current_tombstone_ratio: float


class SlotSheafCrowdingConsolidator:
    """Crowding-optimal bi-temporal retriever & 3-tier Merkle sleep consolidator."""

    def __init__(
        self,
        *,
        rrf_k: int = 60,
        confidence_z: float = 3.0,
        gaba_threshold_cos: float = 0.45,
        causal_margin_s: float = 120.0,
        sheaf_tension_penalty_scale: int = 8_000,
    ) -> None:
        if rrf_k < 0:
            raise ValueError("rrf_k must be >= 0")
        if confidence_z < 0.0 or not math.isfinite(confidence_z):
            raise ValueError("confidence_z must be finite and >= 0.0")
        self.rrf_k = int(rrf_k)
        self.confidence_z = float(confidence_z)
        self.gaba_threshold_cos = float(gaba_threshold_cos)
        self.causal_margin_s = float(causal_margin_s)
        self.sheaf_tension_penalty_scale = int(sheaf_tension_penalty_scale)
        self._index_cache_engine_ref: Optional[weakref.ReferenceType[BiTemporalAGMEngine]] = None
        self._index_cache_mutation_seq: Optional[int] = None
        self._index_cache_val: Optional[_CorpusSearchIndex] = None

    def _get_or_build_index(self, engine: BiTemporalAGMEngine) -> _CorpusSearchIndex:
        if (
            self._index_cache_val is not None
            and self._index_cache_engine_ref is not None
            and self._index_cache_engine_ref() is engine
            and self._index_cache_mutation_seq == engine.mutation_seq
        ):
            return self._index_cache_val

        eps = engine.episodes
        n = len(eps)
        postings_mut: Dict[str, List[Tuple[int, float]]] = defaultdict(list)
        doc_lens: List[float] = []
        unit_embs: List[Tuple[float, ...]] = []
        active_flags: List[bool] = []
        active_ids: List[str] = []
        slot_inact: Counter[str] = Counter()

        for idx, ep in enumerate(eps):
            toks = _extract_episode_tokens(ep)
            doc_lens.append(float(sum(toks.values())))
            for tok, cnt in toks.items():
                if cnt > 0:
                    postings_mut[tok].append((idx, float(cnt)))
            unit_embs.append(tuple(vec_normalize(ep.embedding)) if ep.embedding else ())
            is_act = ep.is_active
            active_flags.append(is_act)
            if is_act:
                active_ids.append(ep.episode_id)
            else:
                slot_inact[ep.slot_key] += 1

        inact_cnt = n - len(active_ids)
        avgdl = max(sum(doc_lens) / float(n), 1e-9) if n > 0 else 1.0
        idx_obj = _CorpusSearchIndex(
            episodes=eps,
            ep_ids=tuple(ep.episode_id for ep in eps),
            ep_by_id={ep.episode_id: ep for ep in eps},
            postings={k: tuple(v) for k, v in postings_mut.items()},
            doc_lens=tuple(doc_lens),
            avgdl=avgdl,
            unit_embeddings=tuple(unit_embs),
            active_flags=tuple(active_flags),
            active_id_set=frozenset(active_ids),
            inactive_count=inact_cnt,
            max_slot_inactive_m=max(slot_inact.values(), default=0),
            current_tombstone_ratio=(inact_cnt / float(n)) if n > 0 else 0.0,
        )
        self._index_cache_engine_ref = weakref.ref(engine)
        self._index_cache_mutation_seq = engine.mutation_seq
        self._index_cache_val = idx_obj
        return idx_obj

    @staticmethod
    def _rank_bm25_and_dense_indexed(
        idx_obj: _CorpusSearchIndex,
        query_text: str,
        query_embedding: Sequence[float],
    ) -> Tuple[List[int], List[int]]:
        ep_ids = idx_obj.ep_ids
        n = len(ep_ids)
        if n == 0:
            return [], []

        q_toks = tokenize_text(query_text)
        k1, b = 1.5, 0.75
        avgdl = idx_obj.avgdl
        doc_lens = idx_obj.doc_lens
        postings = idx_obj.postings

        bm25_scores = [0.0] * n
        for qt in q_toks.keys():
            plist = postings.get(qt)
            if not plist:
                continue
            df = len(plist)
            idf = math.log((n - df + 0.5) / (df + 0.5) + 1.0)
            k1_plus_1_idf = idf * (k1 + 1.0)
            k1_base = k1 * (1.0 - b)
            k1_b_inv_avg = (k1 * b) / avgdl
            for doc_i, tf in plist:
                bm25_scores[doc_i] += (k1_plus_1_idf * tf) / (tf + k1_base + k1_b_inv_avg * doc_lens[doc_i])

        q_unit = vec_normalize(query_embedding) if query_embedding else []
        dense_scores = [0.0] * n
        if q_unit:
            q_dim = len(q_unit)
            unit_embs = idx_obj.unit_embeddings
            for i in range(n):
                ue = unit_embs[i]
                if len(ue) == q_dim:
                    dense_scores[i] = sum(map(mul, ue, q_unit))

        lex_order = sorted(range(n), key=lambda i: (-bm25_scores[i], ep_ids[i]))
        dense_order = sorted(range(n), key=lambda i: (-dense_scores[i], ep_ids[i]))
        return dense_order, lex_order

    def naive_unfiltered_retrieve(
        self,
        engine: BiTemporalAGMEngine,
        query_text: str,
        query_embedding: Sequence[float],
        top_k: int = 5,
    ) -> CrowdingRetrievalReport:
        """Baseline naive top-`k` retrieval without tombstone over-fetching or filtering."""
        idx_obj = self._get_or_build_index(engine)
        eps = idx_obj.episodes
        if not eps or top_k <= 0:
            return CrowdingRetrievalReport((), MappingProxyType({}), (), top_k, 0, 0.0, 0, False)

        dense_idx, lex_idx = self._rank_bm25_and_dense_indexed(idx_obj, query_text, query_embedding)
        pool_w = min(len(eps), max(top_k * 6, 32))
        ep_ids = idx_obj.ep_ids
        dense_top = [ep_ids[i] for i in dense_idx[:pool_w]]
        lex_top = [ep_ids[i] for i in lex_idx[:pool_w]]
        fused = fixed_point_rrf_fuse(dense_top, lex_top, rrf_k=self.rrf_k)[:top_k]

        ep_by_id = idx_obj.ep_by_id
        selected = tuple(cid for cid, _ in fused)
        stale_at_1 = bool(selected and not ep_by_id[selected[0]].is_active)
        return CrowdingRetrievalReport(
            selected_ids=selected,
            scores_micro=MappingProxyType(dict(fused)),
            vetoed_ids=(),
            target_k=top_k,
            overfetch_k_eff=top_k,
            observed_tombstone_ratio=idx_obj.current_tombstone_ratio,
            filtered_tombstone_count=0,
            stale_trap_at_1=stale_at_1,
        )

    def retrieve_active(
        self,
        engine: BiTemporalAGMEngine,
        query_text: str,
        query_embedding: Sequence[float],
        *,
        valid_at: Optional[float] = None,
        tx_at: Optional[float] = None,
        top_k: int = 5,
        deadends: Sequence[DeadendRecord] = (),
        sheaf_node_tensions: Optional[Mapping[str, float]] = None,
        confidence_z: Optional[float] = None,
    ) -> CrowdingRetrievalReport:
        """Crowding-optimal bi-temporal active retrieval with Fixed-Point Integer 4-Signal RRF."""
        idx_obj = self._get_or_build_index(engine)
        eps = idx_obj.episodes
        if not eps or top_k <= 0:
            return CrowdingRetrievalReport((), MappingProxyType({}), (), top_k, 0, 0.0, 0, False)

        z_val = self.confidence_z if confidence_z is None else float(confidence_z)
        resolved_tx = tx_at if tx_at is not None else (engine.max_tx_timestamp or 0.0)
        n_total = len(eps)

        # Fast-path for current active horizon (`valid_at is None and tx_at is None`)
        if valid_at is None and tx_at is None:
            visible_flags: Sequence[bool] = idx_obj.active_flags
            inactive_count = idx_obj.inactive_count
            max_slot_m = idx_obj.max_slot_inactive_m
            vis_id_set: Set[str] | frozenset[str] = idx_obj.active_id_set
        else:
            def _is_visible(ep: BiTemporalEpisode) -> bool:
                if valid_at is not None:
                    return ep.is_visible_as_of(valid_at, resolved_tx)
                return ep.contains_tx(resolved_tx)

            vis_list = [_is_visible(ep) for ep in eps]
            visible_flags = vis_list
            inactive_count = n_total - sum(1 for flag in vis_list if flag)
            slot_inactive_counts: Counter[str] = Counter()
            for ep, is_vis in zip(eps, vis_list):
                if not is_vis:
                    slot_inactive_counts[ep.slot_key] += 1
            max_slot_m = max(slot_inactive_counts.values(), default=0)
            vis_id_set = {ep.episode_id for ep, flg in zip(eps, vis_list) if flg}

        rho = min(0.999999, inactive_count / float(n_total)) if inactive_count < n_total else 0.999999

        k_binomial = compute_safe_overfetch_k(
            target_k=top_k,
            tombstone_ratio=rho,
            confidence_z=z_val,
            max_corpus_size=n_total,
        )
        k_eff = min(n_total, max(k_binomial, top_k + max_slot_m))

        dense_idx, lex_idx = self._rank_bm25_and_dense_indexed(idx_obj, query_text, query_embedding)

        # Adaptive HNSW-style efSearch doubling fallback if extreme local clustering occurs
        cur_k = k_eff
        while True:
            dense_vis = [eps[i].episode_id for i in dense_idx[:cur_k] if visible_flags[i]]
            lex_vis = [eps[i].episode_id for i in lex_idx[:cur_k] if visible_flags[i]]
            union_vis = set(dense_vis) | set(lex_vis)
            if len(union_vis) >= top_k or cur_k >= n_total:
                break
            cur_k = min(n_total, max(cur_k * 2, cur_k + 1))

        filtered_tombstones = (cur_k - len(dense_vis)) + (cur_k - len(lex_vis))

        # Signal 3: 2-hop active graph proximity over bi-temporal edges
        seed_ids = list(dict.fromkeys(dense_vis[:5] + lex_vis[:5]))
        hop_map_raw = engine.neighbors_2hop_as_of(seed_ids, valid_at=valid_at, tx_at=resolved_tx)
        ep_by_id = idx_obj.ep_by_id
        hop_map = {nid: hop for nid, hop in hop_map_raw.items() if nid in vis_id_set}

        # Signal 4: Causal GABA hard veto + Sheaf H^1 tension soft inhibition
        vetoed_ids: Set[str] = set()
        soft_penalties_micro: Dict[str, int] = defaultdict(int)
        candidates_all = set(dense_vis) | set(lex_vis) | set(hop_map.keys())

        for cid in candidates_all:
            ep = ep_by_id[cid]
            ep_unit = vec_normalize(ep.embedding) if ep.embedding else []
            for d_rec in deadends:
                if ep_unit and len(d_rec.dense_vec) == len(ep_unit):
                    sim = vec_dot(ep_unit, vec_normalize(d_rec.dense_vec))
                    thresh = max(self.gaba_threshold_cos, d_rec.threshold_cos)
                    if sim >= thresh and ep.tx_from < (d_rec.discovered_ts_s - self.causal_margin_s):
                        vetoed_ids.add(cid)
                        break

            if sheaf_node_tensions and cid not in vetoed_ids:
                tension = float(sheaf_node_tensions.get(cid, sheaf_node_tensions.get(ep.slot_key, 0.0)))
                if tension > 0.0 and math.isfinite(tension):
                    soft_penalties_micro[cid] += int(round(tension * self.sheaf_tension_penalty_scale))

        fused = fixed_point_rrf_fuse(
            dense_ranked_ids=dense_vis,
            lexical_ranked_ids=lex_vis,
            graph_hop_distances=hop_map,
            gaba_penalties_micro=soft_penalties_micro,
            gaba_vetoed_ids=vetoed_ids,
            rrf_k=self.rrf_k,
            drop_vetoed=True,
        )
        top_fused = fused[:top_k]
        selected = tuple(cid for cid, _ in top_fused)
        stale_at_1 = bool(selected and selected[0] not in vis_id_set)

        return CrowdingRetrievalReport(
            selected_ids=selected,
            scores_micro=MappingProxyType(dict(top_fused)),
            vetoed_ids=tuple(sorted(vetoed_ids)),
            target_k=top_k,
            overfetch_k_eff=cur_k,
            observed_tombstone_ratio=round(inactive_count / float(n_total), 6),
            filtered_tombstone_count=filtered_tombstones,
            stale_trap_at_1=stale_at_1,
        )

    def sleep_consolidate(
        self,
        episodes: Sequence[BiTemporalEpisode] | BiTemporalAGMEngine,
        *,
        findings: Sequence[FindingRecord] = (),
        sunflower_k: int = 3,
        min_procedural_support: int = 2,
    ) -> SleepConsolidationReport:
        """Execute 3-Tier Sleep Consolidation (`Episodic -> SemanticCapsule -> ProceduralCapsule`).

        Every consolidated `SemanticCapsule` and `ProceduralCapsule` commits its exact
        input lineage via `RFC 8785` JCS serialization and an `RFC 6962` SHA-256 Merkle
        tree (`0x00` leaf / `0x01` internal node), with self-verified inclusion proofs.
        """
        ep_list = list(episodes.episodes) if isinstance(episodes, BiTemporalAGMEngine) else list(episodes)

        by_slot: Dict[str, List[BiTemporalEpisode]] = defaultdict(list)
        for ep in ep_list:
            by_slot[ep.slot_key].append(ep)

        semantic_capsules: List[SemanticCapsule] = []
        active_head_total = 0
        archived_tombstone_total = 0
        all_proofs_ok = True

        for slot_key in sorted(by_slot.keys()):
            chain = sorted(by_slot[slot_key], key=lambda e: (e.tx_from, e.valid_from, e.episode_id))
            actives = [e for e in chain if e.is_active]
            tombstones = [e for e in chain if not e.is_active]
            active_head_total += len(actives)
            archived_tombstone_total += len(tombstones)

            head = actives[-1] if actives else chain[-1]
            leaf_bytes_list = [jcs_canonicalize(e.to_dict()) for e in chain]
            root_hex = build_rfc6962_merkle_root(leaf_bytes_list)

            proof_map: Dict[str, MerkleInclusionProof] = {}
            for idx, e in enumerate(chain):
                pf = generate_rfc6962_inclusion_proof(idx, leaf_bytes_list)
                if not verify_rfc6962_inclusion_proof(
                    leaf_bytes_list[idx], pf, root_hex, expected_tree_size=len(leaf_bytes_list)
                ):
                    all_proofs_ok = False
                proof_map[e.episode_id] = pf

            # Extract invariant cross-version lexical atoms
            atom_counts: Counter[str] = Counter()
            for e in chain:
                for tok in _extract_episode_tokens(e).keys():
                    if len(tok) >= 3 and not tok.isdigit():
                        atom_counts[tok] += 1
            threshold_hits = max(1, (len(chain) + 1) // 2)
            invariant_atoms = tuple(
                sorted(
                    (tok for tok, cnt in atom_counts.items() if cnt >= threshold_hits),
                    key=lambda t: (-atom_counts[t], t),
                )[:8]
            )

            v_from = min(e.valid_from for e in chain)
            v_until = None if any(e.valid_until is None for e in actives) else max(
                (e.valid_until for e in chain if e.valid_until is not None),
                default=None,
            )
            summary = MappingProxyType(
                {
                    "slot_key": slot_key,
                    "active_head_id": head.episode_id,
                    "chain_length": len(chain),
                    "superseded_count": len(tombstones),
                    "head_payload": dict(head.payload),
                }
            )
            cap_id = "semcap_" + hashlib.sha256(f"{slot_key}:{root_hex}".encode("utf-8")).hexdigest()[:12]
            semantic_capsules.append(
                SemanticCapsule(
                    capsule_id=cap_id,
                    slot_key=slot_key,
                    active_head_id=head.episode_id,
                    consolidated_episode_ids=tuple(e.episode_id for e in chain),
                    superseded_count=len(tombstones),
                    valid_from=v_from,
                    valid_until=v_until,
                    invariant_atoms=invariant_atoms,
                    canonical_summary=summary,
                    embedding=head.embedding,
                    merkle_root_hex=root_hex,
                    inclusion_proofs=MappingProxyType(proof_map),
                )
            )

        # Optional Sunflower Delta-System petal consolidation over redundant agent findings
        pruned_petal_count = 0
        if findings:
            pruner = SunflowerMemoryPruner(k=sunflower_k, min_core_size=2, retain_petals=1)
            sf_rep = pruner.prune(findings)
            pruned_petal_count = len(sf_rep.pruned_ids)
            finding_by_id = {f.bulgu_id: f for f in findings}
            for cluster in sf_rep.clusters:
                members = [finding_by_id[ mid ] for mid in cluster.member_ids]
                leaf_bytes = [
                    jcs_canonicalize(
                        {
                            "bulgu_id": m.bulgu_id,
                            "konu_id": m.konu_id,
                            "alan": m.alan,
                            "baslik": m.baslik,
                            "ozet": m.ozet,
                            "puan": m.puan,
                        }
                    )
                    for m in members
                ]
                c_root = build_rfc6962_merkle_root(leaf_bytes)
                c_proofs: Dict[str, MerkleInclusionProof] = {}
                for idx, m in enumerate(members):
                    pf = generate_rfc6962_inclusion_proof(idx, leaf_bytes)
                    if not verify_rfc6962_inclusion_proof(
                        leaf_bytes[idx], pf, c_root, expected_tree_size=len(leaf_bytes)
                    ):
                        all_proofs_ok = False
                    c_proofs[m.bulgu_id] = pf

                semantic_capsules.append(
                    SemanticCapsule(
                        capsule_id=f"semcap_sf_{cluster.core_id.lower()}",
                        slot_key=f"sunflower:{cluster.konu_scope}:{cluster.core_id}",
                        active_head_id=cluster.canonical_bulgu_id,
                        consolidated_episode_ids=cluster.member_ids,
                        superseded_count=len(cluster.pruned_petal_ids),
                        valid_from=0.0,
                        valid_until=None,
                        invariant_atoms=cluster.core_atoms,
                        canonical_summary=MappingProxyType(
                            {
                                "core_id": cluster.core_id,
                                "canonical_id": cluster.canonical_bulgu_id,
                                "pruned_petals": len(cluster.pruned_petal_ids),
                            }
                        ),
                        embedding=(),
                        merkle_root_hex=c_root,
                        inclusion_proofs=MappingProxyType(c_proofs),
                    )
                )

        # Tier 3: Distill recurring cross-capsule patterns into ProceduralCapsules
        pattern_groups: Dict[Tuple[str, ...], List[SemanticCapsule]] = defaultdict(list)
        for cap in semantic_capsules:
            # Extract action/operator signature from head payload or top invariant atoms
            head_payload = cap.canonical_summary.get("head_payload", {})
            op_tag = None
            if isinstance(head_payload, Mapping):
                op_tag = head_payload.get("operator") or head_payload.get("resolution") or head_payload.get("policy")
            if isinstance(op_tag, str) and op_tag.strip():
                sig = tuple(_WORD_RE.findall(op_tag.lower())[:4])
            else:
                sig = cap.invariant_atoms[:3]
            if sig:
                pattern_groups[sig].append(cap)

        procedural_capsules: List[ProceduralCapsule] = []
        for sig in sorted(pattern_groups.keys()):
            caps = pattern_groups[sig]
            if len(caps) < max(1, min_procedural_support):
                continue
            proc_leaves = [
                jcs_canonicalize(
                    {
                        "capsule_id": c.capsule_id,
                        "slot_key": c.slot_key,
                        "merkle_root_hex": c.merkle_root_hex,
                        "active_head_id": c.active_head_id,
                    }
                )
                for c in caps
            ]
            p_root = build_rfc6962_merkle_root(proc_leaves)
            p_proofs: Dict[str, MerkleInclusionProof] = {}
            for idx, c in enumerate(caps):
                pf = generate_rfc6962_inclusion_proof(idx, proc_leaves)
                if not verify_rfc6962_inclusion_proof(
                    proc_leaves[idx], pf, p_root, expected_tree_size=len(proc_leaves)
                ):
                    all_proofs_ok = False
                p_proofs[c.capsule_id] = pf

            src_eps = tuple(eid for c in caps for eid in c.consolidated_episode_ids)
            proc_id = "proccap_" + hashlib.sha256(f"{':'.join(sig)}:{p_root}".encode("utf-8")).hexdigest()[:12]
            procedural_capsules.append(
                ProceduralCapsule(
                    procedure_id=proc_id,
                    pattern_signature=sig,
                    recommended_operator="_".join(sig),
                    supporting_capsule_ids=tuple(c.capsule_id for c in caps),
                    source_episode_ids=src_eps,
                    support_count=len(caps),
                    merkle_root_hex=p_root,
                    inclusion_proofs=MappingProxyType(p_proofs),
                )
            )

        total_inputs = len(ep_list) + len(findings)
        surviving_findings = len(findings) - pruned_petal_count
        retained_count = len(semantic_capsules) + max(0, surviving_findings - sum(
            1 for c in semantic_capsules if c.slot_key.startswith("sunflower:")
        ))
        # Exact compression ratio: fraction of raw episodic/petal items folded into capsules
        folded_items = archived_tombstone_total + pruned_petal_count
        compression_ratio = (folded_items / float(total_inputs)) if total_inputs > 0 else 0.0

        batch_leaves = [
            jcs_canonicalize({"capsule_id": c.capsule_id, "merkle_root_hex": c.merkle_root_hex})
            for c in semantic_capsules
        ] + [
            jcs_canonicalize({"procedure_id": p.procedure_id, "merkle_root_hex": p.merkle_root_hex})
            for p in procedural_capsules
        ]
        batch_root = build_rfc6962_merkle_root(batch_leaves)

        return SleepConsolidationReport(
            input_episode_count=len(ep_list),
            input_finding_count=len(findings),
            active_head_count=active_head_total,
            archived_tombstone_count=archived_tombstone_total,
            pruned_sunflower_petal_count=pruned_petal_count,
            semantic_capsules=tuple(semantic_capsules),
            procedural_capsules=tuple(procedural_capsules),
            retained_artifact_count=retained_count,
            compression_ratio=round(compression_ratio, 6),
            all_merkle_proofs_verified=all_proofs_ok,
            batch_merkle_root_hex=batch_root,
        )
