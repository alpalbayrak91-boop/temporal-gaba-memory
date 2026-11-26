"""Cellular Sheaf Coboundary Auditor (v0.3.0) for temporal-gaba-memory.

Zero-dependency pure Python 3.10+ stdlib implementation of cellular sheaf
cohomology diagnostics over temporal multi-agent memory graphs.

Detects global cyclic inconsistencies (H^1 obstructions and non-trivial
GL(d)/O(d) cycle holonomy) across 3+ agent or source cycles that remain
invisible to local pairwise consistency checks and scalar +/-1 signed
frustration auditors.

Mathematical foundations:
- Hansen & Ghrist (2019), Toward a Spectral Theory of Cellular Sheaves,
  J. Appl. Comput. Topol. (arXiv:1808.01513).
- Hansen & Ghrist (2020), Opinion Dynamics on Discourse Sheaves
  (arXiv:2005.12798).
- Guadagnuolo et al. (2026), A Bochner-Weitzenbock Decomposition for
  Cellular Sheaf Laplacians, Curvature and Neural Sheaf Diffusion
  (arXiv:2610.28455).
- Volk (2026), Incremental Sheaf Cohomology on Cellular Complexes
  (arXiv:2606.04227).

License: MIT
"""
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
import hashlib
import json
import math
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

__all__ = [
    "SHEAF_MODULE_VERSION",
    "SheafNode",
    "SheafEdge",
    "PairwiseEdgeAudit",
    "CycleObstruction",
    "ComponentSheafAudit",
    "SheafAuditReport",
    "CellularSheafCoboundaryAuditor",
    "run_sheaf_reference_benchmark",
]

SHEAF_MODULE_VERSION = "0.3.0"

Matrix = Tuple[Tuple[float, ...], ...]
Vector = Tuple[float, ...]


def _to_matrix(mat: Sequence[Sequence[float]], name: str) -> Matrix:
    if not mat:
        raise ValueError(f"{name} must have at least one row")
    rows: List[Tuple[float, ...]] = []
    width: Optional[int] = None
    for r_idx, row in enumerate(mat):
        vals = tuple(float(v) for v in row)
        if not vals:
            raise ValueError(f"{name} row {r_idx} must be non-empty")
        if width is None:
            width = len(vals)
        elif len(vals) != width:
            raise ValueError(f"{name} must be rectangular (row {r_idx} width {len(vals)} != {width})")
        for v in vals:
            if not math.isfinite(v):
                raise ValueError(f"{name} contains non-finite value: {v!r}")
        rows.append(vals)
    return tuple(rows)


def _to_vector(vec: Sequence[float], name: str) -> Vector:
    vals = tuple(float(v) for v in vec)
    if not vals:
        raise ValueError(f"{name} must be non-empty")
    for v in vals:
        if not math.isfinite(v):
            raise ValueError(f"{name} contains non-finite value: {v!r}")
    return vals


def _identity_matrix(dim: int, scale: float = 1.0) -> Matrix:
    if dim < 1:
        raise ValueError("dim must be >= 1")
    return tuple(
        tuple(scale if i == j else 0.0 for j in range(dim))
        for i in range(dim)
    )


def _mat_mul(a: Sequence[Sequence[float]], b: Sequence[Sequence[float]]) -> List[List[float]]:
    m = len(a)
    k = len(a[0])
    n = len(b[0])
    out = [[0.0] * n for _ in range(m)]
    for i in range(m):
        ai = a[i]
        oi = out[i]
        for t in range(k):
            ait = ai[t]
            if ait == 0.0:
                continue
            bt = b[t]
            for j in range(n):
                oi[j] += ait * bt[j]
    return out


def _mat_vec(a: Sequence[Sequence[float]], x: Sequence[float]) -> List[float]:
    return [sum(a[i][j] * x[j] for j in range(len(x))) for i in range(len(a))]


def _vec_norm2(x: Sequence[float]) -> float:
    return math.sqrt(sum(v * v for v in x))


def _frobenius_diff_identity(mat: Sequence[Sequence[float]]) -> float:
    n = len(mat)
    acc = 0.0
    for i in range(n):
        for j in range(n):
            diff = mat[i][j] - (1.0 if i == j else 0.0)
            acc += diff * diff
    return math.sqrt(acc)


def _invert_square_matrix(mat: Sequence[Sequence[float]], tol: float = 1e-12) -> Optional[List[List[float]]]:
    """Gauss-Jordan matrix inversion with partial pivoting. Returns None if singular."""
    n = len(mat)
    if n == 0 or any(len(r) != n for r in mat):
        return None
    aug = [list(map(float, mat[i])) + [1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    for col in range(n):
        piv_row = max(range(col, n), key=lambda r: abs(aug[r][col]))
        piv_val = aug[piv_row][col]
        if abs(piv_val) <= tol:
            return None
        if piv_row != col:
            aug[col], aug[piv_row] = aug[piv_row], aug[col]
        inv_p = 1.0 / aug[col][col]
        row_c = aug[col]
        for j in range(col, 2 * n):
            row_c[j] *= inv_p
        for r in range(n):
            if r == col:
                continue
            factor = aug[r][col]
            if factor == 0.0:
                continue
            row_r = aug[r]
            for j in range(col, 2 * n):
                row_r[j] -= factor * row_c[j]
    return [row[n:] for row in aug]


def _householder_qr_pivoted(
    mat: Sequence[Sequence[float]],
    tol: float = 1e-9,
) -> Tuple[List[List[float]], List[List[float]], List[int], int]:
    """Column-pivoted Householder QR factorization A[:, piv] = Q_r @ R_r.

    Returns:
        q_r: m x r orthonormal column basis of im(A)
        r_r: r x n upper-trapezoidal factor in pivoted column order
        piv: length-n column permutation indices
        rank: numerical rank r
    """
    m = len(mat)
    n = len(mat[0]) if m > 0 else 0
    if m == 0 or n == 0:
        return [], [], list(range(n)), 0

    a = [list(map(float, row)) for row in mat]
    piv = list(range(n))
    col_norms_sq = [sum(a[i][j] * a[i][j] for i in range(m)) for j in range(n)]
    reflectors: List[Tuple[int, List[float], float]] = []
    max_steps = min(m, n)
    max_diag = 0.0
    rank = 0

    for k in range(max_steps):
        # Recompute residual column norms from row k downward for high accuracy
        best_col = k
        best_sq = -1.0
        for j in range(k, n):
            s = sum(a[i][j] * a[i][j] for i in range(k, m))
            col_norms_sq[j] = s
            if s > best_sq:
                best_sq = s
                best_col = j
        if best_sq <= 0.0:
            break
        sigma = math.sqrt(best_sq)
        if k == 0:
            max_diag = sigma
        thresh = max(tol, tol * max_diag * math.sqrt(float(max(m, n))))
        if sigma <= thresh:
            break
        if best_col != k:
            piv[k], piv[best_col] = piv[best_col], piv[k]
            col_norms_sq[k], col_norms_sq[best_col] = col_norms_sq[best_col], col_norms_sq[k]
            for i in range(m):
                a[i][k], a[i][best_col] = a[i][best_col], a[i][k]

        x = [a[i][k] for i in range(k, m)]
        norm_x = math.sqrt(sum(v * v for v in x))
        if norm_x <= thresh:
            break
        sign = 1.0 if x[0] >= 0.0 else -1.0
        v = list(x)
        v[0] += sign * norm_x
        v_norm_sq = sum(val * val for val in v)
        if v_norm_sq <= 1e-30:
            break
        beta = 2.0 / v_norm_sq
        reflectors.append((k, v, beta))
        # Apply H_k = I - beta v v^T to columns k..n-1
        for j in range(k, n):
            dot = sum(v[i - k] * a[i][j] for i in range(k, m))
            scale = beta * dot
            for i in range(k, m):
                a[i][j] -= scale * v[i - k]
        a[k][k] = -sign * norm_x
        for i in range(k + 1, m):
            a[i][k] = 0.0
        rank += 1

    # Materialize thin Q_r (m x rank) by applying reflectors in reverse to I[:, :rank]
    q_r = [[1.0 if i == j else 0.0 for j in range(rank)] for i in range(m)]
    for k, v, beta in reversed(reflectors):
        for j in range(k, rank):
            dot = sum(v[i - k] * q_r[i][j] for i in range(k, m))
            scale = beta * dot
            for i in range(k, m):
                q_r[i][j] -= scale * v[i - k]

    r_r = [a[i][:] for i in range(rank)]
    return q_r, r_r, piv, rank


def _solve_min_norm_lstsq(
    mat: Sequence[Sequence[float]],
    rhs: Sequence[float],
    tol: float = 1e-9,
) -> Tuple[List[float], List[float], List[float], int, List[List[float]]]:
    """Minimum-norm least squares x* = A^+ rhs and exact Hodge decomposition.

    Returns:
        x_star: minimum-norm least squares solution in R^n
        exact_proj: proj_{im(A)}(rhs) in R^m
        harmonic_res: proj_{ker(A^T)}(rhs) = rhs - exact_proj in R^m
        rank: numerical rank of A
        q_r: orthonormal basis of im(A) (m x rank)
    """
    m = len(mat)
    n = len(mat[0]) if m > 0 else 0
    if m == 0 or n == 0:
        return [0.0] * n, [0.0] * m, list(map(float, rhs)), 0, []

    q_r, r_r, piv, rank = _householder_qr_pivoted(mat, tol=tol)
    if rank == 0:
        return [0.0] * n, [0.0] * m, list(map(float, rhs)), 0, q_r

    # c = Q_r^T rhs (length rank)
    c = [sum(q_r[i][j] * rhs[i] for i in range(m)) for j in range(rank)]
    exact_proj = [sum(q_r[i][j] * c[j] for j in range(rank)) for i in range(m)]
    harmonic_res = [rhs[i] - exact_proj[i] for i in range(m)]

    # Factor R_r^T (n x rank) via unpivoted/pivoted QR to obtain exact min-norm x*
    r_t = [[r_r[j][i] for j in range(rank)] for i in range(n)]
    q2, r2, piv2, rank2 = _householder_qr_pivoted(r_t, tol=tol)
    # R_r^T[:, piv2] = Q2 @ R2  =>  (R_r)[piv2, :] = R2^T @ Q2^T
    # Permute c according to piv2
    c_perm = [c[piv2[i]] for i in range(rank2)]
    # Forward solve lower-triangular R2^T z = c_perm
    z = [0.0] * rank2
    for i in range(rank2):
        s = c_perm[i] - sum(r2[j][i] * z[j] for j in range(i))
        diag = r2[i][i]
        z[i] = (s / diag) if abs(diag) > 1e-15 else 0.0
    # x_piv = Q2 @ z
    x_piv = [sum(q2[i][j] * z[j] for j in range(rank2)) for i in range(n)]
    x_star = [0.0] * n
    for j in range(n):
        x_star[piv[j]] = x_piv[j]
    return x_star, exact_proj, harmonic_res, rank, q_r


def _jacobi_symmetric_eigenvalues(
    mat: Sequence[Sequence[float]],
    max_sweeps: int = 24,
) -> List[float]:
    """Cyclic Jacobi eigenvalues for small/medium symmetric PSD matrices (ascending)."""
    n = len(mat)
    if n == 0:
        return []
    a = [list(map(float, row)) for row in mat]
    for _ in range(max_sweeps):
        max_off = 0.0
        for p in range(n):
            for q in range(p + 1, n):
                apq = a[p][q]
                if abs(apq) > max_off:
                    max_off = abs(apq)
                if abs(apq) <= 1e-15:
                    continue
                diff = a[p][p] - a[q][q]
                theta = (math.pi / 4.0) if abs(diff) <= 1e-15 else 0.5 * math.atan2(2.0 * apq, diff)
                c = math.cos(theta)
                s = math.sin(theta)
                for r in range(n):
                    if r != p and r != q:
                        arp, arq = a[r][p], a[r][q]
                        a[r][p] = a[p][r] = c * arp + s * arq
                        a[r][q] = a[q][r] = -s * arp + c * arq
                app, aqq = a[p][p], a[q][q]
                a[p][p] = c * c * app + 2.0 * s * c * apq + s * s * aqq
                a[q][q] = s * s * app - 2.0 * s * c * apq + c * c * aqq
                a[p][q] = a[q][p] = 0.0
        if max_off <= 1e-13:
            break
    eigs = sorted(a[i][i] for i in range(n))
    return [0.0 if abs(v) <= 1e-12 else v for v in eigs]


@dataclass(frozen=True)
class SheafNode:
    """An agent, knowledge source, or memory vertex carrying a real vector stalk R^d."""

    node_id: str
    stalk_dim: int = 1
    state: Optional[Vector] = None
    valid_from_s: Optional[float] = None
    valid_until_s: Optional[float] = None
    metadata: Tuple[Tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if not self.node_id or not self.node_id.strip():
            raise ValueError("node_id must be a non-empty string")
        if self.state is not None:
            vec = _to_vector(self.state, f"SheafNode({self.node_id}).state")
            object.__setattr__(self, "state", vec)
            if self.stalk_dim != len(vec):
                object.__setattr__(self, "stalk_dim", len(vec))
        if self.stalk_dim < 1:
            raise ValueError(f"SheafNode({self.node_id}).stalk_dim must be >= 1")
        if (
            self.valid_from_s is not None
            and self.valid_until_s is not None
            and self.valid_until_s <= self.valid_from_s
        ):
            raise ValueError(
                f"SheafNode({self.node_id}) valid_until_s must exceed valid_from_s"
            )

    def is_active_at(self, as_of_s: Optional[float]) -> bool:
        if as_of_s is None:
            return True
        if self.valid_from_s is not None and as_of_s < self.valid_from_s:
            return False
        if self.valid_until_s is not None and as_of_s >= self.valid_until_s:
            return False
        return True


@dataclass(frozen=True)
class SheafEdge:
    """An oriented pairwise consistency constraint R_target x_target - R_source x_source = offset."""

    edge_id: str
    source: str
    target: str
    restriction_source: Matrix
    restriction_target: Matrix
    offset: Optional[Vector] = None
    weight: float = 1.0
    valid_from_s: Optional[float] = None
    valid_until_s: Optional[float] = None
    label: str = ""

    def __post_init__(self) -> None:
        if not self.edge_id or not self.edge_id.strip():
            raise ValueError("edge_id must be a non-empty string")
        if not self.source or not self.target:
            raise ValueError("source and target must be non-empty strings")
        rs = _to_matrix(self.restriction_source, f"SheafEdge({self.edge_id}).restriction_source")
        rt = _to_matrix(self.restriction_target, f"SheafEdge({self.edge_id}).restriction_target")
        if len(rs) != len(rt):
            raise ValueError(
                f"SheafEdge({self.edge_id}) edge stalk mismatch: "
                f"{len(rs)} rows in restriction_source vs {len(rt)} in restriction_target"
            )
        object.__setattr__(self, "restriction_source", rs)
        object.__setattr__(self, "restriction_target", rt)
        d_e = len(rs)
        if self.offset is None:
            object.__setattr__(self, "offset", tuple(0.0 for _ in range(d_e)))
        else:
            off = _to_vector(self.offset, f"SheafEdge({self.edge_id}).offset")
            if len(off) != d_e:
                raise ValueError(
                    f"SheafEdge({self.edge_id}) offset length {len(off)} != edge stalk dim {d_e}"
                )
            object.__setattr__(self, "offset", off)
        w = float(self.weight)
        if not math.isfinite(w) or w <= 0.0:
            raise ValueError(f"SheafEdge({self.edge_id}) weight must be finite and > 0")
        object.__setattr__(self, "weight", w)
        if (
            self.valid_from_s is not None
            and self.valid_until_s is not None
            and self.valid_until_s <= self.valid_from_s
        ):
            raise ValueError(
                f"SheafEdge({self.edge_id}) valid_until_s must exceed valid_from_s"
            )

    @property
    def edge_stalk_dim(self) -> int:
        return len(self.restriction_source)

    @property
    def source_dim(self) -> int:
        return len(self.restriction_source[0])

    @property
    def target_dim(self) -> int:
        return len(self.restriction_target[0])

    def is_active_at(self, as_of_s: Optional[float]) -> bool:
        if as_of_s is None:
            return True
        if self.valid_from_s is not None and as_of_s < self.valid_from_s:
            return False
        if self.valid_until_s is not None and as_of_s >= self.valid_until_s:
            return False
        return True

    @classmethod
    def identity(
        cls,
        edge_id: str,
        source: str,
        target: str,
        dim: int = 1,
        offset: Optional[Sequence[float]] = None,
        weight: float = 1.0,
        valid_from_s: Optional[float] = None,
        valid_until_s: Optional[float] = None,
        label: str = "",
    ) -> "SheafEdge":
        """Constraint: x_target - x_source = offset."""
        eye = _identity_matrix(dim, 1.0)
        return cls(
            edge_id=edge_id,
            source=source,
            target=target,
            restriction_source=eye,
            restriction_target=eye,
            offset=tuple(offset) if offset is not None else None,
            weight=weight,
            valid_from_s=valid_from_s,
            valid_until_s=valid_until_s,
            label=label,
        )

    @classmethod
    def signed(
        cls,
        edge_id: str,
        source: str,
        target: str,
        sign: int,
        dim: int = 1,
        weight: float = 1.0,
        valid_from_s: Optional[float] = None,
        valid_until_s: Optional[float] = None,
        label: str = "",
    ) -> "SheafEdge":
        """Signed-graph embedding: x_target = sign * x_source (sign in {-1, +1})."""
        if sign not in (-1, 1):
            raise ValueError("sign must be -1 or +1")
        return cls(
            edge_id=edge_id,
            source=source,
            target=target,
            restriction_source=_identity_matrix(dim, float(sign)),
            restriction_target=_identity_matrix(dim, 1.0),
            offset=None,
            weight=weight,
            valid_from_s=valid_from_s,
            valid_until_s=valid_until_s,
            label=label,
        )

    @classmethod
    def linear_transport(
        cls,
        edge_id: str,
        source: str,
        target: str,
        transport_matrix: Sequence[Sequence[float]],
        offset: Optional[Sequence[float]] = None,
        weight: float = 1.0,
        valid_from_s: Optional[float] = None,
        valid_until_s: Optional[float] = None,
        label: str = "",
    ) -> "SheafEdge":
        """Affine transport constraint: x_target = T @ x_source + offset."""
        t_mat = _to_matrix(transport_matrix, "transport_matrix")
        d_out = len(t_mat)
        return cls(
            edge_id=edge_id,
            source=source,
            target=target,
            restriction_source=t_mat,
            restriction_target=_identity_matrix(d_out, 1.0),
            offset=tuple(offset) if offset is not None else None,
            weight=weight,
            valid_from_s=valid_from_s,
            valid_until_s=valid_until_s,
            label=label,
        )

    @classmethod
    def projection_agreement(
        cls,
        edge_id: str,
        source: str,
        target: str,
        proj_source: Sequence[Sequence[float]],
        proj_target: Sequence[Sequence[float]],
        offset: Optional[Sequence[float]] = None,
        weight: float = 1.0,
        valid_from_s: Optional[float] = None,
        valid_until_s: Optional[float] = None,
        label: str = "",
    ) -> "SheafEdge":
        """Heterogeneous subspace constraint: P_target x_target - P_source x_source = offset."""
        return cls(
            edge_id=edge_id,
            source=source,
            target=target,
            restriction_source=_to_matrix(proj_source, "proj_source"),
            restriction_target=_to_matrix(proj_target, "proj_target"),
            offset=tuple(offset) if offset is not None else None,
            weight=weight,
            valid_from_s=valid_from_s,
            valid_until_s=valid_until_s,
            label=label,
        )


@dataclass(frozen=True)
class PairwiseEdgeAudit:
    """Local satisfiability and observed residual for a single edge."""

    edge_id: str
    source: str
    target: str
    locally_consistent: bool
    intrinsic_residual_norm: float
    observed_residual_norm: Optional[float]
    harmonic_stress_norm: float


@dataclass(frozen=True)
class CycleObstruction:
    """Detected cyclic obstruction in the cellular sheaf (focus on length >= 3)."""

    cycle_nodes: Tuple[str, ...]
    cycle_edges: Tuple[str, ...]
    cycle_length: int
    is_global_3plus_cycle: bool
    locally_pairwise_consistent: bool
    obstruction_kind: str  # "affine_h1" | "holonomy_frustration" | "mixed_h1_holonomy"
    h1_obstruction_norm: float
    holonomy_defect: Optional[float]
    affine_transport_defect: Optional[float]
    dim_h0_cycle: int
    dim_h1_cycle: int
    min_cycle_laplacian_eig: Optional[float]


@dataclass(frozen=True)
class ComponentSheafAudit:
    """Exact cohomology, Hodge decomposition, and spectral metrics for one connected component."""

    component_id: int
    node_ids: Tuple[str, ...]
    edge_ids: Tuple[str, ...]
    total_vertex_dof: int
    total_edge_dof: int
    coboundary_rank: int
    dim_h0: int
    dim_h1: int
    euler_invariant_ok: bool
    h1_obstruction_norm: float
    h1_obstruction_energy: float
    observed_total_residual_norm: Optional[float]
    observed_exact_reconcilable_norm: Optional[float]
    observed_harmonic_norm: Optional[float]
    hodge_pythagorean_error: Optional[float]
    min_laplacian_eig: Optional[float]
    second_laplacian_eig: Optional[float]
    reconciled_states: Dict[str, Vector]


@dataclass(frozen=True)
class SheafAuditReport:
    """Full cellular sheaf coboundary audit report for temporal-gaba-memory v0.3.0."""

    module_version: str
    as_of_s: Optional[float]
    active_node_count: int
    active_edge_count: int
    filtered_inactive_nodes: int
    filtered_inactive_edges: int
    all_edges_pairwise_consistent: bool
    is_globally_consistent: bool
    has_local_to_global_paradox: bool
    total_dim_h0: int
    total_dim_h1: int
    global_h1_obstruction_norm: float
    global_h1_obstruction_energy: float
    max_holonomy_defect: float
    pairwise_audits: Tuple[PairwiseEdgeAudit, ...]
    cycle_obstructions: Tuple[CycleObstruction, ...]
    global_3plus_obstructions: Tuple[CycleObstruction, ...]
    components: Tuple[ComponentSheafAudit, ...]
    recommended_cut_edges: Tuple[str, ...]
    cut_restores_global_consistency: bool
    node_tension_scores: Dict[str, float]
    edge_harmonic_stress: Dict[str, float]
    audit_digest_sha256: str

    def to_summary_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable summary suitable for telemetry or CI gates."""
        return {
            "module_version": self.module_version,
            "as_of_s": self.as_of_s,
            "active_node_count": self.active_node_count,
            "active_edge_count": self.active_edge_count,
            "all_edges_pairwise_consistent": self.all_edges_pairwise_consistent,
            "is_globally_consistent": self.is_globally_consistent,
            "has_local_to_global_paradox": self.has_local_to_global_paradox,
            "total_dim_h0": self.total_dim_h0,
            "total_dim_h1": self.total_dim_h1,
            "global_h1_obstruction_norm": round(self.global_h1_obstruction_norm, 9),
            "global_h1_obstruction_energy": round(self.global_h1_obstruction_energy, 9),
            "max_holonomy_defect": round(self.max_holonomy_defect, 9),
            "global_3plus_obstruction_count": len(self.global_3plus_obstructions),
            "recommended_cut_edges": list(self.recommended_cut_edges),
            "cut_restores_global_consistency": self.cut_restores_global_consistency,
            "audit_digest_sha256": self.audit_digest_sha256,
        }


class CellularSheafCoboundaryAuditor:
    """Detects global cyclic contradictions as cellular sheaf H^1 obstructions.

    Given a multi-agent temporal belief graph (V, E) equipped with linear or
    affine restriction maps R_v x_v - R_u x_u = b_e, this auditor:
    1. Filters nodes and edges causally at an optional snapshot `as_of_s`.
    2. Verifies local pairwise satisfiability on every individual edge.
    3. Assembles the weighted cellular coboundary operator delta_w per connected
       component and computes exact H^0 / H^1 dimensions, Hodge decomposition,
       minimum-norm least-squares reconciliation, and sheaf Laplacian spectrum.
    4. Extracts a complete BFS fundamental cycle basis (necessary and sufficient
       for both affine H^1 and non-Abelian GL(d) holonomy obstructions) to isolate
       3+ agent cycles where every pair agrees locally yet global gluing fails.
    5. Computes a minimal-weight obstruction cut that restores global consistency
       and outputs per-node/per-edge tension scores for inhibitory memory gating.
    """

    def __init__(
        self,
        *,  
        tol: float = 1e-8,
        holonomy_tol: float = 1e-7,
        max_spectral_dof: int = 96,
        compute_minimal_cut: bool = True,
        max_cut_steps: int = 32,
    ) -> None:
        if tol <= 0.0 or holonomy_tol <= 0.0:
            raise ValueError("tol and holonomy_tol must be > 0")
        self.tol = float(tol)
        self.holonomy_tol = float(holonomy_tol)
        self.max_spectral_dof = int(max_spectral_dof)
        self.compute_minimal_cut = bool(compute_minimal_cut)
        self.max_cut_steps = int(max_cut_steps)

    def _check_pairwise_edge(
        self,
        edge: SheafEdge,
        src_node: SheafNode,
        dst_node: SheafNode,
    ) -> Tuple[bool, float, Optional[float]]:
        d_e = edge.edge_stalk_dim
        b_e = edge.offset or tuple(0.0 for _ in range(d_e))
        if edge.source == edge.target:
            # Self-loop: (R_t - R_s) x_u = b_e
            d_u = src_node.stalk_dim
            a_loc = [
                [edge.restriction_target[r][c] - edge.restriction_source[r][c] for c in range(d_u)]
                for r in range(d_e)
            ]
        else:
            # Pairwise: [-R_s | R_t] [x_u; x_v] = b_e
            a_loc = [
                [-v for v in edge.restriction_source[r]] + list(edge.restriction_target[r])
                for r in range(d_e)
            ]
        _, _, h_loc, _, _ = _solve_min_norm_lstsq(a_loc, b_e, tol=self.tol)
        intr_norm = _vec_norm2(h_loc)
        locally_ok = intr_norm <= self.tol * max(1.0, _vec_norm2(b_e))

        obs_norm: Optional[float] = None
        if src_node.state is not None and dst_node.state is not None:
            rs_x = _mat_vec(edge.restriction_source, src_node.state)
            rt_x = _mat_vec(edge.restriction_target, dst_node.state)
            diff = [rt_x[r] - rs_x[r] - b_e[r] for r in range(d_e)]
            obs_norm = _vec_norm2(diff)
        return locally_ok, intr_norm, obs_norm

    def _assemble_coboundary(
        self,
        comp_nodes: Sequence[SheafNode],
        comp_edges: Sequence[SheafEdge],
    ) -> Tuple[List[List[float]], List[float], Dict[str, Tuple[int, int]], Dict[str, Tuple[int, int]]]:
        node_slices: Dict[str, Tuple[int, int]] = {}
        col_cursor = 0
        for nd in comp_nodes:
            node_slices[nd.node_id] = (col_cursor, col_cursor + nd.stalk_dim)
            col_cursor += nd.stalk_dim
        total_cols = col_cursor

        edge_slices: Dict[str, Tuple[int, int]] = {}
        rows: List[List[float]] = []
        rhs: List[float] = []
        row_cursor = 0
        for e in comp_edges:
            d_e = e.edge_stalk_dim
            sw = math.sqrt(e.weight)
            s_lo, s_hi = node_slices[e.source]
            t_lo, t_hi = node_slices[e.target]
            b_e = e.offset or tuple(0.0 for _ in range(d_e))
            for r in range(d_e):
                row = [0.0] * total_cols
                for j, val in enumerate(e.restriction_source[r]):
                    row[s_lo + j] -= sw * val
                for j, val in enumerate(e.restriction_target[r]):
                    row[t_lo + j] += sw * val
                rows.append(row)
                rhs.append(sw * b_e[r])
            edge_slices[e.edge_id] = (row_cursor, row_cursor + d_e)
            row_cursor += d_e
        return rows, rhs, node_slices, edge_slices

    def _compute_laplacian_eigs(self, delta_w: Sequence[Sequence[float]], total_cols: int) -> Tuple[Optional[float], Optional[float]]:
        if total_cols == 0 or total_cols > self.max_spectral_dof:
            return None, None
        m = len(delta_w)
        lap = [[0.0] * total_cols for _ in range(total_cols)]
        for r in range(m):
            row = delta_w[r]
            nz = [(j, v) for j, v in enumerate(row) if v != 0.0]
            for i_idx, (i, vi) in enumerate(nz):
                lap[i][i] += vi * vi
                for j, vj in nz[i_idx + 1:]:
                    prod = vi * vj
                    lap[i][j] += prod
                    lap[j][i] += prod
        eigs = _jacobi_symmetric_eigenvalues(lap)
        if not eigs:
            return None, None
        eig0 = eigs[0]
        eig1 = eigs[1] if len(eigs) >= 2 else None
        return eig0, eig1

    def _compute_cycle_holonomy(
        self,
        cycle_nodes: Sequence[str],
        cycle_edges: Sequence[SheafEdge],
        node_map: Mapping[str, SheafNode],
    ) -> Tuple[Optional[float], Optional[float]]:
        """Compose exact affine transport x_{k+1} = T_k x_k + s_k around an oriented cycle."""
        dims = {node_map[nid].stalk_dim for nid in cycle_nodes}
        if len(dims) != 1:
            return None, None
        d = next(iter(dims))
        for e in cycle_edges:
            if e.edge_stalk_dim != d or e.source_dim != d or e.target_dim != d:
                return None, None

        m_cur: List[List[float]] = [[1.0 if i == j else 0.0 for j in range(d)] for i in range(d)]
        a_cur: List[float] = [0.0] * d

        for idx, edge in enumerate(cycle_edges):
            u = cycle_nodes[idx]
            v = cycle_nodes[(idx + 1) % len(cycle_nodes)]
            b_e = edge.offset or tuple(0.0 for _ in range(d))
            if edge.source == u and edge.target == v:
                # Forward along edge: R_t x_v = R_s x_u + b_e  =>  x_v = R_t^-1 R_s x_u + R_t^-1 b_e
                inv_rt = _invert_square_matrix(edge.restriction_target, tol=self.tol)
                if inv_rt is None:
                    return None, None
                t_step = _mat_mul(inv_rt, edge.restriction_source)
                s_step = _mat_vec(inv_rt, b_e)
            elif edge.source == v and edge.target == u:
                # Reverse along edge: R_s x_v = R_t x_u - b_e  =>  x_v = R_s^-1 R_t x_u - R_s^-1 b_e
                inv_rs = _invert_square_matrix(edge.restriction_source, tol=self.tol)
                if inv_rs is None:
                    return None, None
                t_step = _mat_mul(inv_rs, edge.restriction_target)
                s_step = [-val for val in _mat_vec(inv_rs, b_e)]
            else:
                return None, None
            m_cur = _mat_mul(t_step, m_cur)
            a_step = _mat_vec(t_step, a_cur)
            a_cur = [a_step[i] + s_step[i] for i in range(d)]

        hol_defect = _frobenius_diff_identity(m_cur)
        m_inv = _invert_square_matrix(m_cur, tol=self.tol)
        if m_inv is not None:
            hol_defect = max(hol_defect, _frobenius_diff_identity(m_inv))
        aff_defect = _vec_norm2(a_cur)
        return hol_defect, aff_defect

    def _extract_fundamental_cycles(
        self,
        comp_nodes: Sequence[SheafNode],
        comp_edges: Sequence[SheafEdge],
    ) -> List[Tuple[Tuple[str, ...], Tuple[SheafEdge, ...]]]:
        """Deterministic BFS spanning-tree fundamental cycle basis for one component."""
        node_ids = sorted(nd.node_id for nd in comp_nodes)
        adj: Dict[str, List[Tuple[str, SheafEdge]]] = {nid: [] for nid in node_ids}
        self_loops: List[SheafEdge] = []
        for e in sorted(comp_edges, key=lambda x: x.edge_id):
            if e.source == e.target:
                self_loops.append(e)
                continue
            adj[e.source].append((e.target, e))
            adj[e.target].append((e.source, e))
        for nid in adj:
            adj[nid].sort(key=lambda pair: (pair[0], pair[1].edge_id))

        cycles: List[Tuple[Tuple[str, ...], Tuple[SheafEdge, ...]]] = []
        for sl in self_loops:
            cycles.append(((sl.source,), (sl,)))

        root = node_ids[0]
        parent_node: Dict[str, Optional[str]] = {root: None}
        parent_edge: Dict[str, Optional[SheafEdge]] = {root: None}
        depth: Dict[str, int] = {root: 0}
        tree_edge_ids: Set[str] = set()
        q: deque[str] = deque([root])

        while q:
            u = q.popleft()
            for v, e in adj[u]:
                if v not in parent_node:
                    parent_node[v] = u
                    parent_edge[v] = e
                    depth[v] = depth[u] + 1
                    tree_edge_ids.add(e.edge_id)
                    q.append(v)

        for e in sorted(comp_edges, key=lambda x: x.edge_id):
            if e.source == e.target or e.edge_id in tree_edge_ids:
                continue
            u, v = e.source, e.target
            # Reconstruct unique spanning-tree path from u to v via LCA
            path_u_nodes: List[str] = [u]
            path_u_edges: List[SheafEdge] = []
            path_v_nodes: List[str] = [v]
            path_v_edges: List[SheafEdge] = []
            cu, cv = u, v
            while depth[cu] > depth[cv]:
                pe = parent_edge[cu]
                assert pe is not None
                path_u_edges.append(pe)
                cu = parent_node[cu]  # type: ignore[assignment]
                path_u_nodes.append(cu)
            while depth[cv] > depth[cu]:
                pe = parent_edge[cv]
                assert pe is not None
                path_v_edges.append(pe)
                cv = parent_node[cv]  # type: ignore[assignment]
                path_v_nodes.append(cv)
            while cu != cv:
                peu = parent_edge[cu]
                pev = parent_edge[cv]
                assert peu is not None and pev is not None
                path_u_edges.append(peu)
                cu = parent_node[cu]  # type: ignore[assignment]
                path_u_nodes.append(cu)
                path_v_edges.append(pev)
                cv = parent_node[cv]  # type: ignore[assignment]
                path_v_nodes.append(cv)

            # Oriented cycle: u -> ... -> lca -> ... -> v -> u (via chord e)
            cyc_nodes = tuple(path_u_nodes + list(reversed(path_v_nodes[:-1])))
            cyc_edges = tuple(path_u_edges + list(reversed(path_v_edges)) + [e])
            cycles.append((cyc_nodes, cyc_edges))
        return cycles

    def _audit_cycle(
        self,
        cyc_nodes: Tuple[str, ...],
        cyc_edges: Tuple[SheafEdge, ...],
        node_map: Mapping[str, SheafNode],
        pairwise_ok_map: Mapping[str, bool],
    ) -> Optional[CycleObstruction]:
        sub_nodes = [node_map[nid] for nid in sorted(set(cyc_nodes))]
        delta_c, rhs_c, _, _ = self._assemble_coboundary(sub_nodes, cyc_edges)
        total_cols = sum(nd.stalk_dim for nd in sub_nodes)
        total_rows = len(delta_c)
        _, _, h_c, rank_c, _ = _solve_min_norm_lstsq(delta_c, rhs_c, tol=self.tol)
        h1_norm = _vec_norm2(h_c)
        rhs_scale = max(1.0, _vec_norm2(rhs_c))
        has_affine_h1 = h1_norm > self.tol * rhs_scale

        hol_defect, aff_defect = self._compute_cycle_holonomy(cyc_nodes, cyc_edges, node_map)
        has_holonomy = (hol_defect is not None) and (hol_defect > self.holonomy_tol)
        if not has_affine_h1 and not has_holonomy:
            return None

        if has_affine_h1 and has_holonomy:
            kind = "mixed_h1_holonomy"
        elif has_affine_h1:
            kind = "affine_h1"
        else:
            kind = "holonomy_frustration"

        min_eig, _ = self._compute_laplacian_eigs(delta_c, total_cols)
        cyc_len = len(cyc_edges)
        return CycleObstruction(
            cycle_nodes=cyc_nodes,
            cycle_edges=tuple(e.edge_id for e in cyc_edges),
            cycle_length=cyc_len,
            is_global_3plus_cycle=(cyc_len >= 3),
            locally_pairwise_consistent=all(pairwise_ok_map[e.edge_id] for e in cyc_edges),
            obstruction_kind=kind,
            h1_obstruction_norm=h1_norm,
            holonomy_defect=hol_defect,
            affine_transport_defect=aff_defect,
            dim_h0_cycle=total_cols - rank_c,
            dim_h1_cycle=total_rows - rank_c,
            min_cycle_laplacian_eig=min_eig,
        )

    def _evaluate_graph_core(
        self,
        active_nodes: Sequence[SheafNode],
        active_edges: Sequence[SheafEdge],
    ) -> Tuple[
        List[PairwiseEdgeAudit],
        List[CycleObstruction],
        List[ComponentSheafAudit],
        Dict[str, float],
        Dict[str, float],
        float,
        float,
    ]:
        node_map = {nd.node_id: nd for nd in active_nodes}
        pairwise_ok_map: Dict[str, bool] = {}
        pairwise_raw: Dict[str, Tuple[bool, float, Optional[float]]] = {}
        for e in active_edges:
            ok, intr_n, obs_n = self._check_pairwise_edge(e, node_map[e.source], node_map[e.target])
            pairwise_ok_map[e.edge_id] = ok
            pairwise_raw[e.edge_id] = (ok, intr_n, obs_n)

        # Connected components
        adj_nodes: Dict[str, Set[str]] = {nd.node_id: set() for nd in active_nodes}
        for e in active_edges:
            adj_nodes[e.source].add(e.target)
            adj_nodes[e.target].add(e.source)

        visited: Set[str] = set()
        comp_node_lists: List[List[SheafNode]] = []
        for nid in sorted(adj_nodes.keys()):
            if nid in visited:
                continue
            comp_ids: List[str] = []
            dq: deque[str] = deque([nid])
            visited.add(nid)
            while dq:
                cur = dq.popleft()
                comp_ids.append(cur)
                for nxt in sorted(adj_nodes[cur]):
                    if nxt not in visited:
                        visited.add(nxt)
                        dq.append(nxt)
            comp_node_lists.append([node_map[k] for k in sorted(comp_ids)])

        comp_audits: List[ComponentSheafAudit] = []
        all_cycle_obs: List[CycleObstruction] = []
        edge_stress: Dict[str, float] = {e.edge_id: 0.0 for e in active_edges}
        node_tension: Dict[str, float] = {nd.node_id: 0.0 for nd in active_nodes}
        total_h1_energy = 0.0
        max_hol_defect = 0.0

        for c_idx, c_nodes in enumerate(comp_node_lists):
            c_node_set = {nd.node_id for nd in c_nodes}
            c_edges = sorted(
                (e for e in active_edges if e.source in c_node_set and e.target in c_node_set),
                key=lambda x: x.edge_id,
            )
            delta_w, rhs_w, node_slices, edge_slices = self._assemble_coboundary(c_nodes, c_edges)
            n0 = sum(nd.stalk_dim for nd in c_nodes)
            n1 = len(delta_w)
            x_star, _, h_vec, rank, q_r = _solve_min_norm_lstsq(delta_w, rhs_w, tol=self.tol)
            h1_energy = sum(v * v for v in h_vec)
            h1_norm = math.sqrt(h1_energy)
            total_h1_energy += h1_energy

            for e in c_edges:
                r_lo, r_hi = edge_slices[e.edge_id]
                e_norm = _vec_norm2(h_vec[r_lo:r_hi])
                edge_stress[e.edge_id] = e_norm
                node_tension[e.source] += e_norm
                if e.target != e.source:
                    node_tension[e.target] += e_norm

            # Observed-state Hodge decomposition if all nodes in component carry state vectors
            obs_tot: Optional[float] = None
            obs_ex: Optional[float] = None
            obs_hm: Optional[float] = None
            hodge_err: Optional[float] = None
            if all(nd.state is not None for nd in c_nodes) and n1 > 0:
                x_obs: List[float] = []
                for nd in c_nodes:
                    assert nd.state is not None
                    x_obs.extend(nd.state)
                dx = _mat_vec(delta_w, x_obs)
                r_w = [dx[i] - rhs_w[i] for i in range(n1)]
                if rank > 0:
                    c_proj = [sum(q_r[i][j] * r_w[i] for i in range(n1)) for j in range(rank)]
                    r_ex = [sum(q_r[i][j] * c_proj[j] for j in range(rank)) for i in range(n1)]
                else:
                    r_ex = [0.0] * n1
                r_hm = [r_w[i] - r_ex[i] for i in range(n1)]
                obs_tot = _vec_norm2(r_w)
                obs_ex = _vec_norm2(r_ex)
                obs_hm = _vec_norm2(r_hm)
                hodge_err = abs((obs_tot * obs_tot) - (obs_ex * obs_ex + obs_hm * obs_hm))

            eig0, eig1 = self._compute_laplacian_eigs(delta_w, n0)
            reconciled = {
                nd.node_id: tuple(x_star[node_slices[nd.node_id][0]:node_slices[nd.node_id][1]])
                for nd in c_nodes
            }
            dim_h0 = n0 - rank
            dim_h1 = n1 - rank
            comp_audits.append(
                ComponentSheafAudit(
                    component_id=c_idx,
                    node_ids=tuple(nd.node_id for nd in c_nodes),
                    edge_ids=tuple(e.edge_id for e in c_edges),
                    total_vertex_dof=n0,
                    total_edge_dof=n1,
                    coboundary_rank=rank,
                    dim_h0=dim_h0,
                    dim_h1=dim_h1,
                    euler_invariant_ok=((dim_h0 - dim_h1) == (n0 - n1)),
                    h1_obstruction_norm=h1_norm,
                    h1_obstruction_energy=h1_energy,
                    observed_total_residual_norm=obs_tot,
                    observed_exact_reconcilable_norm=obs_ex,
                    observed_harmonic_norm=obs_hm,
                    hodge_pythagorean_error=hodge_err,
                    min_laplacian_eig=eig0,
                    second_laplacian_eig=eig1,
                    reconciled_states=reconciled,
                )
            )

            # Fundamental cycle basis obstruction scan
            if c_edges:
                for cyc_nodes, cyc_edges in self._extract_fundamental_cycles(c_nodes, c_edges):
                    obs = self._audit_cycle(cyc_nodes, cyc_edges, node_map, pairwise_ok_map)
                    if obs is not None:
                        all_cycle_obs.append(obs)
                        if obs.holonomy_defect is not None:
                            max_hol_defect = max(max_hol_defect, obs.holonomy_defect)
                            # Propagate holonomy frustration into node/edge tension for GABA gating
                            per_edge_hol = obs.holonomy_defect / float(max(1, obs.cycle_length))
                            for eid in obs.cycle_edges:
                                edge_stress[eid] = max(edge_stress[eid], per_edge_hol)
                            for nid in set(obs.cycle_nodes):
                                node_tension[nid] += per_edge_hol

        pairwise_audits = [
            PairwiseEdgeAudit(
                edge_id=e.edge_id,
                source=e.source,
                target=e.target,
                locally_consistent=pairwise_raw[e.edge_id][0],
                intrinsic_residual_norm=pairwise_raw[e.edge_id][1],
                observed_residual_norm=pairwise_raw[e.edge_id][2],
                harmonic_stress_norm=edge_stress[e.edge_id],
            )
            for e in sorted(active_edges, key=lambda x: x.edge_id)
        ]
        return (
            pairwise_audits,
            all_cycle_obs,
            comp_audits,
            node_tension,
            edge_stress,
            total_h1_energy,
            max_hol_defect,
        )

    def _find_minimal_cut(
        self,
        active_nodes: Sequence[SheafNode],
        active_edges: Sequence[SheafEdge],
        initial_cycles: Sequence[CycleObstruction],
        initial_h1_energy: float,
    ) -> Tuple[Tuple[str, ...], bool]:
        cur_edges = list(active_edges)
        cur_cycles = list(initial_cycles)
        cur_h1_energy = initial_h1_energy
        cut_ids: List[str] = []

        for _ in range(self.max_cut_steps):
            if not cur_cycles and math.sqrt(cur_h1_energy) <= self.tol:
                return tuple(cut_ids), True
            # Candidate edges: those in any obstructed cycle or carrying non-zero harmonic stress
            cand_ids: Set[str] = set()
            for cyc in cur_cycles:
                cand_ids.update(cyc.cycle_edges)
            if not cand_ids:
                cand_ids = {e.edge_id for e in cur_edges}
            cand_edges = sorted(
                (e for e in cur_edges if e.edge_id in cand_ids),
                key=lambda x: (x.weight, x.edge_id),
            )
            if not cand_edges:
                break

            best_choice: Optional[Tuple[Tuple[int, float, float, str], SheafEdge, List[CycleObstruction], float]] = None
            for cand in cand_edges:
                trial_edges = [e for e in cur_edges if e.edge_id != cand.edge_id]
                _, t_cycles, _, _, _, t_h1_energy, _ = self._evaluate_graph_core(active_nodes, trial_edges)
                # Objective: minimize remaining obstructed cycles, then remaining H^1 energy, then cut weight, then edge_id
                key = (len(t_cycles), round(t_h1_energy, 10), cand.weight, cand.edge_id)
                if best_choice is None or key < best_choice[0]:
                    best_choice = (key, cand, t_cycles, t_h1_energy)

            if best_choice is None:
                break
            _, chosen_edge, next_cycles, next_h1_energy = best_choice
            # Stop if no progress is possible (e.g., intrinsically inconsistent single edge removed already)
            cut_ids.append(chosen_edge.edge_id)
            cur_edges = [e for e in cur_edges if e.edge_id != chosen_edge.edge_id]
            cur_cycles = next_cycles
            cur_h1_energy = next_h1_energy
            if not cur_cycles and math.sqrt(cur_h1_energy) <= self.tol:
                return tuple(cut_ids), True

        clean = (not cur_cycles) and (math.sqrt(cur_h1_energy) <= self.tol)
        return tuple(cut_ids), clean

    def audit(
        self,
        nodes: Sequence[SheafNode],
        edges: Sequence[SheafEdge],
        *,  
        as_of_s: Optional[float] = None,
    ) -> SheafAuditReport:
        """Run full cellular sheaf coboundary audit over (nodes, edges) at snapshot as_of_s."""
        seen_nodes: Dict[str, SheafNode] = {}
        for nd in nodes:
            if nd.node_id in seen_nodes:
                raise ValueError(f"Duplicate SheafNode node_id: {nd.node_id!r}")
            seen_nodes[nd.node_id] = nd

        active_nodes = [nd for nd in nodes if nd.is_active_at(as_of_s)]
        active_node_map = {nd.node_id: nd for nd in active_nodes}
        filtered_nodes = len(nodes) - len(active_nodes)

        seen_edges: Set[str] = set()
        active_edges: List[SheafEdge] = []
        filtered_edges = 0
        for e in edges:
            if e.edge_id in seen_edges:
                raise ValueError(f"Duplicate SheafEdge edge_id: {e.edge_id!r}")
            seen_edges.add(e.edge_id)
            if e.source not in seen_nodes or e.target not in seen_nodes:
                raise ValueError(
                    f"SheafEdge({e.edge_id}) references unknown node: ({e.source!r}, {e.target!r})"
                )
            if not e.is_active_at(as_of_s) or e.source not in active_node_map or e.target not in active_node_map:
                filtered_edges += 1
                continue
            src_nd = active_node_map[e.source]
            dst_nd = active_node_map[e.target]
            if e.source_dim != src_nd.stalk_dim:
                raise ValueError(
                    f"SheafEdge({e.edge_id}) source dim {e.source_dim} != "
                    f"SheafNode({src_nd.node_id}).stalk_dim {src_nd.stalk_dim}"
                )
            if e.target_dim != dst_nd.stalk_dim:
                raise ValueError(
                    f"SheafEdge({e.edge_id}) target dim {e.target_dim} != "
                    f"SheafNode({dst_nd.node_id}).stalk_dim {dst_nd.stalk_dim}"
                )
            active_edges.append(e)

        (
            pairwise_audits,
            cycle_obstructions,
            comp_audits,
            node_tension,
            edge_stress,
            total_h1_energy,
            max_hol_defect,
        ) = self._evaluate_graph_core(active_nodes, active_edges)

        all_pairwise_ok = all(p.locally_consistent for p in pairwise_audits)
        global_h1_norm = math.sqrt(total_h1_energy)
        global_3plus = tuple(c for c in cycle_obstructions if c.is_global_3plus_cycle)
        is_globally_consistent = (
            all_pairwise_ok
            and global_h1_norm <= self.tol
            and len(cycle_obstructions) == 0
        )
        has_local_to_global_paradox = (
            all_pairwise_ok
            and (not is_globally_consistent)
            and any(c.is_global_3plus_cycle and c.locally_pairwise_consistent for c in cycle_obstructions)
        )

        cut_edges: Tuple[str, ...] = ()
        cut_restores = is_globally_consistent
        if (not is_globally_consistent) and self.compute_minimal_cut and active_edges:
            cut_edges, cut_restores = self._find_minimal_cut(
                active_nodes,
                active_edges,
                cycle_obstructions,
                total_h1_energy,
            )

        total_dim_h0 = sum(c.dim_h0 for c in comp_audits)
        total_dim_h1 = sum(c.dim_h1 for c in comp_audits)

        digest_payload = {
            "version": SHEAF_MODULE_VERSION,
            "as_of_s": as_of_s,
            "active_nodes": sorted(active_node_map.keys()),
            "active_edges": sorted(e.edge_id for e in active_edges),
            "dim_h0": total_dim_h0,
            "dim_h1": total_dim_h1,
            "h1_norm": round(global_h1_norm, 9),
            "max_holonomy": round(max_hol_defect, 9),
            "cut_edges": list(cut_edges),
            "is_globally_consistent": is_globally_consistent,
        }
        digest = hashlib.sha256(
            json.dumps(digest_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

        return SheafAuditReport(
            module_version=SHEAF_MODULE_VERSION,
            as_of_s=as_of_s,
            active_node_count=len(active_nodes),
            active_edge_count=len(active_edges),
            filtered_inactive_nodes=filtered_nodes,
            filtered_inactive_edges=filtered_edges,
            all_edges_pairwise_consistent=all_pairwise_ok,
            is_globally_consistent=is_globally_consistent,
            has_local_to_global_paradox=has_local_to_global_paradox,
            total_dim_h0=total_dim_h0,
            total_dim_h1=total_dim_h1,
            global_h1_obstruction_norm=global_h1_norm,
            global_h1_obstruction_energy=total_h1_energy,
            max_holonomy_defect=max_hol_defect,
            pairwise_audits=tuple(pairwise_audits),
            cycle_obstructions=tuple(cycle_obstructions),
            global_3plus_obstructions=global_3plus,
            components=tuple(comp_audits),
            recommended_cut_edges=cut_edges,
            cut_restores_global_consistency=cut_restores,
            node_tension_scores={k: round(v, 9) for k, v in sorted(node_tension.items())},
            edge_harmonic_stress={k: round(v, 9) for k, v in sorted(edge_stress.items())},
            audit_digest_sha256=digest,
        )


def run_sheaf_reference_benchmark() -> Dict[str, Any]:
    """Self-contained benchmark demonstrating 4 canonical sheaf consistency regimes."""
    auditor = CellularSheafCoboundaryAuditor()

    # Regime 1: 3-agent affine drift loop (locally pairwise consistent, global H^1 != 0)
    n_aff = [SheafNode("agent_a", 1), SheafNode("agent_b", 1), SheafNode("agent_c", 1)]
    e_aff = [
        SheafEdge.identity("e_ab", "agent_a", "agent_b", dim=1, offset=(10.0,), weight=1.0),
        SheafEdge.identity("e_bc", "agent_b", "agent_c", dim=1, offset=(10.0,), weight=1.0),
        SheafEdge.identity("e_ca", "agent_c", "agent_a", dim=1, offset=(10.0,), weight=0.5),
    ]
    rep_aff = auditor.audit(n_aff, e_aff)

    # Regime 2: 3-agent SO(2) 90-degree frame rotation holonomy paradox (b = 0)
    rot90 = ((0.0, -1.0), (1.0, 0.0))
    n_rot = [SheafNode("src_1", 2), SheafNode("src_2", 2), SheafNode("src_3", 2)]
    e_rot = [
        SheafEdge.linear_transport("r_12", "src_1", "src_2", rot90, weight=1.0),
        SheafEdge.linear_transport("r_23", "src_2", "src_3", rot90, weight=1.0),
        SheafEdge.linear_transport("r_31", "src_3", "src_1", rot90, weight=0.6),
    ]
    rep_rot = auditor.audit(n_rot, e_rot)

    # Regime 3: 4-agent SO(2) 90-degree flat cycle (90*4 = 360 deg = I_2 -> consistent!)
    n_flat = [SheafNode(f"f_{i}", 2) for i in range(4)]
    e_flat = [
        SheafEdge.linear_transport(f"rf_{i}", f"f_{i}", f"f_{(i + 1) % 4}", rot90, weight=1.0)
        for i in range(4)
    ]
    rep_flat = auditor.audit(n_flat, e_flat)

    # Regime 4: Heterogeneous projection paradox (d_A = 2, d_B = 1, d_C = 1)
    n_het = [SheafNode("ledger_a", 2), SheafNode("summary_b", 1), SheafNode("audit_c", 1)]
    e_het = [
        SheafEdge.projection_agreement("h_ab", "ledger_a", "summary_b", ((1.0, 1.0),), ((1.0,),), offset=(0.0,)),
        SheafEdge.identity("h_bc", "summary_b", "audit_c", dim=1, offset=(0.0,)),
        SheafEdge.projection_agreement("h_ca", "ledger_a", "audit_c", ((1.0, 1.0),), ((1.0,),), offset=(12.0,), weight=0.4),
    ]
    rep_het = auditor.audit(n_het, e_het)

    return {
        "module_version": SHEAF_MODULE_VERSION,
        "affine_3cycle_paradox": rep_aff.to_summary_dict(),
        "so2_3cycle_holonomy_paradox": rep_rot.to_summary_dict(),
        "so2_4cycle_flat_control": rep_flat.to_summary_dict(),
        "heterogeneous_projection_paradox": rep_het.to_summary_dict(),
    }
