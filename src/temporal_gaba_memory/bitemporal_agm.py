"""2D Bi-Temporal AGM Belief-Revision & Append-Only Temporal Graph Engine (Layer 4).

Zero-dependency pure Python 3.10+ standard-library implementation of a
two-dimensional bi-temporal epistemic ledger (`[valid_from, valid_until)` x
`[tx_from, tx_until)`) satisfying Alchourron-Gardenfors-Makinson (AGM) belief
revision postulates alongside Jensen-Lomet transaction-time horizon invariance.

Primary references:
- Alchourron, C. E., Gardenfors, P., & Makinson, D. (1985). On the logic of
  theory change: Partial meet contraction and revision functions. Journal of
  Symbolic Logic, 50(2), 510-530.
- Snodgrass, R. T. (1999). Developing Time-Oriented Database Applications in
  SQL. Morgan Kaufmann; and ISO/IEC 9075-2:2011 (SQL:2011 temporal tables).
- Jensen, C. S., & Lomet, D. B. (2001). Transaction Timestamping in (Temporal)
  Databases. Proceedings of the 27th International Conference on Very Large
  Data Bases (VLDB 2001), 441-450.

License: MIT
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field, replace
import hashlib
import json
import math
from types import MappingProxyType
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

__all__ = [
    "BITEMPORAL_MODULE_VERSION",
    "AGM_CONTRACTION_SENTINEL",
    "BiTemporalEpisode",
    "BiTemporalEdge",
    "RevisionCommitReceipt",
    "BiTemporalAuditReport",
    "BiTemporalAGMEngine",
]

BITEMPORAL_MODULE_VERSION = "1.0.0"
AGM_CONTRACTION_SENTINEL = "__AGM_CONTRACTION__"


def _require_nonempty_str(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string.")
    return value


def _require_finite_float(value: float, name: str) -> float:
    fval = float(value)
    if not math.isfinite(fval):
        raise ValueError(f"{name} must be finite (got {value!r}).")
    return fval


def _freeze_json_value(value: Any, name: str = "payload") -> Any:
    """Recursively validate and freeze JSON-like structures for immutable storage."""
    if value is None or isinstance(value, (bool, str, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{name} contains non-finite float: {value!r}")
        return 0.0 if value == 0.0 else value
    if isinstance(value, Mapping):
        frozen_dict: Dict[str, Any] = {}
        for k, v in sorted(value.items(), key=lambda kv: str(kv[0])):
            if not isinstance(k, str):
                raise ValueError(f"{name} mapping keys must be strings (got {type(k).__name__}).")
            frozen_dict[k] = _freeze_json_value(v, f"{name}.{k}")
        return MappingProxyType(frozen_dict)
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json_value(v, f"{name}[{i}]") for i, v in enumerate(value))
    raise TypeError(f"{name} contains unsupported non-JSON type: {type(value).__name__}")


def _thaw_json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {k: _thaw_json_value(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json_value(v) for v in value]
    return value


def _freeze_embedding(vec: Sequence[float], name: str = "embedding") -> Tuple[float, ...]:
    out: List[float] = []
    for idx, x in enumerate(vec):
        fx = float(x)
        if not math.isfinite(fx):
            raise ValueError(f"{name}[{idx}] must be finite (got {x!r}).")
        out.append(0.0 if fx == 0.0 else fx)
    return tuple(out)


def _half_open_contains(lo: float, hi: Optional[float], pt: float) -> bool:
    """Membership test for half-open interval `[lo, hi)` where `hi=None` means `+infinity`."""
    if pt < lo:
        return False
    if hi is not None and pt >= hi:
        return False
    return True


def _half_open_overlaps(
    a_lo: float,
    a_hi: Optional[float],
    b_lo: float,
    b_hi: Optional[float],
) -> bool:
    """Overlap test for two non-empty half-open intervals `[a_lo, a_hi)` and `[b_lo, b_hi)`."""
    cond1 = True if b_hi is None else (a_lo < b_hi)
    cond2 = True if a_hi is None else (b_lo < a_hi)
    return cond1 and cond2


def _canonical_sha256_hex(obj: Any) -> str:
    raw = json.dumps(_thaw_json_value(obj), sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


@dataclass(frozen=True)
class BiTemporalEpisode:
    """Immutable 2D bi-temporal episode over `[valid_from, valid_until) x [tx_from, tx_until)`."""

    episode_id: str
    slot_key: str
    valid_from: float
    valid_until: Optional[float] = None
    tx_from: float = 0.0
    tx_until: Optional[float] = None
    superseded_by: Optional[str] = None
    payload: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))
    embedding: Tuple[float, ...] = ()
    source_agent: str = "system"
    revision_parent_id: Optional[str] = None

    def __post_init__(self) -> None:
        _require_nonempty_str(self.episode_id, "episode_id")
        _require_nonempty_str(self.slot_key, "slot_key")
        _require_nonempty_str(self.source_agent, "source_agent")
        vf = _require_finite_float(self.valid_from, "valid_from")
        object.__setattr__(self, "valid_from", vf)

        if self.valid_until is not None:
            vu = _require_finite_float(self.valid_until, "valid_until")
            if vu <= vf:
                raise ValueError(
                    f"BiTemporalEpisode({self.episode_id}): valid_until ({vu}) must exceed valid_from ({vf})."
                )
            object.__setattr__(self, "valid_until", vu)

        tf = _require_finite_float(self.tx_from, "tx_from")
        object.__setattr__(self, "tx_from", tf)

        if self.tx_until is not None:
            tu = _require_finite_float(self.tx_until, "tx_until")
            if tu < tf:
                raise ValueError(
                    f"BiTemporalEpisode({self.episode_id}): tx_until ({tu}) cannot precede tx_from ({tf})."
                )
            object.__setattr__(self, "tx_until", tu)

        if (self.tx_until is None) != (self.superseded_by is None):
            raise ValueError(
                f"BiTemporalEpisode({self.episode_id}): tx_until and superseded_by must both be None (active) "
                "or both be populated (closed)."
            )
        if self.superseded_by is not None:
            _require_nonempty_str(self.superseded_by, "superseded_by")

        frozen_payload = _freeze_json_value(dict(self.payload) if self.payload is not None else {}, "payload")
        object.__setattr__(self, "payload", frozen_payload)
        object.__setattr__(self, "embedding", _freeze_embedding(self.embedding, "embedding"))

    @property
    def is_active(self) -> bool:
        return self.tx_until is None

    def contains_valid(self, valid_at: float) -> bool:
        return _half_open_contains(self.valid_from, self.valid_until, _require_finite_float(valid_at, "valid_at"))

    def contains_tx(self, tx_at: float) -> bool:
        return _half_open_contains(self.tx_from, self.tx_until, _require_finite_float(tx_at, "tx_at"))

    def is_visible_as_of(self, valid_at: float, tx_at: float) -> bool:
        return self.contains_valid(valid_at) and self.contains_tx(tx_at)

    def to_snapshot_fact(self) -> Dict[str, Any]:
        """Observable snapshot projection independent of future transaction-close metadata."""
        return {
            "episode_id": self.episode_id,
            "slot_key": self.slot_key,
            "valid_from": self.valid_from,
            "valid_until": self.valid_until,
            "tx_from": self.tx_from,
            "payload": _thaw_json_value(self.payload),
            "embedding": list(self.embedding),
            "source_agent": self.source_agent,
            "revision_parent_id": self.revision_parent_id,
        }

    def to_dict(self) -> Dict[str, Any]:
        out = self.to_snapshot_fact()
        out["tx_until"] = self.tx_until
        out["superseded_by"] = self.superseded_by
        return out


@dataclass(frozen=True)
class BiTemporalEdge:
    """Immutable 2D bi-temporal directed graph edge over `[valid_from, valid_until) x [tx_from, tx_until)`."""

    edge_id: str
    slot_key: str
    source_id: str
    target_id: str
    rel_type: str
    valid_from: float
    valid_until: Optional[float] = None
    tx_from: float = 0.0
    tx_until: Optional[float] = None
    superseded_by: Optional[str] = None
    weight: float = 1.0
    payload: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))
    embedding: Tuple[float, ...] = ()

    def __post_init__(self) -> None:
        _require_nonempty_str(self.edge_id, "edge_id")
        _require_nonempty_str(self.slot_key, "slot_key")
        _require_nonempty_str(self.source_id, "source_id")
        _require_nonempty_str(self.target_id, "target_id")
        _require_nonempty_str(self.rel_type, "rel_type")

        vf = _require_finite_float(self.valid_from, "valid_from")
        object.__setattr__(self, "valid_from", vf)
        if self.valid_until is not None:
            vu = _require_finite_float(self.valid_until, "valid_until")
            if vu <= vf:
                raise ValueError(
                    f"BiTemporalEdge({self.edge_id}): valid_until ({vu}) must exceed valid_from ({vf})."
                )
            object.__setattr__(self, "valid_until", vu)

        tf = _require_finite_float(self.tx_from, "tx_from")
        object.__setattr__(self, "tx_from", tf)
        if self.tx_until is not None:
            tu = _require_finite_float(self.tx_until, "tx_until")
            if tu < tf:
                raise ValueError(
                    f"BiTemporalEdge({self.edge_id}): tx_until ({tu}) cannot precede tx_from ({tf})."
                )
            object.__setattr__(self, "tx_until", tu)

        if (self.tx_until is None) != (self.superseded_by is None):
            raise ValueError(
                f"BiTemporalEdge({self.edge_id}): tx_until and superseded_by must both be None (active) "
                "or both be populated (closed)."
            )
        if self.superseded_by is not None:
            _require_nonempty_str(self.superseded_by, "superseded_by")

        w = _require_finite_float(self.weight, "weight")
        if w < 0.0:
            raise ValueError(f"BiTemporalEdge({self.edge_id}): weight must be >= 0.0 (got {w}).")
        object.__setattr__(self, "weight", w)
        object.__setattr__(
            self,
            "payload",
            _freeze_json_value(dict(self.payload) if self.payload is not None else {}, "payload"),
        )
        object.__setattr__(self, "embedding", _freeze_embedding(self.embedding, "embedding"))

    @property
    def is_active(self) -> bool:
        return self.tx_until is None

    def contains_valid(self, valid_at: float) -> bool:
        return _half_open_contains(self.valid_from, self.valid_until, _require_finite_float(valid_at, "valid_at"))

    def contains_tx(self, tx_at: float) -> bool:
        return _half_open_contains(self.tx_from, self.tx_until, _require_finite_float(tx_at, "tx_at"))

    def is_visible_as_of(self, valid_at: float, tx_at: float) -> bool:
        return self.contains_valid(valid_at) and self.contains_tx(tx_at)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "edge_id": self.edge_id,
            "slot_key": self.slot_key,
            "source_id": self.source_id,
            "target_id": self.target_id,
            "rel_type": self.rel_type,
            "valid_from": self.valid_from,
            "valid_until": self.valid_until,
            "tx_from": self.tx_from,
            "tx_until": self.tx_until,
            "superseded_by": self.superseded_by,
            "weight": self.weight,
            "payload": _thaw_json_value(self.payload),
            "embedding": list(self.embedding),
        }


@dataclass(frozen=True)
class RevisionCommitReceipt:
    """Atomic receipt emitted by `revise_split` or `retract_split`."""

    tx_timestamp: float
    slot_key: str
    inserted_head_id: Optional[str]
    closed_episode_ids: Tuple[str, ...]
    residual_episode_ids: Tuple[str, ...]
    head_episode: Optional[BiTemporalEpisode]
    residual_episodes: Tuple[BiTemporalEpisode, ...]
    closed_episodes: Tuple[BiTemporalEpisode, ...]
    receipt_sha256: str


@dataclass(frozen=True)
class BiTemporalAuditReport:
    """Mechanized verification report for the 5 Jensen-Lomet & AGM invariants."""

    all_passed: bool
    episode_count: int
    edge_count: int
    active_episode_count: int
    tombstone_episode_count: int
    boundedness_ok: bool
    active_closed_consistency_ok: bool
    horizon_invariance_ok: bool
    single_active_per_slot_ok: bool
    supersession_zero_gap_ok: bool
    violations: Tuple[str, ...]
    audit_digest_sha256: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "all_passed": self.all_passed,
            "episode_count": self.episode_count,
            "edge_count": self.edge_count,
            "active_episode_count": self.active_episode_count,
            "tombstone_episode_count": self.tombstone_episode_count,
            "boundedness_ok": self.boundedness_ok,
            "active_closed_consistency_ok": self.active_closed_consistency_ok,
            "horizon_invariance_ok": self.horizon_invariance_ok,
            "single_active_per_slot_ok": self.single_active_per_slot_ok,
            "supersession_zero_gap_ok": self.supersession_zero_gap_ok,
            "violations": list(self.violations),
            "audit_digest_sha256": self.audit_digest_sha256,
        }


@dataclass(frozen=True)
class _SplitAuditRecord:
    """Internal lineage record enabling exact cov_puts verification across Allen splits."""

    tx_timestamp: float
    slot_key: str
    closed_intervals: Tuple[Tuple[str, float, Optional[float]], ...]
    emitted_intervals: Tuple[Tuple[str, float, Optional[float]], ...]
    is_contraction: bool


@dataclass(frozen=True)
class _HorizonCheckpoint:
    """Pre-commit historical snapshot digest recorded to verify `obs_run` horizon invariance."""

    tx_probe: float
    valid_probe: float
    slot_key: str
    expected_fact_ids: Tuple[str, ...]
    expected_digest_sha256: str


class BiTemporalAGMEngine:
    """Append-only 2D Bi-Temporal AGM Belief-Revision & Temporal Graph Engine.

    Guarantees:
    1. Historical valid-time intervals `[valid_from, valid_until)` and payloads are
       never mutated in place.
    2. `revise_split(...)` atomically closes overlapped active versions at the exact
       successor transaction timestamp (`old.tx_until == new.tx_from`, zero gap) and
       emits left/right residual episodes whenever the revision overrides a proper
       sub-interval of valid time (satisfying AGM Success `K*2`, Consistency `K*5`,
       and Minimal Change / Snapshot Preservation).
    3. `query_as_of(valid_at, tx_at)` answers 2D point-in-time queries with zero
       future-write leakage.
    4. `audit_tx_invariants()` mechanically verifies all 5 Jensen-Lomet & AGM
       temporal database invariants over the full ledger.
    """

    def __init__(self) -> None:
        self._episodes: List[BiTemporalEpisode] = []
        self._ep_idx_by_id: Dict[str, int] = {}
        self._slot_ep_indices: Dict[str, List[int]] = defaultdict(list)

        self._edges: List[BiTemporalEdge] = []
        self._edge_idx_by_id: Dict[str, int] = {}
        self._slot_edge_indices: Dict[str, List[int]] = defaultdict(list)

        self._commit_tx_history: List[float] = []
        self._max_tx_timestamp: Optional[float] = None
        self._ep_seq: int = 0
        self._edge_seq: int = 0
        self._mutation_seq: int = 0
        self._split_records: List[_SplitAuditRecord] = []
        self._horizon_checkpoints: List[_HorizonCheckpoint] = []

    @property
    def mutation_seq(self) -> int:
        """Monotonically increasing counter incremented on every state-mutating operation."""
        return self._mutation_seq

    @property
    def max_tx_timestamp(self) -> Optional[float]:
        return self._max_tx_timestamp

    @property
    def episode_count(self) -> int:
        return len(self._episodes)

    @property
    def edge_count(self) -> int:
        return len(self._edges)

    @property
    def last_episode_id(self) -> Optional[str]:
        return self._episodes[-1].episode_id if self._episodes else None

    @property
    def episodes(self) -> Tuple[BiTemporalEpisode, ...]:
        return tuple(self._episodes)

    @property
    def edges(self) -> Tuple[BiTemporalEdge, ...]:
        return tuple(self._edges)

    def get_episode(self, episode_id: str) -> Optional[BiTemporalEpisode]:
        idx = self._ep_idx_by_id.get(episode_id)
        return self._episodes[idx] if idx is not None else None

    def _resolve_monotonic_tx(self, tx_from: Optional[float], tx_at: Optional[float]) -> float:
        if tx_from is not None and tx_at is not None and float(tx_from) != float(tx_at):
            raise ValueError(f"Conflicting tx_from ({tx_from}) and tx_at ({tx_at}) arguments.")
        chosen = tx_from if tx_from is not None else tx_at
        if chosen is None:
            tx_val = 1.0 if self._max_tx_timestamp is None else (self._max_tx_timestamp + 1.0)
        else:
            tx_val = _require_finite_float(chosen, "tx_from")
        if self._max_tx_timestamp is not None and tx_val < self._max_tx_timestamp:
            raise ValueError(
                f"Out-of-order transaction timestamp {tx_val} < current horizon {self._max_tx_timestamp} "
                "(violates Jensen-Lomet monotonic commit ordering)."
            )
        return tx_val

    def _allocate_episode_id(self, requested_id: Optional[str], reserved: Set[str]) -> Tuple[str, int]:
        """Validate or allocate an episode_id without mutating `self._ep_seq` until commit."""
        if requested_id is not None:
            _require_nonempty_str(requested_id, "episode_id")
            if requested_id in self._ep_idx_by_id or requested_id in reserved:
                raise ValueError(f"Duplicate episode_id: {requested_id!r}")
            reserved.add(requested_id)
            return requested_id, self._ep_seq
        seq = self._ep_seq
        while True:
            seq += 1
            cand = f"ep_{seq:06d}"
            if cand not in self._ep_idx_by_id and cand not in reserved:
                reserved.add(cand)
                return cand, seq

    def _allocate_edge_id(self, requested_id: Optional[str], reserved: Set[str]) -> Tuple[str, int]:
        """Validate or allocate an edge_id without mutating `self._edge_seq` until commit."""
        if requested_id is not None:
            _require_nonempty_str(requested_id, "edge_id")
            if requested_id in self._edge_idx_by_id or requested_id in reserved:
                raise ValueError(f"Duplicate edge_id: {requested_id!r}")
            reserved.add(requested_id)
            return requested_id, self._edge_seq
        seq = self._edge_seq
        while True:
            seq += 1
            cand = f"edge_{seq:06d}"
            if cand not in self._edge_idx_by_id and cand not in reserved:
                reserved.add(cand)
                return cand, seq

    def _append_episode_row(self, ep: BiTemporalEpisode) -> None:
        idx = len(self._episodes)
        self._episodes.append(ep)
        self._ep_idx_by_id[ep.episode_id] = idx
        self._slot_ep_indices[ep.slot_key].append(idx)

    def _append_edge_row(self, edge: BiTemporalEdge) -> None:
        idx = len(self._edges)
        self._edges.append(edge)
        self._edge_idx_by_id[edge.edge_id] = idx
        self._slot_edge_indices[edge.slot_key].append(idx)

    def _record_pre_commit_checkpoint(self, slot_key: str, valid_probe: float, tx_commit: float) -> None:
        """Capture an immutable pre-commit witness at `t_probe < tx_commit` for `obs_run` auditing."""
        if self._max_tx_timestamp is None:
            return
        if tx_commit <= self._max_tx_timestamp:
            return
        # Midpoint strictly between previous horizon and new commit timestamp
        tx_probe = 0.5 * (self._max_tx_timestamp + tx_commit)
        if not (self._max_tx_timestamp <= tx_probe < tx_commit):
            return
        snap = self.query_as_of(valid_at=valid_probe, tx_at=tx_probe, slot_key=slot_key)
        fact_ids = tuple(ep.episode_id for ep in snap)
        digest = _canonical_sha256_hex([ep.to_snapshot_fact() for ep in snap])
        self._horizon_checkpoints.append(
            _HorizonCheckpoint(
                tx_probe=tx_probe,
                valid_probe=valid_probe,
                slot_key=slot_key,
                expected_fact_ids=fact_ids,
                expected_digest_sha256=digest,
            )
        )

    def revise_split(
        self,
        slot_key: str,
        valid_from: float,
        valid_until: Optional[float] = None,
        tx_from: Optional[float] = None,
        payload: Optional[Mapping[str, Any]] = None,
        embedding: Sequence[float] = (),
        *,
        episode_id: Optional[str] = None,
        source_agent: str = "system",
        tx_at: Optional[float] = None,
    ) -> RevisionCommitReceipt:
        """Atomically revise `slot_key` over `[valid_from, valid_until)` at transaction time `tx_from`.

        Implements Snodgrass SQL:2011 valid-time portion split + AGM `K*2` (Success)
        and `K*5` (Consistency) without mutating historical validity in place:
        - Every currently active episode of `slot_key` overlapping `[valid_from, valid_until)`
          has its `tx_until` bound to `tx_val` (`old.tx_until == new_head.tx_from`, zero gap)
          and `superseded_by = new_head.episode_id`.
        - Non-overridden left (`[old.valid_from, valid_from)`) and right
          (`[valid_until, old.valid_until)`) valid-time sub-intervals are re-emitted as
          fresh active residual episodes at `tx_from = tx_val`.
        """
        _require_nonempty_str(slot_key, "slot_key")
        vf = _require_finite_float(valid_from, "valid_from")
        vu: Optional[float] = None
        if valid_until is not None:
            vu = _require_finite_float(valid_until, "valid_until")
            if vu <= vf:
                raise ValueError(f"valid_until ({vu}) must strictly exceed valid_from ({vf}).")

        tx_val = self._resolve_monotonic_tx(tx_from, tx_at)
        reserved_ep_ids: Set[str] = set()
        head_id, next_ep_seq = self._allocate_episode_id(episode_id, reserved_ep_ids)

        # Pre-validate new episode fields before touching state (atomic failure guarantee)
        tentative_head = BiTemporalEpisode(
            episode_id=head_id,
            slot_key=slot_key,
            valid_from=vf,
            valid_until=vu,
            tx_from=tx_val,
            tx_until=None,
            superseded_by=None,
            payload=payload or {},
            embedding=embedding,
            source_agent=source_agent,
            revision_parent_id=None,
        )

        slot_indices = self._slot_ep_indices.get(slot_key, [])
        overlapping_indices: List[int] = []
        for idx in slot_indices:
            ep = self._episodes[idx]
            if ep.is_active and _half_open_overlaps(ep.valid_from, ep.valid_until, vf, vu):
                overlapping_indices.append(idx)

        overlapping_indices.sort(key=lambda idx: (self._episodes[idx].valid_from, self._episodes[idx].episode_id))

        closed_updates: List[Tuple[int, BiTemporalEpisode]] = []
        closed_eps: List[BiTemporalEpisode] = []
        residual_eps: List[BiTemporalEpisode] = []
        closed_intervals: List[Tuple[str, float, Optional[float]]] = []

        # Phase 1: Construct and validate all closed rows and residual episodes in local memory
        for idx in overlapping_indices:
            old_ep = self._episodes[idx]
            closed_ep = replace(old_ep, tx_until=tx_val, superseded_by=head_id)
            closed_updates.append((idx, closed_ep))
            closed_eps.append(closed_ep)
            closed_intervals.append((old_ep.episode_id, old_ep.valid_from, old_ep.valid_until))

            # Left residual: [old.valid_from, vf)
            if old_ep.valid_from < vf:
                left_id, _ = self._allocate_episode_id(
                    f"{head_id}_resL_{len(residual_eps)}", reserved_ep_ids
                )
                left_ep = BiTemporalEpisode(
                    episode_id=left_id,
                    slot_key=slot_key,
                    valid_from=old_ep.valid_from,
                    valid_until=vf,
                    tx_from=tx_val,
                    tx_until=None,
                    superseded_by=None,
                    payload=old_ep.payload,
                    embedding=old_ep.embedding,
                    source_agent=old_ep.source_agent,
                    revision_parent_id=old_ep.episode_id,
                )
                residual_eps.append(left_ep)

            # Right residual: [vu, old.valid_until)
            if vu is not None and (old_ep.valid_until is None or vu < old_ep.valid_until):
                right_id, _ = self._allocate_episode_id(
                    f"{head_id}_resR_{len(residual_eps)}", reserved_ep_ids
                )
                right_ep = BiTemporalEpisode(
                    episode_id=right_id,
                    slot_key=slot_key,
                    valid_from=vu,
                    valid_until=old_ep.valid_until,
                    tx_from=tx_val,
                    tx_until=None,
                    superseded_by=None,
                    payload=old_ep.payload,
                    embedding=old_ep.embedding,
                    source_agent=old_ep.source_agent,
                    revision_parent_id=old_ep.episode_id,
                )
                residual_eps.append(right_ep)

        parent_id = closed_eps[0].episode_id if closed_eps else None
        final_head = replace(tentative_head, revision_parent_id=parent_id)

        # Phase 2: Atomic state commit (cannot fail after Phase 1 validation)
        self._record_pre_commit_checkpoint(slot_key=slot_key, valid_probe=vf, tx_commit=tx_val)
        self._ep_seq = next_ep_seq
        for idx, closed_ep in closed_updates:
            self._episodes[idx] = closed_ep
        for res_ep in residual_eps:
            self._append_episode_row(res_ep)
        self._append_episode_row(final_head)

        self._commit_tx_history.append(tx_val)
        self._max_tx_timestamp = tx_val
        self._mutation_seq += 1

        emitted_intervals: List[Tuple[str, float, Optional[float]]] = [
            (res_ep.episode_id, res_ep.valid_from, res_ep.valid_until) for res_ep in residual_eps
        ]
        emitted_intervals.append((final_head.episode_id, final_head.valid_from, final_head.valid_until))
        self._split_records.append(
            _SplitAuditRecord(
                tx_timestamp=tx_val,
                slot_key=slot_key,
                closed_intervals=tuple(closed_intervals),
                emitted_intervals=tuple(emitted_intervals),
                is_contraction=False,
            )
        )

        closed_ids = tuple(ep.episode_id for ep in closed_eps)
        residual_ids = tuple(ep.episode_id for ep in residual_eps)
        receipt_hash = _canonical_sha256_hex(
            {
                "tx_timestamp": tx_val,
                "slot_key": slot_key,
                "inserted_head_id": final_head.episode_id,
                "closed_episode_ids": list(closed_ids),
                "residual_episode_ids": list(residual_ids),
            }
        )
        return RevisionCommitReceipt(
            tx_timestamp=tx_val,
            slot_key=slot_key,
            inserted_head_id=final_head.episode_id,
            closed_episode_ids=closed_ids,
            residual_episode_ids=residual_ids,
            head_episode=final_head,
            residual_episodes=tuple(residual_eps),
            closed_episodes=tuple(closed_eps),
            receipt_sha256=receipt_hash,
        )

    def retract_split(
        self,
        slot_key: str,
        valid_from: float,
        valid_until: Optional[float] = None,
        tx_from: Optional[float] = None,
        *,
        tx_at: Optional[float] = None,
    ) -> RevisionCommitReceipt:
        """AGM Contraction (`K - phi`) over `[valid_from, valid_until)` preserving outside residuals."""
        _require_nonempty_str(slot_key, "slot_key")
        vf = _require_finite_float(valid_from, "valid_from")
        vu: Optional[float] = None
        if valid_until is not None:
            vu = _require_finite_float(valid_until, "valid_until")
            if vu <= vf:
                raise ValueError(f"valid_until ({vu}) must strictly exceed valid_from ({vf}).")

        tx_val = self._resolve_monotonic_tx(tx_from, tx_at)

        slot_indices = self._slot_ep_indices.get(slot_key, [])
        overlapping_indices = [
            idx
            for idx in slot_indices
            if self._episodes[idx].is_active
            and _half_open_overlaps(self._episodes[idx].valid_from, self._episodes[idx].valid_until, vf, vu)
        ]
        overlapping_indices.sort(key=lambda idx: (self._episodes[idx].valid_from, self._episodes[idx].episode_id))

        reserved_ep_ids: Set[str] = set()
        next_ep_seq = self._ep_seq
        closed_updates: List[Tuple[int, BiTemporalEpisode]] = []
        closed_eps: List[BiTemporalEpisode] = []
        residual_eps: List[BiTemporalEpisode] = []
        closed_intervals: List[Tuple[str, float, Optional[float]]] = []

        for idx in overlapping_indices:
            old_ep = self._episodes[idx]
            closed_ep = replace(old_ep, tx_until=tx_val, superseded_by=AGM_CONTRACTION_SENTINEL)
            closed_updates.append((idx, closed_ep))
            closed_eps.append(closed_ep)
            closed_intervals.append((old_ep.episode_id, old_ep.valid_from, old_ep.valid_until))

            if old_ep.valid_from < vf:
                while True:
                    next_ep_seq += 1
                    cand = f"ep_{next_ep_seq:06d}"
                    if cand not in self._ep_idx_by_id and cand not in reserved_ep_ids:
                        reserved_ep_ids.add(cand)
                        left_id = cand
                        break
                residual_eps.append(
                    BiTemporalEpisode(
                        episode_id=left_id,
                        slot_key=slot_key,
                        valid_from=old_ep.valid_from,
                        valid_until=vf,
                        tx_from=tx_val,
                        payload=old_ep.payload,
                        embedding=old_ep.embedding,
                        source_agent=old_ep.source_agent,
                        revision_parent_id=old_ep.episode_id,
                    )
                )
            if vu is not None and (old_ep.valid_until is None or vu < old_ep.valid_until):
                while True:
                    next_ep_seq += 1
                    cand = f"ep_{next_ep_seq:06d}"
                    if cand not in self._ep_idx_by_id and cand not in reserved_ep_ids:
                        reserved_ep_ids.add(cand)
                        right_id = cand
                        break
                residual_eps.append(
                    BiTemporalEpisode(
                        episode_id=right_id,
                        slot_key=slot_key,
                        valid_from=vu,
                        valid_until=old_ep.valid_until,
                        tx_from=tx_val,
                        payload=old_ep.payload,
                        embedding=old_ep.embedding,
                        source_agent=old_ep.source_agent,
                        revision_parent_id=old_ep.episode_id,
                    )
                )

        self._record_pre_commit_checkpoint(slot_key=slot_key, valid_probe=vf, tx_commit=tx_val)
        self._ep_seq = next_ep_seq
        for idx, closed_ep in closed_updates:
            self._episodes[idx] = closed_ep
        for res_ep in residual_eps:
            self._append_episode_row(res_ep)

        self._commit_tx_history.append(tx_val)
        self._max_tx_timestamp = tx_val
        self._mutation_seq += 1
        self._split_records.append(
            _SplitAuditRecord(
                tx_timestamp=tx_val,
                slot_key=slot_key,
                closed_intervals=tuple(closed_intervals),
                emitted_intervals=tuple((r.episode_id, r.valid_from, r.valid_until) for r in residual_eps),
                is_contraction=True,
            )
        )

        closed_ids = tuple(ep.episode_id for ep in closed_eps)
        residual_ids = tuple(ep.episode_id for ep in residual_eps)
        receipt_hash = _canonical_sha256_hex(
            {
                "tx_timestamp": tx_val,
                "slot_key": slot_key,
                "inserted_head_id": None,
                "closed_episode_ids": list(closed_ids),
                "residual_episode_ids": list(residual_ids),
            }
        )
        return RevisionCommitReceipt(
            tx_timestamp=tx_val,
            slot_key=slot_key,
            inserted_head_id=None,
            closed_episode_ids=closed_ids,
            residual_episode_ids=residual_ids,
            head_episode=None,
            residual_episodes=tuple(residual_eps),
            closed_episodes=tuple(closed_eps),
            receipt_sha256=receipt_hash,
        )

    def revise_edge_split(
        self,
        source_id: str,
        target_id: str,
        rel_type: str,
        valid_from: float,
        valid_until: Optional[float] = None,
        tx_from: Optional[float] = None,
        *,
        slot_key: Optional[str] = None,
        edge_id: Optional[str] = None,
        weight: float = 1.0,
        payload: Optional[Mapping[str, Any]] = None,
        embedding: Sequence[float] = (),
        tx_at: Optional[float] = None,
    ) -> BiTemporalEdge:
        """Atomically revise a bi-temporal directed edge with valid-time residual splitting."""
        resolved_slot = slot_key or f"{source_id}--{rel_type}-->{target_id}"
        _require_nonempty_str(resolved_slot, "slot_key")
        vf = _require_finite_float(valid_from, "valid_from")
        vu: Optional[float] = None
        if valid_until is not None:
            vu = _require_finite_float(valid_until, "valid_until")
            if vu <= vf:
                raise ValueError(f"valid_until ({vu}) must strictly exceed valid_from ({vf}).")

        tx_val = self._resolve_monotonic_tx(tx_from, tx_at)
        reserved_edge_ids: Set[str] = set()
        new_edge_id, next_edge_seq = self._allocate_edge_id(edge_id, reserved_edge_ids)
        head_edge = BiTemporalEdge(
            edge_id=new_edge_id,
            slot_key=resolved_slot,
            source_id=source_id,
            target_id=target_id,
            rel_type=rel_type,
            valid_from=vf,
            valid_until=vu,
            tx_from=tx_val,
            tx_until=None,
            superseded_by=None,
            weight=weight,
            payload=payload or {},
            embedding=embedding,
        )

        slot_indices = self._slot_edge_indices.get(resolved_slot, [])
        overlapping = [
            idx
            for idx in slot_indices
            if self._edges[idx].is_active
            and _half_open_overlaps(self._edges[idx].valid_from, self._edges[idx].valid_until, vf, vu)
        ]
        overlapping.sort(key=lambda idx: (self._edges[idx].valid_from, self._edges[idx].edge_id))

        closed_edge_updates: List[Tuple[int, BiTemporalEdge]] = []
        residuals: List[BiTemporalEdge] = []
        for idx in overlapping:
            old_e = self._edges[idx]
            closed_edge_updates.append((idx, replace(old_e, tx_until=tx_val, superseded_by=new_edge_id)))
            if old_e.valid_from < vf:
                res_left_id, _ = self._allocate_edge_id(
                    f"{new_edge_id}_resL_{len(residuals)}", reserved_edge_ids
                )
                residuals.append(
                    BiTemporalEdge(
                        edge_id=res_left_id,
                        slot_key=resolved_slot,
                        source_id=old_e.source_id,
                        target_id=old_e.target_id,
                        rel_type=old_e.rel_type,
                        valid_from=old_e.valid_from,
                        valid_until=vf,
                        tx_from=tx_val,
                        weight=old_e.weight,
                        payload=old_e.payload,
                        embedding=old_e.embedding,
                    )
                )
            if vu is not None and (old_e.valid_until is None or vu < old_e.valid_until):
                res_right_id, _ = self._allocate_edge_id(
                    f"{new_edge_id}_resR_{len(residuals)}", reserved_edge_ids
                )
                residuals.append(
                    BiTemporalEdge(
                        edge_id=res_right_id,
                        slot_key=resolved_slot,
                        source_id=old_e.source_id,
                        target_id=old_e.target_id,
                        rel_type=old_e.rel_type,
                        valid_from=vu,
                        valid_until=old_e.valid_until,
                        tx_from=tx_val,
                        weight=old_e.weight,
                        payload=old_e.payload,
                        embedding=old_e.embedding,
                    )
                )

        self._edge_seq = next_edge_seq
        for idx, closed_e in closed_edge_updates:
            self._edges[idx] = closed_e
        for r_edge in residuals:
            self._append_edge_row(r_edge)
        self._append_edge_row(head_edge)
        self._commit_tx_history.append(tx_val)
        self._max_tx_timestamp = tx_val
        self._mutation_seq += 1
        return head_edge

    def query_as_of(
        self,
        valid_at: float,
        tx_at: float,
        slot_key: Optional[str] = None,
    ) -> List[BiTemporalEpisode]:
        """Point-in-time 2D query returning exact state known at `tx_at` about `valid_at`."""
        v_pt = _require_finite_float(valid_at, "valid_at")
        t_pt = _require_finite_float(tx_at, "tx_at")

        if slot_key is not None:
            cand_indices = self._slot_ep_indices.get(slot_key, [])
            matches = [
                self._episodes[idx]
                for idx in cand_indices
                if self._episodes[idx].is_visible_as_of(v_pt, t_pt)
            ]
        else:
            matches = [ep for ep in self._episodes if ep.is_visible_as_of(v_pt, t_pt)]

        matches.sort(key=lambda ep: (ep.slot_key, ep.valid_from, ep.episode_id))
        return matches

    def query_edges_as_of(
        self,
        valid_at: Optional[float],
        tx_at: float,
        slot_key: Optional[str] = None,
        source_id: Optional[str] = None,
    ) -> List[BiTemporalEdge]:
        """Point-in-time query for directed edges at `(valid_at, tx_at)` (or tx-active when `valid_at=None`)."""
        v_pt: Optional[float] = _require_finite_float(valid_at, "valid_at") if valid_at is not None else None
        t_pt = _require_finite_float(tx_at, "tx_at")

        if slot_key is not None:
            cands: Iterable[BiTemporalEdge] = (
                self._edges[idx] for idx in self._slot_edge_indices.get(slot_key, [])
            )
        else:
            cands = self._edges

        out = [
            e
            for e in cands
            if (e.is_visible_as_of(v_pt, t_pt) if v_pt is not None else e.contains_tx(t_pt))
            and (source_id is None or e.source_id == source_id)
        ]
        out.sort(key=lambda e: (e.slot_key, e.valid_from, e.edge_id))
        return out

    def neighbors_2hop_as_of(
        self,
        seed_ids: Sequence[str],
        valid_at: Optional[float],
        tx_at: float,
    ) -> Dict[str, int]:
        """Compute active 1-hop and 2-hop out/in graph proximity at `(valid_at, tx_at)`."""
        if not self._edges or not seed_ids:
            return {}
        vis_edges = self.query_edges_as_of(valid_at=valid_at, tx_at=tx_at)
        adj: Dict[str, List[str]] = defaultdict(list)
        for e in vis_edges:
            if e.weight > 0.0:
                adj[e.source_id].append(e.target_id)
                adj[e.target_id].append(e.source_id)

        seed_set = set(seed_ids)
        hops: Dict[str, int] = {}
        hop1_frontier: List[str] = []
        for s in sorted(seed_set):
            for n1 in sorted(set(adj.get(s, ()))):
                if n1 not in seed_set and n1 not in hops:
                    hops[n1] = 1
                    hop1_frontier.append(n1)
        for n1 in hop1_frontier:
            for n2 in sorted(set(adj.get(n1, ()))):
                if n2 not in seed_set and n2 not in hops:
                    hops[n2] = 2
        return hops

    def active_episodes(self, slot_key: Optional[str] = None) -> List[BiTemporalEpisode]:
        if slot_key is not None:
            eps = [self._episodes[idx] for idx in self._slot_ep_indices.get(slot_key, []) if self._episodes[idx].is_active]
        else:
            eps = [ep for ep in self._episodes if ep.is_active]
        eps.sort(key=lambda ep: (ep.slot_key, ep.valid_from, ep.episode_id))
        return eps

    def tombstone_episodes(self, slot_key: Optional[str] = None) -> List[BiTemporalEpisode]:
        if slot_key is not None:
            eps = [
                self._episodes[idx]
                for idx in self._slot_ep_indices.get(slot_key, [])
                if not self._episodes[idx].is_active
            ]
        else:
            eps = [ep for ep in self._episodes if not ep.is_active]
        eps.sort(key=lambda ep: (ep.slot_key, ep.tx_from, ep.episode_id))
        return eps

    def supersession_chain(self, slot_key: str) -> List[BiTemporalEpisode]:
        eps = [self._episodes[idx] for idx in self._slot_ep_indices.get(slot_key, [])]
        eps.sort(key=lambda ep: (ep.tx_from, ep.valid_from, ep.episode_id))
        return eps

    def tombstone_ratio(self, as_of_tx: Optional[float] = None) -> float:
        """Compute exact fraction of committed episodes that are closed tombstones."""
        if as_of_tx is None:
            total = len(self._episodes)
            if total == 0:
                return 0.0
            closed = sum(1 for ep in self._episodes if not ep.is_active)
            return closed / float(total)

        t_pt = _require_finite_float(as_of_tx, "as_of_tx")
        committed = [ep for ep in self._episodes if ep.tx_from <= t_pt]
        if not committed:
            return 0.0
        closed_by_t = sum(1 for ep in committed if ep.tx_until is not None and ep.tx_until <= t_pt)
        return closed_by_t / float(len(committed))

    @staticmethod
    def _sample_elementary_tx_points(tx_boundaries: Sequence[float]) -> List[float]:
        pts = sorted(set(tx_boundaries))
        if not pts:
            return [0.0]
        samples: List[float] = [pts[0] - 1.0]
        for i, p in enumerate(pts):
            samples.append(p)
            if i + 1 < len(pts):
                samples.append(0.5 * (p + pts[i + 1]))
        samples.append(pts[-1] + 1.0)
        return samples

    @staticmethod
    def _intervals_cover_target(
        pieces: Sequence[Tuple[float, Optional[float]]],
        target_lo: float,
        target_hi: Optional[float],
    ) -> bool:
        """Verify that union of half-open `pieces` covers `[target_lo, target_hi)` without gaps."""
        relevant = [iv for iv in pieces if _half_open_overlaps(iv[0], iv[1], target_lo, target_hi)]
        if not relevant:
            return False
        relevant.sort(key=lambda iv: (iv[0], float("inf") if iv[1] is None else iv[1]))
        cursor: Optional[float] = target_lo
        for lo, hi in relevant:
            if cursor is None:
                return True
            if lo > cursor:
                return False
            if hi is None:
                return True
            if hi > cursor:
                cursor = hi
            if target_hi is not None and cursor >= target_hi:
                return True
        if target_hi is None:
            return cursor is None
        return cursor is not None and cursor >= target_hi

    def audit_tx_invariants(self) -> BiTemporalAuditReport:
        """Mechanically audit all 5 Jensen-Lomet (2001) & AGM (1985) temporal invariants."""
        violations: List[str] = []

        # 1. Boundedness: tx_from <= tx_until (closed) and valid_from < valid_until (finite)
        boundedness_ok = True
        for ep in self._episodes:
            if not math.isfinite(ep.valid_from) or not math.isfinite(ep.tx_from):
                boundedness_ok = False
                violations.append(f"boundedness: non-finite start timestamp on {ep.episode_id}")
            if ep.valid_until is not None and not (ep.valid_from < ep.valid_until):
                boundedness_ok = False
                violations.append(f"boundedness: invalid valid interval on {ep.episode_id}")
            if ep.tx_until is not None and not (ep.tx_from <= ep.tx_until):
                boundedness_ok = False
                violations.append(f"boundedness: tx_until < tx_from on {ep.episode_id}")

        for edge in self._edges:
            if edge.valid_until is not None and not (edge.valid_from < edge.valid_until):
                boundedness_ok = False
                violations.append(f"boundedness: invalid valid interval on edge {edge.edge_id}")
            if edge.tx_until is not None and not (edge.tx_from <= edge.tx_until):
                boundedness_ok = False
                violations.append(f"boundedness: tx_until < tx_from on edge {edge.edge_id}")

        # 2. Active/Closed consistency: (tx_until is None) == (superseded_by is None) & valid pointer
        active_closed_ok = True
        for ep in self._episodes:
            if (ep.tx_until is None) != (ep.superseded_by is None):
                active_closed_ok = False
                violations.append(f"active_closed: mismatched tx_until/superseded_by on {ep.episode_id}")
            elif ep.superseded_by is not None and ep.superseded_by != AGM_CONTRACTION_SENTINEL:
                if ep.superseded_by not in self._ep_idx_by_id:
                    active_closed_ok = False
                    violations.append(
                        f"active_closed: dangling superseded_by={ep.superseded_by} on {ep.episode_id}"
                    )

        for edge in self._edges:
            if (edge.tx_until is None) != (edge.superseded_by is None):
                active_closed_ok = False
                violations.append(f"active_closed: mismatched tx_until/superseded_by on edge {edge.edge_id}")
            elif edge.superseded_by is not None and edge.superseded_by != AGM_CONTRACTION_SENTINEL:
                if edge.superseded_by not in self._edge_idx_by_id:
                    active_closed_ok = False
                    violations.append(
                        f"active_closed: dangling superseded_by={edge.superseded_by} on edge {edge.edge_id}"
                    )

        # 3. Horizon invariance (`obs_run`): monotonic commit log + exact pre-commit witness replay
        horizon_ok = True
        for i in range(1, len(self._commit_tx_history)):
            if self._commit_tx_history[i] < self._commit_tx_history[i - 1]:
                horizon_ok = False
                violations.append(
                    f"horizon_invariance: commit log regressed at index {i} "
                    f"({self._commit_tx_history[i]} < {self._commit_tx_history[i - 1]})"
                )
        for cp in self._horizon_checkpoints:
            post_snap = self.query_as_of(valid_at=cp.valid_probe, tx_at=cp.tx_probe, slot_key=cp.slot_key)
            post_ids = tuple(ep.episode_id for ep in post_snap)
            post_digest = _canonical_sha256_hex([ep.to_snapshot_fact() for ep in post_snap])
            if post_ids != cp.expected_fact_ids or post_digest != cp.expected_digest_sha256:
                horizon_ok = False
                violations.append(
                    f"horizon_invariance: historical leakage at tx={cp.tx_probe}, valid={cp.valid_probe}, "
                    f"slot={cp.slot_key}"
                )

        # 4. Single-active-per-slot uniqueness (`func_snapshot`): exact elementary-cell sweep over R^2
        single_active_ok = True
        for slot_key, indices in self._slot_ep_indices.items():
            slot_eps = [self._episodes[idx] for idx in indices]
            tx_bounds: List[float] = []
            for ep in slot_eps:
                tx_bounds.append(ep.tx_from)
                if ep.tx_until is not None:
                    tx_bounds.append(ep.tx_until)
            for t_sample in self._sample_elementary_tx_points(tx_bounds):
                vis_tx = [ep for ep in slot_eps if ep.contains_tx(t_sample)]
                vis_tx.sort(key=lambda ep: (ep.valid_from, ep.episode_id))
                for k in range(1, len(vis_tx)):
                    prev_ep = vis_tx[k - 1]
                    curr_ep = vis_tx[k]
                    if prev_ep.valid_until is None or curr_ep.valid_from < prev_ep.valid_until:
                        single_active_ok = False
                        violations.append(
                            f"func_snapshot: overlapping active valid intervals on slot={slot_key} "
                            f"at tx={t_sample} ({prev_ep.episode_id} vs {curr_ep.episode_id})"
                        )
                        break
                if not single_active_ok:
                    break

        for slot_key, indices in self._slot_edge_indices.items():
            slot_edges = [self._edges[idx] for idx in indices]
            tx_bounds = []
            for e in slot_edges:
                tx_bounds.append(e.tx_from)
                if e.tx_until is not None:
                    tx_bounds.append(e.tx_until)
            for t_sample in self._sample_elementary_tx_points(tx_bounds):
                vis_tx_e = [e for e in slot_edges if e.contains_tx(t_sample)]
                vis_tx_e.sort(key=lambda e: (e.valid_from, e.edge_id))
                for k in range(1, len(vis_tx_e)):
                    prev_e = vis_tx_e[k - 1]
                    curr_e = vis_tx_e[k]
                    if prev_e.valid_until is None or curr_e.valid_from < prev_e.valid_until:
                        single_active_ok = False
                        violations.append(
                            f"func_snapshot: overlapping active edge intervals on slot={slot_key} "
                            f"at tx={t_sample}"
                        )
                        break
                if not single_active_ok:
                    break

        # 5. Supersession zero-gap continuity (`cov_puts`): old.tx_until == succ.tx_from & split union
        zero_gap_ok = True
        for ep in self._episodes:
            if ep.superseded_by is not None and ep.superseded_by != AGM_CONTRACTION_SENTINEL:
                succ_idx = self._ep_idx_by_id.get(ep.superseded_by)
                if succ_idx is None:
                    zero_gap_ok = False
                    violations.append(f"cov_puts: missing successor {ep.superseded_by} for {ep.episode_id}")
                else:
                    succ = self._episodes[succ_idx]
                    if ep.tx_until != succ.tx_from:
                        zero_gap_ok = False
                        violations.append(
                            f"cov_puts: non-zero tx gap/overlap between {ep.episode_id} "
                            f"(tx_until={ep.tx_until}) and {succ.episode_id} (tx_from={succ.tx_from})"
                        )
                    if ep.slot_key != succ.slot_key:
                        zero_gap_ok = False
                        violations.append(
                            f"cov_puts: cross-slot supersession {ep.slot_key} -> {succ.slot_key}"
                        )
            if ep.revision_parent_id is not None:
                p_idx = self._ep_idx_by_id.get(ep.revision_parent_id)
                if p_idx is None:
                    zero_gap_ok = False
                    violations.append(f"cov_puts: missing revision_parent_id={ep.revision_parent_id}")
                else:
                    parent_ep = self._episodes[p_idx]
                    if parent_ep.tx_until != ep.tx_from:
                        zero_gap_ok = False
                        violations.append(
                            f"cov_puts: residual/head {ep.episode_id} tx_from={ep.tx_from} != "
                            f"parent {parent_ep.episode_id} tx_until={parent_ep.tx_until}"
                        )

        for edge in self._edges:
            if edge.superseded_by is not None and edge.superseded_by != AGM_CONTRACTION_SENTINEL:
                succ_e_idx = self._edge_idx_by_id.get(edge.superseded_by)
                if succ_e_idx is None or edge.tx_until != self._edges[succ_e_idx].tx_from:
                    zero_gap_ok = False
                    violations.append(f"cov_puts: edge tx continuity violation on {edge.edge_id}")

        for sr in self._split_records:
            if sr.is_contraction:
                continue
            pieces = [(lo, hi) for _, lo, hi in sr.emitted_intervals]
            for old_id, old_lo, old_hi in sr.closed_intervals:
                if not self._intervals_cover_target(pieces, old_lo, old_hi):
                    zero_gap_ok = False
                    violations.append(
                        f"cov_puts: valid-time split hole after superseding {old_id} at tx={sr.tx_timestamp}"
                    )

        all_ok = (
            boundedness_ok
            and active_closed_ok
            and horizon_ok
            and single_active_ok
            and zero_gap_ok
        )
        active_cnt = sum(1 for ep in self._episodes if ep.is_active)
        tomb_cnt = len(self._episodes) - active_cnt
        digest = _canonical_sha256_hex(
            {
                "all_passed": all_ok,
                "episode_count": len(self._episodes),
                "edge_count": len(self._edges),
                "active_episode_count": active_cnt,
                "boundedness_ok": boundedness_ok,
                "active_closed_consistency_ok": active_closed_ok,
                "horizon_invariance_ok": horizon_ok,
                "single_active_per_slot_ok": single_active_ok,
                "supersession_zero_gap_ok": zero_gap_ok,
                "violations": violations,
            }
        )
        return BiTemporalAuditReport(
            all_passed=all_ok,
            episode_count=len(self._episodes),
            edge_count=len(self._edges),
            active_episode_count=active_cnt,
            tombstone_episode_count=tomb_cnt,
            boundedness_ok=boundedness_ok,
            active_closed_consistency_ok=active_closed_ok,
            horizon_invariance_ok=horizon_ok,
            single_active_per_slot_ok=single_active_ok,
            supersession_zero_gap_ok=zero_gap_ok,
            violations=tuple(violations),
            audit_digest_sha256=digest,
        )
