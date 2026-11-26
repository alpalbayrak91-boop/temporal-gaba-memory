"""
AlpLab Bulut v15 - Disiplinlerarasi Bellek ve Celiski Denetim Motoru
Paket: VIZYON-ICRA-SUNFLOWER-MEMORY-PRUNER-20261125
Kaynak Izomorfizm: VIZYON-BILIM-20261125-02 (Erdos #20 Sunflower Lemma & Erdos #23 / Toulouse Frustration)

Iki cekirdek deterministik operator + cift SQLite/PG17 depolama adaptoru:
  1. SunflowerMemoryPruner (Erdos-Rado Delta-System): Ajan bulgularindaki ortak cekirdek (core Y)
     etrafinda tekrarlayan ayrik yapraklari (petals >= k) saptayip kanonik cekirdege indirger.
  2. FrustrationCutAuditor (Toulouse Spin-Glass Frustration): Celisen hipotez cizgesinde
     Phi_0(G) = |E| - maxcut(G) (ve isaretli cizge genellemesi Phi_signed) hesaplar,
     minimum engellenmis kenar kesimini ve tanik tek donguleri uretir.
  3. Dual SQLite / PostgreSQL 17 Adaptoru + 200-Bulgu Sentetik vs. Gercek Sema Kiyaslama Paketi.

KURAL-3 UYUMU: Tum para birimleri yalnizca duz metin USD olarak ifade edilir (0 chr(36)).
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
import hashlib
import itertools
import json
import math
import re
import sqlite3
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

# ---------------------------------------------------------------------------
# Kanonik Sema Sabitleri (Canli PG17 public.alplab_ortak_bulgular / temporal_edges)
# ---------------------------------------------------------------------------

ORTAK_BULGULAR_COLUMNS: Tuple[str, ...] = (
    "bulgu_id",
    "kaynak_ref",
    "kaynak_ajan",
    "konu_id",
    "alan",
    "baslik",
    "ozet",
    "kanit",
    "puan",
    "durum",
    "denetim_notu",
    "kullanildi",
    "created_at",
    "updated_at",
)

TEMPORAL_EDGES_COLUMNS: Tuple[str, ...] = (
    "id",
    "source_key",
    "target_key",
    "rel_type",
    "weight",
    "evidence",
    "valid_from",
    "valid_until",
    "is_active",
    "superseded_by",
    "importance",
    "confidence",
    "source_agent",
    "metadata",
    "created_at",
)

DURUM_PRIORITY: Dict[str, int] = {
    "DOGRULANDI": 4,
    "ADAY": 3,
    "ZAYIF": 2,
    "CURUTULDU": 1,
}

CONTRADICTION_REL_TYPES: frozenset[str] = frozenset({
    "negatif_kanit",
    "contradicts",
    "refutes",
    "celisir",
    "curuttu",
    "supersedes_conflict",
    "itiraz",
})

SUPPORT_REL_TYPES: frozenset[str] = frozenset({
    "dogruladi",
    "DOGRULAR",
    "dogrulandi",
    "supports",
    "temel_aldi",
})

TR_EN_STOPWORDS: frozenset[str] = frozenset({
    "ve", "veya", "ile", "icin", "olan", "olarak", "bir", "bu", "su", "da", "de",
    "ki", "mi", "mu", "ama", "ancak", "fakat", "gibi", "daha", "cok", "az", "en",
    "her", "tum", "tumunu", "uzerinde", "uzerinden", "icinde", "gore", "sonra",
    "once", "kadar", "edildi", "yapildi", "bildirildi", "onerildi", "sunuldu",
    "raporlandi", "saptandi", "gecti", "kaldi", "yok", "var", "olan", "ilgili",
    "the", "and", "for", "with", "from", "that", "this", "into", "over", "under",
    "are", "was", "were", "has", "have", "been", "not", "all", "any", "its",
    "bulgu", "teslim", "taslak", "ozet", "yeni", "iki", "uc", "dort", "bes",
})

REFUTATION_CUES: frozenset[str] = frozenset({
    "yanlis", "celisen", "celiski", "hatali", "curutuldu", "curutme", "tutarsiz",
    "uyusmazlik", "hayalet", "sisiril", "bozuldu", "gecersiz", "sahte",
    "contradict", "refute", "false", "wrong", "mismatch", "invalid", "broken",
})

_TR_TRANSLATION = str.maketrans({
    "ç": "c", "Ç": "c",
    "ğ": "g", "Ğ": "g",
    "ı": "i", "İ": "i", "I": "i",
    "ö": "o", "Ö": "o",
    "ş": "s", "Ş": "s",
    "ü": "u", "Ü": "u",
})

_TOKEN_RE = re.compile(r"[a-z0-9_]{3,}")


def normalize_text(text: str) -> str:
    """Turkce/Ingilizce karakterleri kanonik kucuk ASCII bicimine cevirir."""
    if not text:
        return ""
    return text.translate(_TR_TRANSLATION).lower()


def canonical_sha256(payload: Any) -> str:
    """Deterministik JSON serilestirmesi uzerinden SHA-256 makbuzu uretir."""
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def check_zero_dollar_symbol(text: str) -> bool:
    """KURAL-3 denetimi: metinde chr(36) bulunmadigini dogrular."""
    return chr(36) not in text


# ---------------------------------------------------------------------------
# Veri Modelleri
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FindingRecord:
    """public.alplab_ortak_bulgular tablosu ile birebir 14-kolon uyumlu bulgu kaydi."""
    bulgu_id: str
    konu_id: str
    alan: str
    baslik: str
    ozet: str
    puan: float = 5.0
    durum: str = "ADAY"
    kaynak_ref: str = ""
    kaynak_ajan: str = "web_gemini"
    kanit: str = ""
    denetim_notu: str = "{}"
    kullanildi: int = 0
    created_at: str = "2026-11-25T00:00:00+00:00"
    updated_at: str = "2026-11-25T00:00:00+00:00"
    explicit_atoms: Optional[frozenset[str]] = None

    def to_db_row(self) -> Tuple[Any, ...]:
        return (
            self.bulgu_id,
            self.kaynak_ref,
            self.kaynak_ajan,
            self.konu_id,
            self.alan,
            self.baslik,
            self.ozet,
            self.kanit,
            float(self.puan),
            self.durum,
            self.denetim_notu,
            int(self.kullanildi),
            self.created_at,
            self.updated_at,
        )

    @classmethod
    def from_mapping(cls, row: Dict[str, Any]) -> "FindingRecord":
        return cls(
            bulgu_id=str(row["bulgu_id"]),
            kaynak_ref=str(row.get("kaynak_ref") or ""),
            kaynak_ajan=str(row.get("kaynak_ajan") or ""),
            konu_id=str(row.get("konu_id") or ""),
            alan=str(row.get("alan") or ""),
            baslik=str(row["baslik"]),
            ozet=str(row["ozet"]),
            kanit=str(row.get("kanit") or ""),
            puan=float(row.get("puan", 0.0)),
            durum=str(row.get("durum") or "ADAY"),
            denetim_notu=str(row.get("denetim_notu") or "{}"),
            kullanildi=int(row.get("kullanildi") or 0),
            created_at=str(row.get("created_at") or ""),
            updated_at=str(row.get("updated_at") or ""),
        )


@dataclass(frozen=True)
class HypothesisEdge:
    """Hipotez / bulgu cizgesi kenari (sign=-1 celiski, sign=+1 destek)."""
    source_key: str
    target_key: str
    sign: int = -1
    weight: float = 1.0
    rel_type: str = "negatif_kanit"
    evidence: str = ""

    def canonical_pair(self) -> Tuple[str, str]:
        return (self.source_key, self.target_key) if self.source_key <= self.target_key else (self.target_key, self.source_key)


@dataclass(frozen=True)
class SunflowerCluster:
    """Erdos-Rado k-aycicegi (Delta-sistem) kanonik cekirdek konsolidasyon kaydi."""
    core_id: str
    konu_scope: str
    core_atoms: Tuple[str, ...]
    canonical_bulgu_id: str
    retained_ids: Tuple[str, ...]
    pruned_petal_ids: Tuple[str, ...]
    member_ids: Tuple[str, ...]
    petal_deltas: Dict[str, Tuple[str, ...]]
    exact_disjoint_verified: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "core_id": self.core_id,
            "konu_scope": self.konu_scope,
            "core_atoms": list(self.core_atoms),
            "canonical_bulgu_id": self.canonical_bulgu_id,
            "retained_ids": list(self.retained_ids),
            "pruned_petal_ids": list(self.pruned_petal_ids),
            "member_ids": list(self.member_ids),
            "petal_deltas": {k: list(v) for k, v in sorted(self.petal_deltas.items())},
            "exact_disjoint_verified": self.exact_disjoint_verified,
        }


@dataclass(frozen=True)
class SunflowerPruneReport:
    run_id: str
    k_threshold: int
    min_core_size: int
    retain_petals: int
    input_count: int
    surviving_ids: Tuple[str, ...]
    pruned_ids: Tuple[str, ...]
    clusters: Tuple[SunflowerCluster, ...]
    compression_ratio: float
    post_prune_sunflower_free: bool
    iterations: int
    elapsed_ms: float
    cost_usd: float
    receipt_sha256: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "k_threshold": self.k_threshold,
            "min_core_size": self.min_core_size,
            "retain_petals": self.retain_petals,
            "input_count": self.input_count,
            "surviving_count": len(self.surviving_ids),
            "pruned_count": len(self.pruned_ids),
            "cluster_count": len(self.clusters),
            "compression_ratio": round(self.compression_ratio, 6),
            "post_prune_sunflower_free": self.post_prune_sunflower_free,
            "iterations": self.iterations,
            "elapsed_ms": round(self.elapsed_ms, 3),
            "cost_usd": round(self.cost_usd, 4),
            "receipt_sha256": self.receipt_sha256,
            "clusters": [c.to_dict() for c in self.clusters],
        }


@dataclass(frozen=True)
class BlockFrustrationResult:
    block_index: int
    nodes: Tuple[str, ...]
    edge_count: int
    total_weight: float
    frustration_weight: float
    maxcut_weight: float
    is_bipartite_balanced: bool
    is_exact: bool
    solver_method: str


@dataclass(frozen=True)
class FrustrationAuditReport:
    audit_id: str
    node_count: int
    edge_count: int
    contradiction_edge_count: int
    support_edge_count: int
    total_weight: float
    frustration_index_phi0: float
    maxcut_value: float
    frustration_density: float
    is_exact: bool
    triangle_count: int
    is_triangle_free: bool
    erdos23_n2_over_25_bound: float
    erdos23_bound_satisfied: Optional[bool]
    spins: Dict[str, int]
    frustrated_edges: Tuple[Dict[str, Any], ...]
    satisfied_edges: Tuple[Dict[str, Any], ...]
    witness_cycles: Tuple[Tuple[str, ...], ...]
    quarantine_candidates: Tuple[Dict[str, Any], ...]
    blocks: Tuple[BlockFrustrationResult, ...]
    cut_clears_all_frustration: bool
    elapsed_ms: float
    cost_usd: float
    receipt_sha256: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "audit_id": self.audit_id,
            "node_count": self.node_count,
            "edge_count": self.edge_count,
            "contradiction_edge_count": self.contradiction_edge_count,
            "support_edge_count": self.support_edge_count,
            "total_weight": round(self.total_weight, 6),
            "frustration_index_phi0": round(self.frustration_index_phi0, 6),
            "maxcut_value": round(self.maxcut_value, 6),
            "frustration_density": round(self.frustration_density, 6),
            "is_exact": self.is_exact,
            "triangle_count": self.triangle_count,
            "is_triangle_free": self.is_triangle_free,
            "erdos23_n2_over_25_bound": round(self.erdos23_n2_over_25_bound, 6),
            "erdos23_bound_satisfied": self.erdos23_bound_satisfied,
            "frustrated_edge_count": len(self.frustrated_edges),
            "witness_cycle_count": len(self.witness_cycles),
            "cut_clears_all_frustration": self.cut_clears_all_frustration,
            "elapsed_ms": round(self.elapsed_ms, 3),
            "cost_usd": round(self.cost_usd, 4),
            "receipt_sha256": self.receipt_sha256,
            "frustrated_edges": list(self.frustrated_edges),
            "witness_cycles": [list(c) for c in self.witness_cycles],
            "quarantine_candidates": list(self.quarantine_candidates),
        }


# ---------------------------------------------------------------------------
# Erdos #20 & Erdos #23 Matematiksel Referans Teoremleri
# ---------------------------------------------------------------------------

def erdos20_f2_exact(k: int) -> int:
    """
    Chvatal-Hanson (1976) / Erdos #20 exact f(2, k):
    k-aycicegi icermeyen 2-birimli en buyuk ailenin +1 siniri.
      - Tek k:  k^2 - k + 1
      - Cift k: k^2 - (3/2)k + 1
    """
    if k < 2:
        raise ValueError("k must be >= 2")
    if k % 2 == 1:
        return k * k - k + 1
    return k * k - (3 * k) // 2 + 1


def erdos23_b_delta_exact_small(n: int) -> Optional[int]:
    """
    Erdos #23 exact b_Delta(N) = max { |E(G)| - maxcut(G) : G ucgensiz, |V|=N }
    VIZYON-BILIM-20261125-02 kanitli kucuk degerler (N = 1..10).
    """
    table = {1: 0, 2: 0, 3: 0, 4: 0, 5: 1, 6: 1, 7: 1, 8: 2, 9: 2, 10: 4}
    return table.get(n)


def find_exact_sunflower_in_family(
    sets: Sequence[frozenset[Any]],
    k: int,
    min_core_size: int = 0,
) -> Optional[Tuple[frozenset[Any], Tuple[int, ...]]]:
    """
    Genel kume ailesi icinde exact k-aycicegi (Delta-sistem) arar.
    Teorem: k >= 2 icin herhangi bir exact k-ayciceginin cekirdegi Y,
    daima en az bir ikili kesisim A_i & A_j olarak gerceklesir.
    """
    if k < 2 or len(sets) < k:
        return None
    n = len(sets)
    candidate_cores: Set[frozenset[Any]] = set()
    if min_core_size == 0:
        candidate_cores.add(frozenset())
    for i in range(n):
        si = sets[i]
        for j in range(i + 1, n):
            inter = si & sets[j]
            if len(inter) >= min_core_size:
                candidate_cores.add(inter)

    ordered_cores = sorted(
        candidate_cores,
        key=lambda y: (-len(y), sorted(str(x) for x in y)),
    )
    for core in ordered_cores:
        members = [idx for idx, s in enumerate(sets) if core.issubset(s)]
        if len(members) < k:
            continue
        petals = [sets[idx] - core for idx in members]
        chosen_local = _max_disjoint_petal_subset(petals, list(range(len(members))), exact_limit=28)
        if len(chosen_local) >= k:
            chosen_global = tuple(members[i] for i in chosen_local[:k])
            return core, chosen_global
    return None


def _max_disjoint_petal_subset(
    petals: Sequence[frozenset[Any]],
    priority_order: Sequence[int],
    exact_limit: int = 28,
) -> List[int]:
    """
    Yaprak cakisim cizgesinde maksimum bagimsiz kumeyi (pairwise disjoint petals)
    deterministik oncelik sirasiyla bulur.
    |S| <= exact_limit icin bitmask tabanli kesin dal-ve-sinir, ustu icin dereceli acgozlu.
    """
    m = len(petals)
    if m == 0:
        return []
    if m == 1:
        return [priority_order[0]]

    # Uyumluluk matrisi: compat_mask[i] -> i ile yapraklari ayrik olan dugumler
    ordered_petals = [petals[i] for i in priority_order]
    compat_mask = [0] * m
    for i in range(m):
        pi = ordered_petals[i]
        for j in range(i + 1, m):
            if pi.isdisjoint(ordered_petals[j]):
                compat_mask[i] |= (1 << j)
                compat_mask[j] |= (1 << i)

    if m <= exact_limit:
        # Greedy initial lower bound (priority order)
        best_clique: List[int] = []
        rem = (1 << m) - 1
        cur: List[int] = []
        while rem:
            lsb = rem & -rem
            v = lsb.bit_length() - 1
            cur.append(v)
            rem &= compat_mask[v]
        best_clique = list(cur)

        # Branch-and-bound max clique in compatibility graph (deterministic)
        def _expand(cand_mask: int, current: List[int]) -> None:
            nonlocal best_clique
            if not cand_mask:
                if len(current) > len(best_clique):
                    best_clique = list(current)
                return
            if len(current) + cand_mask.bit_count() <= len(best_clique):
                return
            mask = cand_mask
            while mask:
                if len(current) + mask.bit_count() <= len(best_clique):
                    return
                lsb = mask & -mask
                v = lsb.bit_length() - 1
                mask ^= lsb
                current.append(v)
                _expand(mask & compat_mask[v], current)
                current.pop()

        _expand((1 << m) - 1, [])
        return [priority_order[v] for v in best_clique]

    # Buyuk kumeler icin deterministik minimum-catisma dereceli acgozlu
    conflict_deg = [m - 1 - compat_mask[i].bit_count() for i in range(m)]
    candidates = sorted(range(m), key=lambda i: (conflict_deg[i], i))
    chosen: List[int] = []
    allowed = (1 << m) - 1
    for v in candidates:
        if allowed & (1 << v):
            chosen.append(v)
            allowed &= compat_mask[v]
    chosen.sort()
    return [priority_order[v] for v in chosen]


# ---------------------------------------------------------------------------
# 1. SunflowerMemoryPruner (Erdos-Rado Delta-System Bellek Budayicisi)
# ---------------------------------------------------------------------------

class SunflowerMemoryPruner:
    """
    Ajan bulgularindaki ortak cekirdek (core Y) etrafinda tekrarlayan ayrik yapraklari
    (petals >= k) deterministik olarak saptayip kanonik cekirdege indirger.
    """

    def __init__(
        self,
        k: int = 3,
        min_core_size: int = 2,
        max_atoms_per_finding: int = 12,
        retain_petals: int = 1,
        group_by_konu: bool = True,
        exact_mis_limit: int = 28,
    ) -> None:
        if k < 2:
            raise ValueError("k threshold must be >= 2")
        if min_core_size < 0:
            raise ValueError("min_core_size must be >= 0")
        if not (1 <= retain_petals < k):
            raise ValueError("retain_petals must satisfy 1 <= retain_petals < k")
        self.k = k
        self.min_core_size = min_core_size
        self.max_atoms_per_finding = max_atoms_per_finding
        self.retain_petals = retain_petals
        self.group_by_konu = group_by_konu
        self.exact_mis_limit = exact_mis_limit

    @staticmethod
    def extract_raw_tokens(finding: FindingRecord) -> List[str]:
        """Baslik, ozet ve kanit alanlarindan kanonik kavram atomlarini cikarir."""
        if finding.explicit_atoms is not None:
            return sorted(finding.explicit_atoms)
        parts = [
            normalize_text(finding.baslik),
            normalize_text(finding.ozet),
        ]
        if finding.kanit and not finding.kanit.startswith("gorev:"):
            parts.append(normalize_text(finding.kanit))
        tokens: List[str] = []
        for chunk in parts:
            for tok in _TOKEN_RE.findall(chunk):
                if tok in TR_EN_STOPWORDS or tok.isdigit():
                    continue
                if tok.startswith("http") or tok in {"com", "org", "blob", "main", "tools", "tests", "formal", "agent", "hub", "projects", "home", "alp"}:
                    continue
                tokens.append(tok)
        return tokens

    def build_atom_sets(self, findings: Sequence[FindingRecord]) -> Dict[str, frozenset[str]]:
        """Her bulguyu sabit ust sinirli (<= max_atoms_per_finding) deterministik atom kumesine esler."""
        raw_map: Dict[str, List[str]] = {}
        doc_freq: Counter[str] = Counter()
        for f in findings:
            if f.explicit_atoms is not None:
                raw_map[f.bulgu_id] = sorted(f.explicit_atoms)
                for tok in f.explicit_atoms:
                    doc_freq[tok] += 1
            else:
                toks = self.extract_raw_tokens(f)
                raw_map[f.bulgu_id] = toks
                for tok in set(toks):
                    doc_freq[tok] += 1

        atom_sets: Dict[str, frozenset[str]] = {}
        for f in findings:
            if f.explicit_atoms is not None:
                atom_sets[f.bulgu_id] = frozenset(f.explicit_atoms)
                continue
            title_tokens = set(_TOKEN_RE.findall(normalize_text(f.baslik)))
            tf = Counter(raw_map[f.bulgu_id])
            # Kanonik skor: baslikta gecme bonusu + tf + orta-siklik (cekirdek/yaprak dengesi)
            ranked = sorted(
                tf.keys(),
                key=lambda t: (
                    -(2 if t in title_tokens else 0),
                    -tf[t],
                    -min(doc_freq[t], 8),
                    t,
                ),
            )
            atom_sets[f.bulgu_id] = frozenset(ranked[: self.max_atoms_per_finding])
        return atom_sets

    @staticmethod
    def _canonical_rank_key(f: FindingRecord) -> Tuple[int, float, str, str]:
        return (
            -DURUM_PRIORITY.get(f.durum.upper(), 0),
            -float(f.puan),
            f.created_at or "9999-99-99",
            f.bulgu_id,
        )

    def prune(self, findings: Sequence[FindingRecord]) -> SunflowerPruneReport:
        """
        Deterministik, idempotent Erdos-Rado aycicegi budayicisi.
        Her turda en yuksek ozgulluklu cekirdek Y etrafinda |petals| >= k olan
        exact ayrik yaprakli ayciceklerini kanonik cekirdege indirger.
        """
        t0 = time.perf_counter()
        ordered_findings = sorted(findings, key=self._canonical_rank_key)
        by_id = {f.bulgu_id: f for f in ordered_findings}
        atom_sets = self.build_atom_sets(ordered_findings)

        # Konu bazli veya kuresel gruplama
        scopes: Dict[str, List[str]] = defaultdict(list)
        for f in ordered_findings:
            scope_key = (f.konu_id or "__GENEL__") if self.group_by_konu else "__GLOBAL__"
            scopes[scope_key].append(f.bulgu_id)

        clusters: List[SunflowerCluster] = []
        pruned_set: Set[str] = set()
        total_iterations = 0

        for scope_key in sorted(scopes.keys()):
            active = list(scopes[scope_key])
            while len(active) >= self.k:
                total_iterations += 1
                best_candidate = self._find_best_sunflower_in_scope(active, atom_sets)
                if best_candidate is None:
                    break
                core, member_ids = best_candidate
                # Deterministik kanonik temsilci siralamasi
                sorted_members = sorted(member_ids, key=lambda bid: self._canonical_rank_key(by_id[bid]))
                retained = tuple(sorted_members[: self.retain_petals])
                pruned = tuple(sorted_members[self.retain_petals :])

                # Kesin ayrik yaprak dogrulamasi (P_i & P_j == empty)
                petals_map = {bid: tuple(sorted(atom_sets[bid] - core)) for bid in sorted_members}
                disjoint_ok = True
                for i in range(len(sorted_members)):
                    si = set(petals_map[sorted_members[i]])
                    for j in range(i + 1, len(sorted_members)):
                        if not si.isdisjoint(petals_map[sorted_members[j]]):
                            disjoint_ok = False
                            break
                    if not disjoint_ok:
                        break

                core_atoms = tuple(sorted(core))
                core_id = "SF-" + canonical_sha256({"scope": scope_key, "core": core_atoms})[:12].upper()
                clusters.append(
                    SunflowerCluster(
                        core_id=core_id,
                        konu_scope=scope_key,
                        core_atoms=core_atoms,
                        canonical_bulgu_id=retained[0],
                        retained_ids=retained,
                        pruned_petal_ids=pruned,
                        member_ids=tuple(sorted_members),
                        petal_deltas={bid: petals_map[bid] for bid in pruned},
                        exact_disjoint_verified=disjoint_ok,
                    )
                )
                pruned_local = set(pruned)
                pruned_set.update(pruned_local)
                active = [bid for bid in active if bid not in pruned_local]

        surviving_ids = tuple(f.bulgu_id for f in ordered_findings if f.bulgu_id not in pruned_set)
        pruned_ids = tuple(sorted(pruned_set))

        # Budama sonrasi k-ayciceksizlik garantisi dogrulamasi
        post_free = True
        for scope_key in sorted(scopes.keys()):
            rem_scope = [bid for bid in scopes[scope_key] if bid not in pruned_set]
            if self._find_best_sunflower_in_scope(rem_scope, atom_sets) is not None:
                post_free = False
                break

        input_count = len(ordered_findings)
        compression_ratio = (len(pruned_ids) / input_count) if input_count > 0 else 0.0
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        receipt_payload = {
            "k": self.k,
            "min_core_size": self.min_core_size,
            "retain_petals": self.retain_petals,
            "input_ids": [f.bulgu_id for f in ordered_findings],
            "surviving_ids": list(surviving_ids),
            "pruned_ids": list(pruned_ids),
            "clusters": [c.to_dict() for c in clusters],
        }
        receipt = canonical_sha256(receipt_payload)
        run_id = "SFP-" + receipt[:12].upper()

        return SunflowerPruneReport(
            run_id=run_id,
            k_threshold=self.k,
            min_core_size=self.min_core_size,
            retain_petals=self.retain_petals,
            input_count=input_count,
            surviving_ids=surviving_ids,
            pruned_ids=pruned_ids,
            clusters=tuple(clusters),
            compression_ratio=compression_ratio,
            post_prune_sunflower_free=post_free,
            iterations=total_iterations,
            elapsed_ms=elapsed_ms,
            cost_usd=0.0,
            receipt_sha256=receipt,
        )

    def _find_best_sunflower_in_scope(
        self,
        active_ids: Sequence[str],
        atom_sets: Dict[str, frozenset[str]],
    ) -> Optional[Tuple[frozenset[str], List[str]]]:
        """Aktif kapsam icinde en buyuk / en ozgul exact k-aycicegini bulur."""
        n = len(active_ids)
        if n < self.k:
            return None

        # Tum gerceklesebilir cekirdekler ikili kesisimlerden cikar (exact)
        core_pair_hits: Counter[frozenset[str]] = Counter()
        if self.min_core_size == 0:
            core_pair_hits[frozenset()] = 1
        for i in range(n):
            si = atom_sets[active_ids[i]]
            if len(si) < self.min_core_size:
                continue
            for j in range(i + 1, n):
                inter = si & atom_sets[active_ids[j]]
                if len(inter) >= self.min_core_size:
                    core_pair_hits[inter] += 1

        if not core_pair_hits:
            return None

        # Ozgul cekirdeklere (|Y| buyuk) ve genis desteklilere oncelik
        ordered_cores = sorted(
            core_pair_hits.keys(),
            key=lambda y: (-len(y), -core_pair_hits[y], tuple(sorted(y))),
        )

        best_result: Optional[Tuple[Tuple[int, int, Tuple[str, ...]], frozenset[str], List[str]]] = None
        for core in ordered_cores:
            # Cekirdegi kapsayan bulgular (active_ids zaten kanonik oncelik sirali)
            superset_ids = [bid for bid in active_ids if core.issubset(atom_sets[bid])]
            if len(superset_ids) < self.k:
                continue
            petals = [atom_sets[bid] - core for bid in superset_ids]
            chosen_idx = _max_disjoint_petal_subset(
                petals,
                list(range(len(superset_ids))),
                exact_limit=self.exact_mis_limit,
            )
            if len(chosen_idx) >= self.k:
                chosen_ids = [superset_ids[idx] for idx in chosen_idx]
                score_key = (len(chosen_ids), len(core), tuple(sorted(core)))
                if best_result is None or (score_key[0], score_key[1]) > (best_result[0][0], best_result[0][1]):
                    best_result = (score_key, core, chosen_ids)
                    # En ozgul ve en buyuk cekirdegi bulduysak erken cikis
                    if len(core) >= self.min_core_size and len(chosen_ids) == len(superset_ids):
                        return core, chosen_ids

        if best_result is None:
            return None
        return best_result[1], best_result[2]


# ---------------------------------------------------------------------------
# 2. FrustrationCutAuditor (Toulouse Engellenmislik Indeksi / Erdos #23)
# ---------------------------------------------------------------------------

class FrustrationCutAuditor:
    """
    Birbiriyle celisen hipotez cizgesinde Toulouse engellenmislik indeksini
    Phi_0(G) = |E| - maxcut(G) (ve isaretli cizge genellemesi Phi_signed) hesaplar.

    Algoritma katmanlari:
      1. Iteratif Tarjan iki-bagli bilesen (block) ayristirmasi:
         Phi(G) = sum_B Phi(B) (kesin toplamsallik).
      2. O(|V|+|E|) iki-parcali / dengeli hizli yol (Phi(B) = 0).
      3. |V_B| <= exact_max_block_nodes icin Z_2 simetri kirimli kesin dal-ve-sinir.
      4. Buyuk yogun bloklar icin deterministik coklu-baslangic Kernighan-Lin 1-opt/2-opt.
      5. Her engellenmis kenar icin en kisa tek dongu tanigi + Erdos #23 N^2/25 denetimi.
    """

    def __init__(self, exact_max_block_nodes: int = 20, heuristic_restarts: int = 16) -> None:
        self.exact_max_block_nodes = exact_max_block_nodes
        self.heuristic_restarts = heuristic_restarts

    @staticmethod
    def extract_contradiction_edges_from_findings(
        findings: Sequence[FindingRecord],
        explicit_edges: Sequence[HypothesisEdge] = (),
        include_support_edges: bool = False,
    ) -> List[HypothesisEdge]:
        """
        Acik zamansal kenarlari (alplab_temporal_edges) ve bulgular arasi
        yapisal celiski sinyallerini deterministik olarak cizge kenarlarina cevirir.
        """
        merged: Dict[Tuple[str, str], HypothesisEdge] = {}

        # 1. Acik kenarlar
        for e in explicit_edges:
            if e.source_key == e.target_key:
                continue
            if e.sign > 0 and not include_support_edges:
                continue
            u, v = e.canonical_pair()
            if (u, v) not in merged or abs(e.weight) > abs(merged[(u, v)].weight):
                merged[(u, v)] = HypothesisEdge(
                    source_key=u,
                    target_key=v,
                    sign=-1 if e.sign < 0 else 1,
                    weight=float(e.weight),
                    rel_type=e.rel_type,
                    evidence=e.evidence,
                )

        # 2. Yapisal bulgu celiskisi cikarimi (CURUTULDU veya refutasyon ipucu + ortak nadir kavram)
        pruner = SunflowerMemoryPruner(k=2, min_core_size=1)
        atoms = pruner.build_atom_sets(findings)
        doc_freq: Counter[str] = Counter()
        for s in atoms.values():
            for tok in s:
                doc_freq[tok] += 1
        rare_cutoff = max(2, int(len(findings) * 0.15))

        def _is_refuting(f: FindingRecord) -> bool:
            if f.durum.upper() == "CURUTULDU":
                return True
            text = normalize_text(f.baslik + " " + f.ozet)
            return any(cue in text for cue in REFUTATION_CUES)

        # Ozel problem/varlik capalari (orn. erdos #184, kiosk, kredi tavan)
        erdos_re = re.compile(r"erdos\s*#?\s*(\d+)")
        erdos_anchors: Dict[str, Set[str]] = {}
        for f in findings:
            txt = normalize_text(f.baslik + " " + f.ozet)
            erdos_anchors[f.bulgu_id] = set(erdos_re.findall(txt))

        ordered = sorted(findings, key=lambda x: x.bulgu_id)
        for i in range(len(ordered)):
            fi = ordered[i]
            ref_i = _is_refuting(fi)
            for j in range(i + 1, len(ordered)):
                fj = ordered[j]
                ref_j = _is_refuting(fj)
                if ref_i == ref_j:
                    continue
                # Ayni Erdos problem numarasi uzerinde dogrudan celiski
                shared_erdos = erdos_anchors[fi.bulgu_id] & erdos_anchors[fj.bulgu_id]
                shared_rare = {t for t in (atoms[fi.bulgu_id] & atoms[fj.bulgu_id]) if doc_freq[t] <= rare_cutoff}
                same_konu = (fi.konu_id == fj.konu_id and fi.konu_id != "")
                if shared_erdos or (same_konu and len(shared_rare) >= 2):
                    u, v = (fi.bulgu_id, fj.bulgu_id) if fi.bulgu_id <= fj.bulgu_id else (fj.bulgu_id, fi.bulgu_id)
                    if (u, v) not in merged:
                        reason = (
                            f"shared_erdos:{','.join(sorted(shared_erdos))}"
                            if shared_erdos
                            else f"shared_rare_atoms:{','.join(sorted(shared_rare)[:4])}"
                        )
                        merged[(u, v)] = HypothesisEdge(
                            source_key=u,
                            target_key=v,
                            sign=-1,
                            weight=1.0,
                            rel_type="inferred_contradiction",
                            evidence=reason,
                        )
        return [merged[k] for k in sorted(merged.keys())]

    def audit(
        self,
        nodes: Sequence[str],
        edges: Sequence[HypothesisEdge],
        finding_meta: Optional[Dict[str, FindingRecord]] = None,
    ) -> FrustrationAuditReport:
        """Tam Toulouse engellenmislik denetimini calistirir."""
        t0 = time.perf_counter()
        node_set: Set[str] = set(nodes)
        # Cift/ters kenarlari deterministik birlestir
        edge_map: Dict[Tuple[str, str], HypothesisEdge] = {}
        for e in edges:
            if e.source_key == e.target_key:
                continue
            if e.weight <= 0:
                raise ValueError("Edge weight must be positive")
            u, v = e.canonical_pair()
            node_set.add(u)
            node_set.add(v)
            edge_map[(u, v)] = HypothesisEdge(
                source_key=u,
                target_key=v,
                sign=-1 if e.sign < 0 else 1,
                weight=float(e.weight),
                rel_type=e.rel_type,
                evidence=e.evidence,
            )

        sorted_nodes = sorted(node_set)
        clean_edges = [edge_map[k] for k in sorted(edge_map.keys())]

        adj: Dict[str, List[Tuple[str, HypothesisEdge]]] = {u: [] for u in sorted_nodes}
        for e in clean_edges:
            adj[e.source_key].append((e.target_key, e))
            adj[e.target_key].append((e.source_key, e))
        for u in adj:
            adj[u].sort(key=lambda item: item[0])

        # Ucgen sayimi (saf celiski alt-cizgesi uzerinde Erdos #23 kontrolu)
        neg_adj: Dict[str, Set[str]] = {u: set() for u in sorted_nodes}
        for e in clean_edges:
            if e.sign < 0:
                neg_adj[e.source_key].add(e.target_key)
                neg_adj[e.target_key].add(e.source_key)
        triangle_count = 0
        for e in clean_edges:
            if e.sign < 0:
                triangle_count += len(neg_adj[e.source_key] & neg_adj[e.target_key])
        triangle_count //= 3

        # Iteratif Tarjan ile iki-bagli bilesen (block) ayristirmasi
        blocks_edges = self._biconnected_edge_blocks(sorted_nodes, adj)

        # Her blogu ayri coz, sonra blok-kesme agaci uzerinden spinleri uyumlu hizala
        block_results: List[BlockFrustrationResult] = []
        block_spins_list: List[Dict[str, int]] = []
        total_frustration = 0.0
        total_weight = sum(e.weight for e in clean_edges)

        for idx, b_edges in enumerate(blocks_edges):
            b_nodes = sorted({e.source_key for e in b_edges} | {e.target_key for e in b_edges})
            b_weight = sum(e.weight for e in b_edges)
            b_spins, b_frust, is_bip, is_exact, method = self._solve_block(b_nodes, b_edges)
            block_spins_list.append(b_spins)
            total_frustration += b_frust
            block_results.append(
                BlockFrustrationResult(
                    block_index=idx,
                    nodes=tuple(b_nodes),
                    edge_count=len(b_edges),
                    total_weight=b_weight,
                    frustration_weight=b_frust,
                    maxcut_weight=b_weight - b_frust,
                    is_bipartite_balanced=is_bip,
                    is_exact=is_exact,
                    solver_method=method,
                )
            )

        # Blok spinlerini ortak eklem dugumlerinde (articulation points) Z_2 cevirme ile birlestir
        global_spins = self._stitch_block_spins(sorted_nodes, block_spins_list)

        # Kenarlari engellenmis / memnun olarak sinifla
        frustrated_list: List[Dict[str, Any]] = []
        satisfied_list: List[Dict[str, Any]] = []
        satisfied_adj: Dict[str, List[Tuple[str, int]]] = {u: [] for u in sorted_nodes}
        frustrated_weight_check = 0.0
        contradiction_edges = 0
        support_edges = 0

        for e in clean_edges:
            if e.sign < 0:
                contradiction_edges += 1
            else:
                support_edges += 1
            is_satisfied = (e.sign * global_spins[e.source_key] * global_spins[e.target_key]) == 1
            ed = {
                "source_key": e.source_key,
                "target_key": e.target_key,
                "sign": e.sign,
                "weight": e.weight,
                "rel_type": e.rel_type,
                "evidence": e.evidence,
            }
            if is_satisfied:
                satisfied_list.append(ed)
                satisfied_adj[e.source_key].append((e.target_key, e.sign))
                satisfied_adj[e.target_key].append((e.source_key, e.sign))
            else:
                frustrated_list.append(ed)
                frustrated_weight_check += e.weight

        # Her engellenmis kenar icin en kisa tanik engellenmis donguyu (tek dongu) cikar
        witness_cycles: List[Tuple[str, ...]] = []
        for fe in frustrated_list:
            cyc = self._find_witness_cycle(fe["source_key"], fe["target_key"], satisfied_adj)
            if cyc is not None:
                witness_cycles.append(cyc)

        # Kesim dogrulamasi: engellenmis kenarlar cikarilinca kalan alt-cizge 100% dengeli (Phi=0) mi?
        cut_clears = self._verify_balanced_signed_graph(
            sorted_nodes,
            [e for e in clean_edges if (e.sign * global_spins[e.source_key] * global_spins[e.target_key]) == 1],
        )

        # Karantina adaylarini sirala (en cok engellenmis kenara dokunan, en dusuk guvenli bulgu one)
        incident_frust: Counter[str] = Counter()
        for fe in frustrated_list:
            incident_frust[fe["source_key"]] += 1
            incident_frust[fe["target_key"]] += 1

        quarantine: List[Dict[str, Any]] = []
        for node, deg in incident_frust.items():
            meta = finding_meta.get(node) if finding_meta else None
            puan = float(meta.puan) if meta else 5.0
            durum = meta.durum if meta else "ADAY"
            quarantine.append({
                "node_key": node,
                "frustrated_degree": deg,
                "puan": puan,
                "durum": durum,
                "spin_camp": global_spins[node],
            })
        quarantine.sort(
            key=lambda q: (
                -q["frustrated_degree"],
                DURUM_PRIORITY.get(str(q["durum"]).upper(), 2),
                q["puan"],
                q["node_key"],
            )
        )

        n_nodes = len(sorted_nodes)
        all_exact = all(b.is_exact for b in block_results)
        is_pure_contradiction = (support_edges == 0)
        is_tri_free = (triangle_count == 0) and is_pure_contradiction
        erdos23_bound = (n_nodes * n_nodes) / 25.0
        erdos23_sat: Optional[bool] = (frustrated_weight_check <= erdos23_bound + 1e-9) if is_tri_free else None
        maxcut_val = total_weight - frustrated_weight_check
        density = (frustrated_weight_check / total_weight) if total_weight > 0 else 0.0
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        receipt_payload = {
            "nodes": sorted_nodes,
            "edges": [[e.source_key, e.target_key, e.sign, e.weight] for e in clean_edges],
            "phi0": round(frustrated_weight_check, 6),
            "maxcut": round(maxcut_val, 6),
            "frustrated_edges": [[fe["source_key"], fe["target_key"]] for fe in frustrated_list],
            "is_exact": all_exact,
        }
        receipt = canonical_sha256(receipt_payload)
        audit_id = "FCA-" + receipt[:12].upper()

        return FrustrationAuditReport(
            audit_id=audit_id,
            node_count=n_nodes,
            edge_count=len(clean_edges),
            contradiction_edge_count=contradiction_edges,
            support_edge_count=support_edges,
            total_weight=total_weight,
            frustration_index_phi0=frustrated_weight_check,
            maxcut_value=maxcut_val,
            frustration_density=density,
            is_exact=all_exact,
            triangle_count=triangle_count,
            is_triangle_free=is_tri_free,
            erdos23_n2_over_25_bound=erdos23_bound,
            erdos23_bound_satisfied=erdos23_sat,
            spins=global_spins,
            frustrated_edges=tuple(frustrated_list),
            satisfied_edges=tuple(satisfied_list),
            witness_cycles=tuple(witness_cycles),
            quarantine_candidates=tuple(quarantine),
            blocks=tuple(block_results),
            cut_clears_all_frustration=cut_clears,
            elapsed_ms=elapsed_ms,
            cost_usd=0.0,
            receipt_sha256=receipt,
        )

    @staticmethod
    def _biconnected_edge_blocks(
        nodes: Sequence[str],
        adj: Dict[str, List[Tuple[str, HypothesisEdge]]],
    ) -> List[List[HypothesisEdge]]:
        """Iteratif Tarjan algoritmasi ile cizgeyi kenar-ayrik iki-bagli bloklara boler."""
        disc: Dict[str, int] = {}
        low: Dict[str, int] = {}
        timer = 0
        edge_stack: List[HypothesisEdge] = []
        blocks: List[List[HypothesisEdge]] = []

        for root in nodes:
            if root in disc:
                continue
            timer += 1
            disc[root] = low[root] = timer
            # Stack tuple: (u, parent, next_neighbor_idx)
            dfs_stack: List[Tuple[str, Optional[str], int]] = [(root, None, 0)]
            while dfs_stack:
                u, parent, idx = dfs_stack[-1]
                nbrs = adj[u]
                if idx < len(nbrs):
                    dfs_stack[-1] = (u, parent, idx + 1)
                    v, edge = nbrs[idx]
                    if v == parent:
                        continue
                    if v not in disc:
                        edge_stack.append(edge)
                        timer += 1
                        disc[v] = low[v] = timer
                        dfs_stack.append((v, u, 0))
                    elif disc[v] < disc[u]:
                        edge_stack.append(edge)
                        if disc[v] < low[u]:
                            low[u] = disc[v]
                else:
                    dfs_stack.pop()
                    if parent is not None:
                        if low[u] < low[parent]:
                            low[parent] = low[u]
                        if low[u] >= disc[parent]:
                            block: List[HypothesisEdge] = []
                            while edge_stack:
                                e = edge_stack.pop()
                                block.append(e)
                                if (e.source_key == parent and e.target_key == u) or (
                                    e.source_key == u and e.target_key == parent
                                ):
                                    break
                            if block:
                                blocks.append(block)
        return blocks

    @staticmethod
    def _verify_balanced_signed_graph(
        nodes: Sequence[str],
        edges: Sequence[HypothesisEdge],
    ) -> bool:
        """Isaretli cizgenin 0 engellenme ile 2-renklenebilir (dengeli) oldugunu dogrular."""
        adj: Dict[str, List[Tuple[str, int]]] = {u: [] for u in nodes}
        for e in edges:
            adj[e.source_key].append((e.target_key, e.sign))
            adj[e.target_key].append((e.source_key, e.sign))
        color: Dict[str, int] = {}
        for start in nodes:
            if start in color:
                continue
            color[start] = 1
            q = deque([start])
            while q:
                u = q.popleft()
                for v, sgn in adj[u]:
                    expected = color[u] * sgn
                    if v not in color:
                        color[v] = expected
                        q.append(v)
                    elif color[v] != expected:
                        return False
        return True

    def _solve_block(
        self,
        b_nodes: Sequence[str],
        b_edges: Sequence[HypothesisEdge],
    ) -> Tuple[Dict[str, int], float, bool, bool, str]:
        """Tek bir iki-bagli blogu (bipartite / exact BnB / KL heuristic) cozer."""
        # 1. Hizli iki-parcali / dengeli cizge kontrolu O(|V|+|E|)
        b_adj: Dict[str, List[Tuple[str, int, float]]] = {u: [] for u in b_nodes}
        for e in b_edges:
            b_adj[e.source_key].append((e.target_key, e.sign, e.weight))
            b_adj[e.target_key].append((e.source_key, e.sign, e.weight))

        color: Dict[str, int] = {b_nodes[0]: 1}
        q = deque([b_nodes[0]])
        balanced = True
        while q and balanced:
            u = q.popleft()
            for v, sgn, _ in b_adj[u]:
                exp = color[u] * sgn
                if v not in color:
                    color[v] = exp
                    q.append(v)
                elif color[v] != exp:
                    balanced = False
                    break
        if balanced and len(color) == len(b_nodes):
            return color, 0.0, True, True, "bipartite_bfs_exact"

        # Dugumleri dereceye gore azalan sirala (deterministik)
        ordered = sorted(b_nodes, key=lambda u: (-len(b_adj[u]), u))
        idx_of = {u: i for i, u in enumerate(ordered)}
        n = len(ordered)
        num_adj: List[List[Tuple[int, int, float]]] = [[] for _ in range(n)]
        for e in b_edges:
            iu, iv = idx_of[e.source_key], idx_of[e.target_key]
            num_adj[iu].append((iv, e.sign, e.weight))
            num_adj[iv].append((iu, e.sign, e.weight))

        # Baslangic ust siniri: deterministik coklu-baslangic 1-opt / 2-opt yerel arama
        best_spins_arr, best_frust = self._local_search_maxcut(n, num_adj)

        # 2. Kesin dal-ve-sinir (Z_2 simetri kirimli: spin[0] = +1)
        if n <= self.exact_max_block_nodes or (n <= 24 and len(b_edges) <= 60):
            back_edges: List[List[Tuple[int, int, float]]] = [[] for _ in range(n)]
            for u in range(n):
                for v, sgn, w in num_adj[u]:
                    if v < u:
                        back_edges[u].append((v, sgn, w))

            cur_spins = [0] * n
            cur_spins[0] = 1
            best_val = best_frust
            best_arr = list(best_spins_arr)
            eps = 1e-12

            def _bnb(depth: int, cur_frust: float) -> None:
                nonlocal best_val, best_arr
                if cur_frust >= best_val - eps:
                    return
                if depth == n:
                    best_val = cur_frust
                    best_arr = list(cur_spins)
                    return
                be = back_edges[depth]
                # Iki dalin anlik maliyetini hesapla, ucuz dali once dene
                cost_pos = 0.0
                cost_neg = 0.0
                for v, sgn, w in be:
                    if sgn * cur_spins[v] == -1:
                        cost_pos += w
                    else:
                        cost_neg += w
                if cost_pos <= cost_neg:
                    order_try = ((1, cost_pos), (-1, cost_neg))
                else:
                    order_try = ((-1, cost_neg), (1, cost_pos))
                for spin_val, add_c in order_try:
                    nxt = cur_frust + add_c
                    if nxt < best_val - eps:
                        cur_spins[depth] = spin_val
                        _bnb(depth + 1, nxt)

            _bnb(1, 0.0)
            return (
                {ordered[i]: best_arr[i] for i in range(n)},
                best_val,
                False,
                True,
                "branch_and_bound_exact",
            )

        return (
            {ordered[i]: best_spins_arr[i] for i in range(n)},
            best_frust,
            False,
            False,
            "multistart_1opt_2opt_heuristic",
        )

    def _local_search_maxcut(
        self,
        n: int,
        num_adj: Sequence[Sequence[Tuple[int, int, float]]],
    ) -> Tuple[List[int], float]:
        """Deterministik coklu-baslangic 1-opt ve komsu 2-opt engellenme minimizasyonu."""
        def _eval(spins: Sequence[int]) -> float:
            frust = 0.0
            for u in range(n):
                su = spins[u]
                for v, sgn, w in num_adj[u]:
                    if u < v and (sgn * su * spins[v]) == -1:
                        frust += w
            return frust

        best_spins = [1] * n
        best_frust = float("inf")

        for r in range(self.heuristic_restarts):
            if r == 0:
                # Acgozlu BFS baslatma
                spins = [0] * n
                spins[0] = 1
                for u in range(1, n):
                    score = 0.0
                    for v, sgn, w in num_adj[u]:
                        if v < u:
                            score += sgn * spins[v] * w
                    spins[u] = 1 if score >= 0.0 else -1
            else:
                # Deterministik karma baslatma (LCG benzeri, harici random durumu yok)
                spins = [1] + [1 if (((i * 2654435761 + r * 40503) >> 7) & 1) == 0 else -1 for i in range(1, n)]

            # 1-opt iyilestirme
            improved = True
            while improved:
                improved = False
                for u in range(1, n):
                    gain = 0.0
                    su = spins[u]
                    for v, sgn, w in num_adj[u]:
                        if (sgn * su * spins[v]) == -1:
                            gain += w
                        else:
                            gain -= w
                    if gain > 1e-12:
                        spins[u] = -su
                        improved = True

            # Komsu 2-opt iyilestirme (ikili spin cevirme)
            improved2 = True
            while improved2:
                improved2 = False
                for u in range(1, n):
                    su = spins[u]
                    for v, sgn_uv, w_uv in num_adj[u]:
                        if v <= u:
                            continue
                        sv = spins[v]
                        # u ve v ayni anda cevrildiginde (u,v) kenari degismez; dis kenarlar degisir
                        delta = 0.0
                        for nb, sgn, w in num_adj[u]:
                            if nb == v:
                                continue
                            delta += w if (sgn * su * spins[nb]) == -1 else -w
                        for nb, sgn, w in num_adj[v]:
                            if nb == u:
                                continue
                            delta += w if (sgn * sv * spins[nb]) == -1 else -w
                        if delta > 1e-12:
                            spins[u] = -su
                            spins[v] = -sv
                            improved2 = True
                            break
                    if improved2:
                        break

            fval = _eval(spins)
            if fval < best_frust - 1e-12:
                best_frust = fval
                best_spins = list(spins)
        return best_spins, best_frust

    @staticmethod
    def _stitch_block_spins(
        sorted_nodes: Sequence[str],
        block_spins_list: Sequence[Dict[str, int]],
    ) -> Dict[str, int]:
        """Blok spinlerini eklem noktalarinda Z_2 simetrisiyle birlestirir."""
        global_spins: Dict[str, int] = {}
        remaining = list(range(len(block_spins_list)))
        # Dugum -> blok indeksleri haritasi
        node_to_blocks: Dict[str, List[int]] = defaultdict(list)
        for b_idx, b_map in enumerate(block_spins_list):
            for u in b_map:
                node_to_blocks[u].append(b_idx)

        visited_blocks: Set[int] = set()
        for b_idx in remaining:
            if b_idx in visited_blocks:
                continue
            # Yeni bagli bilesen
            q = deque([b_idx])
            visited_blocks.add(b_idx)
            for u, s in block_spins_list[b_idx].items():
                global_spins[u] = s
            while q:
                cur_b = q.popleft()
                for u in block_spins_list[cur_b]:
                    for nxt_b in node_to_blocks[u]:
                        if nxt_b in visited_blocks:
                            continue
                        visited_blocks.add(nxt_b)
                        flip = 1 if block_spins_list[nxt_b][u] == global_spins[u] else -1
                        for v, sv in block_spins_list[nxt_b].items():
                            global_spins[v] = sv * flip
                        q.append(nxt_b)

        for u in sorted_nodes:
            if u not in global_spins:
                global_spins[u] = 1
        return global_spins

    @staticmethod
    def _find_witness_cycle(
        u: str,
        v: str,
        satisfied_adj: Dict[str, List[Tuple[str, int]]],
    ) -> Optional[Tuple[str, ...]]:
        """Engellenmis (u, v) kenarini kapatan en kisa memnun yol (tanik tek dongu) bulur."""
        parent: Dict[str, Optional[str]] = {u: None}
        q = deque([u])
        while q:
            cur = q.popleft()
            if cur == v:
                path: List[str] = []
                node: Optional[str] = v
                while node is not None:
                    path.append(node)
                    node = parent[node]
                path.reverse()
                return tuple(path)
            for nxt, _ in satisfied_adj.get(cur, ()): 
                if nxt not in parent:
                    parent[nxt] = cur
                    q.append(nxt)
        return None


# ---------------------------------------------------------------------------
# 3. Cift SQLite / PostgreSQL 17 Depolama Adaptoru
# ---------------------------------------------------------------------------

SQLITE_SCHEMA_DDL = """
CREATE TABLE IF NOT EXISTS alplab_ortak_bulgular (
    bulgu_id TEXT PRIMARY KEY,
    kaynak_ref TEXT,
    kaynak_ajan TEXT,
    konu_id TEXT,
    alan TEXT,
    baslik TEXT NOT NULL,
    ozet TEXT NOT NULL,
    kanit TEXT,
    puan REAL NOT NULL DEFAULT 0.0,
    durum TEXT NOT NULL DEFAULT 'ADAY',
    denetim_notu TEXT,
    kullanildi INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS alplab_temporal_edges (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_key TEXT NOT NULL,
    target_key TEXT NOT NULL,
    rel_type TEXT NOT NULL,
    weight REAL NOT NULL DEFAULT 1.0,
    evidence TEXT,
    valid_from TEXT,
    valid_until TEXT,
    is_active INTEGER NOT NULL DEFAULT 1,
    superseded_by TEXT,
    importance TEXT DEFAULT 'P2',
    confidence REAL DEFAULT 1.0,
    source_agent TEXT,
    metadata TEXT DEFAULT '{}',
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS alplab_sunflower_pruning_runs (
    run_id TEXT PRIMARY KEY,
    k_threshold INTEGER NOT NULL,
    min_core_size INTEGER NOT NULL,
    retain_petals INTEGER NOT NULL,
    input_count INTEGER NOT NULL,
    surviving_count INTEGER NOT NULL,
    pruned_count INTEGER NOT NULL,
    compression_ratio REAL NOT NULL,
    post_prune_sunflower_free INTEGER NOT NULL,
    cost_usd REAL NOT NULL DEFAULT 0.0,
    receipt_sha256 TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS alplab_sunflower_clusters (
    run_id TEXT NOT NULL,
    core_id TEXT NOT NULL,
    konu_scope TEXT NOT NULL,
    core_atoms_json TEXT NOT NULL,
    canonical_bulgu_id TEXT NOT NULL,
    petal_count INTEGER NOT NULL,
    pruned_ids_json TEXT NOT NULL,
    petal_deltas_json TEXT NOT NULL,
    exact_disjoint_verified INTEGER NOT NULL,
    PRIMARY KEY (run_id, core_id)
);

CREATE TABLE IF NOT EXISTS alplab_frustration_audit_runs (
    audit_id TEXT PRIMARY KEY,
    node_count INTEGER NOT NULL,
    edge_count INTEGER NOT NULL,
    contradiction_edge_count INTEGER NOT NULL,
    support_edge_count INTEGER NOT NULL,
    frustration_index_phi0 REAL NOT NULL,
    maxcut_value REAL NOT NULL,
    is_exact INTEGER NOT NULL,
    triangle_count INTEGER NOT NULL,
    erdos23_bound REAL NOT NULL,
    cut_clears_all_frustration INTEGER NOT NULL,
    cost_usd REAL NOT NULL DEFAULT 0.0,
    receipt_sha256 TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS alplab_frustration_cut_edges (
    audit_id TEXT NOT NULL,
    source_key TEXT NOT NULL,
    target_key TEXT NOT NULL,
    sign INTEGER NOT NULL,
    weight REAL NOT NULL,
    rel_type TEXT NOT NULL,
    is_frustrated INTEGER NOT NULL,
    witness_cycle_json TEXT,
    PRIMARY KEY (audit_id, source_key, target_key)
);
"""


class MemoryGraphStore:
    """
    SQLite3 ve PostgreSQL 17 (psycopg) ile uyumlu cift-lehce bellek/cizge deposu.
    """

    def __init__(self, conn: Any, dialect: str = "sqlite") -> None:
        if dialect not in {"sqlite", "pg17"}:
            raise ValueError("dialect must be 'sqlite' or 'pg17'")
        self.conn = conn
        self.dialect = dialect

    def _ph(self, count: int) -> str:
        token = "?" if self.dialect == "sqlite" else "%s"
        return ", ".join([token] * count)

    def init_sqlite_schema(self) -> None:
        if self.dialect != "sqlite":
            raise RuntimeError("init_sqlite_schema is for sqlite dialect; use sql/konu4_sunflower_frustration_memory_pg17.sql for PG17")
        self.conn.executescript(SQLITE_SCHEMA_DDL)
        self.conn.commit()

    def verify_schema_parity(self) -> Dict[str, Any]:
        """Depodaki alplab_ortak_bulgular ve alplab_temporal_edges kolon paritesini dogrular."""
        cur = self.conn.cursor()
        if self.dialect == "sqlite":
            cur.execute("PRAGMA table_info(alplab_ortak_bulgular)")
            ob_cols = tuple(r[1] for r in cur.fetchall())
            cur.execute("PRAGMA table_info(alplab_temporal_edges)")
            te_cols = tuple(r[1] for r in cur.fetchall())
        else:
            cur.execute(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_schema = 'public' AND table_name = 'alplab_ortak_bulgular' "
                "ORDER BY ordinal_position"
            )
            ob_cols = tuple(r[0] for r in cur.fetchall())
            cur.execute(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_schema = 'public' AND table_name = 'alplab_temporal_edges' "
                "ORDER BY ordinal_position"
            )
            te_cols = tuple(r[0] for r in cur.fetchall())
        return {
            "dialect": self.dialect,
            "ortak_bulgular_exact_match": ob_cols == ORTAK_BULGULAR_COLUMNS,
            "temporal_edges_exact_match": te_cols == TEMPORAL_EDGES_COLUMNS,
            "ortak_bulgular_columns": list(ob_cols),
            "temporal_edges_columns": list(te_cols),
        }

    def upsert_findings(self, findings: Sequence[FindingRecord]) -> int:
        cols_sql = ", ".join(ORTAK_BULGULAR_COLUMNS)
        ph = self._ph(len(ORTAK_BULGULAR_COLUMNS))
        sql = (
            f"INSERT INTO alplab_ortak_bulgular ({cols_sql}) VALUES ({ph}) "
            "ON CONFLICT(bulgu_id) DO UPDATE SET "
            "baslik = excluded.baslik, ozet = excluded.ozet, puan = excluded.puan, "
            "durum = excluded.durum, updated_at = excluded.updated_at"
        )
        cur = self.conn.cursor()
        cur.executemany(sql, [f.to_db_row() for f in findings])
        self.conn.commit()
        return len(findings)

    def insert_temporal_edges(self, edges: Sequence[HypothesisEdge], created_at: str = "2026-11-25T18:00:00+00:00") -> int:
        ph = self._ph(10)
        sql = (
            "INSERT INTO alplab_temporal_edges "
            "(source_key, target_key, rel_type, weight, evidence, valid_from, is_active, importance, confidence, created_at) "
            f"VALUES ({ph})"
        )
        rows = [
            (
                e.source_key,
                e.target_key,
                e.rel_type,
                float(e.weight),
                e.evidence,
                created_at,
                1 if self.dialect == "sqlite" else True,
                "P2",
                1.0,
                created_at,
            )
            for e in edges
        ]
        cur = self.conn.cursor()
        cur.executemany(sql, rows)
        self.conn.commit()
        return len(rows)

    def load_findings(self, konu_id: Optional[str] = None, limit: int = 1000) -> List[FindingRecord]:
        cols_sql = ", ".join(ORTAK_BULGULAR_COLUMNS)
        cur = self.conn.cursor()
        if konu_id:
            ph = "?" if self.dialect == "sqlite" else "%s"
            cur.execute(
                f"SELECT {cols_sql} FROM alplab_ortak_bulgular WHERE konu_id = {ph} ORDER BY created_at DESC, bulgu_id ASC LIMIT {ph}",
                (konu_id, int(limit)),
            )
        else:
            ph = "?" if self.dialect == "sqlite" else "%s"
            cur.execute(
                f"SELECT {cols_sql} FROM alplab_ortak_bulgular ORDER BY created_at DESC, bulgu_id ASC LIMIT {ph}",
                (int(limit),),
            )
        rows = cur.fetchall()
        out: List[FindingRecord] = []
        for r in rows:
            mapping = dict(zip(ORTAK_BULGULAR_COLUMNS, r))
            out.append(FindingRecord.from_mapping(mapping))
        return out

    def load_contradiction_edges(self, include_support: bool = False) -> List[HypothesisEdge]:
        rel_pool = sorted(CONTRADICTION_REL_TYPES | (SUPPORT_REL_TYPES if include_support else set()))
        ph = self._ph(len(rel_pool))
        active_cond = "is_active = 1" if self.dialect == "sqlite" else "is_active = TRUE"
        sql = (
            "SELECT source_key, target_key, rel_type, weight, COALESCE(evidence, '') "
            f"FROM alplab_temporal_edges WHERE {active_cond} AND rel_type IN ({ph}) "
            "ORDER BY id ASC"
        )
        cur = self.conn.cursor()
        cur.execute(sql, tuple(rel_pool))
        edges: List[HypothesisEdge] = []
        for src, dst, rel, w, ev in cur.fetchall():
            sgn = -1 if rel in CONTRADICTION_REL_TYPES else 1
            edges.append(
                HypothesisEdge(
                    source_key=str(src),
                    target_key=str(dst),
                    sign=sgn,
                    weight=float(w if w is not None else 1.0),
                    rel_type=str(rel),
                    evidence=str(ev or ""),
                )
            )
        return edges

    def save_sunflower_report(self, report: SunflowerPruneReport, created_at: str = "2026-11-25T18:00:00+00:00") -> None:
        cur = self.conn.cursor()
        ph_run = self._ph(12)
        cur.execute(
            "INSERT INTO alplab_sunflower_pruning_runs "
            "(run_id, k_threshold, min_core_size, retain_petals, input_count, surviving_count, "
            "pruned_count, compression_ratio, post_prune_sunflower_free, cost_usd, receipt_sha256, created_at) "
            f"VALUES ({ph_run}) ON CONFLICT(run_id) DO NOTHING",
            (
                report.run_id,
                report.k_threshold,
                report.min_core_size,
                report.retain_petals,
                report.input_count,
                len(report.surviving_ids),
                len(report.pruned_ids),
                float(report.compression_ratio),
                (1 if report.post_prune_sunflower_free else 0) if self.dialect == "sqlite" else report.post_prune_sunflower_free,
                float(report.cost_usd),
                report.receipt_sha256,
                created_at,
            ),
        )
        ph_cl = self._ph(9)
        for c in report.clusters:
            cur.execute(
                "INSERT INTO alplab_sunflower_clusters "
                "(run_id, core_id, konu_scope, core_atoms_json, canonical_bulgu_id, petal_count, "
                "pruned_ids_json, petal_deltas_json, exact_disjoint_verified) "
                f"VALUES ({ph_cl}) ON CONFLICT(run_id, core_id) DO NOTHING",
                (
                    report.run_id,
                    c.core_id,
                    c.konu_scope,
                    json.dumps(list(c.core_atoms), ensure_ascii=True),
                    c.canonical_bulgu_id,
                    len(c.member_ids),
                    json.dumps(list(c.pruned_petal_ids), ensure_ascii=True),
                    json.dumps({k: list(v) for k, v in sorted(c.petal_deltas.items())}, ensure_ascii=True),
                    (1 if c.exact_disjoint_verified else 0) if self.dialect == "sqlite" else c.exact_disjoint_verified,
                ),
            )
        self.conn.commit()

    def save_frustration_report(self, report: FrustrationAuditReport, created_at: str = "2026-11-25T18:00:00+00:00") -> None:
        cur = self.conn.cursor()
        ph_run = self._ph(14)
        b_conv = (lambda b: 1 if b else 0) if self.dialect == "sqlite" else (lambda b: bool(b))
        cur.execute(
            "INSERT INTO alplab_frustration_audit_runs "
            "(audit_id, node_count, edge_count, contradiction_edge_count, support_edge_count, "
            "frustration_index_phi0, maxcut_value, is_exact, triangle_count, erdos23_bound, "
            "cut_clears_all_frustration, cost_usd, receipt_sha256, created_at) "
            f"VALUES ({ph_run}) ON CONFLICT(audit_id) DO NOTHING",
            (
                report.audit_id,
                report.node_count,
                report.edge_count,
                report.contradiction_edge_count,
                report.support_edge_count,
                float(report.frustration_index_phi0),
                float(report.maxcut_value),
                b_conv(report.is_exact),
                report.triangle_count,
                float(report.erdos23_n2_over_25_bound),
                b_conv(report.cut_clears_all_frustration),
                float(report.cost_usd),
                report.receipt_sha256,
                created_at,
            ),
        )
        ph_edge = self._ph(8)
        witness_by_edge: Dict[Tuple[str, str], List[str]] = {}
        for fe, cyc in zip(report.frustrated_edges, report.witness_cycles):
            witness_by_edge[(fe["source_key"], fe["target_key"])] = list(cyc)
        for is_f, edge_list in ((True, report.frustrated_edges), (False, report.satisfied_edges)):
            for ed in edge_list:
                cyc_json = json.dumps(witness_by_edge.get((ed["source_key"], ed["target_key"]), []), ensure_ascii=True)
                cur.execute(
                    "INSERT INTO alplab_frustration_cut_edges "
                    "(audit_id, source_key, target_key, sign, weight, rel_type, is_frustrated, witness_cycle_json) "
                    f"VALUES ({ph_edge}) ON CONFLICT(audit_id, source_key, target_key) DO NOTHING",
                    (
                        report.audit_id,
                        ed["source_key"],
                        ed["target_key"],
                        int(ed["sign"]),
                        float(ed["weight"]),
                        str(ed["rel_type"]),
                        b_conv(is_f),
                        cyc_json,
                    ),
                )
        self.conn.commit()


# ---------------------------------------------------------------------------
# 4. 200-Bulgu Sentetik Uretici + Canli PG17 Gercek Sema Kiyaslama Korpusu
# ---------------------------------------------------------------------------

def generate_synthetic_200_corpus(seed: int = 20261125) -> Dict[str, Any]:
    """
    Gercekci 200-bulguluk sentetik alplab_ortak_bulgular korpusu + gomulu exact
    sunflower aileleri ve exact Toulouse engellenmislik cizgesi uretir.

    Yapi:
      - 12 gomulu exact k-aycicegi (toplam 60 bulgu, k = 3..7, beklenen budanan = 48)
      - 140 ayciceksiz bulgu (100 tekil + 20 adet k-1=2 cift = 40 bulgu)
      - Gomulu celiski cizgesi (32 dugum, 46 kenar, 5 blok):
          * Blok 0: Iki-parcali agac/cift (Phi_0 = 0)
          * Blok 1: Ucgen K_3 (Phi_0 = 1)
          * Blok 2: Besgen C_5 (Phi_0 = 1, ucgensiz)
          * Blok 3: Wagner V_8 (Phi_0 = 2, ucgensiz, Erdos #23 N=8 ekstremal)
          * Blok 4: C_5[I_2] sisirmesi (Phi_0 = 4, ucgensiz, Erdos #23 N=10 ekstremal)
        Toplam kesin Phi_0(G) = 0 + 1 + 1 + 2 + 4 = 8, maxcut(G) = 46 - 8 = 38.
    """
    findings: List[FindingRecord] = []

    # 12 gercekci AlpLab cekirdegi -> toplam 60 yaprak (12 * 5 = 60, budanan = 48)
    planted_sunflowers = [
        ("KONU-3-LEAN4-META-DENETIM", "yazilim", ("proofgate", "inv08", "soundness"), 5),
        ("KONU-3-LEAN4-META-DENETIM", "yazilim", ("debruijn", "lexer", "trust0"), 4),
        ("KONU-5-FIRSAT-KOTA-RADARI", "ekonomi", ("hibe", "portfoy", "knapsack"), 7),
        ("KONU-5-FIRSAT-KOTA-RADARI", "yazilim", ("quotagate", "retry429", "weibull"), 5),
        ("KONU-8-ALPESS-PLATFORM", "muhendislik", ("alpess", "navlun", "deadhead"), 6),
        ("KONU-8-ALPESS-PLATFORM", "yazilim", ("pgbouncer", "rls_initplan", "outbox"), 4),
        ("KONU-9-MIKRO-URUN-KREDI", "yazilim", ("muhur", "rfc6962", "watermark"), 6),
        ("KONU-9-MIKRO-URUN-KREDI", "yazilim", ("tastetwin", "csp3_vitrin", "creditlock"), 5),
        ("KONU-1-BULUT-OS", "yazilim", ("pitstop", "fencing", "watchdog"), 5),
        ("KONU-4-ONCU-ARGE-VITRIN", "yapay_zeka", ("vendi_vs2", "egraph", "ast_synth"), 5),
        ("KONU-10-DISIPLINLERARASI-BILIM", "matematik", ("kuramoto", "antiphase", "syncgate"), 4),
        ("KONU-12-NE-YAPABILIRIZ", "matematik", ("erdos_oeis", "witness", "small_n"), 4),
    ]
    assert sum(sz for _, _, _, sz in planted_sunflowers) == 60

    durum_cycle = ("DOGRULANDI", "ADAY", "ZAYIF", "ADAY", "ZAYIF", "DOGRULANDI", "ZAYIF")
    ajan_pool = ("research_agent_01", "kota_doldurucu", "agy", "web_worker_slot", "research_agent_02")
    idx = 0
    expected_pruned = 0

    for sf_i, (konu, alan, core, size) in enumerate(planted_sunflowers):
        expected_pruned += (size - 1)
        for p_i in range(size):
            idx += 1
            bid = f"SYN-{idx:03d}"
            # Her yaprak kesin ayrik iki ozgun atom alir
            petal_a = f"petal_{sf_i:02d}_{p_i:02d}_alpha"
            petal_b = f"wave_{sf_i:02d}_{p_i:02d}_beta"
            durum = durum_cycle[p_i % len(durum_cycle)]
            puan = round(9.0 - (p_i * 0.7) + ((seed + idx) % 3) * 0.1, 2)
            baslik = f"{' '.join(core)} {petal_a}"
            ozet = f"Deterministik sentetik bulgu {bid}: {' '.join(core)} uzerinde {petal_a} ve {petal_b} olcumu (0.0000 USD)."
            findings.append(
                FindingRecord(
                    bulgu_id=bid,
                    kaynak_ref=f"gorev:syn-{idx:03d}",
                    kaynak_ajan=ajan_pool[idx % len(ajan_pool)],
                    konu_id=konu,
                    alan=alan,
                    baslik=baslik,
                    ozet=ozet,
                    kanit=f"tools/syn_sf_{sf_i:02d}_{p_i:02d}.py",
                    puan=puan,
                    durum=durum,
                    denetim_notu=json.dumps({"kanit": 2, "yenilik": 2, "kullanilabilirlik": 2, "mukerrer_mi": p_i > 0}),
                    kullanildi=0,
                    created_at=f"2026-11-25T{10 + (idx // 60):02d}:{idx % 60:02d}:00+00:00",
                    updated_at=f"2026-11-25T{10 + (idx // 60):02d}:{idx % 60:02d}:00+00:00",
                    explicit_atoms=frozenset(core) | {petal_a, petal_b},
                )
            )

    # 20 adet k=2 cifti (40 bulgu): k=3 esiginin altinda oldugundan HICBIRI budanmamali!
    konu_list = (
        "KONU-1-BULUT-OS",
        "KONU-3-LEAN4-META-DENETIM",
        "KONU-4-ONCU-ARGE-VITRIN",
        "KONU-5-FIRSAT-KOTA-RADARI",
        "KONU-8-ALPESS-PLATFORM",
        "KONU-9-MIKRO-URUN-KREDI",
        "KONU-10-DISIPLINLERARASI-BILIM",
        "KONU-12-NE-YAPABILIRIZ",
    )
    for pair_i in range(20):
        konu = konu_list[pair_i % len(konu_list)]
        pair_core = (f"subthresh_core_{pair_i:02d}a", f"subthresh_core_{pair_i:02d}b")
        for j in range(2):
            idx += 1
            bid = f"SYN-{idx:03d}"
            uniq = f"subthresh_leaf_{pair_i:02d}_{j}"
            findings.append(
                FindingRecord(
                    bulgu_id=bid,
                    kaynak_ref=f"gorev:syn-{idx:03d}",
                    kaynak_ajan=ajan_pool[idx % len(ajan_pool)],
                    konu_id=konu,
                    alan="yazilim",
                    baslik=f"{pair_core[0]} {pair_core[1]} {uniq}",
                    ozet=f"Esik-alti ikili bulgu {bid} (0.0000 USD).",
                    kanit=f"tools/syn_pair_{pair_i:02d}_{j}.py",
                    puan=7.5 - j * 0.5,
                    durum="DOGRULANDI" if j == 0 else "ADAY",
                    created_at=f"2026-11-25T13:{idx % 60:02d}:00+00:00",
                    updated_at=f"2026-11-25T13:{idx % 60:02d}:00+00:00",
                    explicit_atoms=frozenset(pair_core) | {uniq, f"aux_{idx:03d}"},
                )
            )

    # 100 tamamen ozgun tekil bulgu (sifir cekirdek cakismasi -> %0 yanlis budama testi)
    for s_i in range(100):
        idx += 1
        bid = f"SYN-{idx:03d}"
        konu = konu_list[s_i % len(konu_list)]
        atoms = frozenset({f"uniq_{s_i:03d}_x", f"uniq_{s_i:03d}_y", f"uniq_{s_i:03d}_z", f"uniq_{s_i:03d}_w"})
        findings.append(
            FindingRecord(
                bulgu_id=bid,
                kaynak_ref=f"gorev:syn-{idx:03d}",
                kaynak_ajan=ajan_pool[idx % len(ajan_pool)],
                konu_id=konu,
                alan="bilim",
                baslik=f"Ozgun kesif {bid} uniq_{s_i:03d}_x",
                ozet=f"Tekil ozgun arastirma kaydi {bid} (0.0000 USD).",
                kanit=f"tools/syn_uniq_{s_i:03d}.py",
                puan=round(6.0 + (s_i % 4) * 0.8, 2),
                durum="DOGRULANDI" if s_i % 2 == 0 else "ADAY",
                created_at=f"2026-11-25T{14 + (s_i // 60):02d}:{s_i % 60:02d}:00+00:00",
                updated_at=f"2026-11-25T{14 + (s_i // 60):02d}:{s_i % 60:02d}:00+00:00",
                explicit_atoms=atoms,
            )
        )

    assert len(findings) == 200

    # Gomulu exact Toulouse celiski cizgesi (SYN-101..SYN-132 uzerinde 5 bagimsiz blok)
    edges: List[HypothesisEdge] = []

    # Blok 0: Iki-parcali C_4 + dal (6 dugum, 6 kenar, Phi_0 = 0)
    b0 = [f"SYN-{i:03d}" for i in range(101, 107)]
    for u, v in [(b0[0], b0[1]), (b0[1], b0[2]), (b0[2], b0[3]), (b0[3], b0[0]), (b0[0], b0[4]), (b0[4], b0[5])]:
        edges.append(HypothesisEdge(u, v, sign=-1, weight=1.0, rel_type="negatif_kanit", evidence="bipartite_block"))

    # Blok 1: Ucgen K_3 (3 dugum, 3 kenar, Phi_0 = 1)
    b1 = [f"SYN-{i:03d}" for i in range(107, 110)]
    for u, v in itertools.combinations(b1, 2):
        edges.append(HypothesisEdge(u, v, sign=-1, weight=1.0, rel_type="negatif_kanit", evidence="triangle_block"))

    # Blok 2: Besgen C_5 (5 dugum, 5 kenar, Phi_0 = 1, ucgensiz)
    b2 = [f"SYN-{i:03d}" for i in range(110, 115)]
    for i in range(5):
        edges.append(HypothesisEdge(b2[i], b2[(i + 1) % 5], sign=-1, weight=1.0, rel_type="negatif_kanit", evidence="c5_block"))

    # Blok 3: Wagner V_8 (8 dugum, 12 kenar, Phi_0 = 2, ucgensiz, b_Delta(8) = 2)
    b3 = [f"SYN-{i:03d}" for i in range(115, 123)]
    for i in range(8):
        edges.append(HypothesisEdge(b3[i], b3[(i + 1) % 8], sign=-1, weight=1.0, rel_type="negatif_kanit", evidence="wagner_v8_rim"))
    for i in range(4):
        edges.append(HypothesisEdge(b3[i], b3[i + 4], sign=-1, weight=1.0, rel_type="negatif_kanit", evidence="wagner_v8_diag"))

    # Blok 4: C_5[I_2] sisirmesi (10 dugum, 20 kenar, Phi_0 = 4, ucgensiz, b_Delta(10) = 4)
    b4 = [f"SYN-{i:03d}" for i in range(123, 133)]
    groups = [(b4[2 * i], b4[2 * i + 1]) for i in range(5)]
    for i in range(5):
        g_a = groups[i]
        g_b = groups[(i + 1) % 5]
        for u in g_a:
            for v in g_b:
                edges.append(HypothesisEdge(u, v, sign=-1, weight=1.0, rel_type="negatif_kanit", evidence="c5_blowup_i2"))

    return {
        "findings": findings,
        "edges": edges,
        "ground_truth": {
            "total_findings": 200,
            "planted_sunflower_clusters": 12,
            "expected_pruned_count": expected_pruned,
            "expected_surviving_count": 200 - expected_pruned,
            "expected_phi0": 8.0,
            "expected_maxcut": float(len(edges) - 8),
            "expected_edge_count": len(edges),
        },
    }


def build_live_pg17_real_schema_snapshot() -> Dict[str, Any]:
    """
    Canli PG17 public.alplab_ortak_bulgular tablosundan (2026-11-25, 124 satir) alinan
    birebir 14-kolonlu gercek sema orneklemi (24 temsilci kayit + gercek celiski ciftleri).
    """
    rows: List[FindingRecord] = [
        # KONU-3 ProofGate INV-08 sunflower ailesi (4 yaprak: v7, v8, v9, v10)
        FindingRecord("BLG-2FF06701", "KONU-3-LEAN4-META-DENETIM", "yazilim", "ProofGate INV-08 v10.0.0: D1-D6 Aciklari, Lean 4.36.0 (31/31) + COR_4/Exact-LDLT", "v9.0.0 uzerindeki 6 acik COR_4 besgen politop exact Fraction LDLT canli dogrulandi.", 9.0, "DOGRULANDI", "gorev:2ff06701", "research_agent_01", "formal/ProofGateInv08V10HardenedSoundness.lean", "{}", 0, "2026-11-23T18:45:55+00:00", "2026-11-23T18:45:55+00:00"),
        FindingRecord("BLG-425BB3C6", "KONU-3-LEAN4-META-DENETIM", "yazilim", "ProofGate INV-08 v9.0.0 Hardened Soundness (Lean 4.36.0 24 Teorem + Alpha-CAS)", "ProofGateInv08V9HardenedSoundness.lean Lean 4.36.0 bench36 F1=1.0 PBT 25000 verdi.", 9.0, "DOGRULANDI", "gorev:425bb3c6", "research_agent_01", "formal/ProofGateInv08V9HardenedSoundness.lean", "{}", 0, "2026-11-23T17:47:53+00:00", "2026-11-23T17:47:53+00:00"),
        FindingRecord("BLG-E8CCD06D", "KONU-3-LEAN4-META-DENETIM", "yazilim", "ProofGate INV-08 v8.0.0: Lean 4.36.0 (25/25 Teorem) + Genel-k Boros-Prekopa Cholesky", "ProofGateInv08V8HardenedSoundness.lean 25/25 temiz Boros-Prekopa Pivoted Cholesky PSD.", 8.5, "DOGRULANDI", "gorev:e8ccd06d", "research_agent_01", "formal/ProofGateInv08V8HardenedSoundness.lean", "{}", 0, "2026-11-23T17:02:05+00:00", "2026-11-23T17:02:05+00:00"),
        FindingRecord("BLG-B9F61511", "KONU-3-LEAN4-META-DENETIM", "yazilim", "ProofGate INV-08 v7.0.0 Lean 4.36.0 ve Cholesky/Moment-LP Dogrulamasi", "ProofGateInv08V7HardenedSoundness.lean 0 sorry Moment-LP 191 bps diskte dogrulandi.", 8.0, "DOGRULANDI", "gorev:b9f61511", "research_agent_01", "formal/ProofGateInv08V7HardenedSoundness.lean", "{}", 0, "2026-11-23T16:00:37+00:00", "2026-11-23T16:00:37+00:00"),
        # KONU-5 Hibe portfoy & kota motoru sunflower ailesi (4 yaprak)
        FindingRecord("BLG-65289985", "KONU-5-FIRSAT-KOTA-RADARI", "ekonomi", "Hibe portfoyu ve kota motoru", "Basvuru suresine ve sinav takvimine gore hibe portfoyleri onerildi. OpenRouter Groq motoru.", 7.0, "ADAY", "gorev:65289985", "kota_doldurucu", "tools/is_fon_157_nov25_microgrant_hf_gcp_openrouter_groq_engine.py", "{}", 0, "2026-11-25T13:25:42+00:00", "2026-11-25T13:25:42+00:00"),
        FindingRecord("BLG-D66F0204", "KONU-5-FIRSAT-KOTA-RADARI", "yapay_zeka", "Kota tahmini ve hibe likiditesi denetimi", "Hibe firsatlari erisim kosullarina gore ayrildi. Dual regime NIM delta prefix guard.", 6.0, "ADAY", "gorev:d66f0204", "kota_doldurucu", "tools/is_fon_150_nim_delta_and_dual_regime_grant_guard.py", "{}", 0, "2026-11-25T11:28:28+00:00", "2026-11-25T11:28:28+00:00"),
        FindingRecord("BLG-ED6D75C8", "KONU-5-FIRSAT-KOTA-RADARI", "yazilim", "Hibe portfoyu ve sureli kredi yonlendirme modeli taslagi", "Hibe basvurulari icin sirt cantasi portfoy modeli ve sureli bulut kredi yonlendirici.", 5.0, "ZAYIF", "gorev:ed6d75c8", "kota_doldurucu", "gorev:ed6d75c8", "{}", 0, "2026-11-24T10:00:00+00:00", "2026-11-24T10:00:00+00:00"),
        FindingRecord("BLG-2AC889B2", "KONU-5-FIRSAT-KOTA-RADARI", "yapay_zeka", "Hibe Takvimi ve Model Gecisi", "Hibe takvimi guncellendi sirkete bagli bakiye ayrildi kapanacak modeller yonlendirme.", 6.0, "ADAY", "gorev:2ac889b2", "kota_doldurucu", "gorev:2ac889b2", "{}", 0, "2026-11-25T15:26:12+00:00", "2026-11-25T15:26:12+00:00"),
        # KONU-9 Muhur / TasteTwin kredi vitrin ailesi (4 yaprak)
        FindingRecord("BLG-AC150320", "KONU-9-MIKRO-URUN-KREDI", "yazilim", "Muhur ve Tahta Lab icin statik vitrin ve kredi sayaci", "Mikro-USD tamsayi defteriyle rezerv tahsil iade akisi harcama tavani devre kesici.", 7.0, "DOGRULANDI", "gorev:ac150320", "web_gemini", "tools/konu9_muhur_tahtalab_vitrin_sayac_verifier.py", "{}", 0, "2026-11-24T09:00:00+00:00", "2026-11-24T09:00:00+00:00"),
        FindingRecord("BLG-E1115CEE", "KONU-9-MIKRO-URUN-KREDI", "yazilim", "Muhur dogrulayicisinda alti kusur", "Muhur dogrulayicisinda alti kusur ve yuzde 80 harcama kilidi RFC6962 inclusion.", 6.0, "ADAY", "gorev:e1115cee", "kota_doldurucu", "tools/konu9_muhur_v2_rfc6962_vitrin_verifier.py", "{}", 0, "2026-11-25T13:25:34+00:00", "2026-11-25T13:25:34+00:00"),
        FindingRecord("BLG-2BB8F570", "KONU-9-MIKRO-URUN-KREDI", "yazilim", "TasteTwin kredi siniri ve tarayici guvenligi", "TasteTwin prototipinde alti kusur kredi siniri CSP3 2PCR forensic verifier.", 6.0, "ADAY", "gorev:2bb8f570", "kota_doldurucu", "tools/konu9_tastetwin_v36_csp3_forensic_verifier.py", "{}", 0, "2026-11-25T14:25:49+00:00", "2026-11-25T14:25:49+00:00"),
        FindingRecord("BLG-FAB955FC", "KONU-9-MIKRO-URUN-KREDI", "yazilim", "Muhur ve TasteTwin icin yayin paketi", "Cloudflare Workers icin kod dogrulama araci ve yayin adimlari hazirlandi.", 7.0, "ADAY", "gorev:fab955fc", "web_worker_slot", "gorev:fab955fc", "{}", 0, "2026-11-25T13:25:28+00:00", "2026-11-25T13:25:28+00:00"),
        # Gercek celiski ciftleri (Erdos #184, Erdos #36, Kredi Tavan 53710 vs 270 USD, Kiosk)
        FindingRecord("BLG-ERDOS184X", "KONU-10-DISIPLINLERARASI-BILIM", "matematik", "Erdos #184 teslimi yanlis probleme ait", "Hesaplanan diziler ayrik alt-kume toplamli ama Erdos #184 graf donguler problemidir, yanlis eslesme.", 2.0, "CURUTULDU", "gorev:erdos184x", "research_agent_03", "", "{}", 0, "2026-11-23T12:00:00+00:00", "2026-11-23T12:00:00+00:00"),
        FindingRecord("BLG-6CDAE132", "KONU-5-FIRSAT-KOTA-RADARI", "matematik", "Erdos 184: ayrik alt kume toplamli kumelerin sayi dizileri", "1..n icindeki tum alt kume toplamlari farkli kumelerin en buyuk boyutu Erdos 184 icin hesaplandi.", 3.0, "ZAYIF", "gorev:6cdae132", "kota_doldurucu", "gorev:6cdae132", "{}", 0, "2026-11-23T11:00:00+00:00", "2026-11-23T11:00:00+00:00"),
        FindingRecord("BLG-ERDOS36X", "KONU-10-DISIPLINLERARASI-BILIM", "matematik", "Erdos #36 teslimindeki A006561 eslesmesi yanlis", "A006561 duzgun n-genin kosegen kesisim sayisidir minimum ortusme problemiyle ilgisi yok.", 1.0, "CURUTULDU", "gorev:erdos36x", "ajan", "", "{}", 0, "2026-11-23T10:00:00+00:00", "2026-11-23T10:00:00+00:00"),
        FindingRecord("BLG-518D1B69", "KONU-5-FIRSAT-KOTA-RADARI", "yazilim", "Fon manifestolarinda Go dogrulayici hatalari ve kredi mutabakati", "floss-fund go-funding-json 5 GUID hatasi ve kredi defteri 11 satir mutabik kilindi.", 6.0, "CURUTULDU", "gorev:518d1b69", "kota_doldurucu", "https://github.com/alp-oss/proofgate", "{}", 0, "2026-11-24T08:00:00+00:00", "2026-11-24T08:00:00+00:00"),
        FindingRecord("BLG-3E4BB7AA", "KONU-5-FIRSAT-KOTA-RADARI", "yapay_zeka", "Ucretsiz model kotalari ve hibe kayitlarinda hata duzeltmeleri", "Celisen iki denetleyicideki hatalar ayiklanmis NLnet Commons Fund kapandigi ZeroGPU 38 kat sisirildi.", 5.0, "ZAYIF", "gorev:3e4bb7aa", "kota_doldurucu", "https://docs.sambanova.ai", "{}", 0, "2026-11-24T07:00:00+00:00", "2026-11-24T07:00:00+00:00"),
        FindingRecord("BLG-B0DA7F31", "KONU-9-MIKRO-URUN-KREDI", "yazilim", "Bulut kredi basvurularinin katmanli credit_grants tablosuna aktarilmasi", "Canli tablo 53710 USD kullanilabilir kredi gosteriyor gercek korumali tavan 270 USD tespit edildi.", 6.0, "ADAY", "gorev:b0da7f31", "web_worker_slot", "gorev:b0da7f31", "{}", 0, "2026-11-24T06:00:00+00:00", "2026-11-24T06:00:00+00:00"),
        FindingRecord("BLG-73005FF5", "KONU-5-FIRSAT-KOTA-RADARI", "ekonomi", "IS-FON-75 Sirketsiz Hibe/OSS Kredi Arbitraji ve PG17 Escrow Denetimi", "PG17 escrow defterinde 50910/53710 USD hayalet bakiye ayrildi 12s Knapsack 4811.49 USD LCB-90.", 8.5, "DOGRULANDI", "gorev:73005ff5", "research_agent_04", "tools/is_fon_75_nov25_dec17_jan3_arbitrage_and_escrow_auditor.py", "{}", 0, "2026-11-23T18:04:04+00:00", "2026-11-23T18:04:04+00:00"),
        FindingRecord("BLG-50E5B47E", "KONU-9-MIKRO-URUN-KREDI", "urun", "Robotik tatli kiosku maliyet ve porsiyon simulatoru", "Onceki bulgudaki 6.50 USD fiyat 196 porsiyon basabas degerleri curutuldu.", 5.0, "CURUTULDU", "gorev:50e5b47e", "web_gemini", "tools/konu9_kiosk_simulator_verifier.py", "{}", 0, "2026-11-24T05:00:00+00:00", "2026-11-24T05:00:00+00:00"),
        # Ozgun dogrulanmis bulgular (korunmali)
        FindingRecord("BLG-D4A9869E", "KONU-3-LEAN4-META-DENETIM", "yazilim", "8-Katmanli Deterministik Konu-Sapmasi Pre-Submit On-Kapisi", "MAHO-RECON DLQ kok nedeni cozuldu konu_sapmasi_pre_submit_guard 100 recall 0 FPR.", 9.5, "DOGRULANDI", "gorev:d4a9869e", "research_agent_05", "tools/konu_sapmasi_pre_submit_guard.py", "{}", 0, "2026-11-23T17:47:53+00:00", "2026-11-23T17:47:53+00:00"),
        FindingRecord("BLG-E6262570", "KONU-8-ALPESS-PLATFORM", "yazilim", "PG17 Alpess RLS InitPlan + PgBouncer prepare_threshold Kapi Denetcisi", "Canli PG17 19 alpess tablosunda 0 FORCE RLS ve 3 politikasiz aktif RLS tuzagi dogrulandi.", 9.0, "DOGRULANDI", "gorev:e6262570", "research_agent_06", "tools/alpess_pg17_rls_pgbouncer_gate.py", "{}", 0, "2026-11-23T18:04:04+00:00", "2026-11-23T18:04:04+00:00"),
        FindingRecord("BLG-D6CBDEEA", "KONU-2-NORONIK-HAFIZA", "yazilim", "Noronik Zincir Denetcisi ve PG17 TIER-A/B/C Tespiti", "PG17 alplab_episodes 1150 eskitilmis 0 dongu maks derinlik 89 TIER_A 5 birebir kopya.", 8.5, "DOGRULANDI", "gorev:d6cbdeea", "research_agent_06", "tools/noronik_zincir_denetci.py", "{}", 0, "2026-11-24T12:00:00+00:00", "2026-11-24T12:00:00+00:00"),
        FindingRecord("BLG-F6FA888C", "KONU-8-ALPESS-PLATFORM", "hukuk", "Turkiye Karayolu Yuk Sigortasi CMR/FFL/TTK ve SDR Limitleri", "TCMB SDR/USD 8.33 SDR/kg = 11.19 USD TIO m.19 FFL TTK 1478/1472 dogrulandi.", 9.0, "DOGRULANDI", "gorev:f6fa888c", "research_agent_07", "veri/derin_arastirma/DR-KONU-ALPESS-09.md", "{}", 0, "2026-11-23T19:03:09+00:00", "2026-11-23T19:03:09+00:00"),
    ]

    # Canli PG17 gozlenen acik celiski kenarlari (orn. 3-yonlu kredi defteri celiskisi + Erdos #184)
    explicit_edges = [
        HypothesisEdge("BLG-ERDOS184X", "BLG-6CDAE132", sign=-1, weight=1.0, rel_type="negatif_kanit", evidence="Erdos #184 problem tanimi celiskisi"),
        HypothesisEdge("BLG-518D1B69", "BLG-3E4BB7AA", sign=-1, weight=1.0, rel_type="negatif_kanit", evidence="Hibe/fon manifestosu tutarsizligi"),
        HypothesisEdge("BLG-3E4BB7AA", "BLG-73005FF5", sign=-1, weight=1.0, rel_type="negatif_kanit", evidence="Escrow/kota bakiye farki"),
        HypothesisEdge("BLG-518D1B69", "BLG-73005FF5", sign=-1, weight=1.0, rel_type="negatif_kanit", evidence="Kredi defteri 11-satir mutabakat vs hayalet bakiye"),
    ]
    return {
        "findings": rows,
        "edges": explicit_edges,
        "live_pg17_total_rows": 124,
        "live_pg17_durum_dist": {"DOGRULANDI": 56, "ZAYIF": 42, "ADAY": 22, "CURUTULDU": 4},
    }


def run_comparative_benchmark(seed: int = 20261125) -> Dict[str, Any]:
    """
    200-bulgu sentetik korpus ile canli PG17 gercek sema orneklemini SQLite3 uzerinden
    uc-uca calistirir, sema paritesini, SunflowerMemoryPruner ve FrustrationCutAuditor
    sonuclarini karsilastirmali raporlar.
    """
    syn = generate_synthetic_200_corpus(seed=seed)
    real = build_live_pg17_real_schema_snapshot()

    pruner = SunflowerMemoryPruner(k=3, min_core_size=2, retain_petals=1, group_by_konu=True)
    auditor = FrustrationCutAuditor(exact_max_block_nodes=20)

    # 1. Sentetik 200-bulgu SQLite round-trip
    syn_conn = sqlite3.connect(":memory:")
    syn_store = MemoryGraphStore(syn_conn, dialect="sqlite")
    syn_store.init_sqlite_schema()
    syn_parity = syn_store.verify_schema_parity()
    syn_store.upsert_findings(syn["findings"])
    syn_store.insert_temporal_edges(syn["edges"])
    syn_loaded_findings = syn_store.load_findings(limit=500)
    # Explicit atom haritasini sentetik kesin dogrulama icin koru
    atom_lookup = {f.bulgu_id: f.explicit_atoms for f in syn["findings"]}
    syn_findings_exact = [
        FindingRecord(
            **{**f.__dict__, "explicit_atoms": atom_lookup.get(f.bulgu_id)}
        )
        for f in syn_loaded_findings
    ]
    syn_sf_report = pruner.prune(syn_findings_exact)
    syn_store.save_sunflower_report(syn_sf_report)

    syn_loaded_edges = syn_store.load_contradiction_edges()
    syn_meta = {f.bulgu_id: f for f in syn_loaded_findings}
    syn_frust_report = auditor.audit([f.bulgu_id for f in syn_loaded_findings], syn_loaded_edges, finding_meta=syn_meta)
    syn_store.save_frustration_report(syn_frust_report)

    # 2. Gercek PG17 semali orneklem SQLite round-trip
    real_conn = sqlite3.connect(":memory:")
    real_store = MemoryGraphStore(real_conn, dialect="sqlite")
    real_store.init_sqlite_schema()
    real_parity = real_store.verify_schema_parity()
    real_store.upsert_findings(real["findings"])
    real_store.insert_temporal_edges(real["edges"])
    real_loaded_findings = real_store.load_findings(limit=500)
    real_sf_report = pruner.prune(real_loaded_findings)
    real_store.save_sunflower_report(real_sf_report)

    real_explicit_edges = real_store.load_contradiction_edges()
    real_combined_edges = auditor.extract_contradiction_edges_from_findings(real_loaded_findings, real_explicit_edges)
    real_meta = {f.bulgu_id: f for f in real_loaded_findings}
    real_frust_report = auditor.audit([f.bulgu_id for f in real_loaded_findings], real_combined_edges, finding_meta=real_meta)
    real_store.save_frustration_report(real_frust_report)

    syn_conn.close()
    real_conn.close()

    return {
        "benchmark_version": "1.0.0",
        "paket_kodu": "VIZYON-ICRA-SUNFLOWER-MEMORY-PRUNER-20261125",
        "cost_usd": 0.0,
        "schema_parity": {
            "synthetic_sqlite": syn_parity["ortak_bulgular_exact_match"] and syn_parity["temporal_edges_exact_match"],
            "real_snapshot_sqlite": real_parity["ortak_bulgular_exact_match"] and real_parity["temporal_edges_exact_match"],
            "ortak_bulgular_col_count": len(ORTAK_BULGULAR_COLUMNS),
            "temporal_edges_col_count": len(TEMPORAL_EDGES_COLUMNS),
        },
        "synthetic_200": {
            "ground_truth": syn["ground_truth"],
            "sunflower": syn_sf_report.to_dict(),
            "frustration": syn_frust_report.to_dict(),
        },
        "real_pg17_schema_sample": {
            "live_pg17_total_rows": real["live_pg17_total_rows"],
            "sample_rows": len(real["findings"]),
            "sunflower": real_sf_report.to_dict(),
            "frustration": real_frust_report.to_dict(),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="AlpLab SunflowerMemoryPruner & FrustrationCutAuditor CLI"
    )
    parser.add_argument("--benchmark", action="store_true", help="200-bulgu sentetik + gercek sema karsilastirmasini calistir")
    parser.add_argument("--seed", type=int, default=20261125, help="Deterministik sentetik tohum")
    args = parser.parse_args()

    if args.benchmark or True:
        result = run_comparative_benchmark(seed=args.seed)
        out_json = json.dumps(result, indent=2, ensure_ascii=True)
        assert check_zero_dollar_symbol(out_json), "KURAL-3 violation detected"
        print(out_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
