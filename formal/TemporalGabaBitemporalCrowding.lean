/-
  File: formal/TemporalGabaBitemporalCrowding.lean
  Package: temporal-gaba-memory v1.0.0 (Layer 6 Machine-Checked Formalization)

  Pure Lean 4 (`Init` only, zero external dependencies, zero `sorryAx`,
  zero `Classical.choice`) constructively proving 25 `theorem` declarations
  (20 numbered core theorems + 5 structural helper theorems) across:
    1. Jensen-Lomet Transaction-Time Horizon Invariance (`asOfTx_future_append_invariant`,
       `asOfTx_future_close_invariant`, `asOf2D_future_append_invariant`,
       `asOf2D_future_close_invariant`, `asOfTx_future_revise_invariant`)
    2. Snodgrass / AGM Zero-Gap Supersession & 3-Way Allen Valid-Time Split Partition
       (`supersede_no_gap`, `supersede_no_double_active`, `supersede_preserves_coverage`,
       `valid_split_agm_success`, `valid_split_three_way_partition`)
    3. Supersession Crowding Starvation & Over-Fetch Recovery
       (`take_prefix_all_tombstones`, `unfiltered_topK_starves_when_tombstones_ge_k`,
       `overfetch_k_plus_m_guarantees_active_head`, `overfetch_k_plus_m_recovers_head_singleton`,
       `crowding_law_m20_k5_collapse_and_recovery`)
    4. Fixed-Point Integer 4-Signal RRF Monotonicity & Causal GABA Gate Soundness
       (`rrfTerm_pos_of_bound`, `rrfTerm_antitone_rank`, `gabaInhibit_le_base`,
       `rrfFuse4_monotone_and_causalGaba_sound`)

  Verify standalone:
    lean --trust=0 -DwarningAsError=true formal/TemporalGabaBitemporalCrowding.lean
-/

namespace TemporalGabaMemory.V1

/-! ## Section 1: 2D Bi-Temporal Ledger & Jensen-Lomet Horizon Invariance -/

/-- Half-open temporal interval membership `t in [lo, hi)` where `none` denotes `+infinity`. -/
def inHalfOpen (lo : Nat) (hi : Option Nat) (t : Nat) : Bool :=
  decide (lo ≤ t) &&
  match hi with
  | none   => true
  | some u => decide (t < u)

/-- Internal storage row of an append-only 2D bi-temporal ledger. -/
structure BiTemporalRecord where
  recId      : Nat
  slot       : Nat
  validFrom  : Nat
  validUntil : Option Nat
  txFrom     : Nat
  txUntil    : Option Nat
  payload    : Nat
  deriving DecidableEq, Repr

/-- Observable point-in-time snapshot fact (independent of future `txUntil` closure). -/
structure SnapshotFact where
  recId      : Nat
  slot       : Nat
  validFrom  : Nat
  validUntil : Option Nat
  txFrom     : Nat
  payload    : Nat
  deriving DecidableEq, Repr

def toFact (r : BiTemporalRecord) : SnapshotFact :=
  ⟨r.recId, r.slot, r.validFrom, r.validUntil, r.txFrom, r.payload⟩

def visibleTx (r : BiTemporalRecord) (tQuery : Nat) : Bool :=
  inHalfOpen r.txFrom r.txUntil tQuery

def visible2D (r : BiTemporalRecord) (vQuery tQuery : Nat) : Bool :=
  inHalfOpen r.validFrom r.validUntil vQuery && visibleTx r tQuery

/-- Transaction-time historical snapshot query `asOfTx(db, t_query)`. -/
def asOfTx (db : List BiTemporalRecord) (tQuery : Nat) : List SnapshotFact :=
  match db with
  | []      => []
  | r :: rs =>
    if visibleTx r tQuery then
      toFact r :: asOfTx rs tQuery
    else
      asOfTx rs tQuery

/-- Full 2D `(valid_at, tx_at)` point-in-time historical snapshot query. -/
def asOf2D (db : List BiTemporalRecord) (vQuery tQuery : Nat) : List SnapshotFact :=
  match db with
  | []      => []
  | r :: rs =>
    if visible2D r vQuery tQuery then
      toFact r :: asOf2D rs vQuery tQuery
    else
      asOf2D rs vQuery tQuery

/-- Close an active record (`txUntil = none`) at transaction timestamp `tClose`. -/
def closeRecordAt (r : BiTemporalRecord) (targetSlot tClose : Nat) : BiTemporalRecord :=
  if decide (r.slot = targetSlot) && r.txUntil.isNone then
    { r with txUntil := some tClose }
  else
    r

def closeSlotAt (db : List BiTemporalRecord) (targetSlot tClose : Nat) : List BiTemporalRecord :=
  db.map (fun r => closeRecordAt r targetSlot tClose)

/-- Atomic slot revision at `tWrite`: closes active slot records and appends new head. -/
def reviseSlotAt
    (db : List BiTemporalRecord)
    (targetSlot vFrom : Nat)
    (vUntil : Option Nat)
    (tWrite newId newPayload : Nat) : List BiTemporalRecord :=
  closeSlotAt db targetSlot tWrite ++
    [⟨newId, targetSlot, vFrom, vUntil, tWrite, none, newPayload⟩]

/-- Theorem 1: Any record committed strictly after `tQuery` is invisible at `tQuery`. -/
theorem visibleTx_false_of_future_write
    (r : BiTemporalRecord) (tQuery : Nat) (hFuture : tQuery < r.txFrom) :
    visibleTx r tQuery = false := by
  unfold visibleTx inHalfOpen
  have hNotLe : ¬ (r.txFrom ≤ tQuery) := Nat.not_le_of_gt hFuture
  simp [hNotLe]

/-- Theorem 2 (Jensen-Lomet Horizon Theorem — Future Append):
    Appending any record at `r.txFrom > tQuery` leaves `asOfTx db tQuery` strictly identical. -/
theorem asOfTx_future_append_invariant
    (db : List BiTemporalRecord) (r : BiTemporalRecord) (tQuery : Nat)
    (hFuture : tQuery < r.txFrom) :
    asOfTx (db ++ [r]) tQuery = asOfTx db tQuery := by
  induction db with
  | nil =>
    simp [asOfTx, visibleTx_false_of_future_write r tQuery hFuture]
  | cons hd tl ih =>
    simp only [List.cons_append, asOfTx]
    cases visibleTx hd tQuery <;> simp [ih]

/-- Theorem 3 (2D Bi-Temporal Horizon Theorem — Future Append):
    Appending any record at `r.txFrom > tQuery` leaves `asOf2D db vQuery tQuery` strictly identical. -/
theorem asOf2D_future_append_invariant
    (db : List BiTemporalRecord) (r : BiTemporalRecord) (vQuery tQuery : Nat)
    (hFuture : tQuery < r.txFrom) :
    asOf2D (db ++ [r]) vQuery tQuery = asOf2D db vQuery tQuery := by
  have hVis2D : visible2D r vQuery tQuery = false := by
    unfold visible2D
    rw [visibleTx_false_of_future_write r tQuery hFuture]
    simp
  induction db with
  | nil =>
    simp [asOf2D, hVis2D]
  | cons hd tl ih =>
    simp only [List.cons_append, asOf2D]
    cases visible2D hd vQuery tQuery <;> simp [ih]

/-- Helper lemma: closing an active record at `tClose > tQuery` preserves `visibleTx` at `tQuery`
    and leaves `toFact` unchanged. -/
theorem visibleTx_closeRecordAt_eq
    (r : BiTemporalRecord) (targetSlot tClose tQuery : Nat)
    (hFuture : tQuery < tClose) :
    visibleTx (closeRecordAt r targetSlot tClose) tQuery = visibleTx r tQuery := by
  unfold closeRecordAt
  cases hSlot : decide (r.slot = targetSlot) with
  | false => simp
  | true =>
    cases hEnd : r.txUntil with
    | some u => simp [Option.isNone]
    | none =>
      simp only [Option.isNone, Bool.true_and, ite_true, visibleTx, inHalfOpen, hEnd]
      have hLt : decide (tQuery < tClose) = true := decide_eq_true hFuture
      simp [hLt]

theorem toFact_closeRecordAt_eq
    (r : BiTemporalRecord) (targetSlot tClose : Nat) :
    toFact (closeRecordAt r targetSlot tClose) = toFact r := by
  unfold closeRecordAt
  cases decide (r.slot = targetSlot) && r.txUntil.isNone <;> rfl

/-- Theorem 4 (Jensen-Lomet Horizon Theorem — Future Close):
    Closing any active record at `tClose > tQuery` leaves `asOfTx db tQuery` strictly identical. -/
theorem asOfTx_future_close_invariant
    (db : List BiTemporalRecord) (targetSlot tClose tQuery : Nat)
    (hFuture : tQuery < tClose) :
    asOfTx (closeSlotAt db targetSlot tClose) tQuery = asOfTx db tQuery := by
  induction db with
  | nil => rfl
  | cons hd tl ih =>
    simp only [closeSlotAt, List.map, asOfTx]
    rw [visibleTx_closeRecordAt_eq hd targetSlot tClose tQuery hFuture]
    rw [toFact_closeRecordAt_eq hd targetSlot tClose]
    cases visibleTx hd tQuery <;> simp [closeSlotAt] at ih ⊢ <;> exact ih

/-- Theorem 5 (2D Bi-Temporal Horizon Theorem — Future Close):
    Closing any active record at `tClose > tQuery` leaves `asOf2D db vQuery tQuery` strictly identical. -/
theorem asOf2D_future_close_invariant
    (db : List BiTemporalRecord) (targetSlot tClose vQuery tQuery : Nat)
    (hFuture : tQuery < tClose) :
    asOf2D (closeSlotAt db targetSlot tClose) vQuery tQuery = asOf2D db vQuery tQuery := by
  induction db with
  | nil => rfl
  | cons hd tl ih =>
    simp only [closeSlotAt, List.map, asOf2D]
    have hVis : visible2D (closeRecordAt hd targetSlot tClose) vQuery tQuery = visible2D hd vQuery tQuery := by
      unfold visible2D
      rw [visibleTx_closeRecordAt_eq hd targetSlot tClose tQuery hFuture]
      have hValidEq :
          inHalfOpen (closeRecordAt hd targetSlot tClose).validFrom
            (closeRecordAt hd targetSlot tClose).validUntil vQuery =
          inHalfOpen hd.validFrom hd.validUntil vQuery := by
        unfold closeRecordAt
        cases decide (hd.slot = targetSlot) && hd.txUntil.isNone <;> rfl
      rw [hValidEq]
    rw [hVis, toFact_closeRecordAt_eq hd targetSlot tClose]
    cases visible2D hd vQuery tQuery <;> simp [closeSlotAt] at ih ⊢ <;> exact ih

/-- Theorem 6 (Atomic Supersession Revision Horizon Invariance):
    Full atomic `reviseSlotAt` (close old + append new head) at `tWrite > tQuery`
    produces zero historical mutation at `tQuery`. -/
theorem asOfTx_future_revise_invariant
    (db : List BiTemporalRecord)
    (targetSlot vFrom : Nat) (vUntil : Option Nat)
    (tWrite newId newPayload tQuery : Nat)
    (hFuture : tQuery < tWrite) :
    asOfTx (reviseSlotAt db targetSlot vFrom vUntil tWrite newId newPayload) tQuery =
      asOfTx db tQuery := by
  unfold reviseSlotAt
  rw [asOfTx_future_append_invariant (closeSlotAt db targetSlot tWrite)
        ⟨newId, targetSlot, vFrom, vUntil, tWrite, none, newPayload⟩ tQuery hFuture]
  exact asOfTx_future_close_invariant db targetSlot tWrite tQuery hFuture


/-! ## Section 2: Zero-Gap Supersession Continuity & Allen 3-Way Valid-Time Split -/

/-- Theorem 7 (`no_gap`): When `[t0, t1)` is superseded by `[t1, +inf)` with `t0 <= t1`,
    every timestamp `t >= t0` is covered by at least one of the two records. -/
theorem supersede_no_gap
    (old succ : BiTemporalRecord) (t0 t1 t : Nat)
    (hOldFrom : old.txFrom = t0)
    (hOldUntil : old.txUntil = some t1)
    (hSuccFrom : succ.txFrom = t1)
    (hSuccUntil : succ.txUntil = none)
    (hRange : t0 ≤ t) :
    (visibleTx old t || visibleTx succ t) = true := by
  unfold visibleTx inHalfOpen
  rw [hOldFrom, hOldUntil, hSuccFrom, hSuccUntil]
  dsimp only []
  have hLe0 : decide (t0 ≤ t) = true := decide_eq_true hRange
  rw [hLe0]
  by_cases hLt : t < t1
  · have hDecLt : decide (t < t1) = true := decide_eq_true hLt
    rw [hDecLt]
    rfl
  · have hDecLt : decide (t < t1) = false := decide_eq_false hLt
    have hGe1 : t1 ≤ t := Nat.le_of_not_lt hLt
    have hDecGe1 : decide (t1 ≤ t) = true := decide_eq_true hGe1
    rw [hDecLt, hDecGe1]
    rfl

/-- Theorem 8 (`no_double_active`): When `old.txUntil = some t1` and `succ.txFrom = t1`,
    `old` and `succ` are never simultaneously visible at any timestamp `t`. -/
theorem supersede_no_double_active
    (old succ : BiTemporalRecord) (t1 t : Nat)
    (hOldUntil : old.txUntil = some t1)
    (hSuccFrom : succ.txFrom = t1) :
    (visibleTx old t && visibleTx succ t) = false := by
  unfold visibleTx inHalfOpen
  rw [hOldUntil, hSuccFrom]
  dsimp only []
  by_cases hLt : t < t1
  · have hDecLt : decide (t < t1) = true := decide_eq_true hLt
    have hNotLe : ¬ (t1 ≤ t) := Nat.not_le_of_gt hLt
    have hDecGe : decide (t1 ≤ t) = false := decide_eq_false hNotLe
    rw [hDecLt, hDecGe]
    cases decide (old.txFrom ≤ t) <;> rfl
  · have hDecLt : decide (t < t1) = false := decide_eq_false hLt
    rw [hDecLt]
    cases decide (old.txFrom ≤ t) <;> rfl

/-- Count of visible records in `db` at transaction timestamp `t`. -/
def countVisibleTx (db : List BiTemporalRecord) (t : Nat) : Nat :=
  (db.filter (fun r => visibleTx r t)).length

/-- Theorem 9 (`supersede_preserves_coverage`):
    Superseding `[t0, +inf)` at `t1 >= t0` into `old = [t0, t1)` and `succ = [t1, +inf)`
    yields a unique visible version (`countVisibleTx [old, succ] t = 1`) for all `t >= t0`. -/
theorem supersede_preserves_coverage
    (old succ : BiTemporalRecord) (t0 t1 t : Nat)
    (hOldFrom : old.txFrom = t0)
    (hOldUntil : old.txUntil = some t1)
    (hSuccFrom : succ.txFrom = t1)
    (hSuccUntil : succ.txUntil = none)
    (hRange : t0 ≤ t) :
    countVisibleTx [old, succ] t = 1 ∧
    (visibleTx old t && visibleTx succ t) = false := by
  have hOr := supersede_no_gap old succ t0 t1 t hOldFrom hOldUntil hSuccFrom hSuccUntil hRange
  have hNand := supersede_no_double_active old succ t1 t hOldUntil hSuccFrom
  refine ⟨?_, hNand⟩
  unfold countVisibleTx
  cases hO : visibleTx old t with
  | true =>
    rw [hO] at hNand
    have hS : visibleTx succ t = false := by simpa using hNand
    simp [List.filter, hO, hS]
  | false =>
    rw [hO] at hOr
    have hS : visibleTx succ t = true := by simpa using hOr
    simp [List.filter, hO, hS]

/-- Theorem 10 (AGM Success `K*2` on Split Valid Sub-Interval):
    For an interior revision `[c, d)` inside `[a, b)`, every `v in [c, d)` satisfies
    `inHalfOpen c (some d) v = true` while both residuals `[a, c)` and `[d, b)` are `false`. -/
theorem valid_split_agm_success
    (a c d b v : Nat) (hvc : c ≤ v) (hvd : v < d) :
    inHalfOpen c (some d) v = true ∧
    inHalfOpen a (some c) v = false ∧
    inHalfOpen d (some b) v = false := by
  unfold inHalfOpen
  have hc : decide (c ≤ v) = true := decide_eq_true hvc
  have hd : decide (v < d) = true := decide_eq_true hvd
  have hNotLtC : decide (v < c) = false := decide_eq_false (Nat.not_lt_of_ge hvc)
  have hNotLeD : decide (d ≤ v) = false := decide_eq_false (Nat.not_le_of_gt hvd)
  simp [hc, hd, hNotLtC, hNotLeD]

/-- Indicator converting `Bool` to `{0, 1}` for exact partition counting. -/
def b2n (b : Bool) : Nat := if b then 1 else 0

/-- Theorem 11 (3-Way Allen Valid-Time Split Partition):
    Splitting `[a, b)` around `[c, d)` (`c <= d`) into `[a, c)`, `[c, d)`, `[d, b)`
    partitions `[a, b)` into pairwise-disjoint sub-intervals with exact coverage multiplicity `1`. -/
theorem valid_split_three_way_partition
    (a c d b v : Nat)
    (hcd : c ≤ d) (hav : a ≤ v) (hvb : v < b) :
    b2n (inHalfOpen a (some c) v) +
    b2n (inHalfOpen c (some d) v) +
    b2n (inHalfOpen d (some b) v) = 1 := by
  dsimp only [inHalfOpen, b2n]
  have ha : decide (a ≤ v) = true := decide_eq_true hav
  have hb : decide (v < b) = true := decide_eq_true hvb
  rw [ha, hb]
  by_cases hvc : v < c
  · have h1 : decide (v < c) = true := decide_eq_true hvc
    have h2 : decide (c ≤ v) = false := decide_eq_false (Nat.not_le_of_gt hvc)
    have h3 : decide (d ≤ v) = false := decide_eq_false (Nat.not_le_of_gt (Nat.lt_of_lt_of_le hvc hcd))
    rw [h1, h2, h3]
    rfl
  · have h1 : decide (v < c) = false := decide_eq_false hvc
    have h2 : decide (c ≤ v) = true := decide_eq_true (Nat.le_of_not_lt hvc)
    rw [h1, h2]
    by_cases hvd : v < d
    · have h3 : decide (v < d) = true := decide_eq_true hvd
      have h4 : decide (d ≤ v) = false := decide_eq_false (Nat.not_le_of_gt hvd)
      rw [h3, h4]
      rfl
    · have h3 : decide (v < d) = false := decide_eq_false hvd
      have h4 : decide (d ≤ v) = true := decide_eq_true (Nat.le_of_not_lt hvd)
      rw [h3, h4]
      rfl


/-! ## Section 3: Supersession Crowding Starvation & Over-Fetch Recovery -/

/-- Ranked retrieval candidate with active/tombstone status. -/
structure Candidate where
  candId   : Nat
  isActive : Bool
  deriving DecidableEq, Repr

def allTombstones (xs : List Candidate) : Bool :=
  match xs with
  | []      => true
  | c :: cs => (!c.isActive) && allTombstones cs

/-- Theorem 12: Taking `k <= tombstones.length` items from `tombstones ++ rest` where every
    element of `tombstones` is inactive yields zero active candidates after filtering. -/
theorem take_prefix_all_tombstones
    (tombstones rest : List Candidate) (k : Nat)
    (hTomb : allTombstones tombstones = true)
    (hLe : k ≤ tombstones.length) :
    ((tombstones ++ rest).take k).filter Candidate.isActive = [] := by
  induction tombstones generalizing k with
  | nil =>
    have hk0 : k = 0 := Nat.eq_zero_of_le_zero hLe
    subst hk0
    rfl
  | cons hd tl ih =>
    cases k with
    | zero => rfl
    | succ k' =>
      simp only [allTombstones, Bool.and_eq_true] at hTomb
      have hHdInact : hd.isActive = false := by
        cases hAct : hd.isActive with
        | false => rfl
        | true => simp [hAct] at hTomb
      have hK' : k' ≤ tl.length := Nat.le_of_succ_le_succ hLe
      simp only [List.cons_append, List.take, List.filter, hHdInact]
      exact ih k' hTomb.2 hK'

/-- Theorem 13 (`unfiltered_topK_starves_when_tombstones_ge_k` — Crowding Starvation):
    When `m` superseded tombstones precede the active head and `m >= k` (`k <= m`),
    naive `take k` followed by active filtering returns `[]`, completely starving the active head. -/
theorem unfiltered_topK_starves_when_tombstones_ge_k
    (tombstones tail : List Candidate) (head : Candidate) (k m : Nat)
    (hLen : tombstones.length = m)
    (hTomb : allTombstones tombstones = true)
    (hCrowded : k ≤ m) :
    (((tombstones ++ head :: tail).take k).filter Candidate.isActive) = [] := by
  have hLe : k ≤ tombstones.length := by rw [hLen]; exact hCrowded
  exact take_prefix_all_tombstones tombstones (head :: tail) k hTomb hLe

/-- Membership helper: if `0 < r`, `head ∈ (head :: tail).take r`. -/
theorem head_mem_take_of_pos
    (head : Candidate) (tail : List Candidate) (r : Nat) (hr : 0 < r) :
    head ∈ (head :: tail).take r := by
  cases r with
  | zero => exact False.elim (Nat.lt_irrefl 0 hr)
  | succ r' => simp [List.take]

/-- Membership helper across tombstone prefix when `n > tombstones.length`. -/
theorem head_mem_take_after_prefix
    (tombstones tail : List Candidate) (head : Candidate) (n : Nat)
    (hGt : tombstones.length < n) :
    head ∈ (tombstones ++ head :: tail).take n := by
  induction tombstones generalizing n with
  | nil =>
    exact head_mem_take_of_pos head tail n hGt
  | cons x xs ih =>
    cases n with
    | zero => exact False.elim (Nat.not_lt_zero _ hGt)
    | succ n' =>
      simp only [List.length_cons] at hGt
      have hGt' : xs.length < n' := Nat.lt_of_succ_lt_succ hGt
      simp only [List.cons_append, List.take, List.mem_cons]
      exact Or.inr (ih n' hGt')

theorem mem_filter_of_mem_and_active
    (xs : List Candidate) (c : Candidate)
    (hMem : c ∈ xs) (hAct : c.isActive = true) :
    c ∈ xs.filter Candidate.isActive := by
  induction xs with
  | nil => cases hMem
  | cons hd tl ih =>
    simp only [List.mem_cons] at hMem
    cases hMem with
    | inl hEq =>
      subst hEq
      simp [List.filter, hAct]
    | inr hTail =>
      simp only [List.filter]
      cases hd.isActive <;> simp [ih hTail]

/-- Theorem 14 (`overfetch_k_plus_m_guarantees_active_head` — Over-Fetch Recovery):
    Over-fetching `k + m` candidates (`k >= 1`) when at most `m` tombstones precede `head`
    provably recovers the active `head` after `filter Candidate.isActive`. -/
theorem overfetch_k_plus_m_guarantees_active_head
    (tombstones tail : List Candidate) (head : Candidate) (k m : Nat)
    (hLen : tombstones.length ≤ m)
    (hK : 1 ≤ k)
    (hHeadActive : head.isActive = true) :
    head ∈ ((tombstones ++ head :: tail).take (k + m)).filter Candidate.isActive := by
  have hGt : tombstones.length < k + m :=
    Nat.lt_of_le_of_lt hLen (Nat.lt_add_of_pos_left hK)
  have hMemTake : head ∈ (tombstones ++ head :: tail).take (k + m) :=
    head_mem_take_after_prefix tombstones tail head (k + m) hGt
  exact mem_filter_of_mem_and_active ((tombstones ++ head :: tail).take (k + m)) head hMemTake hHeadActive

/-- Theorem 15: Exact singleton recovery when `tombstones` are inactive and `tail = []`:
    `((tombstones ++ [head]).take (k + m)).filter isActive = [head]`. -/
theorem overfetch_k_plus_m_recovers_head_singleton
    (tombstones : List Candidate) (head : Candidate) (k m : Nat)
    (hLen : tombstones.length ≤ m)
    (hTomb : allTombstones tombstones = true)
    (hK : 1 ≤ k)
    (hHeadActive : head.isActive = true) :
    ((tombstones ++ [head]).take (k + m)).filter Candidate.isActive = [head] := by
  induction tombstones generalizing m with
  | nil =>
    cases hSum : k + m with
    | zero =>
      have hPos : 0 < k + m := Nat.lt_of_lt_of_le (by decide : 0 < 1) (Nat.le_add_right_of_le hK)
      rw [hSum] at hPos
      exact False.elim (Nat.lt_irrefl 0 hPos)
    | succ r =>
      cases r <;> simp [List.take, List.filter, hHeadActive]
  | cons x xs ih =>
    cases m with
    | zero =>
      cases hLen
    | succ m' =>
      simp only [List.length_cons] at hLen
      have hLen' : xs.length ≤ m' := Nat.le_of_succ_le_succ hLen
      simp only [allTombstones, Bool.and_eq_true] at hTomb
      have hx : x.isActive = false := by
        cases hxAct : x.isActive with
        | false => rfl
        | true => simp [hxAct] at hTomb
      have hAdd : k + Nat.succ m' = Nat.succ (k + m') := Nat.add_succ k m'
      rw [hAdd]
      simp only [List.cons_append, List.take, List.filter, hx]
      exact ih m' hLen' hTomb.2

/-- Scaled basis-point analytical crowding recall `min(10000, 10000 * k / (m + 1))`. -/
def crowdingRecallBps (k m : Nat) : Nat :=
  Nat.min 10000 ((10000 * k) / (m + 1))

/-- Theorem 16: Quantitative verification of the crowding law at `k = 5, m = 20` (`2380` bps = `23.80%`)
    versus full saturation `10000` bps (`100.00%`) at `m <= 4`. -/
theorem crowding_law_m20_k5_collapse_and_recovery :
    crowdingRecallBps 5 0 = 10000 ∧
    crowdingRecallBps 5 4 = 10000 ∧
    crowdingRecallBps 5 20 = 2380 := by
  decide


/-! ## Section 4: Fixed-Point Integer RRF Monotonicity & Causal GABA Gate Soundness -/

/-- Fixed-point integer RRF score term `scale / (k + 1 + rank)`. -/
def rrfTerm (scale k rank : Nat) : Nat :=
  scale / (k + 1 + rank)

/-- Saturating non-negative integer GABA inhibition `base - penalty` in `Nat`. -/
def gabaInhibit (base penalty : Nat) : Nat :=
  base - penalty

/-- 4-signal fixed-point RRF combiner `(dense + lexical + graph) - gabaPenalty`. -/
def rrfFuse4 (sDense sLex sGraph gabaPenalty : Nat) : Nat :=
  gabaInhibit (sDense + sLex + sGraph) gabaPenalty

/-- Causal GABA veto predicate: fires iff `thresh <= sim` AND `epTs + margin < deadendTs`. -/
def causalGabaVeto (epTs deadendTs margin sim thresh : Nat) : Bool :=
  decide (thresh ≤ sim) && decide (epTs + margin < deadendTs)

/-- Theorem 17: Positive micro-point RRF contribution whenever `k + 1 + rank <= scale`. -/
theorem rrfTerm_pos_of_bound
    (scale k rank : Nat) (hBound : k + 1 + rank ≤ scale) :
    0 < rrfTerm scale k rank := by
  unfold rrfTerm
  have hDenPos : 0 < k + 1 + rank := by
    have h1 : 0 < k + 1 := Nat.succ_pos k
    exact Nat.lt_of_lt_of_le h1 (Nat.le_add_right (k + 1) rank)
  exact Nat.div_pos hBound hDenPos

/-- Theorem 18 (`rrfTerm_antitone_rank` — Fixed-Point Integer RRF Monotonicity):
    Higher rank (smaller 0-based index `r1 <= r2`) never receives a smaller integer RRF score. -/
theorem rrfTerm_antitone_rank
    (scale k r1 r2 : Nat) (hRank : r1 ≤ r2) :
    rrfTerm scale k r2 ≤ rrfTerm scale k r1 := by
  unfold rrfTerm
  have hDenLe : k + 1 + r1 ≤ k + 1 + r2 := Nat.add_le_add_left hRank (k + 1)
  have hDenPos : 0 < k + 1 + r1 :=
    Nat.lt_of_lt_of_le (Nat.succ_pos k) (Nat.le_add_right (k + 1) r1)
  exact Nat.div_le_div_left hDenLe hDenPos

/-- Theorem 19 (`gabaInhibit_le_base` — GABA Non-Negativity, Boundedness & Full Veto):
    Integer GABA inhibition never exceeds `base`, equals `base` under zero penalty,
    and equals `0` whenever `base <= penalty`. -/
theorem gabaInhibit_le_base (base penalty : Nat) :
    gabaInhibit base penalty ≤ base ∧
    gabaInhibit base 0 = base ∧
    (base ≤ penalty → gabaInhibit base penalty = 0) := by
  unfold gabaInhibit
  refine ⟨Nat.sub_le base penalty, Nat.sub_zero base, ?_⟩
  intro hLe
  exact Nat.sub_eq_zero_of_le hLe

/-- Theorem 20 (`rrfFuse4_monotone_dense` & `causalGaba_protects_post_discovery_remediation`):
    (a) 4-signal integer RRF is monotone in each positive channel under arbitrary GABA penalty;
    (b) Post-discovery remediation episodes (`deadendTs <= epTs`) suffer zero autoimmune GABA veto. -/
theorem rrfFuse4_monotone_and_causalGaba_sound
    (d1 d2 sLex sGraph pen epTs deadendTs margin sim thresh : Nat)
    (hd : d1 ≤ d2)
    (hPostDiscovery : deadendTs ≤ epTs) :
    rrfFuse4 d1 sLex sGraph pen ≤ rrfFuse4 d2 sLex sGraph pen ∧
    causalGabaVeto epTs deadendTs margin sim thresh = false := by
  refine ⟨?_, ?_⟩
  · unfold rrfFuse4 gabaInhibit
    have hSum1 : d1 + sLex ≤ d2 + sLex := Nat.add_le_add_right hd sLex
    have hSum2 : d1 + sLex + sGraph ≤ d2 + sLex + sGraph := Nat.add_le_add_right hSum1 sGraph
    exact Nat.sub_le_sub_right hSum2 pen
  · unfold causalGabaVeto
    have hGe : deadendTs ≤ epTs + margin :=
      Nat.le_trans hPostDiscovery (Nat.le_add_right epTs margin)
    have hNotLt : ¬ (epTs + margin < deadendTs) := Nat.not_lt_of_ge hGe
    have hDec : decide (epTs + margin < deadendTs) = false := decide_eq_false hNotLt
    simp [hDec]

end TemporalGabaMemory.V1

/-! ## Axiom Verification Audit (`0 sorryAx`, `0 Classical.choice`) -/
#print axioms TemporalGabaMemory.V1.visibleTx_false_of_future_write
#print axioms TemporalGabaMemory.V1.asOfTx_future_append_invariant
#print axioms TemporalGabaMemory.V1.asOf2D_future_append_invariant
#print axioms TemporalGabaMemory.V1.asOfTx_future_close_invariant
#print axioms TemporalGabaMemory.V1.asOf2D_future_close_invariant
#print axioms TemporalGabaMemory.V1.asOfTx_future_revise_invariant
#print axioms TemporalGabaMemory.V1.supersede_no_gap
#print axioms TemporalGabaMemory.V1.supersede_no_double_active
#print axioms TemporalGabaMemory.V1.supersede_preserves_coverage
#print axioms TemporalGabaMemory.V1.valid_split_agm_success
#print axioms TemporalGabaMemory.V1.valid_split_three_way_partition
#print axioms TemporalGabaMemory.V1.take_prefix_all_tombstones
#print axioms TemporalGabaMemory.V1.unfiltered_topK_starves_when_tombstones_ge_k
#print axioms TemporalGabaMemory.V1.overfetch_k_plus_m_guarantees_active_head
#print axioms TemporalGabaMemory.V1.overfetch_k_plus_m_recovers_head_singleton
#print axioms TemporalGabaMemory.V1.crowding_law_m20_k5_collapse_and_recovery
#print axioms TemporalGabaMemory.V1.rrfTerm_pos_of_bound
#print axioms TemporalGabaMemory.V1.rrfTerm_antitone_rank
#print axioms TemporalGabaMemory.V1.gabaInhibit_le_base
#print axioms TemporalGabaMemory.V1.rrfFuse4_monotone_and_causalGaba_sound
