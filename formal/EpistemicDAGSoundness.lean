-- ============================================================================
-- File: formal/EpistemicDAGSoundness.lean
-- Companion Lean 4 Formalization: 9-D Dimensional Vector Algebra (DimVec9),
-- Minkowski-Bienayme Variance Decomposition, Holonomy Obstruction, and
-- DAG Neumann/Bellman DP Uniqueness (22 theorems, 0 sorry, standard axioms).
-- Build: lean --trust=0 formal/EpistemicDAGSoundness.lean
-- ============================================================================

namespace AlpLabV15Konu10Epistemic

structure DimVec9 where
  dL : Int
  dM : Int
  dT : Int
  dI : Int
  dTheta : Int
  dN : Int
  dJ : Int
  dUSD : Int
  dNat : Int
  deriving DecidableEq, Repr

namespace DimVec9

def zero : DimVec9 := ⟨0, 0, 0, 0, 0, 0, 0, 0, 0⟩
def dimUSD : DimVec9 := ⟨0, 0, 0, 0, 0, 0, 0, 1, 0⟩
def dimNat : DimVec9 := ⟨0, 0, 0, 0, 0, 0, 0, 0, 1⟩

def mul (a b : DimVec9) : DimVec9 :=
  ⟨a.dL + b.dL, a.dM + b.dM, a.dT + b.dT, a.dI + b.dI,
   a.dTheta + b.dTheta, a.dN + b.dN, a.dJ + b.dJ,
   a.dUSD + b.dUSD, a.dNat + b.dNat⟩

def inv (a : DimVec9) : DimVec9 :=
  ⟨-a.dL, -a.dM, -a.dT, -a.dI, -a.dTheta, -a.dN, -a.dJ, -a.dUSD, -a.dNat⟩

def div (a b : DimVec9) : DimVec9 := mul a (inv b)

def canAdd (a b : DimVec9) : Bool := decide (a = b)

theorem dim9_mul_comm (a b : DimVec9) : mul a b = mul b a := by
  cases a; cases b
  unfold mul
  congr 1 <;> omega

-- agy 2026-11-19: Lean 4.36.0 yapi projeksiyon indirgemesi icin dim9_mul_assoc/zero/inv kanitlarinda unfold -> dsimp only duzeltmesi (teorem ifadeleri ayni)
theorem dim9_mul_assoc (a b c : DimVec9) : mul (mul a b) c = mul a (mul b c) := by
  cases a; cases b; cases c
  dsimp only [mul]
  congr 1 <;> omega

theorem dim9_mul_zero (a : DimVec9) : mul a zero = a := by
  cases a
  dsimp only [mul, zero]
  congr 1 <;> omega

theorem dim9_mul_inv (a : DimVec9) : mul a (inv a) = zero := by
  cases a
  dsimp only [mul, inv, zero]
  congr 1 <;> omega

theorem dim9_usd_nat_guard_and_eig_unit :
    canAdd dimUSD dimNat = false ∧
    canAdd dimUSD dimUSD = true ∧
    div dimNat dimUSD = ⟨0, 0, 0, 0, 0, 0, 0, -1, 1⟩ := by
  decide

end DimVec9

def naiveMse3 (a1 a2 a3 v_ale : Nat) : Nat :=
  a1 * a1 + a2 * a2 + a3 * a3 + v_ale

def minkowskiBienaymeMse3 (a1 a2 a3 v_ale : Nat) : Nat :=
  (a1 + a2 + a3) * (a1 + a2 + a3) + v_ale

def crossGap3 (a1 a2 a3 : Nat) : Nat :=
  2 * (a1 * a2 + a1 * a3 + a2 * a3)

theorem minkowski_bienayme_exact_decomposition (a1 a2 a3 v_ale : Nat) :
    minkowskiBienaymeMse3 a1 a2 a3 v_ale =
      naiveMse3 a1 a2 a3 v_ale + crossGap3 a1 a2 a3 := by
  unfold minkowskiBienaymeMse3 naiveMse3 crossGap3
  have h1 : (a1 + a2 + a3) * (a1 + a2 + a3) =
      a1 * a1 + a2 * a2 + a3 * a3 + 2 * (a1 * a2 + a1 * a3 + a2 * a3) := by
    have h_exp : (a1 + a2 + a3) * (a1 + a2 + a3) =
        a1 * a1 + a1 * a2 + a1 * a3 +
        (a2 * a1 + a2 * a2 + a2 * a3) +
        (a3 * a1 + a3 * a2 + a3 * a3) := by
      rw [Nat.add_mul, Nat.add_mul,
          Nat.mul_add, Nat.mul_add,
          Nat.mul_add, Nat.mul_add,
          Nat.mul_add, Nat.mul_add]
    have hc12 : a2 * a1 = a1 * a2 := Nat.mul_comm a2 a1
    have hc13 : a3 * a1 = a1 * a3 := Nat.mul_comm a3 a1
    have hc23 : a3 * a2 = a2 * a3 := Nat.mul_comm a3 a2
    rw [h_exp, hc12, hc13, hc23]
    omega
  omega

theorem naive_never_exceeds_minkowski_bienayme (a1 a2 a3 v_ale : Nat) :
    naiveMse3 a1 a2 a3 v_ale <= minkowskiBienaymeMse3 a1 a2 a3 v_ale := by
  rw [minkowski_bienayme_exact_decomposition]
  exact Nat.le_add_right _ _

theorem minkowski_strictly_greater_when_two_positive (a1 a2 a3 v_ale : Nat)
    (h1 : a1 > 0) (h2 : a2 > 0) :
    naiveMse3 a1 a2 a3 v_ale < minkowskiBienaymeMse3 a1 a2 a3 v_ale := by
  rw [minkowski_bienayme_exact_decomposition]
  unfold crossGap3
  have hpos : a1 * a2 > 0 := Nat.mul_pos h1 h2
  omega

theorem ref_pipeline_3_discipline_underestimation_5944_bps :
    naiveMse3 8 15 16 121 = 666 ∧
    minkowskiBienaymeMse3 8 15 16 121 = 1642 ∧
    crossGap3 8 15 16 = 976 ∧
    (976 * 10000 + 821) / 1642 = 5944 := by
  decide

theorem upstream_e1_dominates_direct_e3_eig_per_usd :
    1738 > 1242 ∧
    ((1738 - 1242) * 10000 + 621) / 1242 = 3994 := by
  decide

-- ============================================================================
-- Extension: Minkowski-Bienayme Decomposition, Z^9 Holonomy, and DAG Neumann/Bellman Uniqueness:
-- Elmas ve Genel DAG Topolojilerinde Z^9 Sifir Boyutsal Holonomi (H^1 = 0)
-- ve Neumann Toplam-Etki Degismezligi (T_vu = sum_pi prod w_e)
-- ============================================================================

/-- Yardimci Lemma 1: Bir Z^9 boyut vektorunun kendisine bolumu boyutsuz birimdir. -/
theorem dim9_div_self (a : DimVec9) : DimVec9.div a a = DimVec9.zero := by
  cases a
  dsimp only [DimVec9.div, DimVec9.mul, DimVec9.inv, DimVec9.zero]
  congr 1 <;> omega

/-- Yardimci Lemma 2: Z^9 boyut oranlarinin zincirleme teleskop ozdesligi:
    (b / a) * (c / b) = c / a. -/
theorem dim9_div_mul_div (a b c : DimVec9) :
    DimVec9.mul (DimVec9.div b a) (DimVec9.div c b) = DimVec9.div c a := by
  cases a; cases b; cases c
  dsimp only [DimVec9.div, DimVec9.mul, DimVec9.inv]
  congr 1 <;> omega

/-- Yardimci Lemma 3: Z^9 bolumunun boyutsuz olmasi iki boyutun esitligine denktir. -/
theorem dim9_div_eq_zero_iff (a b : DimVec9) :
    DimVec9.div a b = DimVec9.zero ↔ a = b := by
  constructor
  · intro h
    cases a; cases b
    dsimp only [DimVec9.div, DimVec9.mul, DimVec9.inv, DimVec9.zero] at h
    injection h with h1 h2 h3 h4 h5 h6 h7 h8 h9
    congr 1 <;> omega
  · intro h
    rw [h]
    exact dim9_div_self b

/-- Kenar Z^9 baglantisi (ayrik 1-cochain) uzerinde keyfi uzunluklu bir yonlu yolun
    bilesik Z^9 transfer boyutu. -/
def pathDim {V : Type} (edgeDim : V → V → DimVec9) : V → List V → DimVec9
  | _, [] => DimVec9.zero
  | u, v :: rest => DimVec9.mul (edgeDim u v) (pathDim edgeDim v rest)

/-- Bir yonlu yolun hedef dugumu. -/
def pathEnd {V : Type} : V → List V → V
  | u, [] => u
  | _, v :: rest => pathEnd v rest

/-- Iki paralel yol arasindaki Z^9 boyutsal holonomi (ayrik Wilson dongusu). -/
def loopHolonomy (phi1 phi2 : DimVec9) : DimVec9 :=
  DimVec9.div phi1 phi2

/-- Teorem 11 (Z^9 Teleskop Teoremi): Dugum potansiyelinden tureyen (exact coboundary)
    her Z^9 kenar baglantisi boyunca keyfi uzunluklu her yolun transfer boyutu
    yalnizca uc dugumlerin oranina esittir (ara dugumlerden ve yol uzunlugundan bagimsizdir). -/
theorem exact_connection_path_telescopes {V : Type}
    (pot : V → DimVec9) (edgeDim : V → V → DimVec9)
    (hexact : ∀ x y, edgeDim x y = DimVec9.div (pot y) (pot x)) :
    ∀ (u : V) (rest : List V),
      pathDim edgeDim u rest = DimVec9.div (pot (pathEnd u rest)) (pot u)
  | u, [] => by
      dsimp only [pathDim, pathEnd]
      exact (dim9_div_self (pot u)).symm
  | u, v :: rest => by
      dsimp only [pathDim, pathEnd]
      rw [hexact u v, exact_connection_path_telescopes pot edgeDim hexact v rest]
      exact dim9_div_mul_div (pot u) (pot v) (pot (pathEnd v rest))

/-- Teorem 12 (Elmas ve Genel DAG Sifir Boyutsal Holonomi Teoremi):
    Ayni u -> v uc dugumlerini birlestiren keyfi uzunluklu iki paralel yolun
    Z^9 transfer boyutlari esittir, dongu holonomileri tam sifirdir (H^1 = 0)
    ve koherent Neumann yol toplami boyutsal olarak tip-guvenlidir (canAdd = true). -/
theorem dag_parallel_paths_zero_holonomy_and_addable {V : Type}
    (pot : V → DimVec9) (edgeDim : V → V → DimVec9)
    (hexact : ∀ x y, edgeDim x y = DimVec9.div (pot y) (pot x))
    (u v : V) (rest1 rest2 : List V)
    (h1 : pathEnd u rest1 = v) (h2 : pathEnd u rest2 = v) :
    pathDim edgeDim u rest1 = pathDim edgeDim u rest2 ∧
    loopHolonomy (pathDim edgeDim u rest1) (pathDim edgeDim u rest2) = DimVec9.zero ∧
    DimVec9.canAdd (pathDim edgeDim u rest1) (pathDim edgeDim u rest2) = true := by
  have hp1 : pathDim edgeDim u rest1 = DimVec9.div (pot v) (pot u) := by
    rw [exact_connection_path_telescopes pot edgeDim hexact u rest1, h1]
  have hp2 : pathDim edgeDim u rest2 = DimVec9.div (pot v) (pot u) := by
    rw [exact_connection_path_telescopes pot edgeDim hexact u rest2, h2]
  have heq : pathDim edgeDim u rest1 = pathDim edgeDim u rest2 := by
    rw [hp1, hp2]
  refine ⟨heq, ?_, ?_⟩
  · dsimp only [loopHolonomy]
    rw [heq]
    exact dim9_div_self _
  · dsimp only [DimVec9.canAdd]
    rw [heq]
    exact decide_eq_true rfl

/-- Teorem 13 (Kohomolojik Engel Teoremi): Paralel iki DAG yolunun Z^9 holonomisi
    sifirdan farkliysa bu yollarin toplanmasi yasaktir (canAdd = false) ve
    tum kenarlarla uyumlu hicbir kuresel Z^9 dugum potansiyeli var olamaz. -/
theorem nonzero_holonomy_blocks_node_potential {V : Type}
    (edgeDim : V → V → DimVec9)
    (u v : V) (rest1 rest2 : List V)
    (h1 : pathEnd u rest1 = v) (h2 : pathEnd u rest2 = v)
    (hhol : loopHolonomy (pathDim edgeDim u rest1) (pathDim edgeDim u rest2) ≠ DimVec9.zero) :
    DimVec9.canAdd (pathDim edgeDim u rest1) (pathDim edgeDim u rest2) = false ∧
    ¬ (∃ pot : V → DimVec9, ∀ x y, edgeDim x y = DimVec9.div (pot y) (pot x)) := by
  have hne : pathDim edgeDim u rest1 ≠ pathDim edgeDim u rest2 := by
    intro heq
    apply hhol
    dsimp only [loopHolonomy]
    rw [heq]
    exact dim9_div_self _
  constructor
  · dsimp only [DimVec9.canAdd]
    exact decide_eq_false hne
  · intro ⟨pot, hexact⟩
    have hzero :=
      (dag_parallel_paths_zero_holonomy_and_addable pot edgeDim hexact u v rest1 rest2 h1 h2).2.1
    exact hhol hzero

/-- 0..(n-1) araligi uzerinde tam sayi toplami. -/
def sumNat : Nat → (Nat → Int) → Int
  | 0, _ => 0
  | n + 1, f => sumNat n f + f n

theorem sumNat_zero : ∀ (n : Nat) (f : Nat → Int),
    (∀ i, i < n → f i = 0) → sumNat n f = 0
  | 0, _, _ => rfl
  | n + 1, f, h => by
      dsimp only [sumNat]
      rw [sumNat_zero n f (fun i hi => h i (Nat.lt_succ_of_lt hi)), h n (Nat.lt_succ_self n)]
      omega

theorem sumNat_congr : ∀ (n : Nat) (f g : Nat → Int),
    (∀ i, i < n → f i = g i) → sumNat n f = sumNat n g
  | 0, _, _, _ => rfl
  | n + 1, f, g, h => by
      dsimp only [sumNat]
      rw [sumNat_congr n f g (fun i hi => h i (Nat.lt_succ_of_lt hi)), h n (Nat.lt_succ_self n)]

theorem sumNat_add : ∀ (n : Nat) (f g : Nat → Int),
    sumNat n (fun i => f i + g i) = sumNat n f + sumNat n g
  | 0, _, _ => rfl
  | n + 1, f, g => by
      dsimp only [sumNat]
      rw [sumNat_add n f g]
      omega

theorem sumNat_mul_left (c : Int) : ∀ (n : Nat) (f : Nat → Int),
    sumNat n (fun i => c * f i) = c * sumNat n f
  | 0, _ => by
      dsimp only [sumNat]
      exact (Int.mul_zero c).symm
  | n + 1, f => by
      dsimp only [sumNat]
      rw [sumNat_mul_left c n f, Int.mul_add]

theorem sumNat_succ_left : ∀ (n : Nat) (f : Nat → Int),
    sumNat (n + 1) f = f 0 + sumNat n (fun i => f (i + 1))
  | 0, f => by
      dsimp only [sumNat]
      omega
  | n + 1, f => by
      -- agy 2026-11-19: Lean 4.36.0 dsimp only [sumNat] iki adim actigi icin ih uzerinde esit indirgeme (teorem ifadesi ayni)
      have ih := sumNat_succ_left n f
      dsimp only [sumNat] at ih ⊢
      omega

theorem sumNat_swap : ∀ (m n : Nat) (f : Nat → Nat → Int),
    sumNat m (fun i => sumNat n (fun j => f i j)) =
      sumNat n (fun j => sumNat m (fun i => f i j))
  | 0, n, _ => by
      dsimp only [sumNat]
      exact (sumNat_zero n (fun _ => 0) (fun _ _ => rfl)).symm
  | m + 1, n, f => by
      dsimp only [sumNat]
      rw [sumNat_swap m n f]
      exact (sumNat_add n (fun j => sumNat m (fun i => f i j)) (fun j => f m j)).symm

/-- Kronecker deltasi (birim matris I). -/
def kronDelta (u v : Nat) : Int :=
  if u = v then 1 else 0

/-- k adimli yuruyus matrisi (W^k)(u, v). -/
def walkSum (w : Nat → Nat → Int) (n : Nat) : Nat → Nat → Nat → Int
  | 0, u, v => kronDelta u v
  | k + 1, u, v => sumNat n (fun m => w u m * walkSum w n k m v)

/-- K adima kadar kesilmis Neumann toplam-etki matrisi:
    T^{(K)}(u, v) = sum_{k=0..K} (W^k)(u, v). -/
def neumannT (w : Nat → Nat → Int) (n : Nat) (K u v : Nat) : Int :=
  sumNat (K + 1) (fun k => walkSum w n k u v)

/-- Tek bir yonlu yolun kenar agirliklari carpimi: prod_{e in pi} w_e. -/
def pathWeight (w : Nat → Nat → Int) : Nat → List Nat → Int
  | _, [] => 1
  | u, m :: rest => w u m * pathWeight w m rest

/-- Bir yol koleksiyonu uzerinde agirlik carpimlari toplami: sum_pi prod_{e in pi} w_e. -/
def sumPathWeights (w : Nat → Nat → Int) (u : Nat) : List (List Nat) → Int
  | [] => 0
  | p :: ps => pathWeight w u p + sumPathWeights w u ps

/-- 0..(n-1) uzerinde listeleri birlestirme yardimcisi. -/
def flatMapNat {α : Type} : Nat → (Nat → List α) → List α
  | 0, _ => []
  | n + 1, f => flatMapNat n f ++ f n

/-- u -> v arasindaki tam k adimli tum yonlu yuruyuslerin acik listesi. -/
def pathsOfLen (n : Nat) : Nat → Nat → Nat → List (List Nat)
  | 0, u, v => if u = v then [[]] else []
  | k + 1, _u, v =>
      flatMapNat n (fun m => (pathsOfLen n k m v).map (fun rest => m :: rest))

/-- K adima kadar tum yonlu yuruyuslerin birlesik listesi. -/
def allPathsUpTo (n : Nat) (K u v : Nat) : List (List Nat) :=
  flatMapNat (K + 1) (fun k => pathsOfLen n k u v)

theorem sumPathWeights_append (w : Nat → Nat → Int) (u : Nat) :
    ∀ (xs ys : List (List Nat)),
      sumPathWeights w u (xs ++ ys) = sumPathWeights w u xs + sumPathWeights w u ys
  | [], ys => by
      dsimp only [List.nil_append, sumPathWeights]
      omega
  | p :: ps, ys => by
      dsimp only [List.cons_append, sumPathWeights]
      rw [sumPathWeights_append w u ps ys]
      omega

theorem sumPathWeights_map_cons (w : Nat → Nat → Int) (u m : Nat) :
    ∀ (ps : List (List Nat)),
      sumPathWeights w u (ps.map (fun rest => m :: rest)) =
        w u m * sumPathWeights w m ps
  | [] => by
      dsimp only [List.map, sumPathWeights]
      exact (Int.mul_zero (w u m)).symm
  | p :: ps => by
      dsimp only [List.map, sumPathWeights, pathWeight]
      rw [sumPathWeights_map_cons w u m ps, Int.mul_add]

theorem sumPathWeights_flatMapNat (w : Nat → Nat → Int) (u : Nat) :
    ∀ (n : Nat) (f : Nat → List (List Nat)),
      sumPathWeights w u (flatMapNat n f) =
        sumNat n (fun i => sumPathWeights w u (f i))
  | 0, _ => rfl
  | n + 1, f => by
      dsimp only [flatMapNat, sumNat]
      rw [sumPathWeights_append, sumPathWeights_flatMapNat w u n f]

/-- Teorem 14: Matris kuvveti (W^k)(u, v) tam k adimli tum yollarin agirlik
    carpimlari toplamina birebir esittir. -/
theorem walkSum_eq_sumPathWeights (w : Nat → Nat → Int) (n : Nat) :
    ∀ (k u v : Nat), walkSum w n k u v = sumPathWeights w u (pathsOfLen n k u v)
  | 0, u, v => by
      dsimp only [walkSum, kronDelta, pathsOfLen]
      if h : u = v then
        simp only [h, ↓reduceIte]
        dsimp only [sumPathWeights, pathWeight]
        omega
      else
        simp only [h, ↓reduceIte]
        dsimp only [sumPathWeights]
  | k + 1, u, v => by
      dsimp only [walkSum, pathsOfLen]
      rw [sumPathWeights_flatMapNat]
      apply sumNat_congr
      intro m _
      rw [sumPathWeights_map_cons, walkSum_eq_sumPathWeights w n k m v]

/-- Teorem 15 (Neumann Toplam-Etki Yol-Toplami Ozdesligi: T_vu = sum_pi prod w_e):
    Neumann serisi matrisi neumannT, acik yol kumesi allPathsUpTo uzerindeki
    tum yollarin kenar agirliklari carpimlari toplamina birebir esittir. -/
theorem neumannT_eq_sum_all_paths (w : Nat → Nat → Int) (n K u v : Nat) :
    neumannT w n K u v = sumPathWeights w u (allPathsUpTo n K u v) := by
  dsimp only [neumannT, allPathsUpTo]
  rw [sumPathWeights_flatMapNat]
  apply sumNat_congr
  intro k _
  exact walkSum_eq_sumPathWeights w n k u v

/-- Topolojik olarak siki sirali (ust-ucgensel) DAG kosulu: sifir olmayan her kenar i < j gider. -/
def IsStrictDag (w : Nat → Nat → Int) (n : Nat) : Prop :=
  ∀ i j, i < n → j < n → w i j ≠ 0 → i < j

/-- Teorem 16 (DAG Nilpotentlik Teoremi): Siki sirali n-dugumlu DAG uzerinde
    u + k > v oldugunda (ozellikle k >= n iken) W^k(u, v) = 0 olur. -/
theorem dag_walkSum_nilpotent (w : Nat → Nat → Int) (n : Nat)
    (hdag : IsStrictDag w n) :
    ∀ (k u v : Nat), u < n → v < n → v < u + k → walkSum w n k u v = 0
  | 0, u, v, _, _, hlt => by
      dsimp only [walkSum, kronDelta]
      have hne : u ≠ v := by omega
      simp only [hne, ↓reduceIte]
  | k + 1, u, v, hu, hv, hlt => by
      dsimp only [walkSum]
      apply sumNat_zero
      intro m hm
      if hz : w u m = 0 then
        rw [hz]
        exact Int.zero_mul _
      else
        have hum : u < m := hdag u m hu hm hz
        have hmk : v < m + k := by omega
        rw [dag_walkSum_nilpotent w n hdag k m v hm hv hmk]
        exact Int.mul_zero _

/-- Yardimci: n <= K + 1 iken bir adim eklemek Neumann toplamini degistirmez. -/
theorem dag_neumannT_stabilizes_one (w : Nat → Nat → Int) (n K u v : Nat)
    (hdag : IsStrictDag w n) (hu : u < n) (hv : v < n) (hK : n ≤ K + 1) :
    neumannT w n (K + 1) u v = neumannT w n K u v := by
  dsimp only [neumannT, sumNat]
  have hnil : walkSum w n (K + 1) u v = 0 :=
    dag_walkSum_nilpotent w n hdag (K + 1) u v hu hv (by omega)
  rw [hnil]
  omega

/-- Teorem 17 (Sonlu Kesme Degismezligi): Siki DAG'da Neumann serisi K = n - 1
    ufkunda tam olarak duraganlasir; keyfi d adim eklemek T_vu degerini degistirmez. -/
theorem dag_neumannT_stabilizes (w : Nat → Nat → Int) (n K u v : Nat)
    (hdag : IsStrictDag w n) (hu : u < n) (hv : v < n) (hK : n ≤ K + 1) :
    ∀ d : Nat, neumannT w n (K + d) u v = neumannT w n K u v
  | 0 => rfl
  | d + 1 => by
      have hstep : neumannT w n (K + d + 1) u v = neumannT w n (K + d) u v :=
        dag_neumannT_stabilizes_one w n (K + d) u v hdag hu hv (by omega)
      have hrew : K + (d + 1) = K + d + 1 := by omega
      rw [hrew, hstep, dag_neumannT_stabilizes w n K u v hdag hu hv hK d]

/-- Teorem 18 (Neumann-Bellman / Topolojik DP Degismezligi: (I - W) T = I):
    PG17 O(|V||E|) topolojik DP ozyinelemesi T(u,v) = delta(u,v) + sum_m w(u,m) T(m,v),
    Neumann serisi ve acik yol toplami sum_pi prod w_e ile birebir ozdestir. -/
theorem dag_neumann_bellman_dp_invariance (w : Nat → Nat → Int) (n K u v : Nat)
    (hdag : IsStrictDag w n) (hu : u < n) (hv : v < n) (hK : n ≤ K + 1) :
    neumannT w n K u v =
      kronDelta u v + sumNat n (fun m => w u m * neumannT w n K m v) := by
  rw [← dag_neumannT_stabilizes_one w n K u v hdag hu hv hK]
  dsimp only [neumannT]
  rw [sumNat_succ_left (K + 1) (fun k => walkSum w n k u v)]
  dsimp only [walkSum]
  rw [sumNat_swap (K + 1) n (fun k m => w u m * walkSum w n k m v)]
  congr 1
  apply sumNat_congr
  intro m _
  exact sumNat_mul_left (w u m) (K + 1) (fun k => walkSum w n k m v)

theorem dag_neumann_dp_unique_aux (w : Nat → Nat → Int) (n K v : Nat)
    (F : Nat → Nat → Int)
    (hdag : IsStrictDag w n) (hv : v < n) (hK : n ≤ K + 1)
    (hF : ∀ x, x < n → F x v = kronDelta x v + sumNat n (fun m => w x m * F m v)) :
    ∀ (d u : Nat), u < n → n - 1 - u ≤ d → F u v = neumannT w n K u v
  | 0, u, hu, hd => by
      rw [hF u hu, dag_neumann_bellman_dp_invariance w n K u v hdag hu hv hK]
      congr 1
      apply sumNat_congr
      intro m hm
      if hz : w u m = 0 then
        rw [hz, Int.zero_mul, Int.zero_mul]
      else
        have hum : u < m := hdag u m hu hm hz
        omega
  | d + 1, u, hu, hd => by
      rw [hF u hu, dag_neumann_bellman_dp_invariance w n K u v hdag hu hv hK]
      congr 1
      apply sumNat_congr
      intro m hm
      if hz : w u m = 0 then
        rw [hz, Int.zero_mul, Int.zero_mul]
      else
        have hum : u < m := hdag u m hu hm hz
        have hmd : n - 1 - m ≤ d := by omega
        rw [dag_neumann_dp_unique_aux w n K v F hdag hv hK hF d m hm hmd]

/-- Teorem 19 (Topolojik DP Cozumunun Tekligi): Siki DAG uzerinde (I - W) F = I
    ozyinelemesini saglayan her F fonksiyonu tekildir ve T_vu = sum_pi prod w_e'ye esittir. -/
theorem dag_neumann_dp_unique (w : Nat → Nat → Int) (n K u v : Nat)
    (F : Nat → Nat → Int)
    (hdag : IsStrictDag w n) (hu : u < n) (hv : v < n) (hK : n ≤ K + 1)
    (hF : ∀ x, x < n → F x v = kronDelta x v + sumNat n (fun m => w x m * F m v)) :
    F u v = sumPathWeights w u (allPathsUpTo n K u v) := by
  rw [← neumannT_eq_sum_all_paths]
  exact dag_neumann_dp_unique_aux w n K v F hdag hv hK hF (n - 1 - u) u hu (Nat.le_refl _)

/-- G3 T4 uclu elmas topolojisi (0=A, 1=B, 2=C, 3=E, 4=D; bypass kenari B->C dahil). -/
def t4Weight (wAB wAC wAE wBC wBD wCD wED : Int) (u v : Nat) : Int :=
  match u, v with
  | 0, 1 => wAB
  | 0, 2 => wAC
  | 0, 3 => wAE
  | 1, 2 => wBC
  | 1, 4 => wBD
  | 2, 4 => wCD
  | 3, 4 => wED
  | _, _ => 0

def t4Pot (dA dB dC dE dD : DimVec9) : Nat → DimVec9
  | 0 => dA
  | 1 => dB
  | 2 => dC
  | 3 => dE
  | _ => dD

/-- Teorem 20 (T4 Uclu Elmas 4-Yollu Sembolik Neumann Degismezligi ve Karisik-Uzunluklu
    Z^9 Sifir Holonomi): 2-adimli ve 3-adimli (A->B->C->D) paralel yollarin tamaminda
    T_DA = wAB*wBD + wAB*(wBC*wCD) + wAC*wCD + wAE*wED ve 4 yolun tamami dD/dA boyutundadir. -/
theorem triple_diamond_t4_neumann_4paths_and_holonomy
    (wAB wAC wAE wBC wBD wCD wED : Int)
    (dA dB dC dE dD : DimVec9) :
    neumannT (t4Weight wAB wAC wAE wBC wBD wCD wED) 5 4 0 4 =
      wAB * wBD + wAB * (wBC * wCD) + wAC * wCD + wAE * wED ∧
    sumPathWeights (t4Weight wAB wAC wAE wBC wBD wCD wED) 0 (allPathsUpTo 5 4 0 4) =
      wAB * wBD + wAB * (wBC * wCD) + wAC * wCD + wAE * wED ∧
    loopHolonomy
      (pathDim (fun x y => DimVec9.div (t4Pot dA dB dC dE dD y) (t4Pot dA dB dC dE dD x)) 0 [1, 4])
      (pathDim (fun x y => DimVec9.div (t4Pot dA dB dC dE dD y) (t4Pot dA dB dC dE dD x)) 0 [1, 2, 4]) =
      DimVec9.zero := by
  have hT : neumannT (t4Weight wAB wAC wAE wBC wBD wCD wED) 5 4 0 4 =
      wAB * wBD + wAB * (wBC * wCD) + wAC * wCD + wAE * wED := by
    simp [neumannT, sumNat, walkSum, kronDelta, t4Weight]
    omega
  refine ⟨hT, ?_, ?_⟩
  · rw [← neumannT_eq_sum_all_paths]
    exact hT
  · exact (dag_parallel_paths_zero_holonomy_and_addable
      (t4Pot dA dB dC dE dD)
      (fun x y => DimVec9.div (t4Pot dA dB dC dE dD y) (t4Pot dA dB dC dE dD x))
      (fun _ _ => rfl)
      0 4 [1, 4] [1, 2, 4] rfl rfl).2.1

/-- T4 uclu elmas tam ata kapanisi vs 2. derece IE-2 vs 3. derece IE-3 maliyet operatorleri. -/
def t4CostExact (cA cB cC cE cD : Int) : Int :=
  cA + cB + cC + cE + cD

def t4CostIE2 (cA cB cC cE cD : Int) : Int :=
  ((cA + cB) + (cA + cB + cC) + (cA + cE)) -
    ((cA + cB) + cA + cA) + cD

def t4CostIE3 (cA cB cC cE cD : Int) : Int :=
  t4CostIE2 cA cB cC cE cD + cA

/-- Teorem 21 (T4 IE-2 Cokusu ve 3. Derece Mobius Onarimi — neg_k10_g3_pairwise_ie2_collapse_on_dense_dag):
    T4 uclu elmasinda IE-2 ortak ata A maliyetini tum sembolik degerler icin tam cA kadar
    eksik hesaplar; cA > 0 iken IE-2 kesin eksik tahmin uretir ve 3. derece Mobius terimi
    tam ata kapanisi esitligini geri yukler. -/
theorem triple_diamond_t4_ie2_collapse_and_ie3_restoration
    (cA cB cC cE cD : Int) (hA : 0 < cA) :
    t4CostIE2 cA cB cC cE cD = t4CostExact cA cB cC cE cD - cA ∧
    t4CostIE2 cA cB cC cE cD < t4CostExact cA cB cC cE cD ∧
    t4CostIE3 cA cB cC cE cD = t4CostExact cA cB cC cE cD := by
  dsimp only [t4CostIE2, t4CostIE3, t4CostExact]
  omega

/-- Bozuk holonomili elmas cizge ornegi: A->B->D (USD/nat) ile A->C->D (USD)
    paralel yollari arasinda nat^-1 holonomi engeli olusur. -/
def brokenDiamondEdgeDim (u v : Nat) : DimVec9 :=
  match u, v with
  | 0, 1 => DimVec9.inv DimVec9.dimNat
  | 1, 3 => DimVec9.dimUSD
  | 0, 2 => DimVec9.zero
  | 2, 3 => DimVec9.dimUSD
  | _, _ => DimVec9.zero

/-- Teorem 22 (Karsi-Ornek Kohomolojik Engel): Bozuk holonomili elmas cizgede
    A->B->D ve A->C->D yollari toplanamaz (canAdd = false) ve hicbir kuresel Z^9
    dugum potansiyeli bu kenar boyutlarini gerceklestiremez. -/
theorem broken_diamond_usd_nat_holonomy_obstruction :
    DimVec9.canAdd
      (pathDim brokenDiamondEdgeDim 0 [1, 3])
      (pathDim brokenDiamondEdgeDim 0 [2, 3]) = false ∧
    ¬ (∃ pot : Nat → DimVec9, ∀ x y, brokenDiamondEdgeDim x y = DimVec9.div (pot y) (pot x)) := by
  apply nonzero_holonomy_blocks_node_potential brokenDiamondEdgeDim 0 3 [1, 3] [2, 3] rfl rfl
  decide

#print axioms DimVec9.dim9_mul_comm
#print axioms DimVec9.dim9_mul_assoc
#print axioms DimVec9.dim9_mul_zero
#print axioms DimVec9.dim9_mul_inv
#print axioms DimVec9.dim9_usd_nat_guard_and_eig_unit
#print axioms minkowski_bienayme_exact_decomposition
#print axioms naive_never_exceeds_minkowski_bienayme
#print axioms minkowski_strictly_greater_when_two_positive
#print axioms ref_pipeline_3_discipline_underestimation_5944_bps
#print axioms upstream_e1_dominates_direct_e3_eig_per_usd
#print axioms exact_connection_path_telescopes
#print axioms dag_parallel_paths_zero_holonomy_and_addable
#print axioms nonzero_holonomy_blocks_node_potential
#print axioms walkSum_eq_sumPathWeights
#print axioms neumannT_eq_sum_all_paths
#print axioms dag_walkSum_nilpotent
#print axioms dag_neumannT_stabilizes
#print axioms dag_neumann_bellman_dp_invariance
#print axioms dag_neumann_dp_unique
#print axioms triple_diamond_t4_neumann_4paths_and_holonomy
#print axioms triple_diamond_t4_ie2_collapse_and_ie3_restoration
#print axioms broken_diamond_usd_nat_holonomy_obstruction

end AlpLabV15Konu10Epistemic
