# High-dimensional failure cases: DMRG vs bordered TT-GMRES

Record of where the Fokker–Planck stationary-state solvers break down as the
dimension `d` (and hence `n^d`) is cranked up. The solvers are **correct and
accurate at small `d`** (see `DMRG.py` and `tt_gmres_fokker_planck.py`, which
match the dense null vector to ~1e-4). The cases below are the regimes where
they *fail*, kept for reference. Three solvers appear: DMRG (`LᵀL`), bordered
TT-GMRES (`L + 𝟙𝟙ᵀ`), and border-free inverse iteration (shift-invert on `L`).

## Test problem

Separable over-damped Langevin / gradient system — the **rank-2 (§2.2)**
operator `L = Σᵢ Lᵢ`, `Lᵢ = T ∂²ₓᵢ + ∂ₓᵢ(Vᵢ'·)`:

- Double-well potential on `x₁` (`V₁ = 1.5 (x²−1)²`), harmonic on `x₂…x_d`.
- `T = 1.0`, box `[-3, 3]`, `n = 11` grid points/dim, `maxrank = 6`.
- Built with `gradient_fp_terms` (DMRG.py); MPO bond dims `[1,2,2,1]`.
- The true Boltzmann density `exp(−V/T)` **factorizes ⇒ TT rank 1**, so this is
  the *easiest possible* high-`d` target. Both methods should sail through; the
  failures below are therefore solver/conditioning failures, not rank failures.

Residual reported is `‖Lp‖/‖p‖` measured directly on `L` (the FP operator). The
discrete near-null floor here is `min|eig(L)| ≈ 8e-2`, so a healthy solve lands
near that, not at 0.

---

## Failure 1 — DMRG: cost explosion + accuracy decay

`dmrg_null_vector` forms `LᵀL` (squares the condition number) and runs
ground-state DMRG. Even on the rank-1 target it becomes impractically slow and
*less* accurate as `d` grows.

| d | n^d | wall time | `‖Lp‖/‖p‖` | TT rank |
|---|-----|-----------|------------|---------|
| 8  | 11⁸  ≈ 2.1e8  | **528 s**  | 9.8e-4 | 6 |
| 16 | 11¹⁶ ≈ 4.6e16 | **1299 s** | 2.5e-2 | 6 |

- Time grows steeply (~9 min → ~22 min); reaching `d ≳ 16` is not practical.
- Residual *degrades* with `d` (9.8e-4 → 2.5e-2) — the `LᵀL` squaring plus a
  fixed sweep/rank budget loses accuracy exactly where it costs the most.

## Failure 2 — bordered TT-GMRES: does not recover the null vector at high `d`

The driver solves the bordered system `(L + 𝟙𝟙ᵀ) p = 𝟙` (one inverse-iteration
step). It is fast at every `d`, but the recovered `p` stops being the stationary
density. Two sub-cases by border scaling:

### 2a. Raw border `𝟙𝟙ᵀ` (the shipped `build_operators`)

`𝟙𝟙ᵀ = ⊗ₖ ones(n,n)` has operator norm `n^d`, which swamps `L` (norm ~tens) and
destroys conditioning.

| d | GMRES time | `‖Lp‖/‖p‖` | overlap `\|<DMRG, GMRES>\|` |
|---|------------|------------|------------------------------|
| 8  | 5.4 s  | **2.9e+1** | 0.036 |
| 16 | 0.1 s  | **6.0e+1** | 0.000 |

Residual ≫ 1 (vs the `8e-2` floor) ⇒ `p` is unrelated to the null vector; the
near-zero overlap with the DMRG solution confirms it.

### 2b. Unit border `êêᵀ`, `ê = 𝟙/√(n^d)` (attempted fix — also fails)

Normalizing the border removes the `n^d` magnitude blow-up but makes the lift of
the near-null eigenvalue too small, so the system is near-singular and the solve
still misses:

| d | GMRES time | `‖Lp‖/‖p‖` |
|---|------------|------------|
| 3  | 0.6 s  | 6.5  (was 2.2e-2 with the raw border!) |
| 8  | 7.1 s  | 1.4e+1 |
| 16 | 18 s   | 2.6e+1 |
| 32 | 40 s   | 3.8e+1 |
| 64 | 88 s   | 5.4e+1 |

Note the unit border is *worse at small `d`* — it breaks the previously-working
`d = 3` case — so it is **not** a fix; the shipped code keeps the raw border
(which at least works for small `d`).

### 2c. No border coefficient `c` rescues it (sweep at `d = 8`)

Solving `(L + c·êêᵀ) p = ê` over `c ∈ [1, 1e6]`:

| c | `‖Lp‖/‖p‖` | GMRES converged? |
|---|------------|------------------|
| 1e0 | 1.4e+1 | False |
| 1e1 | 5.7e+1 | False |
| 1e2 | 5.7e+1 | **True** |
| 1e3 | 5.7e+1 | **True** |
| 1e4 | 5.7e+1 | **True** |
| 1e6 | 5.7e+1 | **True** |

The decisive failure: for `c ≥ 1e2` GMRES **converges on the bordered system**
(`conv=True`) yet `‖Lp‖/‖p‖ ≈ 57`. So the linear solve is accurate — the
*formulation* is wrong: its solution is no longer the null vector.

---

## Failure 3 — border-free inverse iteration (does the border do real work?)

Question: must we perturb `L` by the rank-1 `𝟙𝟙ᵀ` at all?  No — `L p = 0` can be
solved by **inverse iteration on `L` itself**, `p_{k+1} = normalize((L − σI)⁻¹ p_k)`,
where the shift `−σ` is just a diagonal term (it keeps `L`'s `[1,2,2,1]`/`[1,4,4,1]`
MPO structure; no dense rank-1 outer product).  Each inner solve uses the same
relaxed TT-GMRES.  Measured at `d = 3` (`fp_stationary_inverse_iteration`):

**Coupled-OU**, eigenvalues `−0.016, −1.95, …` (spectral gap ≈ 120×):

| σ | `‖Lp‖/‖p‖` | `⟨·, dense-null⟩` |
|------|-----------|-------------------|
| 0.0  | 9.8e-1 | 0.960 |
| −0.1 | **1.8e-2** | **0.99995** |
| −0.5 | 3.4e-2 | 0.99997 |

→ recovers the null vector **with no border**, matching the bordered solve.

**Separable double-well**, eigenvalues `+0.08, −0.45, −1.9, …` (gap ≈ 5×):

| σ | `‖Lp‖/‖p‖` | `⟨·, dense-null⟩` |
|--------|-----------|-------------------|
| 0.0    | 1.3e+1 | 0.33 |
| +0.06  | 1.7e+0 | 0.990 |
| +0.075 | 1.6e+1 | 0.63 |

→ works only in a **narrow, fragile** σ-window; inner GMRES stays `conv=False`
with non-monotone residuals (`31→11→19→21→1.7`).

What this reveals:

- **`σ = 0` fails in both cases.**  The inner solve is then on the *near-singular*
  `L` and GMRES stalls.  So the border was **not** mere bookkeeping: for an
  *iterative* inner solver it lifts the singular direction so GMRES can converge.
  Inverse iteration's textbook robustness (Wilkinson: the ill-conditioned solve's
  error lands along the wanted eigenvector) assumes a **direct** solve; GMRES does
  not inherit it.
- **A shift is the border-free substitute, robust only when the gap is large.**
  OU's near-zero eigenvalue is isolated, so a shift big enough to condition the
  solve (σ = −0.5) still targets the null.  The double-well's gap is ~5×, so no σ
  both conditions the inner solve *and* targets the null — the classic shift-invert
  tension (push σ→λ₀ for targeting ⇒ re-singularize the solve).
- **The shift sign must match λ₀.**  FD truncation pushes the near-zero eigenvalue
  to either side: OU's is `−0.016` (needs σ<0), the double-well's is `+0.08`
  (needs σ>0).  Wrong sign ⇒ targets the second eigenvalue instead.

---

## Diagnosis

Two separate things break the high-`d` bordered one-shot, and an earlier version
of this note mis-stated the first one:

1. **Not** a "loss of alignment" of `𝟙` with the null vector.  For an exactly
   singular `L`, Sherman–Morrison gives `(L + 𝟙𝟙ᵀ)⁻¹𝟙 ∝ L⁻¹𝟙 ∝ v₀` — the
   overlap `𝟙ᵀv₀` only sets the *scale* `γ`, **not the direction**.  Any probe
   with `𝟙ᵀv₀ ≠ 0` recovers the exact null direction.  So shrinking `𝟙ᵀv₀` is not
   the failure mechanism.
2. The real causes:
   - **Border magnitude.**  Raw `𝟙𝟙ᵀ` has operator norm `n^d`, which swamps `L`
     (norm ~tens) and destroys the conditioning of the *iterative* GMRES solve on
     `M` (Failure 2a).  Rescaling alone can't fix it (2b/2c): a too-small border
     leaves the system near-singular, a too-large one converges to a vector with
     `‖Lp‖/‖p‖ ≈ 57` — GMRES converges on `M` yet the answer isn't the null vector.
   - **Near-singularity, not singularity.**  The discrete `L` has a *cluster* of
     small eigenvalues (`λ_i` are sums of per-axis eigenvalues), so `M⁻¹𝟙 ∝ L⁻¹𝟙`
     mixes several small-eigenvalue modes, not the null mode alone.  A single shot
     with a fixed probe doesn't separate them; this is the same near-singular inner
     solve that stalls the border-free `σ = 0` run in Failure 3.

**What would be needed (partly explored in Failure 3):**
- Iterate the solve (inverse iteration, above) so the null mode is amplified over
  several steps — works when the spectral gap is comfortable (OU), fragile when it
  is not (double-well).
- A shift-and-invert `(L − σI)` with a correctly-signed, gap-appropriate `σ`.
- **Precondition the inner solve** (or deflate the known left-null `𝟙`) so GMRES
  converges near σ ≈ λ₀ — the missing piece for the small-gap case.
- For DMRG: avoid the `LᵀL` squaring (a non-symmetric/2-site variant) to cut cost
  and stop the accuracy decay.

## Reproduction

- DMRG + GMRES cross-comparison: `bench_highd.py` (repo root), e.g.
  `python3 bench_highd.py 8 16` (DMRG is slow; expect ~9 min at `d=8`).
- Small-`d` *working* baselines: `python3 tt_gmres_fokker_planck.py`
  (overlaps 0.9999 / 0.9998 at `d = 3`).
- Border-free inverse iteration: `fp_stationary_inverse_iteration` in
  `tt_gmres_fokker_planck.py` (Failure 3 tables).
