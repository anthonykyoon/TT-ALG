# High-dimensional failure cases: DMRG vs bordered TT-GMRES

Record of where the two Fokker–Planck stationary-state solvers break down as the
dimension `d` (and hence `n^d`) is cranked up. Both solvers are **correct and
accurate at small `d`** (see `DMRG.py` and `tt_gmres_fokker_planck.py`, which
match the dense null vector to ~1e-4). The cases below are the regimes where
they *fail*, kept for reference.

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

## Diagnosis

The bordered one-shot is a single inverse-iteration step with a **fixed** probe
vector `𝟙`. Its solution is `p ∝ v₀ / (𝟙ᵀv₀)`. As `d` grows, the null vector
`v₀ = ⊗ᵢ ρᵢ` (a product of localized 1-D densities) becomes nearly orthogonal to
the uniform probe: `𝟙ᵀv₀ = Πᵢ(𝟙ₙᵀρᵢ)` drifts away from the scale that keeps the
border balanced against `L`, and the recovered direction is dominated by the
other (non-null) modes that the single step fails to suppress. No global rescale
of the border fixes a per-mode alignment problem.

**What would be needed (not yet implemented):**
- Iterate the solve (true inverse iteration: `L p_{k+1} = p_k`, normalize) so the
  null mode is amplified over several steps instead of one.
- A shift-and-invert `(L − σI)` formulation with a small `σ`, well-scaled `L`.
- Preconditioning for the stiff `1/h²` spectrum.
- For DMRG: avoid the `LᵀL` squaring (e.g. a non-symmetric/2-site variant) to cut
  cost and stop the accuracy decay.

## Reproduction

- DMRG + GMRES cross-comparison: `bench_highd.py` (repo root), e.g.
  `python3 bench_highd.py 8 16` (DMRG is slow; expect ~9 min at `d=8`).
- Small-`d` *working* baselines: `python3 tt_gmres_fokker_planck.py`
  (overlaps 0.9999 / 0.9998 at `d = 3`).
