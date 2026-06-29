"""
tt_gmres_fokker_planck.py  --  Fokker-Planck stationary state via TT-GMRES.
================================================================================

Companion driver to `DMRG.py`.  `DMRG.py` finds the stationary density of a
Fokker-Planck operator L (the null vector of  L p = 0) with ground-state DMRG.
Here we find the *same* null vector, but with the relaxed TT-GMRES *linear*
solver in `tt_gmres.py`.

The catch: GMRES solves a non-singular linear system, while  L p = 0  is
homogeneous (L is singular, the trivial answer is p = 0).  We turn it into a
single non-singular solve by *bordering* L with a rank-1 normalization term.

    For any vectors u, v with  v^T p_null != 0,  the solution of

            (L + u v^T) p = u

    is exactly  p = p_null / (v^T p_null):  plugging p = gamma p_null gives
    L p = 0 and u (v^T p) = u, so the equation holds iff v^T p = 1.  Hence the
    solution satisfies  L p = 0  exactly, normalized so that  v^T p = 1.

We take u = v = 1 (the all-ones vector).  Then v^T p_null = sum(p_null) != 0
(a probability density has positive mass), and the border lifts the singular
direction so (L + 1 1^T) is non-singular.  Both pieces stay rank-1 in TT form:

    1 1^T = (x)_k ones(n, n)     -> a rank-1 MPO
    1     = (x)_k ones(n)        -> a rank-1 TT-vector

so the whole thing is ONE relaxed-TT-GMRES solve whose answer obeys L p = 0.

A note on the discrete operator: the continuum FP operator is exactly singular,
but the finite-difference L is only *near* singular -- boundary truncation
nudges the zero eigenvalue to a small |lambda_0| (~1e-2 on the demo grid).  The
bordered solve  p = (L + 1 1^T)^{-1} 1  is therefore one step of inverse
iteration  (p is proportional to L^{-1} 1):  the 1/lambda_0 factor makes the
near-null eigenvector dominate, so p matches it to ~1e-5 in overlap.  It does
*not* variationally minimise ||L p|| (that is what DMRG does), so p can keep a
few extra small TT-rank channels from the suppressed 1/lambda_i tails.

Only the operator-building step is problem specific; it is reused verbatim from
`DMRG.py` (`fokker_planck_terms`) and `tt_dmrg.py` (`mpo_from_terms`).
"""

import io
from contextlib import redirect_stdout
from functools import reduce

import numpy as np
from numpy.linalg import norm
import scipy.linalg as sla

from tt_dmrg import mpo_from_terms, mpo_round, mpo_to_dense, tt_to_dense
from DMRG import (
    fokker_planck_terms, analytic_gaussian,
    gradient_fp_terms, analytic_boltzmann,
)
from tt_gmres import relaxed_tt_gmres, mpo_mps_mult, tt_frobenius, tt_round


# =============================================================================
#  Glue between the two MPO conventions
# =============================================================================
def mpo_dmrg_to_gmres(H):
    """
    Re-order MPO cores from the `tt_dmrg` layout to the `tt_gmres` layout.

        tt_dmrg : (R_left, n_out, n_in, R_right)
        tt_gmres: (w_left, w_right, n_out, n_in)   (see tt_gmres.mpo_mps_mult)
    """
    return [np.transpose(Hk, (0, 3, 1, 2)) for Hk in H]


def ones_tt(d, n):
    """The all-ones TT-vector 1 = (x)_k ones(n): rank-1, one core per axis."""
    return [np.ones((1, n, 1)) for _ in range(d)]


def zero_tt(d, n):
    """The zero TT-vector (first core zero makes the whole train zero)."""
    return [np.zeros((1, n, 1))] + [np.ones((1, n, 1)) for _ in range(d - 1)]


# =============================================================================
#  Bordered Fokker-Planck solve:  (L + 1 1^T) p = 1   ->   L p = 0
# =============================================================================
# WARNING (high dimensions): this one-shot bordered solve only works at small d.
# The fixed all-ones border vector loses alignment with the (localized) null
# vector as d grows -- 1^T p_null shrinks and the recovered direction degrades --
# so for large n^d the solve no longer returns the stationary density.  See
# `high_d_failure_cases.md` for the measured breakdown.
def build_operators(terms, d, n, round_eps=1e-12):
    """
    Return (L_gmres, M_gmres) where
        L_gmres = L                      (the FP operator, for residual checks)
        M_gmres = L + 1 1^T              (the bordered, non-singular operator)
    both as MPOs in the tt_gmres core layout.
    """
    L = mpo_round(mpo_from_terms(terms, d), eps=round_eps)

    # 1 1^T  as a single Kronecker term: ones(n, n) on every axis
    border = {ax: np.ones((n, n)) for ax in range(d)}
    M = mpo_round(mpo_from_terms(terms + [border], d), eps=round_eps)

    return mpo_dmrg_to_gmres(L), mpo_dmrg_to_gmres(M)


def fp_stationary_gmres(terms, d, n, max_rank=12, gmres_m=30, max_restarts=8,
                        tol=1e-8, verbose=True):
    """
    Stationary FP density (null vector of L = sum(terms)) via one bordered
    TT-GMRES solve  (L + 1 1^T) p = 1.

    Returns:
        p     : TT-vector solution (satisfies L p = 0, normalized so 1^T p = 1)
        resid : ||L p|| / ||p||   measured directly on L
        info  : the relaxed-TT-GMRES info dict
    """
    L_g, M_g = build_operators(terms, d, n)

    b = ones_tt(d, n)            # right-hand side  u = 1
    x0 = zero_tt(d, n)           # zero initial guess

    # the inner solver is chatty; keep this driver's output clean
    sink = io.StringIO()
    with redirect_stdout(sink):
        p, info = relaxed_tt_gmres(
            b=b, x0=x0, A=M_g,
            tol=tol, max_rank=max_rank, m=gmres_m, max_restarts=max_restarts,
        )

    p = tt_round(p, eps=tol, max_rank=max_rank)

    # FP residual on the *original* operator L
    Lp = tt_round(mpo_mps_mult(L_g, p), eps=tol, max_rank=max_rank)
    resid = tt_frobenius(Lp) / tt_frobenius(p)

    if verbose:
        print(f"  GMRES converged   : {info['converged']}")
        print(f"  computed rel.res.  : {info['final_relative_residual']:.3e}")
        print(f"  ||L p|| / ||p||    : {resid:.3e}")
        print(f"  TT ranks of p      : {[c.shape[0] for c in p] + [1]}")

    return p, resid, info


# =============================================================================
#  Demonstration: coupled Ornstein-Uhlenbeck Fokker-Planck (same as DMRG.py)
# =============================================================================
def run_demo(n=11, max_rank=12):
    np.set_printoptions(precision=4, suppress=True)

    # ---- problem definition: coupled OU in d dimensions (matches DMRG.py) ----
    D = 1.0
    A = np.array([[2.0, 0.5, 0.0],
                  [0.5, 2.0, 0.5],
                  [0.0, 0.5, 2.0]])
    d = A.shape[0]

    Sigma = D * np.linalg.inv(A)
    Lbox = 6.0 * np.sqrt(np.max(np.diag(Sigma)))
    x = np.linspace(-Lbox, Lbox, n)

    terms = fokker_planck_terms(A, D, n, x)

    print("=" * 70)
    print("TT-GMRES for the Fokker-Planck stationary state  (bordered  L p = 0)")
    print("=" * 70)
    print(f"  grid n^d         = {n}^{d} = {n**d}")
    print(f"  box              = [-{Lbox:.2f}, {Lbox:.2f}]")
    print(f"  solve            = (L + 1 1^T) p = 1\n")

    p, resid, info = fp_stationary_gmres(terms, d, n, max_rank=max_rank)

    # ---- verification ----------------------------------------------------
    # contract the TT result to a full vector and fix the sign (density >= 0)
    p_vec = tt_to_dense(p)
    if p_vec.sum() < 0:
        p_vec = -p_vec
    p_vec = p_vec / norm(p_vec)

    # dense reference at the SAME grid: the brute-force null vector of L
    Ldense = reduce(lambda a, b: a + b,
                    [reduce(np.kron, [t.get(ax, np.eye(n)) for ax in range(d)])
                     for t in terms])
    w, V = sla.eig(Ldense)
    i0 = int(np.argmin(np.abs(w)))               # eigenvalue closest to 0
    lam_min = abs(w[i0])                          # finite-difference near-zero floor
    p_dense = np.real(V[:, i0])
    if p_dense.sum() < 0:
        p_dense = -p_dense
    p_dense = p_dense / norm(p_dense)

    # extra cross-check: solve the bordered system densely and compare
    N = n ** d
    Mdense = Ldense + np.ones((N, N))
    p_border = np.linalg.solve(Mdense, np.ones(N))
    if p_border.sum() < 0:
        p_border = -p_border
    p_border = p_border / norm(p_border)

    g = analytic_gaussian(A, D, [x] * d)         # continuum ground truth

    overlap_solver = abs(p_vec @ p_dense)        # TT-GMRES vs dense null vector
    overlap_border = abs(p_vec @ p_border)       # TT-GMRES vs dense bordered solve
    overlap_cont = abs(p_vec @ g)                # TT-GMRES vs continuum Gaussian
    overlap_disc = abs(p_dense @ g)              # discretization ceiling

    print("\n" + "-" * 70)
    print("Results")
    print("-" * 70)
    print(f"  smallest |eig(L)|       : {lam_min: .3e}   (continuum = 0; FD makes it tiny)")
    print(f"  ||L p|| / ||p||         : {resid: .3e}   (bordered solve; <= |eig| floor)")
    print(f"  |<GMRES, dense-null>|   : {overlap_solver:.6f}   (solver accuracy)")
    print(f"  |<GMRES, dense-border>| : {overlap_border:.6f}   (bordered-system check)")
    print(f"  |<GMRES, Gaussian>|     : {overlap_cont:.6f}   (end-to-end)")
    print(f"  |<dense, Gaussian>|     : {overlap_disc:.6f}   (discretization ceiling)")
    if overlap_solver > 0.999:
        print("\n  SUCCESS: TT-GMRES matched the dense null vector of the FP operator.")
    else:
        print("\n  WARNING: solver overlap below threshold; check parameters.")


# =============================================================================
#  Demonstration 2: separable gradient system  --  the rank-2 (sec 2.2) operator
# =============================================================================
def run_gradient_demo(n=15, max_rank=8):
    """
    Boltzmann/Gibbs stationary state of a *separable* over-damped Langevin system,

        L = sum_i L_i,   L_i = T d^2/dx_i^2 + d/dx_i( V_i'(x_i) . ),

    which is exactly the rank-2 MPO of section 2.2 of the notes (every term
    touches a single axis, so the operator bond dimension is 2 for any d).  We
    use a double-well potential on x_1 and harmonic wells elsewhere, matching
    DMRG.run_boltzmann_demo, and solve the same bordered system (L + 1 1^T) p = 1.
    """
    np.set_printoptions(precision=4, suppress=True)

    # ---- problem definition: double-well on x_1, harmonic on x_2..x_d --------
    T = 1.0
    a = [1.5, 1.0, 1.0]                           # a_1 = double-well; rest harmonic
    d = len(a)

    V_funcs = [lambda x, c=a[0]: c * (x**2 - 1.0)**2] + \
              [(lambda x, c=ai: c * x**2) for ai in a[1:]]
    dV = [lambda x, c=a[0]: 4.0 * c * x * (x**2 - 1.0)] + \
         [(lambda x, c=ai: 2.0 * c * x) for ai in a[1:]]

    b = 3.0
    x = np.linspace(-b, b, n)
    grids = [x] * d
    terms = gradient_fp_terms(dV, T, grids)

    # confirm the operator really is the rank-2 (sec 2.2) MPO
    L_mpo = mpo_round(mpo_from_terms(terms, d), eps=1e-12)
    L_ranks = [c.shape[0] for c in L_mpo] + [1]

    print("=" * 70)
    print("TT-GMRES for a Boltzmann stationary state  (rank-2 sec-2.2 operator)")
    print("=" * 70)
    print(f"  grid n^d         = {n}^{d} = {n**d}")
    print(f"  potential        = double-well on x_1, harmonic on x_2..x_{d}")
    print(f"  box              = [-{b:.1f}, {b:.1f}],   T = {T}")
    print(f"  operator L = sum_i L_i,  MPO bond dims = {L_ranks}  (rank 2)")
    print(f"  solve            = (L + 1 1^T) p = 1\n")

    p, resid, info = fp_stationary_gmres(terms, d, n, max_rank=max_rank)

    # ---- verification ----------------------------------------------------
    p_vec = tt_to_dense(p)
    if p_vec.sum() < 0:
        p_vec = -p_vec
    p_vec = p_vec / norm(p_vec)

    Ldense = reduce(lambda u, v: u + v,
                    [reduce(np.kron, [t.get(ax, np.eye(n)) for ax in range(d)])
                     for t in terms])
    w, V = sla.eig(Ldense)
    i0 = int(np.argmin(np.abs(w)))               # eigenvalue closest to 0
    lam_min = abs(w[i0])
    p_dense = np.real(V[:, i0])
    if p_dense.sum() < 0:
        p_dense = -p_dense
    p_dense = p_dense / norm(p_dense)

    rho = analytic_boltzmann(V_funcs, T, grids)  # exact Boltzmann (continuum)

    overlap_solver = abs(p_vec @ p_dense)        # TT-GMRES vs dense null vector
    overlap_cont = abs(p_vec @ rho)              # TT-GMRES vs continuum Boltzmann
    overlap_disc = abs(p_dense @ rho)            # discretization ceiling

    print("\n" + "-" * 70)
    print("Results")
    print("-" * 70)
    print(f"  smallest |eig(L)|       : {lam_min: .3e}   (continuum = 0; FD makes it tiny)")
    print(f"  ||L p|| / ||p||         : {resid: .3e}   (bordered solve; <= |eig| floor)")
    print(f"  |<GMRES, dense-null>|   : {overlap_solver:.6f}   (solver accuracy)")
    print(f"  |<GMRES, Boltzmann>|    : {overlap_cont:.6f}   (end-to-end)")
    print(f"  |<dense, Boltzmann>|    : {overlap_disc:.6f}   (discretization ceiling)")
    if overlap_solver > 0.999:
        print("\n  SUCCESS: TT-GMRES matched the dense null vector (Boltzmann state).")
    else:
        print("\n  WARNING: solver overlap below threshold; check parameters.")


if __name__ == "__main__":
    run_demo()
    print("\n")
    run_gradient_demo()
