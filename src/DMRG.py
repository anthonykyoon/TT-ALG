"""
DMRG.py  -- Example: Fokker-Planck stationary state via the generic TT-DMRG engine.
================================================================================

This file is now just a *driver*.  The actual algorithm -- the generic two-site
DMRG ground-state solver and all the TT/MPO linear algebra -- lives in
`tt_dmrg.py` and knows nothing about Fokker-Planck.  Here we only:

    1. build the Fokker-Planck operator L as an MPO  (the problem-specific part),
    2. hand it to `dmrg_null_vector`, which finds the null vector of L by running
       ground-state DMRG on the symmetric PSD operator L^T L,
    3. check the result against the analytic Gaussian (an OU-only ground truth).

To "drop in a different problem" you only replace step 1: build any MPO and call
`dmrg_ground_state` (symmetric operator) or `dmrg_null_vector` (general L).  See
`example_generic()` at the bottom for a minimal, non-Fokker-Planck demo.

Goal of the test problem
------------------------
Find the stationary distribution p_s of a Fokker-Planck (FP) operator

        L p = - div( mu(x) p ) + D * laplacian(p) = 0,

i.e. the null vector of L.  We use a coupled Ornstein-Uhlenbeck process
    dX = -A X dt + sqrt(2 D) dW   (A symmetric, spd),
whose FP operator is

        L p = sum_i  d/dx_i [ (A x)_i p ]  +  D sum_i d^2/dx_i^2  p,

with exact stationary density the Gaussian

        p_s(x)  ~  exp( -1/2 x^T (A/D) x ),

which gives a ground truth to check the DMRG result against.
"""

import numpy as np
from numpy.linalg import norm
from functools import reduce
import scipy.linalg as sla

from tt_dmrg import (
    mpo_from_terms, tt_to_dense, dmrg_null_vector, dmrg_ground_state,
)


# =============================================================================
#  Fokker-Planck operator -> MPO  (the only problem-specific code)
# =============================================================================
def fd_operators(n, h):
    """1-D finite-difference building blocks on a uniform grid of spacing h."""
    I = np.eye(n)
    D2 = (np.diag(np.ones(n - 1), 1) - 2 * np.eye(n) + np.diag(np.ones(n - 1), -1)) / h**2  # d^2/dx^2 : [1,-2,1]/h^2
    D1 = (np.diag(np.ones(n - 1), 1) - np.diag(np.ones(n - 1), -1)) / (2 * h)               # d/dx    : [1,0,-1]/2h
    return I, D1, D2


def fokker_planck_terms(A, D, n, x):
    """
    Build the list of Kronecker-product terms for the coupled-OU FP operator

        L p = sum_i d/dx_i[(A x)_i p] + D sum_i d^2/dx_i^2 p.

    Arguments:
        A : (d x d) drift matrix
        D : scalar diffusion coefficient
        n : grid points per dimension
        x : grid coordinates (length n)
    Returns:
        terms : list of operator dicts suitable for mpo_from_terms
    """
    d = A.shape[0]
    h = x[1] - x[0]
    I, D1, D2 = fd_operators(n, h)
    X = np.diag(x)  # multiplication by the coordinate x_j
    terms = []

    # diffusion  D * d^2/dx_i^2  -> one D2 on axis i, identity elsewhere
    for i in range(d):
        terms.append({i: D * D2})

    # drift  A_ij * d/dx_i ( x_j * . )  -> a derivative on axis i and (if i!=j)
    # a coordinate-multiply on axis j; the coefficient A_ij is folded in
    for i in range(d):
        for j in range(d):
            if abs(A[i, j]) < 1e-14:
                continue
            if i == j:
                terms.append({i: A[i, j] * (D1 @ X)})   # same axis: d/dx_i (x_i .)
            else:
                terms.append({i: A[i, j] * D1, j: X})    # two axes: d/dx_i on i, x_j on j
    return terms


def analytic_gaussian(A, D, grids):
    """Exact OU stationary density  exp(-1/2 x^T (A/D) x)  sampled on the grid."""
    # build all grid points, evaluate the quadratic form x^T (A/D) x at each,
    # then exp(-1/2 .) -> the exact OU stationary density (inverse covariance A/D)
    mesh = np.meshgrid(*grids, indexing='ij')
    pts = np.stack([m.reshape(-1) for m in mesh], axis=1)   # (N, d)
    M = A / D
    quad = np.einsum('ni,ij,nj->n', pts, M, pts)
    g = np.exp(-0.5 * quad)
    return g / norm(g)


# =============================================================================
#  Boltzmann / Gibbs test case: an over-damped Langevin (gradient) system
# =============================================================================
def gradient_fp_terms(dV, T, grids):
    """
    Kronecker terms for the Fokker-Planck operator of a gradient (over-damped
    Langevin) system  dX = -grad V(X) dt + sqrt(2 T) dW,

        L p = sum_i d/dx_i[ (dV/dx_i) p ]  +  T sum_i d^2/dx_i^2 p,

    whose exact stationary density is the Boltzmann / Gibbs distribution
        rho_s(x)  ~  exp( -V(x) / T ).

    We assume a *separable* potential V(x) = sum_i V_i(x_i), so the drift on
    axis i is V_i'(x_i) and couples no other axis -- every term touches a single
    core (the operator is a sum of one-site operators, exactly the L_i form of
    the double-well notes).

    Arguments:
        dV    : list of d callables, dV[i](x) = V_i'(x)  (the 1-D derivative)
        T     : temperature / diffusion coefficient
        grids : list of d grid-coordinate arrays
    Returns:
        terms : list of operator dicts suitable for mpo_from_terms
    """
    d = len(grids)
    terms = []
    for i in range(d):
        x = grids[i]
        n = len(x)
        h = x[1] - x[0]
        _, D1, D2 = fd_operators(n, h)
        terms.append({i: T * D2})                     # diffusion  T d^2/dx_i^2
        terms.append({i: D1 @ np.diag(dV[i](x))})     # drift  d/dx_i( V_i' . )
    return terms


def analytic_boltzmann(V_funcs, T, grids):
    """Boltzmann density exp(-V/T) for a separable V = sum_i V_i, on the grid."""
    mesh = np.meshgrid(*grids, indexing='ij')
    Vtot = sum(V_funcs[i](mesh[i]) for i in range(len(grids)))
    g = np.exp(-Vtot / T).reshape(-1)
    return g / norm(g)


# =============================================================================
#  Demonstration on the coupled Ornstein-Uhlenbeck Fokker-Planck equation
# =============================================================================
def run_demo(n=11, maxrank=6):
    np.set_printoptions(precision=4, suppress=True)

    # ---- problem definition: coupled OU in d dimensions ------------------
    D = 1.0
    A = np.array([[2.0, 0.5, 0.0],
                  [0.5, 2.0, 0.5],
                  [0.0, 0.5, 2.0]])
    d = A.shape[0]

    # grid: box large enough to contain the Gaussian (std ~ sqrt(D/eig(A)))
    Sigma = D * np.linalg.inv(A)
    Lbox = 6.0 * np.sqrt(np.max(np.diag(Sigma)))

    # small grid + small maxrank keep every two-site block under the dense-eigh
    # cutoff in _solve_local, so the demo runs in a second or two (the L^T L
    # squaring makes the sparse eigsh path slow -- see tt_dmrg's module docstring)
    x = np.linspace(-Lbox, Lbox, n)
    terms = fokker_planck_terms(A, D, n, x)
    L = mpo_from_terms(terms, d)                 # Fokker-Planck operator as an MPO

    print("=" * 70)
    print("Two-site DMRG on the Fokker-Planck MPO (via the generic engine)")
    print("=" * 70)
    print(f"  grid n^d         = {n}^{d} = {n**d}")
    print(f"  FP MPO rank      = {L[1].shape[0] if d > 1 else 1}")
    print(f"  box              = [-{Lbox:.2f}, {Lbox:.2f}]\n")

    # the engine forms L^T L, rounds it, and runs ground-state DMRG for us
    G, lam, residual = dmrg_null_vector(L, round_eps=1e-12, init_rank=2, maxrank=maxrank,
                                        eps=1e-9, max_sweeps=6, tol=1e-12, verbose=True)

    # ---- verification ----------------------------------------------------
    # contract the TT result to a full vector and fix sign (density is positive)
    p = tt_to_dense(G)
    if p.sum() < 0:
        p = -p
    p /= norm(p)

    # Dense reference AT THE SAME GRID: the brute-force null vector of L.  This is
    # the apples-to-apples target for the solver -- it carries the same
    # discretization error as the DMRG result, so a correct solver overlaps it
    # ~1 even on a coarse grid.  (Overlap with the continuum Gaussian is lower
    # purely because the grid is coarse -- that's discretization, not the solver.)
    Ldense = reduce(lambda a, b: a + b,
                    [reduce(np.kron, [t.get(ax, np.eye(n)) for ax in range(d)])
                     for t in terms])
    w, V = sla.eig(Ldense)
    i0 = int(np.argmin(np.abs(w)))               # eigenvalue closest to 0
    p_dense = np.real(V[:, i0]); p_dense /= norm(p_dense)

    g = analytic_gaussian(A, D, [x] * d)         # continuum ground truth
    overlap_solver = abs(p @ p_dense)            # DMRG vs dense null vector (same grid)
    overlap_cont = abs(p @ g)                    # DMRG vs continuum Gaussian
    overlap_disc = abs(p_dense @ g)              # discretization ceiling
    ranks = [gk.shape[0] for gk in G] + [1]

    print("\n" + "-" * 70)
    print("Results")
    print("-" * 70)
    print(f"  smallest eig(L^T L)    : {lam: .3e}   (exact stationary = 0)")
    print(f"  ||L p|| / ||p||        : {residual: .3e}")
    print(f"  |<DMRG, dense-null>|   : {overlap_solver:.6f}   (solver accuracy)")
    print(f"  |<DMRG, Gaussian>|     : {overlap_cont:.6f}   (end-to-end)")
    print(f"  |<dense, Gaussian>|    : {overlap_disc:.6f}   (discretization ceiling)")
    print(f"  TT ranks               : {ranks}")
    if overlap_solver > 0.999:
        print("\n  SUCCESS: DMRG matched the dense null vector of the FP operator.")
    else:
        print("\n  WARNING: solver overlap below threshold; check parameters.")


def run_boltzmann_demo(n=15, maxrank=4):
    """
    Boltzmann/Gibbs stationary state of a separable double-well gradient system.

    Potential (cf. the double-well notes):  V(x) = V_1(x_1) + sum_{i>=2} V_i(x_i)
        V_1(x) = a_1 (x^2 - 1)^2     (bimodal: wells at x = +-1)
        V_i(x) = a_i x^2             (harmonic) for i >= 2
    The exact stationary density is the Boltzmann distribution exp(-V/T), and
    because V is separable it *factorizes* over dimensions -- TT rank 1 -- so a
    small maxrank suffices and DMRG should recover it cleanly even though the
    marginal in x_1 is genuinely non-Gaussian (two peaks).
    """
    np.set_printoptions(precision=4, suppress=True)

    # ---- problem definition: double-well in x_1, harmonic elsewhere ------
    T = 1.0
    a = [1.5, 1.0, 1.0]                           # a_1 = double-well; rest harmonic
    d = len(a)

    # per-dim 1-D potentials V_i and their derivatives V_i' (default-arg capture)
    V_funcs = [lambda x, c=a[0]: c * (x**2 - 1.0)**2] + \
              [(lambda x, c=ai: c * x**2) for ai in a[1:]]
    dV = [lambda x, c=a[0]: 4.0 * c * x * (x**2 - 1.0)] + \
         [(lambda x, c=ai: 2.0 * c * x) for ai in a[1:]]

    # grid: box wide enough to contain both wells (x_1) and the harmonic spread
    b = 3.0
    x = np.linspace(-b, b, n)
    grids = [x] * d
    terms = gradient_fp_terms(dV, T, grids)
    L = mpo_from_terms(terms, d)                  # FP operator as an MPO

    print("=" * 70)
    print("Two-site DMRG for a Boltzmann stationary state (double-well gradient)")
    print("=" * 70)
    print(f"  grid n^d         = {n}^{d} = {n**d}")
    print(f"  potential        = double-well on x_1, harmonic on x_2..x_{d}")
    print(f"  box              = [-{b:.1f}, {b:.1f}],   T = {T}\n")

    G, lam, residual = dmrg_null_vector(L, round_eps=1e-12, init_rank=2, maxrank=maxrank,
                                        eps=1e-9, max_sweeps=8, tol=1e-12, verbose=True)

    # ---- verification ----------------------------------------------------
    p = tt_to_dense(G)
    if p.sum() < 0:
        p = -p
    p /= norm(p)

    # dense null vector at the SAME grid (apples-to-apples solver target)
    Ldense = reduce(lambda u, v: u + v,
                    [reduce(np.kron, [t.get(ax, np.eye(n)) for ax in range(d)])
                     for t in terms])
    w, V = sla.eig(Ldense)
    i0 = int(np.argmin(np.abs(w)))               # eigenvalue closest to 0
    p_dense = np.real(V[:, i0]); p_dense /= norm(p_dense)
    if p_dense.sum() < 0:
        p_dense = -p_dense

    rho = analytic_boltzmann(V_funcs, T, grids)  # exact Boltzmann (continuum)
    overlap_solver = abs(p @ p_dense)            # DMRG vs dense null vector (same grid)
    overlap_cont = abs(p @ rho)                  # DMRG vs continuum Boltzmann
    overlap_disc = abs(p_dense @ rho)            # discretization ceiling
    ranks = [gk.shape[0] for gk in G] + [1]

    print("\n" + "-" * 70)
    print("Results")
    print("-" * 70)
    print(f"  smallest eig(L^T L)    : {lam: .3e}   (exact stationary = 0)")
    print(f"  ||L p|| / ||p||        : {residual: .3e}")
    print(f"  |<DMRG, dense-null>|   : {overlap_solver:.6f}   (solver accuracy)")
    print(f"  |<DMRG, Boltzmann>|    : {overlap_cont:.6f}   (end-to-end)")
    print(f"  |<dense, Boltzmann>|   : {overlap_disc:.6f}   (discretization ceiling)")
    print(f"  TT ranks               : {ranks}")
    if overlap_solver > 0.999:
        print("\n  SUCCESS: DMRG matched the dense null vector (Boltzmann state).")
    else:
        print("\n  WARNING: solver overlap below threshold; check parameters.")


def example_generic():
    """
    Minimal 'drop in any problem' demo -- nothing Fokker-Planck about it.

    Build a symmetric MPO for the separable operator  H = sum_i (-D2_i)  (a
    discrete Laplacian-type operator, here with the sign that makes it PSD) and
    ask the engine for its ground state, then cross-check lambda against a dense
    eigensolve.  Swap in any MPO of your own and call the same function.
    """
    from tt_dmrg import mpo_to_dense

    d, n = 3, 8                                  # small: two-site blocks stay dense
    h = 1.0 / (n + 1)
    D2 = (np.diag(np.ones(n - 1), 1) - 2 * np.eye(n) + np.diag(np.ones(n - 1), -1)) / h**2
    terms = [{i: -D2} for i in range(d)]         # H = sum_i (-D2_i): symmetric, PSD
    H = mpo_from_terms(terms, d)

    print("=" * 70)
    print("Generic drop-in demo: ground state of a symmetric separable MPO")
    print("=" * 70)
    G, lam, residual = dmrg_ground_state(H, init_rank=2, maxrank=8, eps=1e-9,
                                         max_sweeps=8, tol=1e-12, which='SA',
                                         verbose=True)

    # dense cross-check on this (small) problem
    w = np.linalg.eigvalsh(mpo_to_dense(H))
    print("\n" + "-" * 70)
    print(f"  DMRG smallest eig   : {lam: .6e}")
    print(f"  dense smallest eig  : {w[0]: .6e}")
    print(f"  residual ||Hp-lp||  : {residual: .3e}")
    print(f"  agreement           : {abs(lam - w[0]):.3e}")


if __name__ == "__main__":
    run_demo()
    print("\n")
    run_boltzmann_demo()
    print("\n")
    example_generic()
