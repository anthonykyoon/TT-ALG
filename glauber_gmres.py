"""
glauber_gmres.py -- Glauber stationary state via *bordered* TT-GMRES.

The equilibrium solves the homogeneous  L rho = 0, but GMRES needs a
non-singular system.  We border L with a rank-1 term (Wielandt deflation):

        (L + 1 1^T) rho = 1.

For any right null vector pi (L pi = 0) with 1^T pi != 0, the unique solution is
rho = pi / (1^T pi):  (L + 1 1^T)(pi/1^T pi) = (0 + 1 (1^T pi))/1^T pi = 1.
The Glauber generator is *exactly* singular (it is a genuine Markov generator,
L pi = 0 with pi the Ising Gibbs measure, and 1^T pi = sum pi > 0), so unlike the
finite-difference Fokker-Planck case there is no near-singular floor -- the
bordered operator is exactly non-singular and one TT-GMRES solve returns pi to
the solver tolerance.  Crucially, GMRES works on the non-symmetric L directly,
never forming L^T L (contrast the DMRG route, which squares the condition number).

Both L and the border stay cheap in TT form:
    1 1^T = (x)_k ones(2, 2)   (rank-1 MPO)      1 = (x)_k ones(2)   (rank-1 TT)

Validated against the exact 1-D Ising Gibbs measure.  Following the request,
this is checked only at small N (the fixed all-ones border is known to lose
alignment with the null vector as N grows; large-N behaviour is not tested here).
"""

import io
from contextlib import redirect_stdout

import numpy as np
from numpy.linalg import norm

from tt_dmrg import mpo_from_terms, mpo_round, tt_to_dense
from tt_gmres import relaxed_tt_gmres, mpo_mps_mult, tt_frobenius, tt_round
from glauber_mpo import glauber_terms, glauber_gamma, gibbs_vector


# --- TT plumbing (n = 2 for spins) -------------------------------------------
def mpo_dmrg_to_gmres(H):
    """Re-order MPO cores  (R_l, n_out, n_in, R_r) -> (R_l, R_r, n_out, n_in)."""
    return [np.transpose(Hk, (0, 3, 1, 2)) for Hk in H]


def ones_tt(N, n=2):
    """All-ones TT-vector 1 = (x)_k ones(n): rank 1."""
    return [np.ones((1, n, 1)) for _ in range(N)]


def zero_tt(N, n=2):
    """Zero TT-vector (a single zero core zeroes the whole train)."""
    return [np.zeros((1, n, 1))] + [np.ones((1, n, 1)) for _ in range(N - 1)]


# =============================================================================
#  Bordered TT-GMRES solve:  (L + 1 1^T) rho = 1   ->   L rho = 0
# =============================================================================
def glauber_stationary_gmres(N, alpha, gamma, max_rank=20, gmres_m=40,
                             max_restarts=12, tol=1e-9, round_eps=1e-12,
                             verbose=True):
    """
    Glauber stationary state (null vector of L) via one bordered TT-GMRES solve.

    Returns (rho, resid, info):
        rho   : TT-vector, satisfies L rho = 0 (normalised so 1^T rho = 1)
        resid : ||L rho|| / ||rho||   measured on the original operator L
        info  : the relaxed-TT-GMRES info dict
    """
    n = 2
    terms = glauber_terms(N, alpha, gamma)
    L = mpo_round(mpo_from_terms(terms, N), eps=round_eps)

    border = {ax: np.ones((n, n)) for ax in range(N)}     # 1 1^T as one Kron term
    M = mpo_round(mpo_from_terms(terms + [border], N), eps=round_eps)

    L_g, M_g = mpo_dmrg_to_gmres(L), mpo_dmrg_to_gmres(M)
    b, x0 = ones_tt(N), zero_tt(N)

    sink = io.StringIO()                                   # the inner solver is chatty
    with redirect_stdout(sink):
        rho, info = relaxed_tt_gmres(b=b, x0=x0, A=M_g, tol=tol,
                                     max_rank=max_rank, m=gmres_m,
                                     max_restarts=max_restarts)

    rho = tt_round(rho, eps=tol, max_rank=max_rank)
    Lrho = tt_round(mpo_mps_mult(L_g, rho), eps=tol, max_rank=max_rank)
    resid = tt_frobenius(Lrho) / tt_frobenius(rho)

    if verbose:
        print(f"    GMRES converged        : {info['converged']}")
        print(f"    reported rel. residual : {info['final_relative_residual']:.3e}")
        print(f"    ||L rho|| / ||rho||    : {resid:.3e}")
        print(f"    TT ranks of rho        : {[c.shape[0] for c in rho] + [1]}")
    return rho, resid, info


# =============================================================================
#  Validation against the exact Ising Gibbs measure  (small N only)
# =============================================================================
def validate(N=8, J=1.0, kT=1.5, alpha=1.0, max_rank=20):
    gamma = glauber_gamma(J, kT)
    print(f"Glauber ring:  N={N}, J={J}, kT={kT}, "
          f"alpha={alpha}, gamma={gamma:.4f}")
    rho_tt, resid, info = glauber_stationary_gmres(N, alpha, gamma,
                                                   max_rank=max_rank)

    rho = tt_to_dense(rho_tt)
    pi = gibbs_vector(N, J, kT)
    if rho @ pi < 0:                        # solution is defined up to sign
        rho = -rho
    rho = rho / rho.sum()                   # normalise as a probability

    overlap = abs(rho @ pi) / (norm(rho) * norm(pi))
    l1 = np.sum(np.abs(rho - pi))
    linf = np.max(np.abs(rho - pi))
    print(f"    <rho_GMRES, pi> cosine : {overlap:.8f}")
    print(f"    ||rho_GMRES - pi||_1   : {l1:.2e}")
    print(f"    ||rho_GMRES - pi||_inf : {linf:.2e}")

    ok = overlap > 1 - 1e-6
    print("  RESULT:", "PASS -- TT-GMRES reproduces the Gibbs measure."
          if ok else "CHECK -- overlap below threshold.")
    return ok


if __name__ == "__main__":
    all_ok = True
    for N in (6, 8):                        # small N only, by request
        all_ok &= validate(N=N)
        print("-" * 64)
    print("ALL PASSED" if all_ok else "SOME CHECKS FAILED")
