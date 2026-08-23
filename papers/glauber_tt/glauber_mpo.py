"""
glauber_mpo.py -- Build the 1-D Glauber generator L as an MPO and validate the
DMRG stationary state against the exact 1-D Ising Gibbs measure.

Physics (ring of N spins, sigma_i in {+1,-1}):
    master equation   d/dt rho(sigma) = (L rho)(sigma)
                     = sum_i [ w_i(sigma^i) rho(sigma^i) - w_i(sigma) rho(sigma) ],
    flip rate         w_i(sigma) = (alpha/2)[ 1 - (gamma/2) sigma_i (sigma_{i-1}+sigma_{i+1}) ],
    Glauber choice    gamma = tanh(2 J / kT).

Operator form (see the write-up, "L as an MPO"):
    L = sum_i [ a S_i + c Z_{i-1} G_i + c G_i Z_{i+1} ],   indices mod N,
    a = alpha/2,  c = alpha*gamma/4,  S = F - I,  G = ZF + Z,
with F the spin flip and Z the diagonal sigma operator.

By detailed balance the stationary state of L is the Ising Gibbs measure
    pi(sigma) ∝ exp( (J/kT) sum_i sigma_i sigma_{i+1} ),
so the DMRG null vector of L should reproduce pi.  This module checks exactly
that, cross-validating the MPO against a brute-force rate-matrix build first.
"""

import numpy as np
from numpy.linalg import norm

from tt_dmrg import (
    mpo_from_terms, mpo_round, mpo_to_dense, tt_to_dense, dmrg_null_vector,
)

# --- local 2x2 operators, ordered basis (sigma = +1, sigma = -1) --------------
I2 = np.eye(2)
F  = np.array([[0.0, 1.0], [1.0, 0.0]])    # spin flip:  +1 <-> -1
Z  = np.array([[1.0, 0.0], [0.0, -1.0]])   # multiply by sigma
S  = F - I2                                # F - I
G  = Z @ F + Z                             # ZF + Z = Z(F+I)


def glauber_gamma(J, kT):
    """The Glauber coupling parameter gamma = tanh(2 J / kT)  (|gamma| < 1)."""
    return np.tanh(2.0 * J / kT)


# =============================================================================
#  The builder
# =============================================================================
def glauber_terms(N, alpha, gamma):
    """Kronecker terms {axis: matrix} for the ring generator L.

    Each dict is one rank-1 Kronecker term (coefficient folded into a matrix);
    the coupling terms wrap around (indices mod N).  Feed to mpo_from_terms.
    """
    a = 0.5 * alpha
    c = 0.25 * alpha * gamma
    terms = []
    for i in range(N):
        terms.append({i: a * S})                         # on-site  a S_i
        terms.append({(i - 1) % N: Z, i: c * G})         # c Z_{i-1} G_i
        terms.append({i: c * G, (i + 1) % N: Z})         # c G_i Z_{i+1}
    return terms


def glauber_mpo(N, alpha, gamma, round_eps=1e-12):
    """The Glauber generator L as a TT-rounded MPO (bulk rank 6 on the ring)."""
    return mpo_round(mpo_from_terms(glauber_terms(N, alpha, gamma), N),
                     eps=round_eps)


# =============================================================================
#  Reference objects (dense, small N) for validation
# =============================================================================
def all_configs(N):
    """(2^N, N) array of +-1 spins, ordered so site 0 is the MOST significant
    index -- matching tt_to_dense / mpo_to_dense (site 0 = outermost)."""
    idx = np.arange(2 ** N)
    shifts = np.arange(N - 1, -1, -1)                 # site k -> bit (N-1-k)
    bits = (idx[:, None] >> shifts[None, :]) & 1      # (2^N, N)
    return 1 - 2 * bits                               # 0 -> +1, 1 -> -1


def config_index(spins):
    """Inverse of all_configs: +-1 vector -> integer index (site 0 = MSB)."""
    bits = (np.asarray(spins) == -1).astype(int)
    idx = 0
    for b in bits:
        idx = (idx << 1) | int(b)
    return idx


def glauber_rate_matrix(N, alpha, gamma):
    """Brute-force generator L (2^N x 2^N) straight from the master equation."""
    dim = 2 ** N
    spins = all_configs(N)
    L = np.zeros((dim, dim))
    for c in range(dim):
        sigma = spins[c]
        total = 0.0
        for i in range(N):
            wi = 0.5 * alpha * (
                1.0 - 0.5 * gamma * sigma[i]
                * (sigma[(i - 1) % N] + sigma[(i + 1) % N])
            )
            total += wi
            flipped = sigma.copy()
            flipped[i] *= -1
            L[config_index(flipped), c] += wi      # rate  c -> flipped
        L[c, c] -= total                           # escape rate on the diagonal
    return L


def gibbs_vector(N, J, kT):
    """Exact Ising Gibbs measure pi (length 2^N, sums to 1), same ordering."""
    spins = all_configs(N)
    bonds = np.sum(spins * np.roll(spins, -1, axis=1), axis=1)   # sum_i s_i s_{i+1}
    w = np.exp((J / kT) * bonds)                                  # ∝ exp(-H/kT)
    return w / w.sum()


# =============================================================================
#  Validation driver
# =============================================================================
def validate(N=8, J=1.0, kT=1.5, alpha=1.0, maxrank=16, seed=0, verbose=True):
    gamma = glauber_gamma(J, kT)
    print(f"Glauber ring:  N={N}, J={J}, kT={kT}, "
          f"alpha={alpha}, gamma=tanh(2J/kT)={gamma:.4f}\n")

    # 1) MPO builder vs brute-force rate matrix (structural correctness)
    L = glauber_mpo(N, alpha, gamma)
    ranks = [c.shape[0] for c in L] + [1]
    L_mpo = mpo_to_dense(L)
    L_rate = glauber_rate_matrix(N, alpha, gamma)
    err_build = np.max(np.abs(L_mpo - L_rate))
    print(f"[1] MPO bond ranks         : {ranks}  (max {max(ranks)})")
    print(f"    ||L_mpo - L_rate||_inf : {err_build:.2e}   "
          f"{'OK' if err_build < 1e-10 else 'MISMATCH'}")

    # 2) Physics check: the Ising Gibbs measure is stationary,  L pi = 0
    pi = gibbs_vector(N, J, kT)
    stat = norm(L_rate @ pi) / norm(pi)
    print(f"[2] ||L pi|| / ||pi||      : {stat:.2e}   "
          f"{'OK' if stat < 1e-10 else 'NOT STATIONARY'}")

    # 3) DMRG null vector of L  ->  should recover pi
    Gtt, lam, res = dmrg_null_vector(
        L, maxrank=maxrank, init_rank=2, eps=1e-10,
        max_sweeps=30, tol=1e-12, seed=seed, verbose=False,
    )
    rho = tt_to_dense(Gtt)
    if rho @ pi < 0:                       # eigenvector is defined up to sign
        rho = -rho
    rho = rho / rho.sum()                  # normalise as a probability (L1 = 1)

    overlap = abs(rho @ pi) / (norm(rho) * norm(pi))
    l1 = np.sum(np.abs(rho - pi))
    linf = np.max(np.abs(rho - pi))
    print(f"[3] DMRG lambda_min        : {lam:.2e}   "
          f"(residual ||L rho||/||rho|| = {res:.2e})")
    print(f"    <rho_DMRG, pi> cosine  : {overlap:.8f}")
    print(f"    ||rho_DMRG - pi||_1    : {l1:.2e}")
    print(f"    ||rho_DMRG - pi||_inf  : {linf:.2e}")

    ok = (err_build < 1e-10 and stat < 1e-10 and overlap > 1 - 1e-6)
    print("\n  RESULT:", "PASS -- DMRG reproduces the Gibbs measure."
          if ok else "CHECK -- see mismatches above.")
    return ok


if __name__ == "__main__":
    all_ok = True
    for N in (6, 8, 10):
        all_ok &= validate(N=N)
        print("-" * 64)
    print("ALL PASSED" if all_ok else "SOME CHECKS FAILED")
