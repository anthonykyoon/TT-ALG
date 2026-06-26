"""
tt_dmrg.py  -- Generic two-site DMRG ground-state solver in Tensor-Train format.
"""

import numpy as np
from numpy.linalg import norm, svd
from functools import reduce
import scipy.linalg as sla
import scipy.sparse.linalg as spla


# =============================================================================
#  1.  TT / MPO utilities  (generic -- no problem assumptions)
# =============================================================================
def mpo_from_terms(terms, d):
    """
    Build a TT-matrix (MPO) for a sum of Kronecker-product operators.

    Each term is a dict {axis: matrix} describing the operator
        coeff * O_0 (x) O_1 (x) ... (x) O_{d-1},
    where any axis not listed defaults to the identity of the right size.
    The MPO is assembled by *direct-summing* the rank-1 terms, giving an exact
    (if non-minimal) representation of TT-rank K = len(terms).  Run mpo_round
    afterwards to collapse it to the true (usually much smaller) rank.

    Arguments:
        terms : list of (n x n) operator dicts, one entry per non-identity axis
        d     : number of axes (cores)
    Returns:
        H : list of d cores, core k has shape (K, n, n, K) (1 on the ends)
    """
    K = len(terms)
    # infer the local dimension n from any operator we can find
    n = None
    for t in terms:
        for M in t.values():
            n = M.shape[0]
    if n is None:
        raise ValueError("could not infer local dimension from terms")
    I = np.eye(n)

    # an axis not mentioned in a term acts as the identity there
    def op(term, axis):
        return term.get(axis, I)

    # The "direct-sum" construction: place the K terms so they live on disjoint
    # bond channels.  The first core fans the K terms out along its right bond;
    # the middle cores keep each term on the bond DIAGONAL (channel t can only
    # talk to channel t); the last core sums the channels back to one.  Chaining
    # the cores therefore reproduces exactly  sum_t (term_t).  Rank = K.
    H = []
    for k in range(d):
        if d == 1:
            core = np.zeros((1, n, n, 1))
            core[0, :, :, 0] = sum(reduce(np.matmul, [op(t, 0)]) for t in terms)
            H.append(core)
            continue
        if k == 0:
            # first core: 1 -> K, write term t into right-bond channel t
            core = np.zeros((1, n, n, K))
            for t, term in enumerate(terms):
                core[0, :, :, t] = op(term, k)
        elif k == d - 1:
            # last core: K -> 1, sum the K channels
            core = np.zeros((K, n, n, 1))
            for t, term in enumerate(terms):
                core[t, :, :, 0] = op(term, k)
        else:
            # middle core: K -> K, diagonal so channels never mix
            core = np.zeros((K, n, n, K))
            for t, term in enumerate(terms):
                core[t, :, :, t] = op(term, k)
        H.append(core)
    return H


def tt_to_dense(G):
    """Contract a TT-vector to a full dense vector of length prod(n_k)."""
    # walk the chain, absorbing one core at a time via the shared bond index;
    # only ever used on small grids for verification
    cur = G[0][0]  # (n_0, r_1)
    for k in range(1, len(G)):
        cur = np.tensordot(cur, G[k], axes=([cur.ndim - 1], [0]))  # (..., n_k, r_{k+1})
    return cur.reshape(-1)


def mpo_to_dense(H):
    """Contract an MPO to a full dense matrix (prod(n_k) x prod(m_k)).

    Verification helper for small problems -- lets a caller cross-check the TT
    operator against a brute-force dense build.
    """
    d = len(H)
    cur = H[0][0]  # (n_0, m_0, R_1)
    for k in range(1, d):
        cur = np.tensordot(cur, H[k], axes=([cur.ndim - 1], [0]))  # (..., n_k, m_k, R_{k+1})
    cur = cur[..., 0]                       # drop the trailing R_d = 1 bond
    ns = [H[k].shape[1] for k in range(d)]
    ms = [H[k].shape[2] for k in range(d)]
    # axes are interleaved (n_0,m_0,n_1,m_1,...); gather rows then columns
    row_axes = list(range(0, 2 * d, 2))
    col_axes = list(range(1, 2 * d, 2))
    cur = cur.transpose(row_axes + col_axes)
    return cur.reshape(int(np.prod(ns)), int(np.prod(ms)))


def tt_frobenius(G):
    """Frobenius norm of a TT-vector via the standard left-to-right recursion."""
    # F is the running 1x1 (then r_k x r_k) overlap of the chain with itself;
    # each step folds in one core:  F <- sum_i  G[:,i,:]^T F G[:,i,:]
    F = np.ones((1, 1))
    for core in G:
        F = np.einsum('riq,rs,sip->qp', core.conj(), F, core)
    # F has collapsed to a scalar = ||p||^2; clamp away tiny negative round-off
    # (e.g. when G is a near-zero residual vector) before the sqrt
    return np.sqrt(max(np.real(F[0, 0]), 0.0))


def tt_scale(G, alpha):
    """Multiply a TT-vector by a scalar (folded into the first core)."""
    out = [g.copy() for g in G]
    out[0] = out[0] * alpha
    return out


def tt_add(A, B):
    """
    Sum of two TT-vectors, A + B, by block-direct-summing the cores; ranks add.
    The first core stacks the right bonds, the last core stacks the left bonds,
    and the middle cores place A and B in disjoint diagonal bond blocks so the
    two trains never mix.  (Result is un-rounded.)
    """
    d = len(A)
    if len(B) != d:
        raise ValueError("tt_add: trains have different length")
    out = []
    for k in range(d):
        rA0, nA, rA1 = A[k].shape
        rB0, nB, rB1 = B[k].shape
        if nA != nB:
            raise ValueError(f"tt_add: physical dim mismatch at core {k}: {nA} vs {nB}")
        dt = np.result_type(A[k].dtype, B[k].dtype)
        if d == 1:
            out.append(A[k] + B[k])
            continue
        if k == 0:
            core = np.zeros((1, nA, rA1 + rB1), dtype=dt)
            core[:, :, :rA1] = A[k]
            core[:, :, rA1:] = B[k]
        elif k == d - 1:
            core = np.zeros((rA0 + rB0, nA, 1), dtype=dt)
            core[:rA0, :, :] = A[k]
            core[rA0:, :, :] = B[k]
        else:
            core = np.zeros((rA0 + rB0, nA, rA1 + rB1), dtype=dt)
            core[:rA0, :, :rA1] = A[k]
            core[rA0:, :, rA1:] = B[k]
        out.append(core)
    return out


def mpo_transpose(H):
    """Transpose every core (swap row/column physical legs) -> the MPO of H^T."""
    return [np.transpose(Hk, (0, 2, 1, 3)) for Hk in H]


def mpo_product(A, B):
    """
    Compose two MPOs:  (A B) acting as matrices.  Core ranks multiply.
    A core (Ra, n, k, Ra'),  B core (Rb, k, m, Rb')  ->  (Ra Rb, n, m, Ra' Rb').
    """
    out = []
    for Ak, Bk in zip(A, B):
        # contract A's COLUMN leg k with B's ROW leg k (matrix product per site);
        # the two bond indices (a,b) merge into a single product bond
        C = np.einsum('ankA,bkmB->abnmAB', Ak, Bk, optimize=True)
        Ra, Rb, n, m, Ra2, Rb2 = C.shape
        out.append(C.reshape(Ra * Rb, n, m, Ra2 * Rb2))
    return out


def mpo_round(H, eps=1e-10):
    """
    TT-round an MPO: right-orthogonalise with QR, then left-to-right truncated
    SVD.  The direct-sum construction of e.g. H^T H gives a hugely non-minimal
    rank; rounding collapses it to the true, much smaller TT rank.
    """
    B = [Hk.copy() for Hk in H]
    d = len(B)
    # ---- pass 1: right-to-left QR, making cores 1..d-1 right-orthonormal ----
    # this puts the train in a canonical form so the following SVDs are optimal
    for k in range(d - 1, 0, -1):
        R0, n, m, R1 = B[k].shape
        # QR of the core unfolded with its left bond as columns -> orthonormal core
        Q, Rmat = np.linalg.qr(B[k].reshape(R0, n * m * R1).T)
        rr = Q.shape[1]
        B[k] = Q.T.reshape(rr, n, m, R1)
        # B[k]_old = Rmat.T @ B[k]_new, so fold Rmat.T into the right bond of B[k-1]
        B[k - 1] = np.einsum('abcd,dr->abcr', B[k - 1], Rmat.T, optimize=True)
    # ---- pass 2: left-to-right truncated SVD, dropping tiny singular values ----
    # this is where the rank actually shrinks
    for k in range(d - 1):
        R0, n, m, R1 = B[k].shape
        U, s, Vh = _truncated_svd(B[k].reshape(R0 * n * m, R1), eps, maxrank=10**6)
        r = U.shape[1]
        B[k] = U.reshape(R0, n, m, r)            # keep the orthonormal left factor
        M = (s[:, None] * Vh)                    # push s*V^T into the next core
        B[k + 1] = np.einsum('rd,dnmS->rnmS', M, B[k + 1], optimize=True)
    return B


def apply_mpo(H, G):
    """Apply an MPO to a TT-vector, returning a new (un-rounded) TT-vector."""
    out = []
    for Hk, Gk in zip(H, G):
        # contract the operator's COLUMN leg m with the vector's physical leg m;
        # the operator bond and vector bond merge -> rank multiplies
        # Hk (R, n, m, R'), Gk (r, m, r') -> (R r, n, R' r')
        T = np.einsum('RnmS,rms->RrnSs', Hk, Gk, optimize=True)
        R, r, n, S, s = T.shape
        out.append(T.reshape(R * r, n, S * s))
    return out


def _check_mpo(H):
    """Validate the *shape* contract of an MPO (not its symmetry)."""
    d = len(H)
    if d == 0:
        raise ValueError("empty MPO")
    for k, Hk in enumerate(H):
        if Hk.ndim != 4:
            raise ValueError(f"core {k} must be 4-D (R,n,m,R'), got shape {Hk.shape}")
    if H[0].shape[0] != 1 or H[-1].shape[3] != 1:
        raise ValueError("MPO boundary bonds must be 1 "
                         f"(got left {H[0].shape[0]}, right {H[-1].shape[3]})")
    for k, Hk in enumerate(H):
        R0, n, m, R1 = Hk.shape
        if n != m:
            raise ValueError(f"core {k} is not square per site (n={n}, m={m}); "
                             "ground-state DMRG needs a square operator")
        if k + 1 < d and R1 != H[k + 1].shape[0]:
            raise ValueError(f"bond mismatch: core {k} right bond {R1} != "
                             f"core {k + 1} left bond {H[k + 1].shape[0]}")


# =============================================================================
#  2.  Two-site DMRG internals
# =============================================================================
# The environments summarise "everything outside the current two sites" into a
# small tensor, so each local solve is cheap.  An environment carries three legs:
# bra-MPS bond, MPO bond, ket-MPS bond -- i.e. one slice of the sandwich <p|H|p>.
def _left_env(LE, Gk, Hk):
    """Grow the left environment by one site (Galerkin: bra = conj(ket))."""
    # absorb core k: contract the old environment with bra core, MPO core, ket core
    return np.einsum('pqr,pia,qijb,rje->abe', LE, Gk.conj(), Hk, Gk, optimize=True)


def _right_env(RE, Gk, Hk):
    """Grow the right environment by one site."""
    # mirror image of _left_env, absorbing a core from the right
    return np.einsum('abe,pia,qijb,rje->pqr', RE, Gk.conj(), Hk, Gk, optimize=True)


def _solve_local(LE, Hk, Hk1, RE, which='SA'):
    """
    Solve the two-site local problem for the smallest ('SA') or largest ('LA')
    eigenvalue of the (assumed symmetric) effective operator.

    Returns (Psi, lambda) with Psi shaped (r_k, n1, n2, r_{k+2}).
    """
    # the two-site block has dimensions (left bond, phys1, phys2, right bond)
    rk = LE.shape[0]
    n1 = Hk.shape[1]
    n2 = Hk1.shape[1]
    rR = RE.shape[0]
    N = rk * n1 * n2 * rR
    shape = (rk, n1, n2, rR)

    if N <= 6000:
        # Build the effective operator densely in ONE contraction (left env, both
        # MPO cores, right env) and take only the wanted eigenpair with a direct
        # symmetric solver.  Direct eigh avoids the slow Lanczos convergence that
        # a huge spectral spread (e.g. 1/h^2 diffusion terms) would cause.
        Hmat = np.einsum('apm,pIiq,qJjs,esf->aIJemijf', LE, Hk, Hk1, RE,
                         optimize=True).reshape(N, N)
        Hmat = 0.5 * (Hmat + Hmat.T)            # symmetrise away round-off
        idx = 0 if which == 'SA' else N - 1     # smallest- vs largest-algebraic
        w, vr = sla.eigh(Hmat, subset_by_index=[idx, idx])
        lam = w[0]
        v = vr[:, 0]
    else:
        # large blocks: never form the N x N matrix; apply it as a matvec and let
        # Lanczos (eigsh) find the wanted edge of the spectrum
        path = np.einsum_path('apm,pIiq,qJjs,esf,mijf->aIJe', LE, Hk, Hk1, RE,
                              np.empty(shape), optimize='optimal')[0]

        def matvec(v):
            W = np.einsum('apm,pIiq,qJjs,esf,mijf->aIJe', LE, Hk, Hk1, RE,
                          v.reshape(shape), optimize=path)
            return W.reshape(-1)

        op = spla.LinearOperator((N, N), matvec=matvec)
        w, vr = spla.eigsh(op, k=1, which=which, maxiter=2000)
        lam = w[0]
        v = vr[:, 0]

    Psi = v.reshape(shape)
    Psi /= norm(Psi)                            # the local block is defined up to scale
    return Psi, lam


def _truncated_svd(M, eps, maxrank):
    """delta-truncated SVD: keep the leading singular values, capped at maxrank."""
    U, s, Vh = svd(M, full_matrices=False)
    if s[0] == 0:
        return U[:, :1], s[:1], Vh[:1, :]
    tail = np.cumsum(s[::-1] ** 2)[::-1]      # tail[r] = sum of discarded sv^2 if we keep r
    delta2 = (eps * norm(s)) ** 2             # discard budget: a fraction eps of the norm
    # smallest r whose discarded weight fits the budget
    r = 1
    for k in range(1, len(s)):
        if tail[k] <= delta2:
            r = k
            break
        r = k + 1
    r = max(1, min(r, maxrank, len(s)))       # never exceed maxrank, never drop to 0
    return U[:, :r], s[:r], Vh[:r, :]


def _random_right_canonical(dims, init_rank, rng):
    """Random TT-vector with per-site dims, then right-orthonormalise every core."""
    d = len(dims)
    ranks = [1] + [init_rank] * (d - 1) + [1]
    G = [rng.standard_normal((ranks[k], dims[k], ranks[k + 1])) for k in range(d)]
    # right-orthonormalise (same QR-sweep idea as mpo_round) so the very first
    # left environment is consistent and the local eigenproblems are well posed
    for k in range(d - 1, 0, -1):
        r0, nk, r1 = G[k].shape
        Q, Rmat = np.linalg.qr(G[k].reshape(r0, nk * r1).T)   # columns of Q orthonormal
        rr = Q.shape[1]
        G[k] = Q.T.reshape(rr, nk, r1)
        G[k - 1] = np.einsum('abc,dc->abd', G[k - 1], Rmat.T) # push R into the left neighbour
    return G


def _ground_state_residual(H, G, lam):
    """Problem-agnostic quality metric  ||H p - lam p|| / ||p||, all in TT format."""
    Hp = apply_mpo(H, G)
    resid = tt_add(Hp, tt_scale(G, -lam))    # H p - lam p as an (un-rounded) TT-vector
    return tt_frobenius(resid) / tt_frobenius(G)


# =============================================================================
#  3.  Public solvers
# =============================================================================
def dmrg_ground_state(H, init_rank=2, maxrank=12, eps=1e-9,
                      max_sweeps=10, tol=1e-8, which='SA', seed=0, verbose=True):
    """
    Two-site DMRG for an extremal eigenpair of a *symmetric* MPO H.

    Arguments:
        H          : symmetric MPO (list of cores). d and per-site n_k are
                     inferred from it; you do NOT pass them.
        init_rank  : starting TT rank
        maxrank    : maximum TT rank kept after each SVD truncation
        eps        : relative SVD truncation tolerance
        max_sweeps : maximum number of left+right sweeps
        tol        : convergence tolerance on the eigenvalue between sweeps
        which      : 'SA' smallest-algebraic (default) or 'LA' largest-algebraic
        seed       : RNG seed for the random initial guess
    Returns:
        G        : optimised TT-vector (eigenvector, up to scale)
        lam      : converged eigenvalue
        residual : ||H p - lam p|| / ||p||  (how well G is an eigenvector of H)
    """
    if which not in ('SA', 'LA'):
        raise ValueError("which must be 'SA' (smallest) or 'LA' (largest)")
    _check_mpo(H)
    d = len(H)
    dims = [H[k].shape[1] for k in range(d)]

    # --- degenerate single-core case: just a dense eigenproblem ---
    if d == 1:
        M = H[0].reshape(dims[0], dims[0])
        M = 0.5 * (M + M.T)
        idx = 0 if which == 'SA' else dims[0] - 1
        w, vr = sla.eigh(M, subset_by_index=[idx, idx])
        lam = w[0]
        G = [vr[:, 0].reshape(1, dims[0], 1)]
        return G, lam, _ground_state_residual(H, G, lam)

    rng = np.random.default_rng(seed)
    G = _random_right_canonical(dims, init_rank, rng)

    # boundary environments are trivial 1x1x1; precompute every right environment
    # once (the left ones get built on the fly during the first sweep)
    LE = [None] * (d + 1)
    RE = [None] * (d + 1)
    LE[0] = np.ones((1, 1, 1))
    RE[d] = np.ones((1, 1, 1))
    for k in range(d - 1, -1, -1):
        RE[k] = _right_env(RE[k + 1], G[k], H[k])

    lam_prev = np.inf
    lam = 0.0
    for sweep in range(max_sweeps):
        # ---- left -> right: optimise bonds 0,1,...,d-2 ----
        for k in range(d - 1):
            # solve sites (k,k+1) given their environments
            Psi, lam = _solve_local(LE[k], H[k], H[k + 1], RE[k + 2], which=which)
            rk, n1, n2, rR = Psi.shape
            # split the optimised block; U is left-orthonormal, s*V^T moves right
            U, s, Vh = _truncated_svd(Psi.reshape(rk * n1, n2 * rR), eps, maxrank)
            r = U.shape[1]
            G[k] = U.reshape(rk, n1, r)
            G[k + 1] = (s[:, None] * Vh).reshape(r, n2, rR)
            LE[k + 1] = _left_env(LE[k], G[k], H[k])   # extend left env past the fixed core

        # ---- right -> left: optimise bonds d-2,...,1,0 ----
        for k in range(d - 2, -1, -1):
            Psi, lam = _solve_local(LE[k], H[k], H[k + 1], RE[k + 2], which=which)
            rk, n1, n2, rR = Psi.shape
            # mirror split: V^T is right-orthonormal, U*s moves left
            U, s, Vh = _truncated_svd(Psi.reshape(rk * n1, n2 * rR), eps, maxrank)
            r = U.shape[1]
            G[k] = (U * s).reshape(rk, n1, r)
            G[k + 1] = Vh.reshape(r, n2, rR)
            RE[k + 1] = _right_env(RE[k + 2], G[k + 1], H[k + 1])  # extend right env

        ranks = [g.shape[0] for g in G] + [1]
        if verbose:
            print(f"  sweep {sweep + 1:2d}:  lambda = {lam: .3e}   "
                  f"max rank = {max(ranks)}")
        if abs(lam - lam_prev) < tol:
            break
        lam_prev = lam

    return G, lam, _ground_state_residual(H, G, lam)


def dmrg_null_vector(L, round_eps=1e-12, **kwargs):
    """
    Smallest-residual ("null") vector of a possibly non-symmetric operator L.

    DMRG variationally minimises a *symmetric* Rayleigh quotient, so a
    non-symmetric L is handled by minimising ||L p|| -- i.e. finding the smallest
    eigenpair of the symmetric PSD operator H = L^T L.  This is the right tool
    for stationary states / kernels (e.g. a Fokker-Planck operator).

    Arguments:
        L         : MPO of the (possibly non-symmetric) operator
        round_eps : TT-rounding tolerance applied to L^T L (its direct-sum
                    construction is wildly non-minimal in rank)
        **kwargs  : forwarded to dmrg_ground_state (init_rank, maxrank, eps,
                    max_sweeps, tol, seed, verbose)
    Returns:
        G        : optimised TT-vector (null vector, up to scale)
        lam      : smallest eigenvalue of L^T L  (~0 for a true null vector)
        residual : ||L p|| / ||p||  measured directly on L

    Note: forming L^T L squares the condition number -- see the module docstring.
    """
    H = mpo_round(mpo_product(mpo_transpose(L), L), eps=round_eps)
    G, lam, _ = dmrg_ground_state(H, which='SA', **kwargs)
    residual = tt_frobenius(apply_mpo(L, G)) / tt_frobenius(G)
    return G, lam, residual
