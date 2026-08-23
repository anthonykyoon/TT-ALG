import time

import numpy as np
from numpy.linalg import norm, svd
import scipy.linalg as sla


def arnoldi(A, r0, m, tol=1e-12):
    n = r0.shape[0]

    V = np.zeros((n, m + 1))
    H = np.zeros((m + 1, m))

    beta = norm(r0)

    if beta == 0:
        raise ValueError("Initial residual is 0")
    
    V[:, 0] = r0 / beta 

    for j in range(m):
        w = A @ V[:, j]

        # Modified Gram-Schmidt
        for i in range(j + 1):
            H[i, j] = np.dot(V[:, i], w)
            w = w - H[i, j] * V[:, i]

        H[j + 1, j] = norm(w)

        if H[j + 1, j] < tol:
            return V[:, :j + 1], H[:j + 2, :j + 1]
            
        V[:, j + 1] = w / H[j + 1, j]

    return V, H


def gmres(A, b, x0=None, m=20, tol=1e-10):
    n = b.shape[0]

    if x0 is None:
        x0 = np.zeros(n)

    r0 = b - A @ x0
    beta = norm(r0)

    if beta < tol:
        return x0, 0.0 
    
    V, H = arnoldi(A, r0, m, tol)

    k = H.shape[1]

    e1 = np.zeros(H.shape[0])
    e1[0] = beta 

    y, *_ = sla.lstsq(H, e1)

    x = x0 + V[:, :k] @ y

    residual = norm(b - A @ x)

    return x, residual

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

def tt_ew_addition(A, B):
    """
    Assumes that A, and B are both TTs of appropiate size and dimension
    does A + B elementwise
    """
    if len(A) != len(B):
        raise ValueError("These TT's are not of appropiate length")
    return [a + b for a, b in zip(A, B)]


def tt_ew_subtraction(A, B):
    """
    Does A - B. Assume A and B are TT of apprpiate size and everything elementwise 
    """
    if len(A) != len(B):
        raise ValueError("These TT's are not of appropiate length")
    return [a - b for a, b in zip(A, B)]

def tt_addition(A,B):
    if len(A) != len(B):
        raise ValueError("TT vectors must have the same number of cores")
    d = len(A)

    if d == 1:
        return [A[0] + B[0]]

    out = []

    for k in range(d):
        rA0, nA, rA1 = A[k].shape
        rB0, nB, rB1 = B[k].shape

        if nA != nB:
            raise ValueError(f"Dimension mismatch at core {k}")

        dtype = np.result_type(A[k].dtype, B[k].dtype)

        if k == 0:
            core = np.zeros((1, nA, rA1 + rB1), dtype=dtype)
            core[:, :, :rA1] = A[k]
            core[:, :, rA1:] = B[k]
        elif k == d - 1:
            core = np.zeros((rA0 + rB0, nA, 1), dtype=dtype)
            core[:rA0, :, :] = A[k]
            core[rA0:, :, :] = B[k]

        else:
            core = np.zeros((rA0 + rB0, nA, rA1 + rB1), dtype=dtype)
            core[:rA0, :, :rA1] = A[k]
            core[rA0:, :, rA1:] = B[k]

        out.append(core)

    return out 
    
def tt_subtraction(A, B):
    """
    A - B for TT/MPS vectors.
    """
    return tt_addition(A, tt_scale(B, -1.0))

def tt_dot(A,B):
    if len(A) != len(B):
        raise ValueError("TT vectors must have the same number of cores")
    
    f = np.ones((1, 1), dtype=np.result_type(A[0].dtype, B[0].dtype))

    for Ak, Bk in zip(A, B):
        # F_new(q,p) = sum_{r,s,i} conj(A(r,i,q)) F(r,s) B(s,i,p)
        f = np.einsum("riq,rs,sip->qp", Ak.conj(), f, Bk, optimize=True)

    return f[0, 0]

def truncated_svd(M, eps, maxrank):
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

def tt_round(G, eps = 1e-10, max_rank = 100):
    B = [core.copy() for core in G]
    d = len(B)

    if d == 1:
        return B

    # Right-to-left QR sweep: make cores 1,...,d-1 right-orthogonal
    for k in range(d - 1, 0, -1):
        r0, n, r1 = B[k].shape

        # Matrix shape: (r0, n*r1)
        # QR on transpose: (n*r1, r0)
        Q, R = np.linalg.qr(B[k].reshape(r0, n * r1).T)

        rr = Q.shape[1]
        B[k] = Q.T.reshape(rr, n, r1)

        # Fold R.T into previous core's right bond
        B[k - 1] = np.einsum("abc,cd->abd", B[k - 1], R.T, optimize=True)

    # Left-to-right SVD sweep: truncate ranks
    for k in range(d - 1):
        r0, n, r1 = B[k].shape

        M = B[k].reshape(r0 * n, r1)
        U, s, Vh = truncated_svd(M, eps=eps, maxrank=max_rank)

        r = U.shape[1]
        B[k] = U.reshape(r0, n, r)

        SVh = s[:, None] * Vh
        B[k + 1] = np.einsum("ab,bcd->acd", SVh, B[k + 1], optimize=True)

    return B

def mpo_mps_mult(A, B):
    """
    A[i] shape: (wL, wR, n_out, n_in)
    B[i] shape: (rL, n_in, rR)

    returns MPS core shape: (wL*rL, n_out, wR*rR)
    """
    if len(A) != len(B):
        raise ValueError("These TT's are not of appropriate length")

    out = []
    for Ai, Bi in zip(A, B):
        T = np.einsum("abst,xty->axsby", Ai, Bi, optimize=True)
        wL, rL, n_out, wR, rR = T.shape
        out.append(T.reshape(wL * rL, n_out, wR * rR))

    return out

def tt_scale(G, alpha):
    """
    tt scalar multiplication
    """
    out = [core.copy() for core in G]
    out[0] = alpha * out[0]
    return out
    
def relaxed_tt_gmres(b, x0, A, tol, max_rank: int, m: int, max_restarts=10,
                     stag_rtol=1e-2, stag_patience=2):
    """Relaxed (inexact) TT-GMRES.

    Convergence status is reported in info["status"], one of:
        "converged"    -- the TRUE relative residual reached `tol`.
        "stagnated"    -- the true residual stopped improving across restarts;
                          the max_rank truncation floor has been hit, so further
                          restarts cannot help.  This is convergence to the
                          rank-limited accuracy, NOT failure -- raise max_rank to
                          drive the residual lower.
        "max_restarts" -- ran out of restarts while still improving.
    info["converged"] stays True only for the first case (reached `tol`).
    Stagnation is declared after `stag_patience` consecutive restarts whose
    relative residual improved by less than `stag_rtol`.
    """
    x = [core.copy() for core in x0]

    b_norm = tt_frobenius(b)
    if b_norm == 0:
        raise ValueError("Right-hand side b has zero norm")

    info = {
        "restart_history": [],
        "converged": False,
        "stagnated": False,
        "status": "max_restarts",   # overwritten on convergence/stagnation
    }
    prev_true_rel_res = None
    stall_count = 0

    for restart in range(max_restarts):    
        Ax0 = mpo_mps_mult(A, x)
        r0 = tt_subtraction(b, Ax0)
        r0 = tt_round(r0, eps=tol, max_rank=max_rank)
        beta = tt_frobenius(r0)
        rel_res0 = beta / b_norm 

        if rel_res0 <= tol:
            info["converged"] = True
            info["status"] = "converged"
            info["final_relative_residual"] = rel_res0
            return x, info

        V = [None] * (m+1)
        V[0] = tt_scale(r0, 1.0 / beta)

        H = np.zeros((m + 1, m), dtype=np.result_type(b[0].dtype, x[0].dtype, float))

        res_tilde_norm = beta
        
        y_last = None
        k_last = 0

        for j in range(m):
            if res_tilde_norm == 0:
                delta = tol 
            else:
                delta = tol / (res_tilde_norm / beta)
            
            delta = min(delta, 1e-1)

            w = mpo_mps_mult(A, V[j])
            w = tt_round(w, eps=delta, max_rank=max_rank)

            for i in range(j + 1):
                H[i, j] = tt_dot(V[i], w)
                w = tt_subtraction(w, tt_scale(V[i], H[i, j]))
            
            w = tt_round(w, eps=delta, max_rank=max_rank)
            H[j + 1, j] = tt_frobenius(w)

            Hj = H[:j + 2, :j + 1]

            e1 = np.zeros(j + 2, dtype=H.dtype)
            e1[0] = beta

            yj, *_ = sla.lstsq(Hj, e1)

            res_tilde_norm = norm(e1 - Hj @ yj)
            rel_res_tilde = res_tilde_norm / b_norm

            y_last = yj
            k_last = j + 1

            print(
                f"restart {restart + 1}, iter {j + 1}: "
                f"computed rel residual = {rel_res_tilde:.3e}, "
                f"delta = {delta:.3e}, "
                f"h_next = {H[j + 1, j]:.3e}"
            )
        
            if H[j + 1, j] < 1e-14:
                    break

            if rel_res_tilde <= tol:
                    break

            V[j + 1] = tt_scale(w, 1.0 / H[j + 1, j])

        if y_last is None:
            raise RuntimeError("GMRES failed before producing a least-squares solution")

        x_new = [core.copy() for core in x]

        for i in range(k_last):
            x_new = tt_addition(x_new, tt_scale(V[i], y_last[i]))

        x_new = tt_round(x_new, eps=tol, max_rank=max_rank)

        Ax_new = mpo_mps_mult(A, x_new)
        r_true = tt_subtraction(b, Ax_new)
        r_true = tt_round(r_true, eps=tol, max_rank=max_rank)

        true_rel_res = tt_frobenius(r_true) / b_norm

        info["restart_history"].append(
            {
                "restart": restart + 1,
                "inner_iters": k_last,
                "computed_relative_residual": float(rel_res_tilde),
                "true_relative_residual": float(true_rel_res),
            }
        
        )
        print(
            f"after restart {restart + 1}: "
            f"true rel residual = {true_rel_res:.3e}"
        )

        x = x_new

        if true_rel_res <= tol:
            info["converged"] = True
            info["status"] = "converged"
            info["final_relative_residual"] = true_rel_res
            return x, info

        # Stagnation: the true residual stopped improving, so the max_rank
        # truncation floor has been reached and further restarts cannot help.
        # This is convergence to the rank-limited accuracy, not failure.
        improved = (prev_true_rel_res is None
                    or true_rel_res < (1.0 - stag_rtol) * prev_true_rel_res)
        stall_count = 0 if improved else stall_count + 1
        prev_true_rel_res = true_rel_res
        if stall_count >= stag_patience:
            info["stagnated"] = True
            info["status"] = "stagnated"
            info["final_relative_residual"] = true_rel_res
            print(f"stagnated: true rel residual plateaued at {true_rel_res:.3e} "
                  f"(raise max_rank to drive it lower)")
            return x, info

    info["final_relative_residual"] = true_rel_res
    return x, info









if __name__ == "__main__":
    print("Running ordinary GMRES test")
    print("=" * 60)

    A_dense = np.array([[4., 1., 0.],
                        [1., 3., 1.],
                        [0., 1., 2.]])

    b_dense = np.array([1., 2., 3.])

    x_dense, res_dense = gmres(A_dense, b_dense, m=3)

    print("GMRES solution:", x_dense)
    print("GMRES residual:", res_dense)
    print("True solution:", np.linalg.solve(A_dense, b_dense))

    print("\nRunning TT-GMRES test")
    print("=" * 60)

    def tt_to_dense(G):
        """
        Convert TT/MPS vector to dense vector.
        Assumes cores have shape (r_left, n, r_right).
        """
        cur = G[0][0]  # shape (n0, r1)

        for k in range(1, len(G)):
            cur = np.tensordot(cur, G[k], axes=([-1], [0]))

        return cur.reshape(-1)

    # Build a tiny 2D Kronecker product system:
    #
    # A = A1 kron A2
    #
    # but stored as an MPO with two rank-1 MPO cores.
    A1 = np.array([[4.0, 1.0],
                   [1.0, 3.0]])

    A2 = np.array([[2.0, 0.5],
                   [0.5, 1.0]])

    A_mpo = [
        A1.reshape(1, 1, 2, 2),
        A2.reshape(1, 1, 2, 2),
    ]

    # True solution is rank-1:
    #
    # x_true(i,j) = a(i) c(j)
    a = np.array([1.0, 2.0])
    c = np.array([3.0, 4.0])

    x_true = [
        a.reshape(1, 2, 1),
        c.reshape(1, 2, 1),
    ]

    # Build b = A x_true in TT format
    b_tt = mpo_mps_mult(A_mpo, x_true)
    b_tt = tt_round(b_tt, eps=1e-12, max_rank=10)

    # Initial guess x0 = zero TT.
    # This represents the full zero vector because the first core is zero.
    x0_tt = [
        np.zeros((1, 2, 1)),
        np.ones((1, 2, 1)),
    ]

    x_tt, info = relaxed_tt_gmres(
        b=b_tt,
        x0=x0_tt,
        A=A_mpo,
        tol=1e-10,
        max_rank=10,
        m=4,
        max_restarts=5,
    )

    x_computed_dense = tt_to_dense(x_tt)
    x_true_dense = tt_to_dense(x_true)

    print("\nTT-GMRES computed x:")
    print(x_computed_dense)

    print("\nTrue x:")
    print(x_true_dense)

    print("\nRelative error:")
    print(norm(x_computed_dense - x_true_dense) / norm(x_true_dense))

    print("\nInfo:")
    print(info)

    print("\nRunning higher-dimensional TT-GMRES test")
    print("=" * 60)

    # ------------------------------------------------------------
    # Helper: convert TT/MPS to dense vector
    # ------------------------------------------------------------
    def tt_to_dense(G):
        cur = G[0][0]  # shape (n0, r1)

        for k in range(1, len(G)):
            cur = np.tensordot(cur, G[k], axes=([-1], [0]))

        return cur.reshape(-1)

    # ------------------------------------------------------------
    # Higher-dimensional test
    #
    # We solve:
    #     A x = b
    #
    # where:
    #     A = A1 kron A2 kron A3 kron A4 kron A5
    #
    # but A is stored as an MPO, and x,b are stored as TT/MPS.
    # ------------------------------------------------------------
    d = 8
    n = 3

    # Local SPD-ish matrices, one per dimension
    local_As = []

    for k in range(d):
        Ak = np.array([
            [2.5 + 0.2 * k, 0.3, 0.0],
            [0.3, 3.0 + 0.1 * k, 0.2],
            [0.0, 0.2, 2.0 + 0.15 * k],
        ])

        local_As.append(Ak)

    # Rank-1 MPO for Kronecker product A1 ⊗ A2 ⊗ ... ⊗ Ad
    A_mpo_hd = [
        Ak.reshape(1, 1, n, n)
        for Ak in local_As
    ]

    # True rank-1 TT solution:
    # x_true(i1,...,id) = g1(i1) g2(i2) ... gd(id)
    # Generated for arbitrary d so the test scales with the dimension.
    rng = np.random.default_rng(0)
    local_xs = [rng.uniform(-1.0, 2.0, size=n) for _ in range(d)]

    x_true_hd = [
        vec.reshape(1, n, 1)
        for vec in local_xs
    ]

    # Build b = A x_true in TT format
    b_hd = mpo_mps_mult(A_mpo_hd, x_true_hd)
    b_hd = tt_round(b_hd, eps=1e-12, max_rank=20)

    # Initial guess x0 = zero TT
    # This represents the zero tensor because the first core is all zeros.
    x0_hd = [
        np.zeros((1, n, 1))
    ] + [
        np.ones((1, n, 1))
        for _ in range(d - 1)
    ]

    x_true_hd_dense = tt_to_dense(x_true_hd)

    print("\nProblem size")
    print("-" * 60)
    print("Dimension d:", d)
    print("Mode size n:", n)
    print("Dense vector size n^d:", n ** d)

    # ------------------------------------------------------------
    # 1) TT solver (relaxed TT-GMRES)
    # ------------------------------------------------------------
    t0 = time.perf_counter()
    x_hd, info_hd = relaxed_tt_gmres(
        b=b_hd,
        x0=x0_hd,
        A=A_mpo_hd,
        tol=1e-10,
        max_rank=20,
        m=10,
        max_restarts=5,
    )
    tt_time = time.perf_counter() - t0

    x_tt_dense = tt_to_dense(x_hd)
    tt_error = norm(x_tt_dense - x_true_hd_dense) / norm(x_true_hd_dense)

    print("\nTT solver (relaxed TT-GMRES)")
    print("-" * 60)
    print(f"Wall-clock time:        {tt_time:.4f} s")
    print(f"Relative error vs true: {tt_error:.3e}")
    print(f"Converged:              {info_hd['converged']}")

    # ------------------------------------------------------------
    # 2) Naive dense solve
    #
    # Forms the full n^d x n^d Kronecker matrix and calls a direct
    # solver. Only feasible while n^d stays small.
    # ------------------------------------------------------------
    DENSE_LIMIT = 20000
    print("\nNaive dense solve")
    print("-" * 60)
    if n ** d <= DENSE_LIMIT:
        t0 = time.perf_counter()
        A_dense_hd = local_As[0]
        for Ak in local_As[1:]:
            A_dense_hd = np.kron(A_dense_hd, Ak)

        b_dense_hd = tt_to_dense(b_hd)
        x_dense_solve = np.linalg.solve(A_dense_hd, b_dense_hd)
        dense_time = time.perf_counter() - t0

        dense_error = norm(x_dense_solve - x_true_hd_dense) / norm(x_true_hd_dense)

        print(f"Wall-clock time:        {dense_time:.4f} s")
        print(f"Relative error vs true: {dense_error:.3e}")

        # ------------------------------------------------------------
        # 3) Comparison
        # ------------------------------------------------------------
        tt_vs_dense = norm(x_tt_dense - x_dense_solve) / norm(x_dense_solve)
        speedup = dense_time / tt_time if tt_time > 0 else float("inf")

        print("\nComparison")
        print("-" * 60)
        print(f"TT vs dense solution agreement: {tt_vs_dense:.3e}")
        print(f"TT time / dense time:           {tt_time:.4f}s / {dense_time:.4f}s")
        print(f"Speedup (dense / TT):           {speedup:.2f}x")
    else:
        print(f"Skipped: n^d = {n ** d} exceeds DENSE_LIMIT = {DENSE_LIMIT}")
        print("(At this dimension only the TT solver is tractable.)")