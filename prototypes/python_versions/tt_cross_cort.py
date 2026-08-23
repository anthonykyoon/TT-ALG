import numpy as np

def householder_reflector(x):
    """
    Construct an n×n Householder reflector H such that H @ x zeros out all but the first entry of x.
    """
    x = x.astype(float).copy()
    n = x.shape[0]
    norm_x = np.linalg.norm(x)
    if norm_x == 0:
        return np.eye(n)
    # choose sign to avoid cancellation
    sign = 1.0 if x[0] >= 0 else -1.0
    v = x.copy()
    v[0] += sign * norm_x
    v /= np.linalg.norm(v)
    return np.eye(n) - 2.0 * np.outer(v, v)


def golub_kahan_bidiagonalization(A):
    """
    Golub–Kahan bidiagonalization of a square matrix A (m×m) via Householder reflections.
    Returns (U, B, V) with U, V orthonormal and B bidiagonal.
    """
    A = A.astype(float)
    m, n = A.shape
    assert m == n, "Matrix must be square for bidiagonalization"
    B = A.copy()
    U = np.eye(m)
    V = np.eye(n)

    for k in range(n):
        # Left reflector to zero out below-diagonal in column k
        Hk = householder_reflector(B[k:, k])
        H = np.eye(m)
        H[k:, k:] = Hk
        B = H @ B
        U = U @ H

        # Right reflector to zero out off-diagonal in row k
        if k < n - 1:
            Gk = householder_reflector(B[k, k+1:])
            G = np.eye(n)
            G[k+1:, k+1:] = Gk
            B = B @ G
            V = V @ G

    return U, B, V


def elementary_symmetric(lambdas):
    """
    Compute elementary symmetric polynomials s_0…s_m of the list `lambdas` of length m.
    Returns an array c of length (m+1) where
      c[j] = sum_{1≤i1<…<ij≤m} (lambdas[i1] * … * lambdas[ij]).
    """
    m = len(lambdas)
    c = np.zeros(m+1, dtype=float)
    c[0] = 1.0
    for λ in lambdas:
        # update backwards
        for j in range(m, 0, -1):
            c[j] += λ * c[j-1]
    return c  # c[0]…c[m]


def derandomized_cross_approximation(A, k):
    """
    Algorithm 3: Derandomized cross approximation (m ≤ n, k ≤ m)
    Input:
      A : (m×n) numpy array, with m ≤ n
      k : number of pivots to select (≤ m)
    Output:
      I, J : sorted lists of row- and column-indices of length k
    """
    A = A.astype(float)
    m, n = A.shape
    assert m <= n, "Require m ≤ n"
    B = A.copy()
    I = []
    J = []

    for t in range(1, k+1):
        # 1) Thin SVD of the current residual B
        U, Sigma, Vh = np.linalg.svd(B, full_matrices=False)
        V = Vh.T  # shape (n, m)

        min_ratio = np.inf
        best_i = best_j = None

        # 2) Loop over all (i, j) to find pivot minimizing the ratio
        for i in range(m):
            for j in range(n):
                if B[i, j] == 0:
                    raise ValueError(f"B[{i},{j}] is zero; pivoting required")
                # form x = Σ V[j,:]^T, y = (1/B[i,j]) Σ U[i,:]^T
                x = Sigma * V[j, :]
                y = (Sigma * U[i, :]) / B[i, j]

                # 3) Form M = Σ − x y^T
                M = np.diag(Sigma) - np.outer(x, y)

                # 4) Bidiagonalize M via Golub–Kahan
                _, B_bidiag, _ = golub_kahan_bidiagonalization(M)

                # 5) Compute singular values of the bidiagonal matrix
                sigma_b = np.linalg.svd(B_bidiag, full_matrices=False, compute_uv=False)

                # 6) Build symmetric polynomials on (σ_b)^2
                coeffs = elementary_symmetric(sigma_b**2)

                # 7) Ratio = c_{m−k+t−1} / c_{m−k+t}
                idx1 = m - k + t - 1
                idx2 = m - k + t
                ratio = coeffs[idx1] / coeffs[idx2]

                if ratio < min_ratio:
                    min_ratio = ratio
                    best_i, best_j = i, j

        # 8) Record pivot and deflate
        I.append(best_i)
        J.append(best_j)
        outer = np.outer(B[:, best_j], B[best_i, :])
        B -= outer / B[best_i, best_j]

    return sorted(I), sorted(J)

def TT_Cross_cort(A: np.ndarray, k: int) -> list[np.ndarray]:
    """
    Tensor-Train cross approximation via derandomized cross (Algorithm 3).
    
    Parameters
    ----------
    A : np.ndarray
        d-dimensional array to decompose.
    k : int
        Number of pivots to select in each unfolding (≤ mode-size).
    
    Returns
    -------
    tt_cores : list of np.ndarray
        List of d cores, where core i has shape (r[i], n[i], r[i+1]),
        with r[0]=r[d]=1 and intermediate r[i] chosen by the cross algorithm.
    """
    # number of modes and their sizes
    d    = A.ndim
    dims = A.shape
    
    # working copy and output list
    W = A.copy()
    tt_cores = []
    
    # TT-rank vector, initialized to all ones
    r = np.ones(d+1, dtype=int)
    r[0] = r[-1] = 1
    
    # loop over first d−1 modes
    for i in range(d-1):
        # reshape W into a matrix of size (r[i]*dims[i]) × (rest)
        rows = r[i] * dims[i]
        cols = W.size // rows
        M = W.reshape(rows, cols)
        
        # pick k pivots via Algorithm 3
        I, J = derandomized_cross_approximation(M, k)
        
        # form C, R and the central submatrix M[I,J]
        C = M[:, J]              # shape = (rows, k)
        R = M[I, :]              # shape = (k, cols)
        Uinv = np.linalg.inv(M[np.ix_(I, J)])  # shape = (k, k)
        
        # next TT-rank
        r[i+1] = len(J)
        
        # core = C @ Uinv, then reshape to (r[i], dims[i], r[i+1])
        core = (C @ Uinv).reshape(r[i], dims[i], r[i+1])
        tt_cores.append(core)
        
        # deflated residual for next unfolding
        W = R
    
    # final core: reshape the last residual
    final_core = W.reshape(r[-2], dims[-1], r[-1])
    tt_cores.append(final_core)
    
    return tt_cores