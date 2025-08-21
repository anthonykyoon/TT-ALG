import numpy as np
from numpy.linalg import norm
import copy

def summation_algorithm(A) -> float:
    A = np.asarray(A)
    print("summ")
    print(A.shape)
    print(A)
    if A.shape[0] != A.shape[1]:
        raise ValueError("A must be square")

    n = A.shape[0]

    if np.issubdtype(A.dtype, np.integer):
        A = A.astype(np.float64)

    I = np.eye(n, dtype=A.dtype)
    B = I.copy()          # B_0 = I
    coeffs = [1]          # leading 1 for λ^n
    c_list = []

    for k in range(1, n + 1):
        ck = -np.trace(A @ B) / k     # c_k = -(1/k) tr(A B_{k-1})
        c_list.append(ck)
        B = A @ B + ck * I            # B_k = A B_{k-1} + c_k I

    coeffs.extend(c_list)
    return np.array(coeffs, dtype=A.dtype)


def build_bidiagonal(alpha, beta):
    """
    Build the (k+1) x k bidiagonal B for Golub–Kahan:
        A V_k = U_{k+1} B

    diag(B)        = alpha[0:k]
    subdiag(B)     = beta[1:k+1]  (i.e., B[i, i-1] for i=1..k)
    """
    alpha = np.asarray(alpha)
    beta  = np.asarray(beta)

    if alpha.ndim != 1 or beta.ndim != 1:
        raise ValueError("alpha and beta must be 1D arrays.")

    k = alpha.shape[0]
    if beta.shape[0] != k + 1:
        raise ValueError(f"beta must have length k+1; got {beta.shape[0]} vs k={k}")

    B = np.zeros((k + 1, k), dtype=np.result_type(alpha, beta))

    # main diagonal
    B[np.arange(k), np.arange(k)] = alpha

    # subdiagonal: positions (i, i-1) for i=1..k
    if k > 0:
        B[np.arange(1, k + 1), np.arange(k)] = beta[1:]

    return B

def golub_kahan_bidiagonal_reduction(A: np.ndarray):
    """
    Perform full Golub–Kahan bidiagonalization:
        A = U @ B @ V.T
    where U and V are orthogonal, and B is bidiagonal.

    Parameters
    ----------
    A : (m, n) ndarray

    Returns
    -------
    U : (m, m) ndarray
    B : (m, n) ndarray  (bidiagonal)
    V : (n, n) ndarray
    """
    A = np.array(A, dtype=float, copy=True)
    m, n = A.shape
    U = np.eye(m)
    V = np.eye(n)

    for i in range(min(m, n)):
        # ---- Left Householder: eliminate below-diagonal entries in column i ----
        x = A[i:, i]
        normx = np.linalg.norm(x)
        if normx == 0:
            continue
        # sign trick for stability
        sign = 1.0 if x[0] >= 0 else -1.0
        u = x.copy()
        u[0] += sign * normx
        u /= np.linalg.norm(u)

        # Apply to A (from the left)
        A[i:, :] -= 2.0 * np.outer(u, u @ A[i:, :])
        # Accumulate into U
        U[:, i:] -= 2.0 * np.outer(U[:, i:] @ u, u)

        if i < n - 1:
            # ---- Right Householder: eliminate off-diagonal entries in row i ----
            x = A[i, i+1:]
            normx = np.linalg.norm(x)
            if normx != 0:
                sign = 1.0 if x[0] >= 0 else -1.0
                v = x.copy()
                v[0] += sign * normx
                v /= np.linalg.norm(v)

                # Apply to A (from the right)
                A[:, i+1:] -= 2.0 * (A[:, i+1:] @ v)[:, None] * v[None, :]
                # Accumulate into V
                V[:, i+1:] -= 2.0 * np.outer(V[:, i+1:] @ v, v)

    return U, A, V


def kressner_algo(A, tolerance, k: int):
    I = []
    J = []
    B = copy.deepcopy((np.asarray(A)))
    m, n = B.shape
    for t in range(0, k):
        i_t = 0
        j_t = 0
        U, s, V = np.linalg.svd(B, compute_uv=True, full_matrices=False)
        Sigma_full = np.zeros((m, n))
        np.fill_diagonal(Sigma_full, s)
        min_rato = np.inf
        for i in range(1, m):
            for j in range(1,n):
                x = Sigma_full @ V[j,:]
                if B[i,j] < tolerance:
                    break
                y = (1 / B[i,j]) * Sigma_full @ U[i,:].T
                matrix_to_be_transfromed = Sigma_full - x @ y.T
                _, C, _ = golub_kahan_bidiagonal_reduction(matrix_to_be_transfromed)
                singular_values = summation_algorithm(C)
                if singular_values[m-k+t] < tolerance:
                    break
                r = singular_values[m-k+t-1] / singular_values[m-k+t]
                if r < min_rato:
                    i_t = i
                    j_t = j
        I.append(i_t)
        J.append(j_t)
        B = B - (1 / B[i_t, j_t]) * np.outer(B[:, j_t], B[i_t, :])
    return I, J
