import numpy as np
from numpy.linalg import norm
import copy

def summation_algorithm(A) -> float:
    A = np.asarray(A)
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

def build_bidiagonal(alpha: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """
    Build the (k+1) x k bidiagonal B for Golub–Kahan:
        A V_k = U_{k+1} B

    B has:
      - main diagonal = alpha[0:k]
      - subdiagonal   = beta[1:k+1]
    """
    alpha = np.asarray(alpha)
    beta = np.asarray(beta)
    k = alpha.shape[0]
    if beta.shape[0] != k + 1:
        raise ValueError(f"beta must have length k+1; got {beta.shape[0]} vs k={k}")

    B = np.zeros((k + 1, k), dtype=np.result_type(alpha, beta))
    np.fill_diagonal(B, alpha)           # main diagonal
    B[1:, :-1] += np.diag(beta[1:])      # subdiagonal (rows 1..k, cols 0..k-1)
    return B

def golab_kahn(A, k : int, tolerance : float, u0 = None, reorth = True, return_UV = False):
    A = np.asarray(A)
    m, n = A.shape
    rng = np.random.default_rng(2)

    if u0 is None:
        u = rng.standard_normal(m)
    else:
        u = np.asarray(u0, dtype = A.dtype)

    beta_prev = norm(u)
    if beta_prev < tolerance:
        raise ValueError("Starting vector has near-zero norm")

    u /= beta_prev

    U = np.zeros((m, k + 1), dtype=A.dtype)
    V = np.zeros((n, k ), dtype=A.dtype)
    alpha = np.zeros(k, dtype=A.dtype)
    beta = np.zeros(k + 1, dtype=A.dtype)
    beta[0] = 0.0
    U[:,0] = u

    def mgs_reorth(x, Q):
        for j in range(Q.shape[1]):
            h = np.dot(Q[:,j].conj(), x)
            x -= h * Q[:,j]
        return x

    for i in range(k):
        r = A.T @ U[:, i]
        if i > 0:
            r -= mgs_reorth(r, V[:, i-1])
        if reorth and i > 0:
            r = mgs_reorth(r, V[:, :i])
        a = norm(r)
        alpha[i] = a
        if a < tolerance:
            V[:,i:] = 0
            beta[i+1] = 0
            alpha[i:] = 0
            break

        p = A @ V[:,i] - alpha[i] * U[:,i]
        if reorth:
            p = mgs_reorth(p, U[:, :i+1])
        b = norm(p)
        beta[i+1] = b
        if b < tolerance:
            U[:, i+1:] = 0
            beta[i+2:] = 0
            alpha[i+1:] = 0
            break
        U[:, i+1] = p / b

    if return_UV:
        return U, V, alpha, beta
    else:
        return alpha, beta


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
                x = Sigma_full @ V[j,:].T
                if B[i,j] < tolerance:
                    break
                y = (1 / B[i,j]) * Sigma_full @ U[i,:].T
                matrix_to_be_transfromed = Sigma_full - x @ y.T
                alpha, beta = golab_kahn(matrix_to_be_transfromed, k = min(m,n), tolerance=tolerance)
                C = build_bidiagonal(alpha, beta)
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
