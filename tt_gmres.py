import numpy as np 
from numpy.linalg import norm, svd 
import scipy.linalg as sla 
import scipy.sparse.linalg as spla 
from functools import reduce 

def arnoldi(A, r0, m):
    n = r0.shape[0]
    V = np.zeros((n, m+1))
    H = np.zeros(m+1, m)
    beta = norm(r0)
    if beta == 0:
        raise ValueError("Intial Residual is 0")
    
    V[:, 0] = r0 / beta 

    for j in range(m):
        w = A @ V[:, j]

        for i in range(j+1):
            H[i,j] = np.dot(V[:,i], w)
            w = w - H[i,j] * V[:,i]

    return "WIP!"

        


def gmres(A, b , x0, m: int, tol):
    n = b.shape[0]

    if x0 is None:
        x0 = np.zeros(n)

    r0 = b - A @ x0
    beta = norm(r0)

    if beta < tol:
        return x0, 0.0 
    
    return "WIP!"