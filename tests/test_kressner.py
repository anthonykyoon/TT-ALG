import numpy as np
import pytest
from kressner import kressner_algo   # change to actual filename


def test_square_matrix_small():
    A = np.array([[2.0, 1.0],
                  [1.0, 3.0]])
    I, J = kressner_algo(A, tolerance=1e-8, k=2)
    assert isinstance(I, list)
    assert isinstance(J, list)
    assert all(isinstance(i, int) for i in I)
    assert all(isinstance(j, int) for j in J)
    assert len(I) == 2
    assert len(J) == 2
    approximation = A[:, J] @ np.linalg.inv(A[np.ix_(I, J)]) @ A[I, :]
    error = np.linalg.norm(A - approximation)
    assert error < 1e-8

def test_rectangular_matrix():
    A = np.array([[1., 2., 10.],
                  [4., 5., 8.]])
    I, J = kressner_algo(A, tolerance=1e-8, k=2)
    assert len(I) == 2
    assert len(J) == 2

def test_random_matrix_reproducible():
    np.random.seed(0)
    A = np.random.randn(4, 4)
    I1, J1 = kressner_algo(A, tolerance=1e-6, k=3)

    np.random.seed(0)
    A = np.random.randn(4, 4)
    I2, J2 = kressner_algo(A, tolerance=1e-6, k=3)

    assert I1 == I2
    assert J1 == J2

def test_invalid_non_square_allowed():
    A = np.random.randn(5, 3)
    k = min(A.shape)
    with pytest.raises(ValueError):
        kressner_algo(A, tolerance=1e-6, k=1)

def test_invalid_inputs():
    A = np.array([1, 2, 3])  # not 2D
    with pytest.raises(ValueError):
        kressner_algo(A, tolerance=1e-6, k=1)