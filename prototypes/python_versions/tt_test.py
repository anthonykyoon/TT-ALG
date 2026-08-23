# test_tt_algos.py

import numpy as np
import random

# ─── Adjust these imports to wherever you put your ported routines ───────────
from tt_cross_cort import derandomized_cross_approximation, TT_Cross_cort
# ─────────────────────────────────────────────────────────────────────────────

def random_tensor(dimension: int) -> np.ndarray:
    """
    Generate a random 5-D tensor with shape (2, 100, rand(1..10), dimension, 10).
    """
    dim_1 = 2
    dim_2 = 100
    dim_3 = random.randint(1, 10)
    dim_4 = dimension
    return np.random.rand(dim_1, dim_2, dim_3, dim_4, 10)


def random_tensor_train(ranks: list[int], dims: list[int]) -> list[np.ndarray]:
    """
    Generate a random TT-train given a rank-vector and a dims-vector.
    ranks must have length = len(dims)+1, and ranks[0]=ranks[-1]=1.
    """
    if len(ranks) != len(dims) + 1:
        raise ValueError("`ranks` must have length len(dims)+1")
    if ranks[0] != 1 or ranks[-1] != 1:
        raise ValueError("First and last TT-rank must be 1")

    tt = []
    for i in range(len(dims)):
        tt.append(np.random.rand(ranks[i], dims[i], ranks[i+1]))
    return tt


def direct_sub(tt_a: list[np.ndarray], tt_b: list[np.ndarray]) -> list[np.ndarray]:
    """
    Elementwise subtraction of two TT-trains of the same format.
    """
    if len(tt_a) != len(tt_b):
        raise ValueError("Both TT-trains must have the same length")
    result = []
    for A_core, B_core in zip(tt_a, tt_b):
        if A_core.shape != B_core.shape:
            raise ValueError("Core shapes must match")
        result.append(A_core - B_core)
    return result


if __name__ == "__main__":
    # ─── Reproducible randomness ────────────────────────────────────────────────
    SEED = 123456
    random.seed(SEED)
    np.random.seed(SEED)
    # ─────────────────────────────────────────────────────────────────────────────

    # 1) Test random_tensor & TT_Cross_cort
    test_tensor = random_tensor(3)
    print("test_tensor shape:", test_tensor.shape)

    cores = TT_Cross_cort(test_tensor, k=10)
    print(f"TT_Cross_cort returned {len(cores)} cores:")
    for idx, core in enumerate(cores, start=1):
        print(f"  core {idx:>2}: shape = {core.shape}")

    # 2) Test random_tensor_train & direct_sub
    test_ranks = [1, 3, 2, 4, 9, 1]
    test_dims  = [  3, 4, 3, 2, 4]
    tt_train = random_tensor_train(test_ranks, test_dims)
    print("random_tensor_train core shapes:", [c.shape for c in tt_train])

    # subtract the TT-train from itself → should get all zeros
    tt_zero = direct_sub(tt_train, tt_train)
    assert all(np.allclose(core, 0) for core in tt_zero)
    print("direct_sub ✓ all‐zero cores as expected")

    # 3) Test matrix cross‐approximation
    test_matrix = np.random.rand(10, 15)
    I, J = derandomized_cross_approximation(test_matrix, k=6)
    print("derandomized_cross_approximation →")
    print("  I =", I)
    print("  J =", J)
