import warnings

import numpy as np
import pytest

from glauber_dynamics import glauber


# Silence the harmless overflow in np.exp(dE/T) at low T (gives p_flip -> 0,
# which is the correct behavior).
pytestmark = pytest.mark.filterwarnings("ignore:overflow encountered in exp")


def all_up(shape):
    return np.ones(shape, dtype=np.int8)


# ---------------------------------------------------------------------------
# Invariants
# ---------------------------------------------------------------------------

def test_spins_stay_in_pm_one():
    rng = np.random.default_rng(0)
    lattice = rng.choice([-1, 1], size=(8, 8)).astype(np.int8)
    out = glauber(lattice.copy(), n_steps=5000, T=2.5)
    assert set(np.unique(out)).issubset({-1, 1})


def test_shape_and_dtype_preserved():
    lattice = all_up((6, 6))
    out = glauber(lattice.copy(), n_steps=1000, T=2.0)
    assert out.shape == (6, 6)
    assert out.dtype == lattice.dtype


def test_zero_steps_is_identity():
    lattice = np.random.default_rng(1).choice([-1, 1], size=(5, 5))
    out = glauber(lattice.copy(), n_steps=0, T=2.0)
    assert np.array_equal(out, lattice)


@pytest.mark.parametrize("shape", [(20,), (8, 8), (4, 4, 4), (3, 3, 3, 3)])
def test_runs_in_1d_2d_3d_4d(shape):
    lattice = all_up(shape)
    out = glauber(lattice.copy(), n_steps=2000, T=2.0)
    assert out.shape == shape
    assert set(np.unique(out)).issubset({-1, 1})


# ---------------------------------------------------------------------------
# Input handling
# ---------------------------------------------------------------------------

def test_accepts_list_input():
    # A plain list is converted to an array internally; should run and return
    # an ndarray (and leaves the caller's list untouched).
    out = glauber([1, -1, 1, -1, 1, -1], n_steps=500, T=2.0)
    assert isinstance(out, np.ndarray)
    assert set(np.unique(out)).issubset({-1, 1})


def test_input_array_not_mutated():
    # The function works on a copy: the caller's array is left untouched and a
    # new object is returned, even at high T where flips almost surely happen.
    lattice = all_up((10, 10))
    returned = glauber(lattice, n_steps=20000, T=100.0)
    assert returned is not lattice  # a fresh array comes back
    assert np.array_equal(lattice, all_up((10, 10)))  # original unchanged
    assert not np.array_equal(returned, lattice)  # but the result evolved


# ---------------------------------------------------------------------------
# Determinism (must pass a fresh copy each call, since the arg is mutated)
# ---------------------------------------------------------------------------

def test_same_seed_same_result():
    base = np.random.default_rng(7).choice([-1, 1], size=(12, 12))
    a = glauber(base.copy(), n_steps=5000, seed=123, T=2.0)
    b = glauber(base.copy(), n_steps=5000, seed=123, T=2.0)
    assert np.array_equal(a, b)


def test_different_seed_different_result():
    base = np.random.default_rng(7).choice([-1, 1], size=(12, 12))
    a = glauber(base.copy(), n_steps=5000, seed=1, T=2.0)
    b = glauber(base.copy(), n_steps=5000, seed=2, T=2.0)
    assert not np.array_equal(a, b)


# ---------------------------------------------------------------------------
# Physics (deterministic via fixed seed; thresholds are generous)
# ---------------------------------------------------------------------------

def test_low_temperature_preserves_order():
    # Deep in the ordered phase, an all-aligned state should stay aligned:
    # flipping a spin against all its neighbors is strongly suppressed.
    lattice = all_up((16, 16))
    out = glauber(lattice, n_steps=50000, seed=42, T=0.5)
    assert abs(out.mean()) > 0.8


def test_high_temperature_destroys_order():
    # Far above T_c (~2.27), thermal noise disorders an initially aligned state.
    lattice = all_up((16, 16))
    out = glauber(lattice, n_steps=50000, seed=42, T=100.0)
    assert abs(out.mean()) < 0.3


# ---------------------------------------------------------------------------
# Vectorized (checkerboard) path
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("shape", [(9,), (8, 9), (3, 3, 3, 3)])
def test_vectorized_requires_even_dims(shape):
    # The checkerboard update is only correct on a bipartite lattice, which
    # under periodic boundaries means every axis must have even length.
    with pytest.raises(ValueError):
        glauber(all_up(shape), n_steps=1000, T=2.0, vectorized=True)


def test_vectorized_zero_steps_is_identity():
    lattice = np.random.default_rng(3).choice([-1, 1], size=(8, 8))
    out = glauber(lattice.copy(), n_steps=0, T=2.0, vectorized=True)
    assert np.array_equal(out, lattice)


def test_vectorized_spins_stay_in_pm_one():
    lattice = np.random.default_rng(4).choice([-1, 1], size=(8, 8, 8))
    out = glauber(lattice.copy(), n_steps=20000, T=2.5, vectorized=True)
    assert set(np.unique(out)).issubset({-1, 1})


def test_vectorized_determinism():
    # Same seed -> identical trajectory (each call gets a fresh copy + seed).
    base = np.random.default_rng(5).choice([-1, 1], size=(8, 8))
    a = glauber(base.copy(), n_steps=10000, seed=11, T=2.0, vectorized=True)
    b = glauber(base.copy(), n_steps=10000, seed=11, T=2.0, vectorized=True)
    assert np.array_equal(a, b)


@pytest.mark.parametrize("T,predicate", [
    (0.5, lambda m: m > 0.8),    # ordered below T_c
    (100.0, lambda m: m < 0.3),  # disordered far above T_c
])
def test_vectorized_matches_scalar_equilibrium(T, predicate):
    # The two update schemes draw different randomness, so they can't match
    # element-wise; they must agree on the *physics*. Both reach the same
    # equilibrium |M| regime on an even lattice.
    shape = (16, 16)
    n_steps = 80000
    scalar = glauber(all_up(shape), n_steps=n_steps, seed=42, T=T)
    vector = glauber(all_up(shape), n_steps=n_steps, seed=42, T=T, vectorized=True)
    assert predicate(abs(scalar.mean()))
    assert predicate(abs(vector.mean()))


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
