import numpy as np 


def glauber(lattice,  n_steps: int, seed = 60637, T = 2.0, rng = None,
            vectorized = False):
    # periodic boundary conditions.
    # Work on a copy so the caller's input is never mutated, regardless of
    # whether a list or an ndarray was passed in.
    lattice = np.array(lattice, copy=True)

    # dimensions
    dimensions = lattice.shape
    num_dims = len(dimensions)

    # Pass an existing Generator (`rng=`) to continue one Markov chain across
    # repeated calls; otherwise a fresh, seeded stream is used.
    if rng is None:
        rng = np.random.default_rng(seed)

    if vectorized:
        return _glauber_vectorized(lattice, n_steps, T, rng)

    for iteration in range(n_steps):
        idx = tuple(rng.integers(0, dim_size) for dim_size in dimensions)
        
        S = 0
        for axis in range(num_dims):
            for direction in (+1, -1):
                neighbor_idx = list(idx)
                neighbor_idx[axis] = (idx[axis] + direction) % dimensions[axis]
                S += lattice[tuple(neighbor_idx)]

        dE = 2 * lattice[idx] * S

        p_flip = 1 / (1 + np.exp(dE / T))
        if rng.random() < p_flip:
            lattice[idx] *= -1

    return lattice


def _glauber_vectorized(lattice, n_steps, T, rng):
    """Checkerboard (red/black) Glauber update.

    Sites of one color are mutually non-adjacent, so an entire color can be
    flipped in one vectorized pass. This requires an *even* side length along
    every axis: under periodic boundaries an odd-length axis is not bipartite
    (its sites cannot be 2-colored without two same-color neighbors meeting),
    so the parallel update would be incorrect.

    `n_steps` is counted in single-site attempts, matching the scalar path:
    one full sweep (both colors) touches every site once, so the number of
    sweeps performed is ``n_steps // lattice.size`` (and n_steps=0 is a no-op).
    """
    dimensions = lattice.shape
    num_dims = lattice.ndim

    if any(d % 2 != 0 for d in dimensions):
        raise ValueError(
            "vectorized=True requires an even length along every axis "
            "(the checkerboard update needs a bipartite lattice under "
            f"periodic boundaries); got shape {dimensions}"
        )

    # Parity 2-coloring: color[i,j,...] = (i + j + ...) % 2.
    coords = np.indices(dimensions)
    color = coords.sum(axis=0) % 2

    n_sites = lattice.size
    n_sweeps = n_steps // n_sites
    for _ in range(n_sweeps):
        for c in (0, 1):
            # Neighbor sum via periodic shifts; recomputed for each color
            # because flipping color 0 changes color 1's local fields.
            S = np.zeros(dimensions, dtype=np.int16)
            for axis in range(num_dims):
                S += np.roll(lattice, 1, axis=axis)
                S += np.roll(lattice, -1, axis=axis)

            dE = 2 * lattice * S
            p_flip = 1.0 / (1.0 + np.exp(dE / T))
            flip = (color == c) & (rng.random(dimensions) < p_flip)
            lattice[flip] *= -1

    return lattice








    
