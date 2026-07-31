"""
glauber_config.py -- solver parameters for glauber_solve.py

Edit the values below to tune the run.  Every key is optional: glauber_solve.py
holds its own built-in defaults and only overrides the keys it finds here, so a
missing key (or a missing file entirely) is harmless.  Delete this file to fall
back to the built-in defaults.
"""

# ---------------------------------------------------------------------------
#  Physical parameters of the Glauber ring
# ---------------------------------------------------------------------------
PHYSICAL = {
    "J":     1.0,   # Ising coupling constant
    "kT":    1.5,   # temperature (k_B T); with J=1 the chain is disordered here
    "alpha": 1.0,   # overall transition-rate timescale (sets units of time only)
}
# The Glauber coupling gamma = tanh(2 J / kT) is derived from J and kT, not set
# here; |gamma| < 1 is required and holds automatically for kT > 0.

# ---------------------------------------------------------------------------
#  Two-site DMRG  (variational null vector of M = L^T L)
# ---------------------------------------------------------------------------
DMRG = {
    "maxrank":    10,     # max TT bond dim kept after each SVD truncation.
                          #   The stationary state has interior TT rank 4, so 12
                          #   is comfortable slack; ranks near 4-6 can trap the
                          #   minimisation in a poor local minimum.
    "init_rank":  2,      # TT rank of the random starting guess
    "eps":        1e-10,  # relative SVD truncation tolerance within a sweep
    "max_sweeps": 15,     # cap on the number of left+right sweeps
    "tol":        1e-9,   # stop early once |lambda| changes by less than this
                          #   between sweeps
    "seed":       0,      # RNG seed for the random initial guess (reproducible)
}

# ---------------------------------------------------------------------------
#  Bordered TT-GMRES  ((L + 1 1^T) rho = 1)
# ---------------------------------------------------------------------------
GMRES = {
    "max_rank":     7,    # max TT rank kept while rounding the Krylov vectors
    "gmres_m":      35,    # Arnoldi/restart length (inner iterations per cycle)
    "max_restarts": 12,    # maximum number of GMRES restarts
    "tol":          1e-9,  # relative-residual convergence tolerance
}
# NOTE: the fixed all-ones border loses alignment with the stationary vector as
# N grows, so TT-GMRES may report "NOT converged" at large N regardless of these
# knobs.  That is a property of the border, not of the tolerances.

# ---------------------------------------------------------------------------
#  Validation / reporting
# ---------------------------------------------------------------------------
VALIDATION = {
    "N_dense_max":     20,    # build the exact 2^N Gibbs vector (for the overlap
                              #   column) only up to this N; beyond it the dense
                              #   reference is skipped.  Memory ~ 2^N * 8 bytes.
    "converged_resid": 1e-4,  # ||L rho|| / ||rho|| below this is labelled
                              #   "converged" in the results table
}
