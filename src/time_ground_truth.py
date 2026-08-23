#!/usr/bin/env python3
"""
time_ground_truth.py -- measure how long it takes to build the exact "ground
truth" reference for the Glauber validation, as a function of chain size N.

The primary ground truth is the analytic 1-D Ising Gibbs measure pi
(gibbs_vector) -- the exact stationary distribution both solvers are scored
against.  With --matrix it also times the brute-force 2^N x 2^N generator L
(glauber_rate_matrix); that one costs O(4^N) memory and hits the wall much
sooner than the Gibbs vector.

Both objects are dense and of size 2^N, so build time grows exponentially in N;
this script shows exactly where that becomes painful.  Timing uses the best of a
few repeats (perf_counter, GC paused) to beat timer noise on the cheap sizes.

Usage:
    python3 time_ground_truth.py                 # sweep N = 4..20 (Gibbs only)
    python3 time_ground_truth.py 4 24            # sweep N = 4..24
    python3 time_ground_truth.py 4 14 --matrix   # also time the dense L (4^N mem)
"""

import sys
import gc
import time

import numpy as np

from glauber_mpo import gibbs_vector, glauber_rate_matrix, glauber_gamma

# physical regime (matches glauber_config.py defaults)
J, kT, ALPHA = 1.0, 1.5, 1.0

# don't attempt a dense object that would need more than this many GB
MEM_LIMIT_GB = 8.0


def best_time(fn, repeats, warmup=True):
    """Best-of-`repeats` wall time for fn(), GC paused; optional warmup call."""
    if warmup:
        fn()
    best = float("inf")
    for _ in range(repeats):
        gc.disable()
        t0 = time.perf_counter()
        fn()
        dt = time.perf_counter() - t0
        gc.enable()
        best = min(best, dt)
    return best


def pick_repeats(N):
    """Cheap builds need repeats to beat timer noise; expensive ones run once."""
    if N <= 12:
        return 7
    if N <= 18:
        return 2
    return 1


def main():
    flags = {a for a in sys.argv[1:] if a.startswith("-")}
    pos = [a for a in sys.argv[1:] if not a.startswith("-")]
    do_matrix = "--matrix" in flags or "-m" in flags
    N_lo = int(pos[0]) if len(pos) > 0 else 4
    N_hi = int(pos[1]) if len(pos) > 1 else 20
    gamma = glauber_gamma(J, kT)

    cols = ["N", "dim=2^N", "gibbs (s)", "s / state"]
    if do_matrix:
        cols += ["rate L (s)", "L mem (GB)"]
    print(f"\nGround-truth build time   J={J}  kT={kT}  alpha={ALPHA}  "
          f"gamma={gamma:.4f}")
    print(f"(best of a few repeats; --matrix also times the dense 2^N x 2^N L, "
          f"capped at {MEM_LIMIT_GB:g} GB)\n")
    w = 13
    print("  ".join(c.ljust(w) for c in cols))
    print("-" * (len(cols) * (w + 2)))

    for N in range(N_lo, N_hi + 1):
        dim = 2 ** N
        reps = pick_repeats(N)
        tg = best_time(lambda n=N: gibbs_vector(n, J, kT), reps)
        row = [str(N), f"{dim:,}", f"{tg:.4e}", f"{tg / dim:.2e}"]

        if do_matrix:
            mem_gb = dim * dim * 8 / 1e9            # dense float64, 2^N x 2^N
            if mem_gb > MEM_LIMIT_GB:
                row += ["skipped", f"{mem_gb:,.1f} (>cap)"]
            else:
                tm = best_time(lambda n=N: glauber_rate_matrix(n, ALPHA, gamma),
                               1, warmup=False)
                row += [f"{tm:.4e}", f"{mem_gb:.3f}"]

        print("  ".join(c.ljust(w) for c in row), flush=True)

    print("\nBoth references scale as 2^N (the Gibbs vector) / 4^N in memory "
          "(the dense L);\nthat exponential growth is the reason the writeup "
          "validates only at modest N.\n")


if __name__ == "__main__":
    main()
