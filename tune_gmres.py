#!/usr/bin/env python3
"""
tune_gmres.py -- grid-search good bordered-TT-GMRES parameters for the Glauber
stationary state.

Why a search helps: this is inexact (truncated) GMRES, so max_rank sets a hard
floor on the achievable residual -- gmres_m and max_restarts cannot beat it.
The efficient search is therefore mostly one-dimensional in max_rank: raise it
until the physical residual ||L rho|| / ||rho|| clears the target, then use just
enough restarts.  The residual is evaluated in TT form, so this works at any N
(no 2^N reference needed); at small N the exact Gibbs overlap is also reported.

Usage:
    python3 tune_gmres.py            # prompts for N
    python3 tune_gmres.py 8          # N from the command line
"""

import sys
import time

import numpy as np
from numpy.linalg import norm

from glauber_mpo import glauber_gamma, gibbs_vector
from glauber_gmres import glauber_stationary_gmres
from tt_dmrg import tt_to_dense

# Physical regime (match glauber_config.py if you want).
J, kT, ALPHA = 1.0, 1.5, 1.0

# Search grid.  max_rank is the binding knob; the others are held generous.
RANK_GRID   = [6, 8, 10, 12, 16, 20, 24]
M_GRID      = [40]        # add more values to sweep restart length too
MAX_RESTARTS = 30         # generous; converges if the rank floor allows
TOL          = 1e-9       # inner GMRES tolerance
TARGET_RESID = 1e-6       # physical ||L rho||/||rho|| we call "good enough"
N_DENSE_MAX  = 16         # build exact Gibbs overlap only up to this N


def prompt_for_N():
    raw = sys.argv[1] if len(sys.argv) > 1 else input("Enter chain size N (>= 2): ").strip()
    try:
        N = int(raw)
    except ValueError:
        print(f"'{raw}' is not an integer."); sys.exit(1)
    if N < 2:
        print("N must be >= 2."); sys.exit(1)
    return N


def main():
    N = prompt_for_N()
    gamma = glauber_gamma(J, kT)
    dense_ok = N <= N_DENSE_MAX
    pi = gibbs_vector(N, J, kT) if dense_ok else None

    print(f"\nGrid search: N={N}, J={J}, kT={kT}, gamma={gamma:.4f}")
    print(f"target physical residual ||L rho||/||rho|| < {TARGET_RESID:.0e}, "
          f"tol={TOL:.0e}, max_restarts={MAX_RESTARTS}")
    print(f"sweeping max_rank in {RANK_GRID}, gmres_m in {M_GRID}\n")

    header = ["max_rank", "gmres_m", "time(s)", "restarts", "status",
              "||L rho||/||rho||", "overlap(pi)", "meets target"]
    rows = []
    candidates = []  # dicts for configs that meet the target

    for gm in M_GRID:
        for mr in RANK_GRID:
            t = time.time()
            rho, resL, info = glauber_stationary_gmres(
                N, ALPHA, gamma, max_rank=mr, gmres_m=gm,
                max_restarts=MAX_RESTARTS, tol=TOL, verbose=False)
            dt = time.time() - t
            nrestarts = len(info["restart_history"])
            ov = "n/a"
            if dense_ok:
                v = tt_to_dense(rho); v = v if v @ pi >= 0 else -v
                ov = f"{abs(v @ pi) / (norm(v) * norm(pi)):.6f}"
            meets = resL < TARGET_RESID
            rows.append([str(mr), str(gm), f"{dt:.1f}", str(nrestarts),
                         info["status"], f"{resL:.2e}", ov, "yes" if meets else "no"])
            if meets:
                candidates.append({"mr": mr, "gm": gm, "dt": dt,
                                   "restarts": nrestarts})
            print(f"  max_rank={mr:2d} gmres_m={gm}: {info['status']:12s} "
                  f"resid={resL:.2e} {'OK' if meets else '--'} "
                  f"({nrestarts} restarts, {dt:.1f}s)", flush=True)

    # Recommendation: max_rank is the binding, cost-dominating knob, and small-N
    # wall times are noise-dominated. So pick by (fewest restarts, then smallest
    # max_rank, then smallest gmres_m) -- integer-valued, noise-free keys that
    # favour the cheapest robust config rather than a microsecond timing fluke.
    best = min(candidates,
               key=lambda c: (c["restarts"], c["mr"], c["gm"])) if candidates else None

    # ---- summary table ----
    widths = [max(len(r[i]) for r in ([header] + rows)) for i in range(len(header))]
    def line(c): return "  ".join(str(x).ljust(w) for x, w in zip(c, widths))
    print("\n" + "=" * 60)
    print(f"RESULTS  (N = {N})")
    print("=" * 60)
    print(line(header))
    print("  ".join("-" * w for w in widths))
    for r in rows:
        print(line(r))

    print()
    if best is None:
        print(f"No config in the grid reached ||L rho||/||rho|| < {TARGET_RESID:.0e}.")
        print("The max_rank floor is above target: extend RANK_GRID upward.")
    else:
        print(f"RECOMMENDED (fewest restarts, then smallest max_rank): "
              f"max_rank={best['mr']}, gmres_m={best['gm']} "
              f"({best['restarts']} restarts, {best['dt']:.1f}s).")
        print("Set these in glauber_config.py under GMRES.")
    print()


if __name__ == "__main__":
    main()
