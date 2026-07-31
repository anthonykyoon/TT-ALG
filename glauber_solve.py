#!/usr/bin/env python3
"""
glauber_solve.py -- interactive Glauber stationary-state solver.

Prompts for a chain size N, solves  L rho = 0  on the Glauber ring with BOTH
    * two-site DMRG   (variational null vector of M = L^T L), and
    * bordered TT-GMRES  ((L + 1 1^T) rho = 1, non-symmetric, no L^T L),
and prints the results as a table.

Reference:
    For small N the exact 1-D Ising Gibbs measure pi (the analytic stationary
    state) is built and each solver's overlap with it is reported.  For large N
    the 2^N reference is infeasible, so only TT-native metrics are shown:
        - the null-vector residual  ||L rho|| / ||rho||   (computed in TT form),
        - the max TT rank and wall-clock time, and
        - a DMRG-vs-GMRES cross-check (two independent methods agreeing ~1 is
          strong evidence both are correct).

Solver parameters live in glauber_config.py; edit that file to tune the run.
Any key missing there (or a missing file) falls back to the defaults below.

Usage:
    python3 glauber_solve.py            # prompts for N
    python3 glauber_solve.py 40         # takes N from the command line
"""

import sys
import time

import numpy as np
from numpy.linalg import norm

from glauber_mpo import glauber_terms, glauber_gamma, gibbs_vector
from tt_dmrg import mpo_from_terms, mpo_round, dmrg_null_vector, tt_to_dense
from glauber_gmres import glauber_stationary_gmres
from tt_gmres import tt_dot, tt_frobenius

# --- built-in defaults; overridden per-key by glauber_config.py if present ----
DEFAULTS = {
    "PHYSICAL":   {"J": 1.0, "kT": 1.5, "alpha": 1.0},
    "DMRG":       {"maxrank": 12, "init_rank": 2, "eps": 1e-10,
                   "max_sweeps": 20, "tol": 1e-9, "seed": 0},
    "GMRES":      {"max_rank": 24, "gmres_m": 40, "max_restarts": 20, "tol": 1e-9},
    "VALIDATION": {"N_dense_max": 20, "converged_resid": 1e-4},
}


def load_config():
    """Merge glauber_config.py over DEFAULTS (per key); tolerate a missing file."""
    cfg = {sec: dict(vals) for sec, vals in DEFAULTS.items()}
    try:
        import glauber_config as user
        found = True
    except Exception as e:                       # missing file or a syntax error
        print(f"(no usable glauber_config.py -- using built-in defaults; {e})")
        return cfg, False
    for sec in cfg:
        overrides = getattr(user, sec, None)
        if isinstance(overrides, dict):
            unknown = set(overrides) - set(cfg[sec])
            if unknown:
                print(f"(warning: unknown {sec} keys ignored: {sorted(unknown)})")
            cfg[sec].update({k: v for k, v in overrides.items() if k in cfg[sec]})
    return cfg, found


def tt_cosine(a, b):
    """Cosine similarity of two TT-vectors, computed without contracting them."""
    return abs(tt_dot(a, b)) / (tt_frobenius(a) * tt_frobenius(b))


def prompt_for_N():
    """Get N from argv[1] if present, otherwise prompt interactively."""
    if len(sys.argv) > 1:
        raw = sys.argv[1]
    else:
        try:
            raw = input("Enter chain size N (integer >= 2): ").strip()
        except EOFError:
            print("No input received.")
            sys.exit(1)
    try:
        N = int(raw)
    except ValueError:
        print(f"'{raw}' is not an integer.")
        sys.exit(1)
    if N < 2:
        print("N must be >= 2.")
        sys.exit(1)
    return N


def print_table(rows, headers):
    """Print a list-of-rows table with aligned, padded columns."""
    cols = list(zip(*([headers] + rows)))
    widths = [max(len(str(c)) for c in col) for col in cols]
    sep = "  "

    def line(cells):
        return sep.join(str(c).ljust(w) for c, w in zip(cells, widths))

    print(line(headers))
    print(sep.join("-" * w for w in widths))
    for r in rows:
        print(line(r))


def sci(x):
    return f"{x:.2e}"


def main():
    cfg, _ = load_config()
    phys, dmrg_cfg, gmres_cfg, val_cfg = (
        cfg["PHYSICAL"], cfg["DMRG"], cfg["GMRES"], cfg["VALIDATION"])
    J, kT, ALPHA = phys["J"], phys["kT"], phys["alpha"]

    N = prompt_for_N()
    gamma = glauber_gamma(J, kT)
    dense_ok = N <= val_cfg["N_dense_max"]

    print(f"\nGlauber ring:  N = {N},  J = {J},  kT = {kT},  alpha = {ALPHA}")
    print(f"gamma = tanh(2J/kT) = {gamma:.4f}")
    print(f"state space = 2^{N} = {2.0**N:.3e}")
    if dense_ok:
        print("exact Ising Gibbs reference: available (overlap validation shown)")
    else:
        print(f"exact Gibbs reference: INFEASIBLE (2^{N} too large) -- "
              "TT-native metrics only")
    if N > 12:
        print("\nNOTE: runtime grows steeply with N. DMRG forms the dense two-site\n"
              "operator of M = L^T L (bond ~36) at every step, so expect seconds for\n"
              f"N<=10, ~1 min near N=12, and several minutes by N~16-20. This reference\n"
              "implementation is meant for modest N; very large N (e.g. 60+) is not\n"
              "practical here. Bordered TT-GMRES may also fail to converge at large N\n"
              "(the fixed all-ones border loses alignment with the stationary vector).")
    print()

    # --- build the Glauber generator as an MPO (rank 6, no exponential cost) ---
    t = time.time()
    L = mpo_round(mpo_from_terms(glauber_terms(N, ALPHA, gamma), N), eps=1e-12)
    t_mpo = time.time() - t
    mpo_rank = max(c.shape[0] for c in L)
    print(f"built Glauber MPO: max bond dim {mpo_rank}  ({t_mpo:.2f}s)\n")

    # --- DMRG: variational null vector of M = L^T L ---
    print("running DMRG ...", flush=True)
    t = time.time()
    G, lam, res_dmrg = dmrg_null_vector(
        L, maxrank=dmrg_cfg["maxrank"], init_rank=dmrg_cfg["init_rank"],
        eps=dmrg_cfg["eps"], max_sweeps=dmrg_cfg["max_sweeps"],
        tol=dmrg_cfg["tol"], seed=dmrg_cfg["seed"], verbose=False)
    t_dmrg = time.time() - t
    rank_dmrg = max(c.shape[0] for c in G)

    # --- bordered TT-GMRES: non-symmetric solve, never forms L^T L ---
    print("running bordered TT-GMRES ...", flush=True)
    t = time.time()
    rho, res_gmres, info = glauber_stationary_gmres(
        N, ALPHA, gamma, max_rank=gmres_cfg["max_rank"],
        gmres_m=gmres_cfg["gmres_m"], max_restarts=gmres_cfg["max_restarts"],
        tol=gmres_cfg["tol"], verbose=False)
    t_gmres = time.time() - t
    rank_gmres = max(c.shape[0] for c in rho)

    # --- optional overlap against the exact Gibbs measure ---
    ov_dmrg = ov_gmres = None
    if dense_ok:
        pi = gibbs_vector(N, J, kT)
        rd = tt_to_dense(G)
        rd = rd if rd @ pi >= 0 else -rd
        rg = tt_to_dense(rho)
        rg = rg if rg @ pi >= 0 else -rg
        ov_dmrg = abs(rd @ pi) / (norm(rd) * norm(pi))
        ov_gmres = abs(rg @ pi) / (norm(rg) * norm(pi))

    # --- two independent solvers should agree (TT-native cross-check) ---
    xcos = tt_cosine(G, rho)

    # ---- results table ----
    print("\n" + "=" * 60)
    print(f"RESULTS  (N = {N})")
    print("=" * 60)
    # Verdict is judged on the PHYSICAL residual ||L rho||/||rho|| -- the quality
    # of the stationary state -- not on the bordered GMRES tolerance. A GMRES run
    # can plateau above its internal tol (max_rank truncation floor) while still
    # delivering an excellent null vector; that is reported as "stagnated", not a
    # failure.
    def verdict(res):
        return "converged" if res < val_cfg["converged_resid"] else "NOT converged"

    def gmres_verdict(res, gmres_info):
        if res < val_cfg["converged_resid"]:
            return "converged"
        return "stagnated" if gmres_info.get("stagnated") else "NOT converged"

    headers = ["method", "time (s)", "||L rho||/||rho||", "max rank",
               "overlap(pi)", "status"]
    rows = [
        ["DMRG", f"{t_dmrg:.1f}", sci(res_dmrg), str(rank_dmrg),
         f"{ov_dmrg:.8f}" if ov_dmrg is not None else "n/a", verdict(res_dmrg)],
        ["TT-GMRES", f"{t_gmres:.1f}", sci(res_gmres), str(rank_gmres),
         f"{ov_gmres:.8f}" if ov_gmres is not None else "n/a",
         gmres_verdict(res_gmres, info)],
    ]
    print_table(rows, headers)

    print("\nsolver detail:")
    print(f"  DMRG      lambda_min = {sci(lam)}")
    print(f"  TT-GMRES  status = {info['status']}   "
          f"bordered rel.res = {sci(info['final_relative_residual'])}   "
          f"physical ||L rho||/||rho|| = {sci(res_gmres)}")
    if info["status"] == "stagnated" and res_gmres >= val_cfg["converged_resid"]:
        print("            (residual hit the max_rank truncation floor; "
              "raise GMRES max_rank in glauber_config.py to go lower)")
    elif info["status"] == "max_restarts" and res_gmres >= val_cfg["converged_resid"]:
        print("            (still converging when restarts ran out; "
              "raise GMRES max_restarts in glauber_config.py)")
    print(f"  cross-check  TT cosine( DMRG , TT-GMRES ) = {xcos:.8f}")
    if not dense_ok:
        print("  (overlap(pi) is 'n/a' because the exact 2^N Gibbs vector was "
              "not built; the residual and cross-check above are the evidence.)")
    print()


if __name__ == "__main__":
    main()
