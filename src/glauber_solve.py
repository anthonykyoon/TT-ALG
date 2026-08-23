#!/usr/bin/env python3
"""
glauber_solve.py -- sweep the Glauber stationary-state solvers over N and save.

Set a maximum chain size N (in glauber_config.py under SWEEP, or on the command
line) and this runs, for every N from N_min up to N_max, both solvers of
    L rho = 0  on the Glauber ring:
    * two-site DMRG   (variational null vector of M = L^T L), and
    * bordered TT-GMRES  ((L + 1 1^T) rho = 1, non-symmetric, no L^T L),
prints a per-N table, and writes ALL results to a CSV file you can inspect later.

Reference:
    For small N the exact 1-D Ising Gibbs measure pi (the analytic stationary
    state) is built and each solver's overlap with it is reported.  For large N
    the 2^N reference is infeasible, so only TT-native metrics are shown:
        - the null-vector residual  ||L rho|| / ||rho||   (computed in TT form),
        - the max TT rank and wall-clock time, and
        - a DMRG-vs-GMRES cross-check (two independent methods agreeing ~1 is
          strong evidence both are correct).

Solver parameters and the sweep range / output file live in glauber_config.py;
edit that file to tune the run.  Any key missing there (or a missing file) falls
back to the defaults below.

Run it from the repo root; a relative results file is written to its own
folder, results/<stem>/<stem>.csv, so each sweep's CSV and plots stay together.

Usage:
    python3 src/glauber_solve.py            # sweep N_min..N_max from the config
    python3 src/glauber_solve.py 12         # override: sweep N_min..12
    python3 src/glauber_solve.py 4 12       # override: sweep N = 4..12
    python3 src/glauber_solve.py 4 12 out.csv   # ... -> results/out/out.csv
"""

import os
import sys
import csv
import time
from datetime import datetime

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
    "SWEEP":      {"N_min": 2, "N_max": 8, "results_file": "glauber_results.csv"},
}

HERE = os.path.dirname(os.path.abspath(__file__))

# Sweep output goes in one folder per sweep, named after the CSV, under results/.
# In the repo the solver lives in src/, so results/ is its sibling; in the flat
# standalone bundle there is no src/, so results/ sits inside the bundle itself.
REPO = os.path.dirname(HERE) if os.path.basename(HERE) == "src" else HERE
RESULTS_DIR = os.path.join(REPO, "results")

# An overlap this close to 1 (i.e. 1 - overlap at or below this) is flagged in the
# CSV as "within machine precision".  Iterative solves land a few 1e-10 short of
# the exact float 1.0, so this lets an exact 1 and a 1 - 4e-10 read as the same
# (effectively perfect) result instead of looking different downstream.
OVERLAP_MACHINE_PREC_TOL = 1e-9


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


def parse_sweep_args(cfg):
    """Resolve (N_min, N_max, results_file) from argv, falling back to config.

    argv forms (all optional):  [N_max] | [N_min N_max] | [N_min N_max file]
    A lone integer is treated as N_max.
    """
    sweep = cfg["SWEEP"]
    N_min, N_max = int(sweep["N_min"]), int(sweep["N_max"])
    results_file = sweep["results_file"]

    ints = [a for a in sys.argv[1:] if a.lstrip("+-").isdigit()]
    non_ints = [a for a in sys.argv[1:] if not a.lstrip("+-").isdigit()]
    if len(ints) == 1:
        N_max = int(ints[0])
    elif len(ints) >= 2:
        N_min, N_max = int(ints[0]), int(ints[1])
    if non_ints:
        results_file = non_ints[0]

    if N_min < 2:
        print("N_min must be >= 2; clamping to 2.")
        N_min = 2
    if N_max < N_min:
        print(f"N_max ({N_max}) < N_min ({N_min}); nothing to do.")
        sys.exit(1)

    # resolve a relative name into its own results folder: results/<stem>/<stem>.csv
    if not os.path.isabs(results_file):
        stem, ext = os.path.splitext(os.path.basename(results_file))
        results_file = os.path.join(RESULTS_DIR, stem, stem + (ext or ".csv"))
    return N_min, N_max, results_file


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
    return f"{x:.2e}" if x is not None else "n/a"


def machine_prec_flag(overlap):
    """'yes'/'no' if overlap is (not) within machine precision of 1; '' if n/a."""
    if overlap is None:
        return ""
    return "yes" if (1.0 - overlap) <= OVERLAP_MACHINE_PREC_TOL else "no"


def dmrg_status(res, converged_resid):
    return "converged" if res < converged_resid else "NOT converged"


def gmres_status(res, info, converged_resid):
    if res < converged_resid:
        return "converged"
    return "stagnated" if info.get("stagnated") else "NOT converged"


def run_one(N, cfg):
    """Run both solvers at a single N and return a structured result dict.

    Each solver is guarded independently: a failure at one N (or in one solver)
    is recorded as an error row and does not abort the sweep.
    """
    phys, dmrg_cfg, gmres_cfg, val_cfg = (
        cfg["PHYSICAL"], cfg["DMRG"], cfg["GMRES"], cfg["VALIDATION"])
    J, kT, ALPHA = phys["J"], phys["kT"], phys["alpha"]
    gamma = glauber_gamma(J, kT)
    dense_ok = N <= val_cfg["N_dense_max"]
    cr = val_cfg["converged_resid"]

    res = {"N": N, "dim": 2 ** N, "gamma": gamma, "dense_ok": dense_ok,
           "dmrg": {"error": None}, "gmres": {"error": None},
           "cross_check_cosine": None, "t_gibbs": None}

    # --- build the Glauber generator as an MPO (rank 6, no exponential cost) ---
    t = time.perf_counter()
    L = mpo_round(mpo_from_terms(glauber_terms(N, ALPHA, gamma), N), eps=1e-12)
    res["t_mpo"] = time.perf_counter() - t
    res["mpo_rank"] = max(c.shape[0] for c in L)

    # --- DMRG: variational null vector of M = L^T L ---
    print("  running DMRG ...", flush=True)
    G = None
    try:
        t = time.perf_counter()
        G, lam, res_dmrg = dmrg_null_vector(
            L, maxrank=dmrg_cfg["maxrank"], init_rank=dmrg_cfg["init_rank"],
            eps=dmrg_cfg["eps"], max_sweeps=dmrg_cfg["max_sweeps"],
            tol=dmrg_cfg["tol"], seed=dmrg_cfg["seed"], verbose=False)
        res["dmrg"].update(
            time=time.perf_counter() - t, residual=res_dmrg,
            max_rank=max(c.shape[0] for c in G), lambda_min=lam, overlap=None,
            status=dmrg_status(res_dmrg, cr))
    except Exception as e:                                   # keep the sweep alive
        res["dmrg"].update(error=str(e), status="ERROR")
        print(f"    DMRG failed: {e}")

    # --- bordered TT-GMRES: non-symmetric solve, never forms L^T L ---
    print("  running bordered TT-GMRES ...", flush=True)
    rho = None
    try:
        t = time.perf_counter()
        rho, res_gmres, info = glauber_stationary_gmres(
            N, ALPHA, gamma, max_rank=gmres_cfg["max_rank"],
            gmres_m=gmres_cfg["gmres_m"], max_restarts=gmres_cfg["max_restarts"],
            tol=gmres_cfg["tol"], verbose=False)
        res["gmres"].update(
            time=time.perf_counter() - t, residual=res_gmres,
            max_rank=max(c.shape[0] for c in rho), overlap=None,
            bordered_relres=info["final_relative_residual"],
            status=gmres_status(res_gmres, info, cr))
    except Exception as e:
        res["gmres"].update(error=str(e), status="ERROR")
        print(f"    TT-GMRES failed: {e}")

    # --- optional overlap against the exact Gibbs measure (small N only) ---
    if dense_ok:
        # time the build of the exact ground-truth stationary state pi itself,
        # so the reference cost sits alongside the solver times per N. This is
        # skipped (t_gibbs stays None) once N exceeds N_dense_max, where the 2^N
        # vector is infeasible to build.
        t = time.perf_counter()
        pi = gibbs_vector(N, J, kT)
        res["t_gibbs"] = time.perf_counter() - t
        if G is not None:
            rd = tt_to_dense(G)
            rd = rd if rd @ pi >= 0 else -rd
            res["dmrg"]["overlap"] = abs(rd @ pi) / (norm(rd) * norm(pi))
        if rho is not None:
            rg = tt_to_dense(rho)
            rg = rg if rg @ pi >= 0 else -rg
            res["gmres"]["overlap"] = abs(rg @ pi) / (norm(rg) * norm(pi))

    # --- two independent solvers should agree (TT-native cross-check) ---
    if G is not None and rho is not None:
        res["cross_check_cosine"] = tt_cosine(G, rho)

    return res


def print_one(res):
    """Print the per-N results table and detail block for one run."""
    N = res["N"]
    print("\n" + "=" * 60)
    print(f"RESULTS  (N = {N},  dim = 2^{N} = {res['dim']:,})")
    print("=" * 60)
    print(f"built Glauber MPO: max bond dim {res['mpo_rank']}  "
          f"({res['t_mpo']:.2f}s)")
    if res.get("t_gibbs") is not None:
        print(f"built exact ground-truth pi (2^{N} Gibbs vector) "
              f"in {res['t_gibbs']:.4f}s")

    def ov(d):
        if d.get("overlap") is None:
            return "n/a"
        tag = " (mp)" if machine_prec_flag(d["overlap"]) == "yes" else ""
        return f"{d['overlap']:.8f}{tag}"

    def cell(d, key, fmt):
        return fmt(d[key]) if d.get("error") is None and key in d else "-"

    headers = ["method", "time (s)", "||L rho||/||rho||", "max rank",
               "overlap(pi)", "status"]
    rows = []
    for name, key in (("DMRG", "dmrg"), ("TT-GMRES", "gmres")):
        d = res[key]
        rows.append([
            name,
            cell(d, "time", lambda x: f"{x:.1f}"),
            cell(d, "residual", sci),
            cell(d, "max_rank", str),
            ov(d),
            d.get("status", "-"),
        ])
    print_table(rows, headers)

    xc = res["cross_check_cosine"]
    print(f"cross-check  TT cosine( DMRG , TT-GMRES ) = "
          f"{xc:.8f}" if xc is not None else
          "cross-check  unavailable (a solver errored)")


def save_results(path, cfg, results):
    """Write one CSV row per (N, method), with a config comment header."""
    phys = cfg["PHYSICAL"]
    stamp = datetime.now().isoformat(timespec="seconds")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    header = ["timestamp", "N", "dim_2^N", "method", "time_s",
              "residual_norm_Lrho", "max_rank", "overlap_pi",
              "overlap_machine_prec", "status", "detail", "cross_check_cosine",
              "ground_truth_build_s"]
    with open(path, "w", newline="") as f:
        f.write(f"# Glauber sweep  generated={stamp}  "
                f"J={phys['J']} kT={phys['kT']} alpha={phys['alpha']} "
                f"gamma={glauber_gamma(phys['J'], phys['kT']):.6f}\n")
        f.write("# detail = DMRG lambda_min / TT-GMRES bordered rel.res; "
                "overlap_pi='n/a' when the 2^N Gibbs reference was not built; "
                "overlap_machine_prec=yes when 1-overlap_pi <= "
                f"{OVERLAP_MACHINE_PREC_TOL:.0e}\n")
        w = csv.writer(f)
        w.writerow(header)
        for res in results:
            N, dim = res["N"], res["dim"]
            xc = "" if res["cross_check_cosine"] is None else \
                f"{res['cross_check_cosine']:.10f}"
            # per-N cost of building the exact ground-truth pi (blank when the
            # 2^N reference was skipped); repeated on both method rows, like xc.
            gt = "" if res.get("t_gibbs") is None else f"{res['t_gibbs']:.6e}"
            for name, key, detail_key in (("DMRG", "dmrg", "lambda_min"),
                                          ("TT-GMRES", "gmres", "bordered_relres")):
                d = res[key]
                if d.get("error") is not None:
                    w.writerow([stamp, N, dim, name, "", "", "", "", "",
                                f"ERROR: {d['error']}", "", xc, gt])
                    continue
                ov = "" if d.get("overlap") is None else f"{d['overlap']:.10f}"
                mp = machine_prec_flag(d.get("overlap"))
                detail = d.get(detail_key)
                w.writerow([stamp, N, dim, name,
                            f"{d['time']:.4f}", f"{d['residual']:.6e}",
                            d["max_rank"], ov, mp, d["status"],
                            "" if detail is None else f"{detail:.6e}", xc, gt])
    return path


def main():
    cfg, _ = load_config()
    phys, val_cfg = cfg["PHYSICAL"], cfg["VALIDATION"]
    J, kT, ALPHA = phys["J"], phys["kT"], phys["alpha"]
    gamma = glauber_gamma(J, kT)
    N_min, N_max, results_file = parse_sweep_args(cfg)

    print(f"\nGlauber sweep:  N = {N_min}..{N_max},  "
          f"J = {J},  kT = {kT},  alpha = {ALPHA},  gamma = {gamma:.4f}")
    print(f"exact Gibbs overlap built for N <= {val_cfg['N_dense_max']} "
          "(TT-native residual + cross-check reported at every N)")
    if N_max > 12:
        print("\nNOTE: runtime grows steeply with N. DMRG forms the dense two-site\n"
              "operator of M = L^T L (bond ~36) at every step, so expect seconds for\n"
              "N<=10, ~1 min near N=12, and several minutes by N~16-20. This reference\n"
              "implementation is meant for modest N; very large N (e.g. 60+) is not\n"
              "practical here. Bordered TT-GMRES may also fail to converge at large N\n"
              "(the fixed all-ones border loses alignment with the stationary vector).")

    results = []
    t_all = time.perf_counter()
    for N in range(N_min, N_max + 1):
        print(f"\n[ N = {N} ]", flush=True)
        res = run_one(N, cfg)
        print_one(res)
        results.append(res)
        # save after each N so partial results survive an interruption at large N
        save_results(results_file, cfg, results)

    dt = time.perf_counter() - t_all
    print("\n" + "=" * 60)
    print(f"DONE: swept N = {N_min}..{N_max} in {dt:.1f}s "
          f"({len(results)} sizes, {2 * len(results)} solver runs).")
    print(f"Results saved to:\n  {results_file}")
    print("Inspect later with e.g.  "
          "python3 -c \"import pandas as pd; "
          "print(pd.read_csv(r'%s', comment='#'))\"" % results_file)
    print("=" * 60)


if __name__ == "__main__":
    main()
