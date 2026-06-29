"""Crank d up for the separable (rank-2, sec 2.2) FP operator and compare
DMRG vs bordered TT-GMRES, with no dense reference (n^d is astronomical)."""
import sys, time
import numpy as np

from DMRG import gradient_fp_terms
from tt_dmrg import mpo_from_terms, mpo_round, dmrg_null_vector
from tt_gmres import tt_frobenius as ttf_g, tt_round as ttr_g, mpo_mps_mult, tt_dot
from tt_gmres_fokker_planck import fp_stationary_gmres, mpo_dmrg_to_gmres


def make_terms(d, n, T=1.0, b=3.0):
    a = [1.5] + [1.0] * (d - 1)            # double-well on x_1, harmonic elsewhere
    dV = [lambda x, c=a[0]: 4.0 * c * x * (x**2 - 1.0)] + \
         [(lambda x, c=ai: 2.0 * c * x) for ai in a[1:]]
    x = np.linspace(-b, b, n)
    return gradient_fp_terms(dV, T, [x] * d)


def overlap(p, q):
    num = abs(tt_dot(p, q))
    return num / (ttf_g(p) * ttf_g(q))


def run(d, n=11, maxrank=6):
    terms = make_terms(d, n)
    L_dmrg = mpo_round(mpo_from_terms(terms, d), eps=1e-12)
    L_g = mpo_dmrg_to_gmres(L_dmrg)

    # --- DMRG ---
    t0 = time.perf_counter()
    G, lam, res_dmrg = dmrg_null_vector(L_dmrg, round_eps=1e-12, init_rank=2,
                                        maxrank=maxrank, eps=1e-9, max_sweeps=6,
                                        tol=1e-12, verbose=False)
    t_dmrg = time.perf_counter() - t0
    rk_dmrg = max(c.shape[0] for c in G)

    # --- GMRES (bordered) ---
    t0 = time.perf_counter()
    p, res_g, info = fp_stationary_gmres(terms, d, n, max_rank=maxrank,
                                         gmres_m=30, max_restarts=6,
                                         tol=1e-8, verbose=False)
    t_gmres = time.perf_counter() - t0
    rk_g = max(c.shape[0] for c in p)

    ov = overlap(G, p)
    print(f"d={d:4d} n={n}  |  DMRG: {t_dmrg:7.2f}s res={res_dmrg:.1e} rk={rk_dmrg:2d}"
          f"  |  GMRES: {t_gmres:7.2f}s res={res_g:.1e} rk={rk_g:2d}"
          f"  |  <DMRG,GMRES>={ov:.6f}", flush=True)


if __name__ == "__main__":
    ds = [int(x) for x in sys.argv[1:]] or [8, 16, 32]
    print("Separable double-well FP  (rank-2 sec-2.2 operator; true density is rank-1)")
    print("-" * 100)
    for d in ds:
        run(d)
