"""Demonstration of Glauber (Monte Carlo) dynamics for the Ising model.

Run:
    python glauber_demo.py

Produces three figures:
  1. glauber_relaxation.png  - magnetization vs. number of steps at a fixed
     low temperature, starting from a random state (order emerges).
  2. glauber_M_vs_T.png      - equilibrium |magnetization| vs. temperature
     for the 2D model, showing the order/disorder transition near the 2D
     Ising critical temperature T_c = 2 / ln(1 + sqrt(2)) ~ 2.269.
  3. glauber_M_vs_T_4d.png   - the same transition for the 4D model ($10^4$
     lattice), whose
     critical temperature is T_c ~ 6.68 (higher, because each site now has
     8 neighbors reinforcing order instead of 4).

The same `glauber` routine drives every case; only the lattice shape changes.
"""

import warnings

import numpy as np

from glauber_dynamics import glauber

# np.exp(dE/T) overflows harmlessly at low T (p_flip -> 0); quiet it for a
# clean demo run.
warnings.filterwarnings("ignore", message="overflow encountered in exp")

T_C_2D = 2.0 / np.log(1.0 + np.sqrt(2.0))  # ~2.269, exact 2D Ising critical temp
T_C_4D = 6.68                              # ~6.68, 4D Ising (numerical estimate)

# Backward-compatible alias for the 2D critical temperature.
T_C = T_C_2D


def magnetization(lattice):
    """Mean spin; +-1 for a fully ordered lattice, ~0 for a disordered one."""
    return lattice.mean()


def relaxation_curve(L=32, dims=2, T=1.5, sweeps=200, seed=60637, vectorized=False):
    """Track |M| as the system relaxes from a random start at low T.

    Measured in full lattice *sweeps* (one sweep = L**dims single-spin update
    attempts), driving a single continuous Markov chain off one RNG stream so
    the relaxation is genuine rather than an artifact of per-chunk re-seeding.
    `dims` sets the lattice dimensionality; the lattice is L**dims spins.
    """
    shape = (L,) * dims
    rng = np.random.default_rng(seed)
    lattice = rng.choice([-1, 1], size=shape).astype(np.int8)
    n_sites = lattice.size

    steps, mags = [], []
    for s in range(sweeps):
        # Hand the same `rng` back in each call to continue one Markov chain.
        lattice = glauber(lattice, n_steps=n_sites, T=T, rng=rng, vectorized=vectorized)
        steps.append((s + 1) * n_sites)
        mags.append(abs(magnetization(lattice)))
    return np.array(steps), np.array(mags)


def m_vs_T(L=32, dims=2, temps=None, equil_steps=300_000, seed=0, vectorized=False):
    """Sweep temperature; report |M| of an initially ordered L**dims lattice."""
    if temps is None:
        temps = np.linspace(1.0, 3.6, 14)

    shape = (L,) * dims
    mags = []
    for k, T in enumerate(temps):
        lattice = np.ones(shape, dtype=np.int8)  # start ordered
        lattice = glauber(lattice, n_steps=equil_steps, seed=seed + k, T=float(T),
                          vectorized=vectorized)
        mags.append(abs(magnetization(lattice)))
        print(f"  T = {T:5.2f}   |M| = {mags[-1]:.3f}")
    return np.asarray(temps), np.asarray(mags)


def main():
    try:
        import matplotlib

        matplotlib.use("Agg")  # headless: write files, no display needed
        import matplotlib.pyplot as plt
    except ImportError:
        plt = None
        print("matplotlib not installed; printing numbers only.\n")

    print(f"2D Ising critical temperature T_c = {T_C_2D:.4f}")
    print(f"4D Ising critical temperature T_c ~ {T_C_4D:.4f}\n")

    print("2D relaxation at T = 1.5 (below T_c), random start:")
    steps, mags = relaxation_curve(dims=2, T=1.5, vectorized=True)
    print(f"  |M| went {mags[0]:.3f} -> {mags[-1]:.3f} over {steps[-1]:,} steps")

    # 4D relaxation for the overlay: L=10 (10^4 sites) at T=4.0, below the 4D
    # T_c. Even side length so the vectorized checkerboard update applies.
    print("4D relaxation at T = 4.0 (below T_c), random start:")
    steps4, mags4 = relaxation_curve(L=10, dims=4, T=4.0, sweeps=300, vectorized=True)
    print(f"  |M| went {mags4[0]:.3f} -> {mags4[-1]:.3f} over {steps4[-1]:,} steps\n")

    print("2D magnetization vs. temperature (ordered start):")
    temps, m = m_vs_T(dims=2, vectorized=True)

    # 4D Ising: L=10 -> 10**4 = 10000 sites, 8 neighbors each. Sweep across the
    # higher 4D transition temperature.
    print("\n4D magnetization vs. temperature (ordered start, 10^4 lattice):")
    temps4, m4 = m_vs_T(L=10, dims=4, temps=np.linspace(4.0, 9.0, 11),
                        equil_steps=2_000_000, vectorized=True)

    if plt is None:
        return

    # Plot vs. sweeps (= steps / sites) so the differently sized 2D and 4D
    # lattices share a comparable x-axis.
    sweeps_2d = np.arange(1, len(mags) + 1)
    sweeps_4d = np.arange(1, len(mags4) + 1)

    plt.figure(figsize=(6, 4))
    plt.plot(sweeps_2d, mags, lw=2, label="2D ($32^2$), T = 1.5")
    plt.plot(sweeps_4d, mags4, lw=2, label="4D ($10^4$), T = 4.0")
    plt.xlabel("lattice sweeps")
    plt.ylabel("|magnetization|")
    plt.title("Relaxation toward order below $T_c$ (2D vs. 4D)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("glauber_relaxation.png", dpi=120)
    print("\nWrote glauber_relaxation.png")

    plt.figure(figsize=(6, 4))
    plt.plot(temps, m, "o-", lw=2)
    plt.axvline(T_C_2D, color="r", ls="--", label=f"$T_c \\approx {T_C_2D:.2f}$")
    plt.xlabel("temperature T")
    plt.ylabel("|magnetization|")
    plt.title("Order/disorder transition (2D Ising, Glauber dynamics)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("glauber_M_vs_T.png", dpi=120)
    print("Wrote glauber_M_vs_T.png")

    plt.figure(figsize=(6, 4))
    plt.plot(temps4, m4, "o-", lw=2, color="C2")
    plt.axvline(T_C_4D, color="r", ls="--", label=f"$T_c \\approx {T_C_4D:.2f}$")
    plt.xlabel("temperature T")
    plt.ylabel("|magnetization|")
    plt.title("Order/disorder transition (4D Ising, $10^4$ lattice)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("glauber_M_vs_T_4d.png", dpi=120)
    print("Wrote glauber_M_vs_T_4d.png")


if __name__ == "__main__":
    main()
