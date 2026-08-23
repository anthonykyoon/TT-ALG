#!/usr/bin/env python3
"""
plot_results.py -- plot a chosen metric from a glauber_solve.py results CSV
against the chain size N.  (Combines the former plot_overlap.py / plot_times.py.)

By default it reads the CSV named in glauber_config.py (SWEEP["results_file"]),
plots one series per solver method, and writes each PNG next to that CSV
(i.e. into the sweep's own results/<stem>/ folder).

Metrics (--metric):
    overlap_pi           overlap with the exact Gibbs measure  (default; adds a
                         second log-scale panel of the infidelity 1 - overlap)
    time_s               wall-clock solve time                 (log y-axis)
    residual_norm_Lrho   physical residual ||L rho|| / ||rho|| (log y-axis)
    max_rank             max TT rank of the solution           (linear y-axis)
    cross_check_cosine   TT cosine(DMRG, TT-GMRES), one series per N (linear)
    ground_truth_build_s time to build the exact reference π (Gibbs) per N (log)

Usage:
    python3 src/plot_results.py                        # overlap + time, config CSV
    python3 src/plot_results.py --metric overlap_pi    # just one of them
    python3 src/plot_results.py results/<run>/<run>.csv # both, from a specific CSV
    python3 src/plot_results.py --metric time_s --linear   # force a linear y-axis
    python3 src/plot_results.py --hue-mp               # color by machine-prec flag
    python3 src/plot_results.py --metric all           # every metric
    python3 src/plot_results.py --list                 # list available metrics
"""

import os
import re
import sys
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))

# Plots are written beside the CSV they came from, i.e. into that sweep's folder
# under results/ (see glauber_solve.py, which uses the same layout).
REPO = os.path.dirname(HERE) if os.path.basename(HERE) == "src" else HERE
RESULTS_DIR = os.path.join(REPO, "results")

# metric -> how to plot it
#   label      : y-axis label
#   scale      : default y-axis scale ('linear' / 'log')
#   per_method : True  -> one series per solver (DMRG, TT-GMRES)
#                False -> a single per-N series (value repeated across methods)
#   kind       : 'overlap' triggers the special two-panel value+infidelity view
METRICS = {
    "overlap_pi": {
        "label": "overlap  ⟨ρ, π⟩ / (‖ρ‖‖π‖)",
        "scale": "linear", "per_method": True, "kind": "overlap"},
    "time_s": {
        "label": "wall-clock time  (s)",
        "scale": "log", "per_method": True, "kind": "line"},
    "residual_norm_Lrho": {
        "label": "physical residual  ‖L ρ‖ / ‖ρ‖",
        "scale": "log", "per_method": True, "kind": "line"},
    "max_rank": {
        "label": "max TT rank",
        "scale": "linear", "per_method": True, "kind": "line"},
    "cross_check_cosine": {
        "label": "cross-check  cosine(DMRG, TT-GMRES)",
        "scale": "linear", "per_method": False, "kind": "line"},
    "ground_truth_build_s": {
        "label": "exact ground-truth π build time  (s)",
        "scale": "log", "per_method": False, "kind": "line"},
}

MARKERS = {"DMRG": "o", "TT-GMRES": "s"}

# Optional hue by the CSV's overlap_machine_prec flag (written by glauber_solve.py):
# 'yes' = overlap within machine precision of 1, 'no' = not, '' = no Gibbs reference.
MP_COLUMN = "overlap_machine_prec"
MP_COLORS = {"yes": "tab:green", "no": "tab:red", "": "lightgray"}
MP_LABELS = {"yes": "within machine precision",
             "no": "NOT within machine prec.",
             "": "no Gibbs reference"}


def overlay_mp_hue(ax, sub, ycol):
    """Overlay markers on (N, ycol) colored by MP_COLUMN; return the flags drawn."""
    flags = sub[MP_COLUMN].fillna("").astype(str)
    seen = []
    for flag in ("yes", "no", ""):
        m = flags == flag
        if m.any():
            ax.scatter(sub["N"][m], sub[ycol][m], c=MP_COLORS[flag],
                       edgecolors="black", linewidths=0.4, s=55, zorder=5)
            seen.append(flag)
    return seen


def mp_legend_handles(seen):
    """Proxy legend handles explaining the machine-precision hue colors."""
    from matplotlib.lines import Line2D
    return [Line2D([0], [0], marker="o", linestyle="", markersize=8,
                   markerfacecolor=MP_COLORS[f], markeredgecolor="black",
                   label=MP_LABELS[f])
            for f in ("yes", "no", "") if f in seen]


def default_csv():
    """The results file named in glauber_config.py, resolved inside results/."""
    name = "glauber_results.csv"
    try:
        import glauber_config as cfg
        name = cfg.SWEEP.get("results_file", name)
    except Exception:
        pass
    if os.path.isabs(name):
        return name
    base = os.path.basename(name)
    return os.path.join(RESULTS_DIR, os.path.splitext(base)[0], base)


def read_header_meta(path):
    """Pull 'J=.. kT=.. alpha=.. gamma=..' from the leading '#' comment lines."""
    meta = {}
    with open(path) as f:
        for line in f:
            if not line.startswith("#"):
                break
            for key, val in re.findall(r"(\w+)=([-\d.eE+]+)", line):
                meta.setdefault(key, val)
    return meta


def load_metric(path, metric):
    """Read the CSV and return a frame with numeric N and the chosen metric."""
    import pandas as pd
    df = pd.read_csv(path, comment="#")
    if metric not in df.columns:
        print(f"column '{metric}' not in {path}; available: "
              f"{', '.join(df.columns)}")
        sys.exit(1)
    df = df[pd.to_numeric(df[metric], errors="coerce").notna()].copy()
    df[metric] = df[metric].astype(float)
    df["N"] = df["N"].astype(int)
    return df


def plot_overlap(df, spec, subtitle, hue_mp=False):
    """Two-panel view: overlap vs N, and infidelity (1 - overlap) on a log axis."""
    import matplotlib.pyplot as plt
    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    has_mp = hue_mp and MP_COLUMN in df.columns
    seen = set()
    for method, sub in df.groupby("method"):
        sub = sub.sort_values("N")
        mk = MARKERS.get(method, "^")
        ax_top.plot(sub["N"], sub["overlap_pi"], marker=mk, label=method)
        infid = 1.0 - sub["overlap_pi"]
        pos = infid > 0                       # log axis needs strictly positive
        if pos.any():
            ax_bot.plot(sub["N"][pos], infid[pos], marker=mk, label=method)
        if (~pos).any():
            ns = ", ".join(map(str, sub["N"][~pos].tolist()))
            ax_bot.plot([], [], " ",
                        label=f"{method}: overlap=1 (machine prec.) at N={ns}")
        if has_mp:
            seen.update(overlay_mp_hue(ax_top, sub, "overlap_pi"))
            sub_pos = sub[pos].assign(_infid=infid[pos])
            if not sub_pos.empty:
                seen.update(overlay_mp_hue(ax_bot, sub_pos, "_infid"))
    ax_top.set_ylabel(spec["label"])
    ax_top.set_title("Solver overlap with the exact Gibbs measure  π"
                     + (f"\n{subtitle}" if subtitle else ""))
    ax_top.grid(True, alpha=0.3)
    if has_mp and seen:
        h, l = ax_top.get_legend_handles_labels()
        extra = mp_legend_handles(seen)
        ax_top.legend(h + extra, l + [e.get_label() for e in extra], fontsize=8)
    else:
        ax_top.legend()
    # log scale needs at least one positive infidelity; otherwise every solver hit
    # overlap == 1 exactly, so keep a linear axis rather than crash on an empty log.
    if (1.0 - df["overlap_pi"] > 0).any():
        ax_bot.set_yscale("log")
    ax_bot.set_xlabel("chain size  N")
    ax_bot.set_ylabel("infidelity  1 − overlap")
    ax_bot.grid(True, which="both", alpha=0.3)
    ax_bot.legend(fontsize=8)
    ax_bot.set_xticks(sorted(df["N"].unique()))
    fig.tight_layout()
    return fig


def plot_line(df, metric, spec, scale, subtitle, hue_mp=False):
    """Single-panel line plot of `metric` vs N (per method, or one per-N series)."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 5))
    has_mp = hue_mp and MP_COLUMN in df.columns and spec["per_method"]
    seen = set()
    if spec["per_method"]:
        for method, sub in df.groupby("method"):
            sub = sub.sort_values("N")
            ax.plot(sub["N"], sub[metric], marker=MARKERS.get(method, "^"),
                    label=method)
            if has_mp:
                seen.update(overlay_mp_hue(ax, sub, metric))
        if has_mp and seen:
            h, l = ax.get_legend_handles_labels()
            extra = mp_legend_handles(seen)
            ax.legend(h + extra, l + [e.get_label() for e in extra], fontsize=8)
        else:
            ax.legend()
    else:                                     # value is per-N; drop the duplicate
        sub = df.drop_duplicates("N").sort_values("N")
        ax.plot(sub["N"], sub[metric], marker="D", color="C2",
                label="DMRG vs TT-GMRES")
        ax.legend()
    ax.set_yscale(scale)
    ax.set_xlabel("chain size  N")
    ax.set_ylabel(spec["label"])
    ax.set_title(f"{metric} vs N" + (f"\n{subtitle}" if subtitle else ""))
    ax.grid(True, which="both", alpha=0.3)
    ax.set_xticks(sorted(df["N"].unique()))
    fig.tight_layout()
    return fig


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", nargs="?", default=None,
                    help="results CSV (default: from glauber_config.py)")
    ap.add_argument("--metric", nargs="+", default=["overlap_pi", "time_s"],
                    choices=list(METRICS) + ["all"], metavar="METRIC",
                    help="one or more columns to plot vs N "
                         "(default: overlap_pi time_s); pass others, or 'all'")
    ap.add_argument("--out", default=None, help="output PNG path")
    ap.add_argument("--linear", action="store_true", help="force a linear y-axis")
    ap.add_argument("--log", action="store_true", help="force a log y-axis")
    ap.add_argument("--show", action="store_true",
                    help="open an interactive window as well as saving")
    ap.add_argument("--hue-mp", "--hue-machine-prec", dest="hue_mp",
                    action="store_true",
                    help="color each point by the CSV's overlap_machine_prec flag "
                         "(green = within machine precision of 1, red = not)")
    ap.add_argument("--list", action="store_true",
                    help="list available metrics and exit")
    args = ap.parse_args()

    if args.list:
        print("available metrics:")
        for name, spec in METRICS.items():
            print(f"  {name:20s} {spec['label']}")
        return

    import matplotlib
    if not args.show:
        matplotlib.use("Agg")                 # headless: just write the file
    import matplotlib.pyplot as plt

    path = args.csv or default_csv()
    if not os.path.exists(path):
        print(f"results CSV not found: {path}\n"
              "run  python3 glauber_solve.py  first, or pass a path.")
        sys.exit(1)

    # expand 'all' and de-duplicate while preserving order
    metrics = (list(METRICS) if "all" in args.metric
               else list(dict.fromkeys(args.metric)))
    if args.out is not None and len(metrics) > 1:
        print("--out is ignored when plotting multiple metrics; auto-naming each.")
        args.out = None

    if args.hue_mp:
        import pandas as pd
        cols = pd.read_csv(path, comment="#", nrows=0).columns
        if MP_COLUMN not in cols:
            print(f"note: --hue-mp requested but '{MP_COLUMN}' not in {path}; "
                  "re-run glauber_solve.py to add it. Plotting without hue.")

    meta = read_header_meta(path)
    subtitle = "  ".join(f"{k}={meta[k]}" for k in ("J", "kT", "alpha", "gamma")
                         if k in meta)
    stem = os.path.splitext(os.path.basename(path))[0]

    saved = []
    for metric in metrics:
        spec = METRICS[metric]
        df = load_metric(path, metric)
        if df.empty:
            print(f"no numeric '{metric}' values in {path} -- skipping.")
            continue
        if spec["kind"] == "overlap":
            fig = plot_overlap(df, spec, subtitle, hue_mp=args.hue_mp)
        else:
            scale = "linear" if args.linear else "log" if args.log else spec["scale"]
            fig = plot_line(df, metric, spec, scale, subtitle, hue_mp=args.hue_mp)
        out = args.out or os.path.join(os.path.dirname(os.path.abspath(path)),
                                       f"{metric}_vs_N_{stem}.png")
        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
        fig.savefig(out, dpi=150)
        saved.append(out)
        print(f"saved plot to:\n  {out}")
        if not args.show:
            plt.close(fig)

    if not saved:
        sys.exit(1)
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
