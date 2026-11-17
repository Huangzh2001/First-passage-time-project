"""Plot a 3 x 2 grid of survival probabilities and FPT densities."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from hashlib import sha256
from inspect import getsource, signature
from pathlib import Path
from time import perf_counter
from zipfile import BadZipFile
import os

# Avoid a separate pool of BLAS threads in each boundary worker.
for variable in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[variable] = "1"

import numpy as np
import scipy

import binomial_moments
import model_definitions
import random_boundary
import random_solvers

MODEL = 0
PATHS = 2000
SEED = 6188
DT_VALUES = (.02, .01)
TIMES = np.linspace(0., 6., 301)
BIN_STRIDE = 10
QUADRATURE = 96
# Each case is gamma, mRNA order L, protein order K, and boundary modes N.
CASES = ((4., 8, 8, 4), (4., 8, 8, 8), (4., 12, 12, 8),
         (8., 12, 12, 8), (16., 12, 12, 8))
BOUNDARIES = (
    ("Jacobi", "jacobi", {"sigma": .7, "kappa": 1., "theta": .25}),
    ("Reflected OU", "ou", {"sigma": .65, "kappa": 1.5, "theta": .75}),
    ("Reflected Brownian", "brownian", {"sigma": 1.}),
)
OUTPUT = Path(__file__).resolve().parent / "results_random_boundary" / "random_boundary_grid"


def hit_rule(m, n, x):
    return m + n + 0.25 * m * x - x


def load_cache(path):
    try:
        with np.load(path, allow_pickle=False) as saved:
            return dict(saved)
    except (FileNotFoundError, OSError, ValueError, EOFError, BadZipFile):
        return None


def solve_boundary(row, times, cases, paths, cache, refresh):
    _, kind, parameters = row
    boundary = random_boundary.RandomBoundary(kind, criterion=hit_rule, **parameters)
    model = model_definitions.network(MODEL)
    simulated = []
    for dt in sorted(DT_VALUES):
        path = cache / f"{kind}_ssa_{dt:.17g}.npz"
        saved = None if refresh else load_cache(path)
        if saved is None:
            hits = random_solvers.sample_hits(model, boundary, times[-1], paths, dt, SEED)
            np.savez(path, hits=hits)
        else:
            hits = saved["hits"]
        simulated.append(np.sort(hits))
    h_fine = 1 - np.searchsorted(simulated[0], times, side="right") / paths
    if len(simulated) > 1:
        h_coarse = 1 - np.searchsorted(simulated[-1], times, side="right") / paths
        gap = np.max(np.abs(h_fine - h_coarse))
        print(f"{kind} | SSA dt check | max survival gap={gap:.4f}", flush=True)

    curves = {}
    # Some versions of solve_moments call boundary.g(t=...), whereas the
    # shipped boundary.g only takes m, n, x. Supply the supported callback.
    kwargs = {}
    if "criterion" in signature(random_solvers.solve_moments).parameters:
        kwargs["criterion"] = lambda t, m, n, x: boundary.g(m, n, x)
    for case in cases:
        gamma, L, K, N = case
        path = cache / f"{kind}_{gamma:.17g}_{L}_{K}_{N}.npz"
        saved = None if refresh else load_cache(path)
        if saved is None:
            start = perf_counter()
            try:
                h, fpt, valid = random_solvers.solve_moments(
                    model, boundary, times, gamma, L, K, N, QUADRATURE, **kwargs)
            except (RuntimeError, FloatingPointError) as exc:
                print(f"{kind} | {case} | solve failed: {exc}", flush=True)
                continue
            print(f"{kind} | {case} | valid={valid} | {perf_counter()-start:.1f} s",
                  flush=True)
            np.savez(path, h=h, fpt=fpt, valid=valid)
        else:
            h, fpt, valid = saved["h"], saved["fpt"], bool(saved["valid"])
            print(f"{kind} | {case} | valid={valid} | cached", flush=True)
        if valid:
            curves[case] = (h, fpt)
    return simulated[0], h_fine, curves


def calculate(times, cases, paths, workers, refresh):
    key = sha256(times.tobytes())
    config = (MODEL, BOUNDARIES, DT_VALUES, paths, SEED, QUADRATURE,
              np.__version__, scipy.__version__)
    key.update(repr(config).encode())
    key.update(getsource(hit_rule).encode())
    for module in (random_boundary, random_solvers, binomial_moments, model_definitions):
        key.update(Path(module.__file__).read_bytes())
    cache = OUTPUT.parent / ".cache" / key.hexdigest()[:20]
    cache.mkdir(parents=True, exist_ok=True)
    if workers == 1:
        return [solve_boundary(row, times, cases, paths, cache, refresh) for row in BOUNDARIES]
    with ProcessPoolExecutor(max_workers=min(workers, len(BOUNDARIES))) as pool:
        jobs = [pool.submit(solve_boundary, row, times, cases, paths, cache, refresh)
                for row in BOUNDARIES]
        return [job.result() for job in jobs]


def draw(results, times, cases):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    try:
        font_manager.findfont("Arial", fallback_to_default=False)
    except ValueError:
        print("Arial is not installed. Using a sans-serif fallback.", flush=True)
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
        "axes.spines.top": False,
        "axes.spines.right": False,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    })

    fig = plt.figure(figsize=(9.5, 8.3), layout="constrained")
    grid = fig.add_gridspec(len(BOUNDARIES)+1, 2, height_ratios=[1, 1, 1, .23])
    left_anchor = right_anchor = None
    ssa_color = "#bdbdbd"
    theory_colors = {4.: "#9467bd", 8.: "#1f77b4", 16.: "#ff7f0e",
                     32.: "#2ca02c", 64.: "#d62728"}
    other_colors = iter(plt.rcParams["axes.prop_cycle"].by_key()["color"])
    style_by_case = {}
    for case in cases:
        gamma, L, K, N = case
        matches = [candidate for candidate in cases if candidate[0] == gamma]
        index = matches.index(case)
        if gamma not in theory_colors:
            theory_colors[gamma] = next(other_colors, "#4d4d4d")
        styles = (":", "--", "-")
        style = "-" if len(matches) == 1 else styles[index % len(styles)]
        label = f"gamma={gamma:g}"
        if len(matches) > 1:
            label += f" (L={L}, K={K}, N={N})"
        style_by_case[case] = (theory_colors[gamma], style, label)
    legend = {f"SSA (dt={min(DT_VALUES):g})": Patch(facecolor=ssa_color, edgecolor="none")}
    indices = np.r_[np.arange(0, times.size - 1, BIN_STRIDE), times.size - 1]
    edges = times[indices]
    shown = set()

    for index, ((row_name, _, _), (hits, h, curves)) in enumerate(zip(BOUNDARIES, results)):
        left = fig.add_subplot(grid[index, 0], sharex=left_anchor, sharey=left_anchor)
        right = fig.add_subplot(grid[index, 1], sharex=left, sharey=right_anchor)
        if index == 0:
            left_anchor, right_anchor = left, right
            left.set_title("Survival probability H(t)")
            right.set_title("FPT density")

        left.step(times, h, where="post", color="#808080", linewidth=1.2, zorder=1)
        counts, _ = np.histogram(hits, bins=edges)
        right.bar(edges[:-1], counts / (len(hits) * np.diff(edges)), width=np.diff(edges),
                  align="edge", color=ssa_color, edgecolor="none", zorder=1)

        for case, (theory_h, theory_fpt) in curves.items():
            color, style, _ = style_by_case[case]
            left.plot(times, theory_h, color=color, linestyle=style, linewidth=1.2)
            right.plot(times, theory_fpt, color=color, linestyle=style, linewidth=1.2)
            shown.add(case)

        left.set_ylabel(row_name)
        left.set_ylim(0, 1.02)
        for ax in (left, right):
            ax.set_xlim(times[0], times[-1])
            ax.tick_params(top=False, right=False, labelbottom=index == len(BOUNDARIES)-1)
            if index == len(BOUNDARIES)-1:
                ax.set_xlabel("Time")

    right_anchor.set_ylim(bottom=0)
    for case in cases:
        if case in shown:
            color, style, label = style_by_case[case]
            legend[label] = Line2D([], [], color=color, linestyle=style)
    footer = fig.add_subplot(grid[-1, :])
    footer.set_axis_off()
    footer.legend(legend.values(), legend.keys(), ncols=min(3, len(legend)),
                  loc="center", frameon=False, fontsize=9)
    for ext in (".pdf", ".svg"):
        fig.savefig(OUTPUT.with_suffix(ext))
    plt.close(fig)


def main(cases=CASES, times=TIMES, paths=PATHS, workers=3, refresh=False):
    draw(calculate(times, cases, paths, workers, refresh), times, cases)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", default=3, type=int)
    parser.add_argument("--refresh", action="store_true", help="ignore cached results")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    main(workers=args.workers, refresh=args.refresh)
