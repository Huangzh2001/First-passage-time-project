"""Plot FPT distributions for fixed, growing, and guarded-pulse thresholds.

Place beside ssa.py, binomial_moments.py, model_definitions.py, and boundary_definitions.py.
"""

from concurrent.futures import ProcessPoolExecutor
from hashlib import sha256
import inspect
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from time import perf_counter
from zipfile import BadZipFile

# Parallel jobs are independent; prevent nested BLAS thread pools.
for variable in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[variable] = "1"

import numpy as np
import scipy

import binomial_moments
import boundary_definitions
import model_definitions
import ssa


FIXED_THRESHOLD = 3
GROWTH = (2.6, 0.048)  # Initial threshold and growth rate.
GROWTH_POSITIVE_SCORE_CAP = 0.24


def threshold_boundary(name):
    def fixed(_time, _mrna, protein):
        return protein - FIXED_THRESHOLD + 0.5

    def growth(time, _mrna, protein):
        initial_threshold, growth_rate = GROWTH
        margin = protein - initial_threshold * np.exp(growth_rate * time)
        positive = np.maximum(margin, 0.0)
        cap = GROWTH_POSITIVE_SCORE_CAP
        return np.minimum(margin, 0.0) + cap * positive / (cap + positive)

    score = {"fixed": fixed, "cell_growth": growth}[name]
    return ssa.FirstEvent(score)


MODEL_VARIANTS = (0, 1, 2)
BOUNDARY_ROWS = (
    ("Fixed threshold", threshold_boundary, "fixed"),
    ("Cell growth", threshold_boundary, "cell_growth"),
    ("Temporary release", boundary_definitions.comparison_boundary, "guarded_pulse"),
)
GAMMAS = (8., 16., 32., 64.)
PATHS = 50_000
WORKERS = 4
RETRY_FAILED = False
SEED = 6188
TIMES = np.linspace(0., 6., 751)
BIN_STRIDE = 10
PLOT_CDF = False
SOLVER_OPTIONS = dict(basis="binomial", rtol=2e-9, atol=1e-12, max_step=.035, timeout=25.)
OUTPUT = Path(__file__).resolve().with_name("fpt_grid_v2_pulse")


def cache_paths():
    # Invalidate numerical results when the model, boundary, solver, or numerical settings change.
    key = sha256()
    for module in (binomial_moments, boundary_definitions, model_definitions, ssa):
        key.update(Path(module.__file__).read_bytes())
    settings = (
        "fpt-grid-v2", MODEL_VARIANTS, tuple(name for _, _, name in BOUNDARY_ROWS), PATHS, SEED,
        FIXED_THRESHOLD, GROWTH, GROWTH_POSITIVE_SCORE_CAP, ssa.HIT_SUBDIVISIONS,
        SOLVER_OPTIONS, np.__version__, scipy.__version__,
    )
    key.update(inspect.getsource(threshold_boundary).encode())
    key.update(repr(settings).encode())
    key.update(TIMES.tobytes())
    return OUTPUT.parent / ".fpt_grid_cache" / key.hexdigest()[:20]


def load_results(path):
    try:
        with np.load(path, allow_pickle=False) as data:
            return dict(data)
    except (FileNotFoundError, OSError, ValueError, EOFError, BadZipFile):
        return None


def save_results(path, **data):
    with NamedTemporaryFile(dir=path.parent, delete=False) as file:
        temporary = Path(file.name)
        try:
            np.savez(file, **data)
            file.close()
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


def load_or_simulate(variant, cache, times, paths, seed):
    path = cache / f"ssa_{variant}.npz"
    empirical = load_results(path)
    if empirical is not None:
        return int(empirical["states"]), empirical["hits"]

    model = model_definitions.network(variant)
    boundaries = [factory(name) for _, factory, name in BOUNDARY_ROWS]
    started = perf_counter()
    # The previous calls used the same seed for each boundary, replaying each
    # trajectory. Evaluate all three hit times in a single SSA call instead.
    hits = ssa.first_passage_times(
        model.rates, model.states, model.initial,
        [boundary.first_hit for boundary in boundaries],
        float(times[-1]), paths, seed,
    )
    save_results(path, states=model.states, hits=hits)
    print(f"{model.name} | SSA, all boundaries | {perf_counter()-started:.1f} s",
          flush=True)
    return model.states, hits


def load_or_solve(variant, row, gamma, cache, times, options, retry_failed):
    _, factory, name = BOUNDARY_ROWS[row]
    path = cache / f"moments_{variant}_{name}_{gamma:.17g}.npz"
    result = load_results(path)
    if result is not None and not (retry_failed and not result["valid"]):
        return result

    model = model_definitions.network(variant)
    started = perf_counter()
    run, status, attempts = binomial_moments.adaptive_solve(
        model, factory(name), times, gamma, **options)
    result = dict(
        status=status, valid=run is not None,
        h=run["h"] if run is not None else np.empty(0),
        density=run["density"] if run is not None else np.empty(0),
    )
    save_results(path, **result)
    elapsed = perf_counter()-started
    print(f"{name} | {model.name} | gamma={gamma:g} | {status} | "
          f"{elapsed:.1f} s | {len(attempts)} orders", flush=True)
    return result


def calculate(workers, retry_failed, gammas):
    cache = cache_paths()
    cache.mkdir(parents=True, exist_ok=True)
    keys = [(variant, row, gamma)
            for gamma in gammas
            for row in range(len(BOUNDARY_ROWS))
            for variant in MODEL_VARIANTS]
    if workers == 1:
        empirical = {v: load_or_simulate(v, cache, TIMES, PATHS, SEED) for v in MODEL_VARIANTS}
        computed = {key: load_or_solve(*key, cache, TIMES, SOLVER_OPTIONS, retry_failed)
                    for key in keys}
    else:
        # Schedule single curves so an expensive case does not stall its entire column.
        with ProcessPoolExecutor(max_workers=min(workers, len(keys))) as pool:
            simulations = {v: pool.submit(load_or_simulate, v, cache, TIMES, PATHS, SEED)
                           for v in MODEL_VARIANTS}
            solutions = {key: pool.submit(load_or_solve, *key, cache, TIMES, SOLVER_OPTIONS,
                                          retry_failed)
                         for key in keys}
            empirical = {v: job.result() for v, job in simulations.items()}
            computed = {key: job.result() for key, job in solutions.items()}
    columns = []
    for variant in MODEL_VARIANTS:
        states, hits = empirical[variant]
        panels = {
            name: (hits[row], [computed[variant, row, gamma] for gamma in gammas])
            for row, (_, _, name) in enumerate(BOUNDARY_ROWS)
        }
        columns.append((variant, states, panels))
    return columns


def draw(columns, gammas):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    try:
        font_manager.findfont("Arial", fallback_to_default=False)
    except ValueError:
        print("Arial is not installed. PDF export will use an available sans-serif fallback.", flush=True)
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    nrows, ncols = len(BOUNDARY_ROWS), len(columns)
    fig = plt.figure(figsize=(3.7 * ncols, 1.5 + 2.4 * nrows), layout="constrained")
    grid = fig.add_gridspec(nrows + 2, ncols, height_ratios=[.58, *([1] * nrows), .16])
    for col, (_, states, _) in enumerate(columns):
        header = fig.add_subplot(grid[0, col])
        header.set_axis_off()
        header.text(.5, .95, f"{states}-state model", transform=header.transAxes,
                    ha="center", va="top")

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    entries = {"SSA": (Line2D([], [], color="#727272") if PLOT_CDF
                       else Patch(facecolor="#bdbdbd", edgecolor="none"))}
    has_unverified_order = False
    for row, (row_label, _, name) in enumerate(BOUNDARY_ROWS):
        edges = TIMES[np.r_[np.arange(0, TIMES.size - 1, BIN_STRIDE), TIMES.size - 1]]
        widths = np.diff(edges)
        y_reference = None
        for col, (_, _, panels) in enumerate(columns):
            ax = fig.add_subplot(grid[row + 1, col], sharex=y_reference, sharey=y_reference)
            if y_reference is None:
                y_reference = ax
            hits, curves = panels[name]
            if PLOT_CDF:
                ax.step(TIMES, np.searchsorted(np.sort(hits), TIMES, side="right") / len(hits),
                        where="post", color="#727272", linewidth=1)
            else:
                counts, _ = np.histogram(hits, bins=edges)
                heights = counts / (len(hits) * widths)
                ax.bar(edges[:-1], heights, width=widths, align="edge",
                       color="#bdbdbd", edgecolor="none", zorder=1)

            for index, (gamma, result) in enumerate(zip(gammas, curves)):
                if not result["valid"]:
                    continue
                verified = result["status"] == "converged"
                color = colors[index % len(colors)]
                style = "-" if verified else "--"
                values = 1 - result["h"] if PLOT_CDF else result["density"]
                ax.plot(TIMES, values, color=color, linestyle=style,
                        linewidth=1.15, zorder=2)
                entries.setdefault(f"gamma={gamma:g}", Line2D([], [], color=color))
                has_unverified_order |= not verified

            ax.set_xlim(TIMES[0], TIMES[-1])
            if PLOT_CDF:
                ax.set_ylim(0, 1.02)
            ax.tick_params(top=False, right=False, labelbottom=True, labelleft=col == 0)
            if col == 0:
                quantity = "FPT cumulative probability" if PLOT_CDF else "FPT density"
                ax.set_ylabel(f"{row_label}\n{quantity}")
            if row == nrows - 1:
                ax.set_xlabel("Time")

    footer = fig.add_subplot(grid[-1, :])
    footer.set_axis_off()
    if has_unverified_order:
        entries["Order convergence not verified"] = Line2D([], [], color="0.35", ls="--")
    footer.legend(list(entries.values()), list(entries), loc="center",
                  ncols=len(entries), frameon=False)
    for extension in (".pdf", ".svg"):
        fig.savefig(OUTPUT.with_suffix(extension))
    plt.close(fig)


def main():
    draw(calculate(WORKERS, RETRY_FAILED, GAMMAS), GAMMAS)


if __name__ == "__main__":
    main()
