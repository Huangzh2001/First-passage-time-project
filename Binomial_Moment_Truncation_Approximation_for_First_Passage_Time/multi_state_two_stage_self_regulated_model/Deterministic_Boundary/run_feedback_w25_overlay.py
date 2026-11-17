"""Compare shortest 25% first-passage windows across thresholds and feedback."""

from concurrent.futures import ProcessPoolExecutor
import csv
from functools import partial
from math import ceil
import os
from pathlib import Path

for name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[name] = "1"

import numpy as np

from binomial_moments import adaptive_solve
from ssa import FirstEvent, TwoStageModel, first_passage_times


# Edit the parameters here, then run this file.
STATE_COUNTS = (2, 4, 8)                        # Promoter states to compare
THRESHOLDS = (2, 4, 6, 8, 10, 14)                          # Protein thresholds, e.g. range(2, 11)
HORIZON = 1200.0                                 # Final observation time
WORKERS = 4                                    # Number of parallel processes
SSA_PATHS = 0                                  # Set to 20_000 to check with SSA
ORDER_CAP = (22, 26)                            # BM truncation caps (L, K)
SOLVER_TIMEOUT = 90.0                           # Seconds per BM order
OUTPUT_DIR = Path(__file__).resolve().parent / "feedback_w25"

FEEDBACK = (
    (-0.9, "Negative", "--", "s"),
    (0.0, "None", ":", "o"),
    (0.9, "Positive", "-", "^"),
)
GAMMAS = (48.0, 64.0)
MASS = 0.25


def make_model(states, strength):
    initial = np.array([0.75 / (states - 1)] * (states - 1) + [0.25])
    # switching_rates = np.zeros((states, states))
    # for i in range(states - 1):
    #     switching_rates[i + 1, i] = 0.75
    #     switching_rates[i, i + 1] = 0.75 * initial[i + 1] / initial[i]

    switching_rates = np.zeros((states, states))
    off_step_rate = (states - 1) / 4.0

    for i in range(states - 1):
        switching_rates[i, i + 1] = off_step_rate

    switching_rates[states - 1, 0] = 0.75

    def switching(state, _mrna, _protein):
        return switching_rates[state]

    def transcription(state, _mrna, protein):
        return 4.4 * (state == states - 1) * (1 + strength * protein / (2 + protein))

    return TwoStageModel(
        f"d={states}, a={strength:g}", (initial, 0, 0), switching, transcription,
        np.full(states, 2.4), np.full(states, 2.5), np.full(states, 0.15),
    )


def boundary_score(_time, _mrna, protein, *, threshold):
    return protein - threshold + 0.5


def fixed_boundary(threshold):
    return FirstEvent(partial(boundary_score, threshold=threshold))


def time_grid(horizon):
    times = [np.array([0.0])]
    for lower, upper, spacing in ((0, 40, 0.04), (40, 240, 0.10), (240, horizon, 0.5)):
        end = min(upper, horizon)
        if end > lower:
            times.append(np.linspace(lower, end, ceil((end - lower) / spacing) + 1)[1:])
    return np.concatenate(times)


def bm_interval(times, survival):
    cdf = np.maximum.accumulate(np.clip(1 - survival, 0, 1))
    probability, first = np.unique(cdf, return_index=True)
    last_start = probability[-1] - MASS
    if last_start < 0:
        return np.nan, np.nan
    # For a piecewise-linear quantile, the width changes slope at these knots.
    knots = np.r_[0, last_start, probability, probability - MASS]
    q = np.clip(knots, 0, last_start)
    start = np.interp(q, probability, times[first])
    end = np.interp(q + MASS, probability, times[first])
    index = np.argmin(end - start)
    return start[index], end[index]


def ssa_interval(hit_times):
    count = ceil(MASS * len(hit_times))
    finite = np.sort(hit_times[np.isfinite(hit_times)])
    if len(finite) < count:
        return np.nan, np.nan
    left = finite[:len(finite) - count + 1]
    right = finite[count - 1:]
    index = np.argmin(right - left)
    return left[index], right[index]


def solve_bm(job):
    states, strength, threshold, horizon, order_cap, timeout = job
    model = make_model(states, strength)
    boundary = fixed_boundary(threshold)
    times = time_grid(horizon)
    start_k = max(8, threshold + 1)
    cap_l, cap_k = order_cap
    solutions = []
    row = dict(
        states=states, threshold=threshold, strength=strength, horizon=horizon,
        bm_status="", bm_lo=np.nan, bm_hi=np.nan, bm_w25=np.nan,
        bm_cdf_at_horizon=np.nan, gamma_w25_gap=np.nan, L="", K="",
        ssa_paths=0, ssa_w25=np.nan, ssa_hit_fraction=np.nan,
        max_bm_ssa_cdf_gap=np.nan,
    )
    for gamma in GAMMAS:
        result, status, attempts = adaptive_solve(
            model, boundary, times, gamma, start=(8, start_k), step=(2, 2),
            cap=(cap_l, max(cap_k, start_k + 4)), basis="binomial",
            order_tol=2e-5, max_step=1.25, rtol=1e-9, timeout=timeout,
        )
        if status != "converged":
            last = attempts[-1]
            row["bm_status"] = f"{status} at gamma={gamma:g} ({last['solver_status']})"
            row["L"], row["K"] = last["L"], last["K"]
            return row, None
        solutions.append(result)

    check, chosen = solutions
    lo, hi = bm_interval(times, chosen["h"])
    check_lo, check_hi = bm_interval(times, check["h"])
    status = "converged" if np.isfinite(hi) else "horizon_below_25_percent"
    if np.isfinite(hi) and hi == horizon:
        status = "interval_ends_at_horizon"
    row.update(bm_status=status, bm_lo=lo, bm_hi=hi, bm_w25=hi - lo,
               bm_cdf_at_horizon=1 - chosen["h"][-1],
               gamma_w25_gap=abs((hi - lo) - (check_hi - check_lo)),
               L=chosen["L"], K=chosen["K"])
    return row, 1 - chosen["h"]


def solve_ssa(job):
    states, strength, thresholds, horizon, paths, seed = job
    model = make_model(states, strength)
    callbacks = [fixed_boundary(b).first_hit for b in thresholds]
    return first_passage_times(
        model.rates, model.states, model.initial, callbacks, horizon, paths, seed,
    )


def draw(rows, states_list, thresholds, folder):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from scipy.interpolate import PchipInterpolator

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10, "pdf.fonttype": 42,
        "axes.spines.top": False, "axes.spines.right": False,
    })
    colors = ["#245ca3", "#188361", "#bc4b29", "#7956a3", "#957319"]
    palette = [*colors, *plt.get_cmap("tab20").colors]
    widths = {(row["states"], row["strength"], row["threshold"]): row["bm_w25"]
              for row in rows if row["bm_status"] == "converged"}
    fig, ax = plt.subplots(figsize=(7.6, 4.6), layout="constrained")
    x = np.array(thresholds)
    for i, states in enumerate(states_list):
        color = palette[i % len(palette)]
        for strength, _, style, marker in FEEDBACK:
            y = np.array([widths.get((states, strength, b), np.nan) for b in thresholds])
            indices = np.flatnonzero(np.isfinite(y))
            segments = np.split(indices, np.flatnonzero(np.diff(indices) > 1) + 1)
            for segment in segments:
                if len(segment) >= 2:
                    grid = np.linspace(x[segment[0]], x[segment[-1]], max(80, 20 * len(segment)))
                    interpolate = PchipInterpolator(x[segment], y[segment])
                    ax.plot(grid, interpolate(grid), color=color, linestyle=style, linewidth=1.6)
            ax.plot(x[indices], y[indices], color=color, linestyle="None", marker=marker,
                    markersize=4.5, markeredgecolor="white", markeredgewidth=0.45)

    model_handles = [
        Line2D([], [], color=palette[i % len(palette)], linewidth=2, label=str(states))
        for i, states in enumerate(states_list)
    ]
    feedback_handles = [
        Line2D([], [], color="#333333", linestyle=style, marker=marker, markersize=4, label=label)
        for _, label, style, marker in FEEDBACK
    ]
    model_legend = ax.legend(handles=model_handles, title="Promoter states", loc="upper left",
                             bbox_to_anchor=(0, 1.23), frameon=False, ncol=min(5, len(states_list)))
    ax.add_artist(model_legend)
    ax.legend(handles=feedback_handles, title="Feedback", loc="upper right",
              bbox_to_anchor=(1, 1.23), frameon=False, ncol=3)
    ax.set(
        xlabel="Fixed protein threshold, B",
        ylabel="Shortest 25% interval, W₂₅ (time units)",
    )
    ax.set_yscale("log")
    ax.set_xticks(x if len(x) <= 10 else x[::ceil(len(x) / 10)])
    ax.grid(axis="y", color="#e4e6e9")
    ax.set_axisbelow(True)
    if len(widths) < len(rows):
        ax.text(0.99, 0.97, f"{len(rows) - len(widths)} BM points excluded",
                transform=ax.transAxes, ha="right", va="top", fontsize=9)
    fig.savefig(folder / "feedback_w25_overlay.pdf")
    plt.close(fig)


def main():
    states_list = sorted(set(STATE_COUNTS))
    thresholds = sorted(set(THRESHOLDS))
    if not states_list or states_list[0] < 2:
        raise ValueError("STATE_COUNTS must contain integers of at least 2.")
    if not thresholds or thresholds[0] < 1:
        raise ValueError("THRESHOLDS must contain positive integers.")
    if WORKERS < 1 or SSA_PATHS < 0 or min(ORDER_CAP) < 12:
        raise ValueError("Use positive WORKERS, nonnegative SSA_PATHS, and ORDER_CAP of at least (12, 12).")
    if not all(np.isfinite(x) and x > 0 for x in (HORIZON, SOLVER_TIMEOUT)):
        raise ValueError("HORIZON and SOLVER_TIMEOUT must be finite and positive.")

    pairs = [(d, a) for d in states_list for a, *_ in FEEDBACK]
    jobs = [(d, a, b, HORIZON, ORDER_CAP, SOLVER_TIMEOUT) for d, a in pairs for b in thresholds]
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        ssa_jobs = {}
        if SSA_PATHS:
            ssa_jobs = {(d, a): pool.submit(solve_ssa, (d, a, thresholds, HORIZON,
                                                      SSA_PATHS, 348071 + 7919 * index))
                        for index, (d, a) in enumerate(pairs)}
        with (OUTPUT_DIR / "feedback_w25_overlay.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = None
            for index, (row, cdf) in enumerate(pool.map(solve_bm, jobs)):
                if SSA_PATHS:
                    hits = ssa_jobs[row["states"], row["strength"]].result()[index % len(thresholds)]
                    lo, hi = ssa_interval(hits)
                    row.update(ssa_paths=SSA_PATHS, ssa_w25=hi - lo,
                               ssa_hit_fraction=np.mean(np.isfinite(hits)))
                    if cdf is not None:
                        ssa_cdf = np.searchsorted(np.sort(hits), time_grid(HORIZON), side="right") / SSA_PATHS
                        row["max_bm_ssa_cdf_gap"] = np.max(np.abs(cdf - ssa_cdf))
                if writer is None:
                    writer = csv.DictWriter(stream, fieldnames=list(row))
                    writer.writeheader()
                writer.writerow(row)
                stream.flush()
                rows.append(row)
                print(f"d={row['states']}, a={row['strength']:+g}, B={row['threshold']}: "
                      f"W25={row['bm_w25']:.5g}, {row['bm_status']}", flush=True)

    failed = sum(row["bm_status"] != "converged" for row in rows)
    draw(rows, states_list, thresholds, OUTPUT_DIR)
    if failed:
        print(f"{failed} BM cases excluded. Check bm_status in the CSV.", flush=True)
    print(f"Output: {OUTPUT_DIR.resolve()}", flush=True)


if __name__ == "__main__":
    main()
