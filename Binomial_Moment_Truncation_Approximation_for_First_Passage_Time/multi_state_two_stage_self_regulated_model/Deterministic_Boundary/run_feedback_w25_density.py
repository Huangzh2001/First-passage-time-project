# """Plot FPT densities and shortest 25% intervals for matched feedback comparisons."""
# from concurrent.futures import ProcessPoolExecutor
# import csv
# from functools import partial
# import os
# from pathlib import Path

# for thread_var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
#     os.environ[thread_var] = "1"

# import numpy as np
# from scipy.integrate import trapezoid

# from binomial_moments import adaptive_solve
# from ssa import FirstEvent, TwoStageModel, first_passage_times


# STATE_COUNTS = (2, 4, 8)
# THRESHOLDS = (2, 4, 6)
# WORKERS = 6
# SSA_PATHS = 20_000  # Set to zero to draw from BM without SSA validation.
# OUTPUT_DIR = Path(__file__).resolve().parent / "feedback_w25_density"
# STRENGTHS = (-0.9, 0.0, 0.9)
# LABELS = ("Negative feedback", "No feedback", "Positive feedback")
# COLORS = ("#216c9e", "#373737", "#b5532a")
# GAMMAS = (48.0, 64.0)
# TIME = np.r_[np.linspace(0, 40, 1001), np.arange(40.1, 240.01, 0.1)]
# MASS = 0.25


# def make_model(states, strength):
#     initial = np.array([0.75 / (states - 1)] * (states - 1) + [0.25])
#     rates = np.zeros((states, states))
#     for i in range(states - 1):
#         rates[i + 1, i] = 0.75
#         rates[i, i + 1] = 0.75 * initial[i + 1] / initial[i]

#     def switching(state, _mrna, _protein):
#         return rates[state]

#     def transcription(state, _mrna, protein):
#         return 4.4 * (state == states - 1) * (1 + strength * protein / (2 + protein))

#     return TwoStageModel(
#         f"d={states}, a={strength:g}", (initial, 0, 0), switching, transcription,
#         np.full(states, 2.4), np.full(states, 2.5), np.full(states, 0.15),
#     )


# def score(_time, _mrna, protein, *, threshold):
#     return protein - threshold + 0.5


# def fixed_boundary(threshold):
#     return FirstEvent(partial(score, threshold=threshold))


# def shortest_bm_interval(survival, mass=MASS):
#     cdf = np.maximum.accumulate(np.clip(1 - survival, 0.0, 1.0))
#     probability, first = np.unique(cdf, return_index=True)
#     last_start = probability[-1] - mass
#     if last_start < 0:
#         raise ValueError("Extend the time grid until the CDF reaches 25%.")
#     # The width of the interpolated quantile interval is piecewise linear in q.
#     # Its minimum is at a knot of Q(q) or Q(q + mass), or at an endpoint.
#     knots = np.r_[0.0, last_start, probability, probability - mass]
#     starts = knots[(knots >= 0) & (knots <= last_start)]
#     lo = np.interp(starts, probability, TIME[first])
#     hi = np.interp(starts + mass, probability, TIME[first])
#     index = np.argmin(hi - lo)
#     return lo[index], hi[index]


# def shortest_ssa_interval(hit_times):
#     count = int(np.ceil(MASS * len(hit_times)))
#     finite = np.sort(hit_times[np.isfinite(hit_times)])
#     if len(finite) < count:
#         raise ValueError("Extend the SSA horizon until 25% of paths have hit.")
#     lo, hi = finite[:len(finite) - count + 1], finite[count - 1:]
#     index = np.argmin(hi - lo)
#     return lo[index], hi[index]


# def solve_group(job):
#     index, states, strength, paths = job
#     model = make_model(states, strength)
#     groups = []
#     for threshold in THRESHOLDS:
#         boundary = fixed_boundary(threshold)
#         solutions = []
#         for gamma in GAMMAS:
#             run, status, attempts = adaptive_solve(
#                 model, boundary, TIME, gamma,
#                 start=(8, max(8, threshold + 1)), step=(2, 2), cap=(22, 22),
#                 basis="binomial", order_tol=2e-5, max_step=1.25,
#                 rtol=1e-9, timeout=60.0,
#             )
#             if status != "converged":
#                 raise RuntimeError(f"{model.name}, B={threshold}, gamma={gamma:g}: {status}; {attempts[-1]}")
#             solutions.append(run)
#         groups.append(solutions)
#         print(f"{model.name}, B={threshold}: BM ready", flush=True)

#     # A path contributes the first hit of every fixed threshold.
#     callbacks = [fixed_boundary(b).first_hit for b in THRESHOLDS]
#     hits = first_passage_times(
#         model.rates, model.states, model.initial, callbacks, TIME[-1], paths, 348071 + index * 7919,
#     ) if paths else None

#     results = []
#     for i, (threshold, (check, chosen)) in enumerate(zip(THRESHOLDS, groups)):
#         survival = chosen["h"]
#         lo, hi = shortest_bm_interval(survival)
#         check_lo, check_hi = shortest_bm_interval(check["h"])
#         ssa_lo = ssa_hi = ssa_gap = np.nan
#         if paths:
#             hit_times = hits[i]
#             ssa_lo, ssa_hi = shortest_ssa_interval(hit_times)
#             cdf = np.searchsorted(np.sort(hit_times), TIME, side="right") / paths
#             ssa_gap = np.max(np.abs(1 - survival - cdf))
#         # Integrate the survival curve to obtain moments of min(T, horizon).
#         s = np.minimum.accumulate(np.clip(survival, 0.0, 1.0))
#         mean = trapezoid(s, TIME)
#         sd = np.sqrt(max(0.0, 2.0 * trapezoid(TIME * s, TIME) - mean**2))
#         _, end50 = shortest_bm_interval(survival, 0.5)
#         summary = dict(
#             bm_mean_240=mean, bm_sd_240=sd, bm_cv_240=sd / mean, bm_end50=end50,
#             states=states, threshold=threshold, strength=strength, L=chosen["L"], K=chosen["K"],
#             bm_lo=lo, bm_hi=hi, bm_width=hi - lo, ssa_paths=paths,
#             ssa_lo=ssa_lo, ssa_hi=ssa_hi, ssa_width=ssa_hi - ssa_lo,
#             max_bm_ssa_cdf_gap=ssa_gap, bm_width_gamma_gap=abs((hi - lo) - (check_hi - check_lo)),
#             bm_cdf_at_horizon=1 - survival[-1],
#         )
#         print(f"{model.name}, B={threshold}: W25={hi - lo:.4f}, "
#               f"SSA={ssa_hi - ssa_lo:.4f}", flush=True)
#         results.append((summary, chosen["density"]))
#     return results

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


# def draw(results, output):
#     import matplotlib as mpl
#     mpl.use("Agg")
#     import matplotlib.pyplot as plt

#     mpl.rcParams.update({
#         "font.family": "sans-serif", "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
#         "font.size": 10, "pdf.fonttype": 42,
#         "axes.spines.top": False, "axes.spines.right": False,
#     })
#     for states in STATE_COUNTS:
#         fig, axes = plt.subplots(len(THRESHOLDS), 3, figsize=(9.7, 2.4 * len(THRESHOLDS)),
#                                  layout="constrained", sharex="row", sharey="row", squeeze=False)
#         fig.suptitle(f"{states} promoter states")
#         for row, threshold in enumerate(THRESHOLDS):
#             curves = [results[states, threshold, strength] for strength in STRENGTHS]
#             # Keep a common scale within each threshold row.
#             largest_end = max(s["bm_end50"] for s, _ in curves)
#             xmax = 5 * np.ceil(max(15, 1.32 * largest_end) / 5)
#             peaks = max(np.max(density[TIME <= xmax]) for _, density in curves)
#             for col, ((info, density), label, color) in enumerate(zip(curves, LABELS, COLORS)):
#                 ax = axes[row, col]
#                 lo, hi = info["bm_lo"], info["bm_hi"]
#                 ax.plot(TIME, density, color=color, lw=1.65)
#                 inside = (TIME > lo) & (TIME < hi)
#                 shading_times = np.r_[lo, TIME[inside], hi]
#                 ax.fill_between(shading_times, np.interp(shading_times, TIME, density),
#                                 color=color, alpha=0.27)
#                 ax.axvline(lo, color=color, alpha=0.55, lw=0.8)
#                 ax.axvline(hi, color=color, alpha=0.55, lw=0.8)
#                 ax.text(0.97, 0.91, f"W25 = {hi-lo:.2f}", ha="right",
#                         va="top", transform=ax.transAxes, fontsize=9)
#                 ax.set(xlim=(0, xmax), ylim=(0, 1.10 * peaks))
#                 ax.tick_params(labelbottom=True)
#                 if row == 0:
#                     ax.set_title(label, color=color, fontsize=11)
#                 if col == 0:
#                     ax.set_ylabel(f"B = {threshold}\nFPT density")
#                 if row == len(THRESHOLDS) - 1:
#                     ax.set_xlabel("Time")
#         fig.savefig(output / f"feedback_w25_{states}_states.pdf")
#         plt.close(fig)


# def main():
#     if SSA_PATHS < 0 or SSA_PATHS == 1 or WORKERS < 1:
#         raise ValueError("SSA_PATHS must be 0 or at least 2, and WORKERS must be positive.")

#     conditions = [(states, strength) for states in STATE_COUNTS for strength in STRENGTHS]
#     jobs = [(i, states, strength, SSA_PATHS) for i, (states, strength) in enumerate(conditions)]
#     with ProcessPoolExecutor(max_workers=WORKERS) as pool:
#         groups = list(pool.map(solve_group, jobs))
#     results = {(s["states"], s["threshold"], s["strength"]): (s, d)
#                for group in groups for s, d in group}
#     OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
#     draw(results, OUTPUT_DIR)
#     with (OUTPUT_DIR / "feedback_w25_intervals.csv").open("w", encoding="utf-8", newline="") as stream:
#         rows = [item[0] for item in results.values()]
#         writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
#         writer.writeheader()
#         writer.writerows(rows)


def draw(results, output):
    import matplotlib as mpl
    mpl.use("Agg")
    import matplotlib.pyplot as plt

    mpl.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "pdf.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    times = time_grid(HORIZON)
    states_list = sorted(set(STATE_COUNTS))
    thresholds = sorted(set(THRESHOLDS))
    colors = ("#216c9e", "#373737", "#b5532a")

    for states in states_list:
        fig, axes = plt.subplots(
            len(thresholds), len(FEEDBACK),
            figsize=(9.7, 2.4 * len(thresholds)),
            layout="constrained", sharex="row", sharey="row", squeeze=False,
        )
        fig.suptitle(f"{states} promoter states")

        for i, threshold in enumerate(thresholds):
            curves = [
                results[states, threshold, strength]
                for strength, *_ in FEEDBACK
            ]
            ends = [
                info["bm_hi"] for info, _ in curves
                if info["bm_status"] == "converged"
            ]
            xmax = float(times[-1])
            if ends:
                xmax = min(xmax, max(15.0, 1.32 * max(ends)))

            visible = times <= xmax
            peak = max(
                (np.max(density[visible]) for _, density in curves if density is not None),
                default=0.0,
            )
            ymax = 1.10 * peak if peak > 0 else 1.0

            for j, ((info, density), (_, label, _, _)) in enumerate(zip(curves, FEEDBACK)):
                ax = axes[i, j]
                color = colors[j % len(colors)]

                if density is not None:
                    ax.plot(times, density, color=color, lw=1.65)

                if density is not None and info["bm_status"] == "converged":
                    lo, hi = info["bm_lo"], info["bm_hi"]
                    inside = (times > lo) & (times < hi)
                    shading_times = np.r_[lo, times[inside], hi]
                    ax.fill_between(
                        shading_times, np.interp(shading_times, times, density),
                        color=color, alpha=0.27,
                    )
                    ax.axvline(lo, color=color, alpha=0.55, lw=0.8)
                    ax.axvline(hi, color=color, alpha=0.55, lw=0.8)
                    text = f"W25 = {info['bm_w25']:.2f}"
                elif density is not None:
                    text = "W25 unavailable within horizon"
                else:
                    text = "BM did not converge"

                ax.text(
                    0.97, 0.91, text,
                    ha="right", va="top", transform=ax.transAxes, fontsize=8,
                )
                ax.set(xlim=(0, xmax), ylim=(0, ymax))
                ax.tick_params(labelbottom=True)
                if i == 0:
                    ax.set_title(label, color=color, fontsize=11)
                if j == 0:
                    ax.set_ylabel(f"B = {threshold}\nFPT density")
                if i == len(thresholds) - 1:
                    ax.set_xlabel("Time")

        fig.savefig(output / f"feedback_w25_{states}_states.pdf")
        plt.close(fig)


def main():
    states_list = sorted(set(STATE_COUNTS))
    thresholds = sorted(set(THRESHOLDS))
    if not states_list or not thresholds:
        raise ValueError("STATE_COUNTS and THRESHOLDS must be nonempty.")
    if WORKERS < 1 or SSA_PATHS < 0:
        raise ValueError("WORKERS must be positive and SSA_PATHS must be nonnegative.")

    pairs = [(d, a) for d in states_list for a, *_ in FEEDBACK]
    jobs = [
        (d, a, b, HORIZON, ORDER_CAP, SOLVER_TIMEOUT)
        for d, a in pairs
        for b in thresholds
    ]

    times = time_grid(HORIZON)
    threshold_index = {b: i for i, b in enumerate(thresholds)}
    output = OUTPUT_DIR / "density"
    output.mkdir(parents=True, exist_ok=True)
    results = {}

    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        ssa_jobs = {}
        if SSA_PATHS:
            ssa_jobs = {
                (d, a): pool.submit(
                    solve_ssa,
                    (d, a, thresholds, HORIZON, SSA_PATHS, 348071 + 7919 * i),
                )
                for i, (d, a) in enumerate(pairs)
            }

        for row, cdf in pool.map(solve_bm, jobs):
            d = row["states"]
            a = row["strength"]
            b = row["threshold"]

            if SSA_PATHS:
                hits = ssa_jobs[d, a].result()[threshold_index[b]]
                lo, hi = ssa_interval(hits)
                row.update(
                    ssa_paths=SSA_PATHS,
                    ssa_w25=hi - lo,
                    ssa_hit_fraction=np.mean(np.isfinite(hits)),
                )
                if cdf is not None:
                    ssa_cdf = np.searchsorted(np.sort(hits), times, side="right") / SSA_PATHS
                    row["max_bm_ssa_cdf_gap"] = np.max(np.abs(cdf - ssa_cdf))

            density = None
            if cdf is not None:
                cdf_for_plot = np.maximum.accumulate(np.clip(cdf, 0.0, 1.0))
                density = np.maximum(np.gradient(cdf_for_plot, times), 0.0)

            results[d, b, a] = (row, density)
            print(f"d={d}, a={a:+g}, B={b}: W25={row['bm_w25']:.5g}, {row['bm_status']}",
                  flush=True)

    rows = [row for row, _ in results.values()]
    with (output / "feedback_w25_intervals.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    draw(results, output)
    print(f"Output: {output.resolve()}", flush=True)

if __name__ == "__main__":
    main()
