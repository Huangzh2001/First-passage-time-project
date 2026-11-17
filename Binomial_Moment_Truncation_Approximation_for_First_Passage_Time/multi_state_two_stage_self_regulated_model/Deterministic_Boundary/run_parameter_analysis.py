"""One-at-a-time BM density sweeps with separate SSA checks."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
from functools import partial
import os
from pathlib import Path


for variable in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[variable] = "1"
import matplotlib; matplotlib.rcParams["pdf.fonttype"] = 42

import numpy as np

from binomial_moments import adaptive_solve
from ssa import FirstEvent, TwoStageModel, estimate_batched


BASE = dict(k_on=0.45, k_off=0.25, transcription=2.2, translation=2.4,
            mrna_decay=2.5, protein_decay=0.15, B=4)
PANELS = (
    ("k_on", r"$k_{\mathrm{on}}$", "Promoter activation"),
    ("k_off", r"$k_{\mathrm{off}}$", "Promoter inactivation"),
    ("transcription", r"$\mu$", "ON-state transcription"),
    ("translation", r"$k_{\mathrm{tl}}$", "Translation"),
    ("mrna_decay", r"$\delta_M$", "mRNA degradation"),
    ("protein_decay", r"$\delta_P$", "Protein degradation"),
    ("B", r"$B$", "Protein threshold"),
)
FACTORS = (0.5, 0.75, 1.0, 1.5, 2.0)
TIMES = np.linspace(0.0, 40.0, 2001)
GAMMAS = (48.0, 64.0)


def make_model(params):
    def switching(state, _mrna, _protein):
        return np.stack((params["k_off"] * (state == 1), params["k_on"] * (state == 0)), axis=-1)

    def transcription(state, _mrna, _protein):
        return params["transcription"] * (state == 1)

    return TwoStageModel(
        "two_state_no_feedback", (0, 0, 0), switching, transcription,
        np.full(2, params["translation"]), np.full(2, params["mrna_decay"]),
        np.full(2, params["protein_decay"]),
    )


def boundary_score(_time, _mrna, protein, *, threshold):
    # Counts are integers, so g >= 0 is exactly P >= threshold.
    return protein - threshold + 0.5


def parameter_set(parameter, factor):
    params = {**BASE, parameter: round(BASE[parameter] * factor, 12)}
    if not float(params["B"]).is_integer() or params["B"] < 1:
        raise ValueError("Choose factors that keep the protein threshold a positive integer.")
    params["B"] = int(params["B"])
    return params


def solve_case(job):
    index, varied, factor, params, paths = job
    model = make_model(params)
    threshold = params["B"]
    boundary = FirstEvent(partial(boundary_score, threshold=threshold))
    runs = []
    for gamma in GAMMAS:
        result, status, attempts = adaptive_solve(
            model, boundary, TIMES, gamma,
            start=(6, max(6, threshold + 1)), step=(2, 2), cap=(22, 24),
            basis="binomial", order_tol=2e-5, max_step=0.065, timeout=50.0,
        )
        if status != "converged":
            raise RuntimeError(f"{varied} x{factor:g}, gamma={gamma:g}: {status}; {attempts[-1]}")
        runs.append(result)

    previous, chosen = runs
    cdf = 1 - chosen["h"]
    ssa_at_end, max_cdf_gap, ssa_median = np.nan, np.nan, np.nan
    if paths:
        survival, hits = estimate_batched(
            model.rates, model.states, model.initial, TIMES,
            paths=paths, seed=271331 + 37 * index,
            first_hit=boundary.first_hit,
        )
        ssa_cdf = 1 - survival
        ssa_at_end = ssa_cdf[-1]
        max_cdf_gap = np.max(np.abs(cdf - ssa_cdf))
        ssa_median = hits[(paths - 1) // 2]

    summary = dict(
        parameter=varied, factor=factor, **params, L=chosen["L"], K=chosen["K"],
        gamma=GAMMAS[-1], max_gamma_cdf_gap=np.max(np.abs(previous["h"] - chosen["h"])),
        max_gamma_density_gap=np.max(np.abs(previous["density"] - chosen["density"])),
        ssa_paths=paths, max_bm_ssa_cdf_gap=max_cdf_gap,
        bm_median=np.interp(0.5, cdf, TIMES) if cdf[-1] >= 0.5 else np.inf,
        ssa_median=ssa_median, bm_cdf_at_40=cdf[-1], ssa_cdf_at_40=ssa_at_end,
    )
    print(f"{varied:>14} x{factor:<4g}  L,K={chosen['L']},{chosen['K']}  "
          f"BM/SSA CDF gap={max_cdf_gap:.4g}", flush=True)
    return summary, chosen["density"]


def plot(results, destination):
    import matplotlib as mpl
    mpl.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    mpl.rcParams.update({
        "font.family": "Helvetica",
"pdf.use14corefonts": False,
"pdf.fonttype": 42,
        "font.size": 10, "pdf.fonttype": 42,
        "axes.spines.top": False, "axes.spines.right": False,
    })
    colors = ("#16769f", "#55a5c0", "#262626", "#c87727", "#a6383d")
    styles = ("--", "-.", "-", "-.", "--")
    settings = list(zip(FACTORS, colors, styles))
    fig, axes = plt.subplots(2, 4, figsize=(10.4, 5.1), layout="constrained",
                             sharex=True, sharey=True)
    for index, (parameter, symbol, title) in enumerate(PANELS):
        axis = axes.flat[index]
        for factor, color, style in settings:
            key = tuple(parameter_set(parameter, factor).values())
            axis.plot(TIMES, results[key][1], color=color, ls=style, lw=1.7 if factor == 1 else 1.4)
        axis.set_title(f"({chr(97+index)}) {title} {symbol}", loc="left", fontsize=9.3)
        axis.set(xlim=(0, 20), xticks=np.arange(0, 21, 5))
        axis.set_xlabel(r"Time $t$")
        axis.tick_params(labelbottom=True)
    visible = TIMES <= 20
    ymax = max(np.max(density[visible]) for _, density in results.values())
    axes[0, 0].set_ylim(0, 1.07 * ymax)
    for axis in axes[:, 0]:
        axis.set_ylabel(r"Density $f_T(t)$")
    axes.flat[-1].set_axis_off()
    handles = [Line2D([], [], color=color, ls=style,
                      label="Baseline" if factor == 1 else f"{factor:g}× baseline")
               for factor, color, style in settings]
    axes.flat[-1].legend(handles=handles, loc="center", frameon=False)
    fig.savefig(destination)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths", type=int, default=20_000, help="SSA trajectories per case (0 to skip)")
    parser.add_argument("--workers", type=int, default=4, help="parallel parameter cases")
    parser.add_argument(
    "--output",
    type=Path,
    default=Path(__file__).resolve().parent / "parameter_analysis",
)
    args = parser.parse_args()
    if args.paths < 0 or args.workers < 1:
        parser.error("--paths must be nonnegative and --workers must be positive")


    cases = {}
    for parameter, _, _ in PANELS:
        for factor in FACTORS:
            params = parameter_set(parameter, factor)
            varied = "baseline" if factor == 1.0 else parameter
            cases.setdefault(tuple(params.values()), (varied, factor, params))


    jobs = [(index, *case, args.paths) for index, case in enumerate(cases.values())]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results = dict(zip(cases, pool.map(solve_case, jobs)))

        
    args.output.mkdir(parents=True, exist_ok=True)
    plot(results, args.output / "two_stage_parameter_fpt.pdf")
    plot(results, args.output / "two_stage_parameter_fpt.svg")
    summaries = [summary for summary, _ in results.values()]
    with (args.output / "parameter_validation.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)


if __name__ == "__main__":
    main()
