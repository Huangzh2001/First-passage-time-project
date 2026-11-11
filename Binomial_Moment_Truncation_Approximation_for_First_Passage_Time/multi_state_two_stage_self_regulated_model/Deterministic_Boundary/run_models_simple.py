from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from binomial_moments import adaptive_solve
from boundary_definitions import smooth_boundary_pair
from model_definitions import network
from ssa import estimate_batched


CASES = (
    (0, "product"),
    (0, "weighted"),
    (1, "saturating"),
    (2, "curved"),
)
GAMMAS = (8., 16., 32., 64.)
PATHS = 50_000
SEED = 6188
RESULTS = Path(__file__).resolve().parent / "results_simple"


def run_case(variant, name):
    model = network(variant)
    boundary = smooth_boundary_pair(name)
    times = np.linspace(0., 8., 1001)
    edges = times[::10]
    folder = RESULTS / f"{model.name}__{name}"
    folder.mkdir(parents=True, exist_ok=True)

    hard, _ = estimate_batched(
        model.rates, model.states, model.initial, times,
        paths=PATHS, seed=SEED, first_hit=boundary.first_hit,
    )
    columns = {"time": times, "H_ssa": hard}
    fig, (ax_h, ax_fpt) = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    ax_h.step(times, hard, where="post", color="black", label="SSA")
    ax_fpt.stairs(-np.diff(hard[::10]) / np.diff(edges), edges, color="black", label="SSA")

    for gamma in GAMMAS:
        run, status, _ = adaptive_solve(
            model, boundary, times, gamma, basis="binomial",
            rtol=2e-9, atol=1e-12, max_step=.035, timeout=25.,
        )
        print(f"{folder.name} | γ={gamma:g} | {status}", flush=True)
        if run is None:
            continue

        label = f"γ={gamma:g}, ({run['L']}, {run['K']})"
        style = "-" if status == "converged" else "--"
        ax_h.plot(times, run["h"], style, label=label)
        ax_fpt.plot(times, run["density"], style, label=label)
        columns[f"H_gamma_{gamma:g}"] = run["h"]
        columns[f"fpt_gamma_{gamma:g}"] = run["density"]

    for axis, title in ((ax_h, "Survival H(t)"), (ax_fpt, "FPT density")):
        axis.set(title=title, xlabel="Time")
        axis.legend()
    for ext in ("png", "svg", "pdf"):
        fig.savefig(folder / f"comparison.{ext}", dpi=180)
    plt.close(fig)

    np.savetxt(folder / "curves.csv", np.column_stack(list(columns.values())),
               delimiter=",", header=",".join(columns), comments="")


if __name__ == "__main__":
    for variant, name in CASES:
        run_case(variant, name)