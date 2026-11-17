"""Compare slow and fast promoter switching using the project's BM and SSA.

Place this file beside binomial_moments.py and ssa.py, then run it.
"""
import argparse
from pathlib import Path

import numpy as np

from binomial_moments import adaptive_solve
from ssa import FirstEvent, TwoStageModel, estimate_batched


THRESHOLD = 4
TIME = np.linspace(0.0, 120.0, 3001)
GAMMAS = (32.0, 48.0, 64.0)
SPEED_FACTORS = (0.08, 5.0)
BASE_ON_RATE = 0.25
BASE_OFF_RATE = 0.75
ON_FRACTION = BASE_ON_RATE / (BASE_ON_RATE + BASE_OFF_RATE)
SEEDS = (1272047, 1759027)


def model(speed, states=2):
    # 单向循环：最后一个状态是唯一的活跃态。
    off_step_rate = BASE_ON_RATE

    on_fraction = off_step_rate / (
        off_step_rate + (states - 1) * BASE_OFF_RATE
    )
    off_fraction = (1.0 - on_fraction) / (states - 1)
    initial = np.array([off_fraction] * (states - 1) + [on_fraction])

    rates = np.zeros((states, states))

    for i in range(states - 1):
        rates[i, i + 1] = off_step_rate

    rates[states - 1, 0] = BASE_OFF_RATE

    def switching(promoter, _mrna, _protein):
        return speed * rates[promoter]

    def transcription(promoter, _mrna, _protein):
        return 4.4 * (promoter == states - 1)

    return TwoStageModel(
        name=f"states={states}, speed={speed:g}",
        initial=(initial, 0, 0),
        switching=switching,
        transcription=transcription,
        translation=np.full(states, 2.4),
        mrna_decay=np.full(states, 2.5),
        protein_decay=np.full(states, 0.15),
    )


def defining_function(_time, _mrna, protein):
    return protein - THRESHOLD + 0.5


def calculate_bm(models, boundary):
    runs = []
    for network in models:
        gamma_runs = []
        for gamma in GAMMAS:
            result, status, _ = adaptive_solve(
                network, boundary, TIME, gamma,
                start=(6, 6), step=(2, 2), cap=(18, 18),
                basis="binomial", max_step=0.06, timeout=45.0,
            )
            if status != "converged":
                raise RuntimeError(f"{network.name}, gamma={gamma:g}: {status}")
            print(f"{network.name} BM gamma={gamma:g}: L={result['L']}, K={result['K']}", flush=True)
            gamma_runs.append(result)
        runs.append(gamma_runs)
    return runs


def calculate_ssa(models, paths, boundary):
    runs = []
    for network, seed in zip(models, SEEDS):
        survival, hits = estimate_batched(
            network.rates, network.states, network.initial, TIME,
            paths=paths, seed=seed, first_hit=boundary.first_hit,
        )
        runs.append((1.0 - survival, hits))
        print(f"{network.name} SSA: {paths:,} paths", flush=True)
    return runs


def diagnostics(bm, ssa, paths):
    width = np.sqrt(np.log(80.0) / (2 * paths))  # 95% simultaneous DKW bands for both SSA CDFs.
    print(f"95% simultaneous SSA CDF half-width = {width:.6f}")
    for speed, gamma_runs, (cdf, hits) in zip(SPEED_FACTORS, bm, ssa):
        bm_cdf = 1.0 - gamma_runs[-1]["h"]
        delta_gamma = np.max(np.abs(gamma_runs[-1]["h"] - gamma_runs[-2]["h"]))
        print(f"speed={speed:g}: max BM-SSA CDF gap={np.max(np.abs(bm_cdf - cdf)):.6f}; "
              f"max BM gamma 48-64 CDF gap={delta_gamma:.6f}")
        for t in (2.0, 4.0, 8.0, 12.0, 16.0, 20.0, 40.0, 120.0):
            print(f"  t={t:g}: BM={np.interp(t, TIME, bm_cdf):.6f}, SSA={np.interp(t, TIME, cdf):.6f}")
        for q in (0.5, 0.9):
            position = int(np.ceil(q * paths)) - 1
            bm_quantile = np.interp(q, bm_cdf, TIME) if q <= bm_cdf[-1] else np.inf
            print(f"  q{q:.2f} BM={bm_quantile:.6f}, SSA={hits[position]:.6f}")
    for label, curves in (
        ("BM", [1.0 - run[-1]["h"] for run in bm]),
        ("SSA", [run[0] for run in ssa]),
    ):
        gap = curves[0] - curves[1]
        leading = np.argmax(gap)
        lagging = np.argmin(gap)
        print(f"{label}: slow minus fast max={gap[leading]:.6f} at t={TIME[leading]:g}, "
              f"min={gap[lagging]:.6f} at t={TIME[lagging]:g}")


def plot(bm, output):
    import matplotlib as mpl
    mpl.use("Agg")
    import matplotlib.pyplot as plt

    mpl.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
        "font.size": 10, "pdf.fonttype": 42,
        "axes.spines.top": False, "axes.spines.right": False,
    })
    colors = ("#20639b", "#b75b29")
    fig, (ax_cdf, ax_pdf) = plt.subplots(1, 2, figsize=(7.45, 3.1), layout="constrained")
    labels = ("Slow switching", "Fast switching")
    for gamma_runs, color, label in zip(bm, colors, labels):
        chosen = gamma_runs[-1]
        ax_cdf.plot(TIME, 1.0 - chosen["h"], color=color, linewidth=1.75, label=label)
        ax_pdf.plot(TIME, chosen["density"], color=color, linewidth=1.75)

    ax_cdf.set(xlim=(0, 60), ylim=(0, 1.03), xlabel="Time", ylabel="Cumulative probability",
               xticks=np.arange(0, 61, 10))
    ax_pdf.set(xlim=(0, 24), ylim=(0, None), xlabel="Time", ylabel="FPT density",
               xticks=np.arange(0, 25, 4))
    ax_cdf.set_title("(a) First-passage CDF", loc="left", fontsize=11)
    ax_pdf.set_title("(b) First-passage density", loc="left", fontsize=11)
    ax_cdf.legend(loc="lower right", frameon=False)
    fig.savefig(output)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--states", type=int, nargs="+", choices=(2, 3, 4),
                    default=(2, 4, 8))
    parser.add_argument("--output", type=Path,
                    default=Path(__file__).resolve().parent / "switch_rate")
    parser.add_argument("--paths", type=int, default=100_000, help="SSA paths per condition")
    args = parser.parse_args()

    if args.paths < 0:
        parser.error("--paths must be nonnegative")
    args.output.mkdir(parents=True, exist_ok=True)

    boundary = FirstEvent(defining_function)
    for states in args.states:
        print(f"\n--- {states} states ---", flush=True)

        models = [model(speed, states=states) for speed in SPEED_FACTORS]
        bm = calculate_bm(models, boundary)

        if args.paths:
            ssa = calculate_ssa(models, args.paths, boundary)
            diagnostics(bm, ssa, args.paths)

        output = args.output / f"strong_switching_{states}_states.pdf"
        plot(bm, output)
        print(f"Saved: {output}", flush=True)


if __name__ == "__main__":
    main()
