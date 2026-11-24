from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from binomial_moments import build_operator, solve_with_survival, is_valid_curve
from model_definitions import network
from ssa import FirstEvent, estimate_batched


def run_fixed_gamma(gamma=16., fixed=20, orders=(1, 2, 4, 8)):
    model = network(0)
    boundary = FirstEvent(
        lambda t, m, n: (np.maximum(2*m, 1.5*n) - 12) / 4
    )
    times = np.linspace(0., 8., 1001)
    edges = times[::10]

    # SSA 只计算一次
    hard, _ = estimate_batched(
        model.rates, model.states, model.initial, times,
        paths=50_000, seed=6188, first_hit=boundary.first_hit,
    )

    def solve(L, K):
        operator = build_operator(
            model.rates, model.states, model.initial, L, K
        )
        run = solve_with_survival(
            operator, L, K, boundary.g, times, gamma,
            basis="binomial", rtol=2e-9, atol=1e-12,
            max_step=.035, timeout=60.,
        )
        mass_gap = np.max(abs(run["h"] - run["h_from_sum"]))
        valid = is_valid_curve(run) and np.isfinite(mass_gap) and mass_gap <= 1e-5

        print(f"L={L}, K={K}: {run['status']}, "
              f"valid={valid}, mass_gap={mass_gap:.3e}", flush=True)

        return run if valid else None

    reference = solve(fixed, fixed)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), layout="constrained")

    # 第一行固定 K、增加 L；第二行固定 L、增加 K。
    for row, name in enumerate(("L", "K")):

        # 绘制 SSA 参考结果
        axes[row, 0].step(
            times, hard, where="post",
            color="black", linestyle="--", label="SSA"
        )
        axes[row, 1].stairs(
            -np.diff(hard[::10]) / np.diff(edges), edges,
            color="black", linestyle="--", label="SSA"
        )

        for order in orders:
            L, K = (order, fixed) if name == "L" else (fixed, order)
            run = reference if L == K == fixed else solve(L, K)

            if run is None:
                continue

            label = f"L={L}, K={K}"
            axes[row, 0].plot(times, run["h"], label=label)
            axes[row, 1].plot(times, run["density"], label=label)

            if reference is not None:
                error = np.max(abs(run["h"] - reference["h"]))
                print(f"  max |H - H_ref| = {error:.3e}", flush=True)

        condition = f"Fixed gamma={gamma:g}, " + (
            f"K={fixed}" if name == "L" else f"L={fixed}"
        )

        axes[row, 0].set(
            title=condition + " | Survival",
            xlabel="Time", ylabel="H(t)"
        )
        axes[row, 1].set(
            title=condition + " | FPT density",
            xlabel="Time", ylabel="Density"
        )

        for ax in axes[row]:
            ax.legend(fontsize=8)

    fig.savefig(Path(__file__).with_name("fixed_gamma_convergence.pdf"))
    plt.close(fig)


if __name__ == "__main__":
    run_fixed_gamma()