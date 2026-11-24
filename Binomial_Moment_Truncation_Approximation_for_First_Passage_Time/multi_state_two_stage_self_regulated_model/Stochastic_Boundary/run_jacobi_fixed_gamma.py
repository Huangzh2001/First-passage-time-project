from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from model_definitions import network
from random_boundary import RandomBoundary
from random_solvers import solve_moments, sample_hits


def run_fixed_gamma(gamma=16., fixed=(12, 12, 8),
                    orders=((1, 2, 4), (1, 2, 4), (1, 2, 4)),
                    paths=50_000):
    times = np.linspace(0., 6., 301)
    model = network(0)
    boundary = RandomBoundary(
        "jacobi", lower=2.5, upper=10.5,
        sigma=.33, kappa=1.1, theta=.19,
        criterion=lambda m, n, x: m+n-x
    )

    # SSA 参考结果只计算一次
    hits = sample_hits(model, boundary, times[-1], paths, .01, 6188)
    hard = 1 - np.searchsorted(hits, times, side="right") / paths

    def solve(L, K, N):
        return solve_moments(
            model, boundary, times, gamma, L, K, N, 96,
            criterion=lambda t, m, n, x: boundary.g(m, n, x)
        )

    reference = solve(*fixed)
    fig, axes = plt.subplots(3, 2, figsize=(11, 10), layout="constrained")

    # 每次只改变一个截断参数
    for row, name in enumerate(("L", "K", "N")):
        axes[row, 0].plot(times, hard, "k--", label="SSA")
        axes[row, 1].stairs(
            -np.diff(hard[::10]) / np.diff(times[::10]),
            times[::10], color="black", linestyle="--", label="SSA"
        )

        for order in orders[row]:
            params = list(fixed)
            params[row] = order
            h, f, valid = reference if tuple(params) == fixed else solve(*params)
            print(f"L,K,N={params}, valid={valid}", flush=True)

            if valid:
                axes[row, 0].plot(times, h, label=f"{name}={order}")
                axes[row, 1].plot(times, f, label=f"{name}={order}")
                if reference[2]:
                    print("Max error:", np.max(abs(h-reference[0])), flush=True)

        axes[row, 0].set_title(f"Varying {name} | Survival")
        axes[row, 1].set_title(f"Varying {name} | FPT density")
        for ax in axes[row]:
            ax.set_xlabel("Time")
            ax.legend(fontsize=8)

    fig.savefig(Path(__file__).with_name("jacobi_fixed_gamma.pdf"))
    plt.close(fig)


if __name__ == "__main__":
    run_fixed_gamma()
