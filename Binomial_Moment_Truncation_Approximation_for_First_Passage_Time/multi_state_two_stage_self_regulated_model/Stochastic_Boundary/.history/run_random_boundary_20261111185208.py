"""与现有 binomial_moments.py、ssa.py、model_definitions.py 放在同一目录运行。"""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from model_definitions import network
from random_boundary import BrownianBoundary
from random_solvers import sample_hits, solve_moments

MODEL = 0
BOUNDARY = BrownianBoundary(lower=3.5, upper=7.5, sigma=1.)
PATHS = 2000
DT_VALUES = (.02, .01)
# 每行为 (gamma, L, K, N)，分别比较边界基数、分子矩阶数和软参数。
CASES = ((4., 8, 8, 4), (4., 8, 8, 8), (4., 12, 12, 8), (8., 12, 12, 8), (16., 12, 12, 8))
QUADRATURE = 96
TIMES = np.linspace(0., 5., 251)
RESULTS = Path(__file__).resolve().parent / 'results_random_brownian'


def main():
    RESULTS.mkdir(exist_ok=True)
    model = network(MODEL)
    fig, (ax_h, ax_fpt) = plt.subplots(1, 2, figsize=(12, 4.5), layout='constrained')
    for dt in DT_VALUES:
        hits = sample_hits(model, BOUNDARY, TIMES[-1], PATHS, dt, 6188)
        h = 1 - np.searchsorted(hits, TIMES, side='right') / PATHS
        label = f'SSA + Brownian, dt={dt:g}'
        ax_h.step(TIMES, h, where='post', label=label)
        edges = TIMES[::10]
        ax_fpt.stairs(
            -np.diff(h[::10]) / np.diff(edges),
            edges,
            label=label,
        )
        np.savetxt(RESULTS / f'hits_dt_{dt:g}.csv', hits, delimiter=',', header='hit', comments='')
        np.savetxt(RESULTS / f'ssa_dt_{dt:g}.csv', np.column_stack((TIMES, h)),
                   delimiter=',', header='time,H', comments='')
    for gamma, L, K, N in CASES:
        h, fpt, valid = solve_moments(model, BOUNDARY, TIMES, gamma, L, K, N, QUADRATURE)
        label = f'gamma={gamma:g}, L={L}, K={K}, N={N}'
        print(f'{label}: valid={valid}', flush=True)
        np.savetxt(RESULTS / f'gamma_{gamma:g}_L{L}_K{K}_N{N}.csv',
                   np.column_stack((TIMES, h, fpt)), delimiter=',', header='time,H,FPT', comments='')
        if valid:
            ax_h.plot(TIMES, h, label=label)
            ax_fpt.plot(TIMES, fpt, label=label)
    for axis, title in ((ax_h, 'Survival H(t)'), (ax_fpt, 'FPT density')):
        axis.set(title=title, xlabel='Time')
        axis.legend(fontsize=7)
    for extension in ('png', 'svg', 'pdf'):
        fig.savefig(RESULTS / f'comparison.{extension}', dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    main()
