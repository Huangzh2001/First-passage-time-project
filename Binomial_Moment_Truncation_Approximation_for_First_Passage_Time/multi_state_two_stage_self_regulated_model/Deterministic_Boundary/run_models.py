"""模型、边界、运行选项和输出都配置在本文件。"""

from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path
import csv

import matplotlib.pyplot as plt
import numpy as np

from binomial_moments import build_operator, survival
from ssa import first_passage_times


T = 6.0
INITIAL = (0, 0, 0)  # 启动子状态、mRNA 数、蛋白质数
GAMMAS = (16, 32, 64, 128, 256)
PATHS, SEED = 80_000, 2026
CUTOFF = (12, 16)
AND_CUTOFF = (16, 22)
RATE_CAP = 1e6


class Promoter:
    def __init__(self, transcription, edges, translation=1.1, mrna_decay=.8, protein_decay=.4):
        self.transcription = np.asarray(transcription)
        self.states = len(transcription)
        self.base = np.zeros((self.states, self.states))
        self.feedback = np.zeros_like(self.base)
        for i, j, rate, strength in edges:
            self.base[i, j], self.feedback[i, j] = rate, strength
        self.translation = translation
        self.mrna_decay = mrna_decay
        self.protein_decay = protein_decay

    def rates(self, i, m, n):
        # 两个求解器共享这套反应率。也可以在这里定义其他非线性调控。
        n_squared = np.square(n, dtype=float)
        hill = n_squared / (9 + n_squared)
        switches = self.base[i] + hill[..., None]*self.feedback[i]
        counts = np.stack([self.transcription[i], self.translation*m,
                           self.mrna_decay*m, self.protein_decay*n], axis=-1)
        return np.concatenate((switches, counts), axis=-1)


# 一条切换边为 (起点, 终点, 基础速率, 蛋白质 Hill 反馈强度)。
MODELS = {
    "refractory_3": Promoter([0, 0, 2.5], [(0, 1, 1, 0), (1, 2, 1.4, 0), (2, 0, .7, 0)]),
    "sequential_6": Promoter([0, 0, 0, 0, 0, 4.0],
        [(0, 1, 2.1, 0), (1, 2, 2.3, 0), (2, 3, 2, 0), (3, 4, 2.2, 0),
         (4, 5, 2.4, 0), (5, 0, .65, 0)]),
    "graded_4": Promoter([0, 0, 1.6, 3.8],
        [(0, 1, 1.2, 0), (1, 0, .45, 0), (1, 2, 1.1, 0),
         (2, 1, .65, 0), (2, 3, .9, 0), (3, 2, .55, 0)]),
    "positive_3": Promoter([0, 0, 3.5],
        [(0, 1, 1.3, 0), (1, 0, .45, 0), (1, 2, .55, 2.4), (2, 1, .8, 0)]),
    "negative_5": Promoter([0, 0, 2.5, .6, 4.2],
        [(0, 1, 1.7, 0), (1, 2, 1.4, 0), (1, 3, .8, 0), (2, 0, .55, 2.8),
         (3, 4, 1.2, 0), (3, 0, .35, 0), (4, 0, .7, 3.2)]),
}


@dataclass
class Boundary:
    thresholds: object  # 函数 t -> (mRNA 阈值, 蛋白质阈值)
    combine: object = np.maximum  # maximum = OR，minimum = AND
    crossings: tuple = ()  # 阈值碰到整数的时刻，包括切触
    jumps: tuple = ()      # 阈值发生不连续跳变的时刻

    def signed_distance(self, t, m, n):
        a, b = self.thresholds(t)
        return self.combine(m-a, n-b)

    def schedule(self):
        # SSA 使用硬边界，因此只需要知道每段里哪些整数计数算越界。
        cuts = np.array(sorted({0., T, *(t for t in (*self.crossings, *self.jumps) if 0 < t < T)}))
        at_cut = np.asarray([self.thresholds(t) for t in cuts])
        between = np.asarray([self.thresholds(t) for t in (cuts[:-1]+cuts[1:])/2])
        # 解析交点代回阈值时，消除浮点运算造成的微小整数偏差。
        at_cut = np.where(np.isclose(at_cut, np.rint(at_cut), rtol=0, atol=1e-12),
                          np.rint(at_cut), at_cut)
        return cuts, at_cut, between, self.combine


def linear_crossings(intercept, slope):
    low, high = sorted((intercept, intercept+slope*T))
    return tuple((count-intercept)/slope for count in range(int(np.ceil(low)), int(np.floor(high))+1))


def periodic_crossings():
    omega = 2*np.pi/3
    for count in range(5, 9):
        phase = np.arccos((count-6.5)/2.1)
        for cycle in range(int(np.ceil(T/3))):
            yield (2*np.pi*cycle + phase)/omega
            yield (2*np.pi*(cycle+1) - phase)/omega


BOUNDARIES = {
    "fixed": Boundary(lambda t: (np.inf, 6)),
    "falling": Boundary(lambda t: (np.inf, 8.4-.7*t),
                        crossings=linear_crossings(8.4, -.7)),
    "periodic": Boundary(lambda t: (np.inf, 6.5+2.1*np.cos(2*np.pi*t/3)),
                          crossings=tuple(periodic_crossings())),
    "step_or": Boundary(lambda t: (5, 8) if t < 2 else (3, 5) if t < 4 else (4, 7),
                        jumps=(2., 4.)),
    "moving_and": Boundary(lambda t: (2.2-.17*t, 4.4+.37*t), np.minimum,
                           crossings=linear_crossings(2.2, -.17) + linear_crossings(4.4, .37)),
}

# 改这里选择要一起跑的模型与边界。--all 会运行全部模型 × 全部边界。
CASES = {
    "refractory_3": ("fixed",),
    "sequential_6": ("falling",),
    "graded_4": ("fixed",),
    "positive_3": ("periodic",),
    "negative_5": ("moving_and", "step_or"),
}


def run_model(name, boundary_names, args, h_csv, fpt_csv):
    model = MODELS[name]
    boundaries = [BOUNDARIES[key] for key in boundary_names]
    schedules = [boundary.schedule() for boundary in boundaries]
    hits = first_passage_times(model.rates, model.states, INITIAL, schedules,
                               T, args.paths, SEED + list(MODELS).index(name))
    edges = np.round(np.linspace(0, T, 121), 12)
    plot_times = np.round(np.linspace(0, T, 601), 12)
    jump_offsets = np.geomspace(1e-7, .05, 25)
    fig, axes = plt.subplots(2, len(boundaries), figsize=(4.1*len(boundaries), 6.5),
                             squeeze=False, layout="constrained")
    operators = {}
    for column, (key, boundary, schedule, hit) in enumerate(zip(boundary_names, boundaries, schedules, hits)):
        cuts = schedule[0][1:-1]
        left = np.maximum(0, cuts - 1e-6)
        jumps = np.array(sorted(t for t in boundary.jumps if 0 < t < T))
        near_jumps = (jumps[:, None] + jump_offsets).ravel()
        times = np.unique(np.r_[plot_times, edges, cuts, left, near_jumps[near_jumps < T]])
        density_segments = np.split(np.arange(len(times)), np.searchsorted(times, jumps))
        bin_index = np.searchsorted(times, edges)

        h_ssa = 1 - np.searchsorted(np.sort(hit), times, side="right")/args.paths
        fpt_ssa = np.histogram(hit, bins=edges)[0]/args.paths
        ax, fpt = axes[:, column]
        ax.plot(times, h_ssa, color="black", lw=1.6, label="SSA")
        fpt.stairs(fpt_ssa/np.diff(edges), edges, fill=True, color="#718e9d", alpha=.4)

        def write_curves(method, gamma, h, mass):
            h_csv.writerows((name, key, method, gamma, t, p) for t, p in zip(times, h))
            fpt_csv.writerows((name, key, method, gamma, a, b, p)
                             for a, b, p in zip(edges[:-1], edges[1:], mass))

        write_curves("SSA", "", h_ssa, fpt_ssa)
        cutoff = AND_CUTOFF if boundary.combine is np.minimum else CUTOFF
        if cutoff not in operators:
            operators[cutoff] = build_operator(model.rates, model.states, INITIAL, *cutoff)
        for gamma in args.gammas:
            h, density = survival(operators[cutoff], boundary.signed_distance, times, gamma,
                                  jumps=boundary.jumps, rate_cap=RATE_CAP, return_density=True)
            mass = -np.diff(h[bin_index])
            write_curves("binomial", gamma, h, mass)
            line, = ax.plot(times, h, lw=1.4, label=f"γ = {gamma:g}")
            for segment in density_segments:
                fpt.plot(times[segment], density[segment], color=line.get_color(), lw=1.4)
            print(f"{name} | {key} | γ={gamma:g} | H(T) {h[-1]:.6f} | SSA {h_ssa[-1]:.6f}",
                  flush=True)

        ax.set(title=key, ylabel="Survival H(t)", xlim=(0, T))
        fpt.set(xlabel="Time", ylabel="First-passage density (1/time)", xlim=(0, T))
        if jumps.size:
            # 跳变后的软密度可能很高，用对称对数轴同时显示峰值和背景。
            fpt.set_yscale("symlog", linthresh=.1)
            fpt.set_title("Symmetric log scale", fontsize=9)
        for panel in (ax, fpt):
            panel.grid(alpha=.2)
    fig.suptitle(name)
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.savefig(args.output/f"{name}.png", dpi=160)
    plt.close(fig)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--all", action="store_true", help="Run all models against all boundaries.")
    parser.add_argument("--paths", type=int, default=PATHS)
    parser.add_argument("--gammas", type=float, nargs="+", default=GAMMAS)
    parser.add_argument("--output", type=Path, default=Path("results"),
                        help="Output directory. Relative paths are based on this script's directory.")
    args = parser.parse_args()
    if args.paths < 1 or any(not np.isfinite(gamma) or gamma <= 0 for gamma in args.gammas):
        parser.error("paths and all gamma values must be positive")
    cases = {name: tuple(BOUNDARIES) for name in MODELS} if args.all else CASES
    args.output = (Path(__file__).resolve().parent / args.output).resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    print(f"结果保存目录：{args.output}", flush=True)
    with (args.output/"survival.csv").open("w", newline="") as h_file, \
         (args.output/"first_passage.csv").open("w", newline="") as fpt_file:
        h_csv, fpt_csv = csv.writer(h_file), csv.writer(fpt_file)
        h_csv.writerow(["model", "boundary", "method", "gamma", "time", "H"])
        fpt_csv.writerow(["model", "boundary", "method", "gamma", "bin_left", "bin_right", "mass"])
        for name, boundary_names in cases.items():
            run_model(name, boundary_names, args, h_csv, fpt_csv)


if __name__ == "__main__":
    main()
