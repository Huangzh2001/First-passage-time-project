"""多状态基因表达的 Gillespie SSA 与硬边界首达。"""

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np


def first_passage_times(rates, states, initial, boundaries, horizon, paths, seed):
    """
    每条轨迹都从给定的初始状态出发。程序反复做三件事：
    1. 根据当前反应速率，随机抽出下一次反应什么时候发生。
    2. 在等反应的这段时间里，mRNA 和蛋白质数量固定，但门槛可能会变，所以用 first_hit 查有没有在反应前就达标。
    3. 如果反应在观察期限内，就随机决定发生哪种反应，更新启动子状态或分子数量，再继续。
    """
    if not np.isfinite(horizon) or horizon < 0:
        raise ValueError("horizon must be finite and nonnegative")

    rng = np.random.default_rng(seed)
    boundaries = tuple(boundaries)

    hits = np.full((len(boundaries), paths), np.inf)

    promoter_initial, m0, n0 = initial
    if np.ndim(promoter_initial) == 0:
        promoter = np.full(paths, promoter_initial, dtype=int)
    else:
        promoter = rng.choice(states, size=paths, p=promoter_initial)

    mrna = np.full(paths, m0, dtype=int)
    protein = np.full(paths, n0, dtype=int)
    time = np.zeros(paths)
    active = np.arange(paths)

    while active.size:
        i, m, n, now = promoter[active], mrna[active], protein[active], time[active]

        propensities = np.asarray(rates(i, m, n), dtype=float)

        if (propensities.shape != (active.size, states + 4)
                or not np.all(np.isfinite(propensities) & (propensities >= 0))):
            raise ValueError("rates must return states + 4 finite nonnegative rates per path")
        
        cumulative = propensities.cumsum(axis=-1)
        total_rate = cumulative[:, -1]

        scale = np.divide(1.0, total_rate, out=np.full_like(total_rate, np.inf),
                          where=total_rate > 0)
        
        jump_time = now + rng.exponential(scale)

        within_horizon = (now < horizon) & (jump_time <= horizon)
        stop = np.where(within_horizon, jump_time, horizon)

        for row, first_hit in zip(hits, boundaries):
            pending = np.isinf(row[active])
            if np.any(pending):
                row[active[pending]] = first_hit(
                    now[pending], stop[pending], m[pending], n[pending],
                    ~within_horizon[pending],
                )

        uniform = rng.random(within_horizon.sum()) * total_rate[within_horizon]
        event = (cumulative[within_horizon] <= uniform[:, None]).sum(axis=1)
        active = active[within_horizon]
        time[active] = jump_time[within_horizon]

        switch = event < states
        promoter[active[switch]] = event[switch]
        
        mrna[active] += (event == states).astype(int) - (event == states + 2)

        protein[active] += (event == states + 1).astype(int) - (event == states + 3)

    return hits

# def interval_first_hit(windows):
#     """用闭区间并集解析检测首达，包括两次反应之间开窗。"""
#     def first_hit(start, stop, m, n, closed):
#         hit = np.full(np.shape(start), np.inf)
#         for lo, hi in windows(m, n):
#             candidate = np.maximum(start, lo)
#             valid = (candidate <= hi) & ((candidate < stop) | ((candidate == stop) & closed))
#             hit = np.minimum(hit, np.where(valid, candidate, np.inf))
#         return hit
#     return first_hit

def estimate_batched(rates, states, initial, times, *,
                     paths, seed, first_hit, batch_size=50000):
    """这个函数把 SSA 算出的每条轨迹的首达时间，汇总成生存曲线 \(H(t)\)"""
    if paths < 1 or batch_size < 1:
        raise ValueError("paths and batch_size must be positive")

    hits = []
    for batch, start in enumerate(range(0, paths, batch_size)):
        result = first_passage_times(
            rates, states, initial, [first_hit], float(times[-1]),
            min(batch_size, paths - start), seed + batch,
        )
        hits.append(result[0])

    ordered = np.sort(np.concatenate(hits))
    survival = 1 - np.searchsorted(ordered, times, side="right") / len(ordered)
    return survival, ordered

# ✅️
@dataclass(frozen=True)
class TwoStageModel:
    """通用两阶段模型，initial 为 (初始启动子或其分布, mRNA, 蛋白)。

    switching 返回非负切换率，对角为零。transcription 返回有界转录率。
    """

    name: str
    initial: tuple
    switching: Callable
    transcription: Callable
    translation: np.ndarray
    mrna_decay: np.ndarray
    protein_decay: np.ndarray

    def __post_init__(self):
            d = len(self.translation)
    
            for name in ("mrna_decay", "protein_decay"):
                if len(getattr(self, name)) != d:
                    raise ValueError(f"{name} must have {d} elements")
    
            switching = np.asarray(
                self.switching(np.arange(d), np.zeros(d), np.zeros(d))
            )
            if switching.shape != (d, d):
                raise ValueError(
                    f"switching must return ({d}, {d}), got {switching.shape}"
                )
            # print("hahahahahaha")

    @property
    def states(self):
        return len(self.translation)

    def rates(self, i, m, n):
        i = np.asarray(i, dtype=int)
        m, n = np.asarray(m, dtype=float), np.asarray(n, dtype=float)
        counts = np.stack((self.transcription(i, m, n), self.translation[i] * m,
                           self.mrna_decay[i] * m, self.protein_decay[i] * n), axis=-1)
        return np.concatenate((self.switching(i, m, n), counts), axis=-1)


# @dataclass(frozen=True)
# class FirstEvent:
#     """同一个首达事件的 g 与解析检测器，调用者保证两者一致。"""

#     g: Callable
#     first_hit: Callable

#     @classmethod
#     def from_windows(cls, windows):
#         """用闭区间并集定义 g 的符号和硬首达，空区间用 (+inf,-inf)。"""
#         def g(t, m, n):
#             scores = [np.minimum(t - lo, hi - t) for lo, hi in windows(m, n)]
#             value = np.maximum.reduce(scores)
#             return np.where(np.isfinite(value), value, -1.)

#         return cls(g, interval_first_hit(windows))

HIT_SUBDIVISIONS = 32
HIT_BISECTION_STEPS = 24

# ✅️
@dataclass(frozen=True)
class FirstEvent:
    g: Callable

    def first_hit(self, start, stop, m, n, closed):
        """搜索区间内第一次满足 g >= 0 的时刻，未找到则返回 inf。"""
        hits = np.full(start.shape, np.inf)
        end = np.where(closed, stop, np.nextafter(stop, start))
        valid = (start < stop) | (closed & (start == stop))

        # 先检查当前时刻，可能已经满足边界。
        hits[valid & (self.g(start, m, n) >= 0)] = start[
            valid & (self.g(start, m, n) >= 0)
        ]

        # 从左向右检查，发现 g 从负变成非负时，用二分细化。
        left = start.copy()
        for step in range(1, HIT_SUBDIVISIONS + 1):
            pending = valid & np.isinf(hits) & (left < end)
            if not np.any(pending):
                break

            idx = np.flatnonzero(pending)
            right = end[idx] if step == HIT_SUBDIVISIONS else (
                start[idx] + (end[idx] - start[idx]) * (step / HIT_SUBDIVISIONS)
            )

            crossed = np.broadcast_to(
                self.g(right, m[idx], n[idx]) >= 0, right.shape
            )
            found = idx[crossed]
            if found.size:
                lo, hi = left[found], right[crossed]
                for _ in range(HIT_BISECTION_STEPS):
                    mid = lo + (hi - lo) / 2
                    reached = self.g(mid, m[found], n[found]) >= 0
                    lo = np.where(reached, lo, mid)
                    hi = np.where(reached, mid, hi)
                hits[found] = hi

            left[idx] = right

        return hits

# ✅️
def exponential_boundary(score, floor, amplitude, decay):
    """Return g(t,m,n) and its analytically matching hard-hit time windows."""

    def boundary(t, m, n):
        return (score(m, n) - (floor + amplitude * np.exp(-t / decay))) / 4

    # def windows(m, n):
    #     level = score(m, n)
    #     start = np.full(level.shape, np.inf)
    #     start[level >= floor + amplitude] = 0.
    #     mask = (level > floor) & (level < floor + amplitude)
    #     start[mask] = decay * np.log(amplitude / (level[mask] - floor))
    #     return [(start, np.where(level > floor, np.inf, -np.inf))]

    return FirstEvent(boundary)


def affine_time_boundary(routes, stop_time=8.):
    """每条路线是若干 a+b*t>=0 的交集，多条路线取并集。

    每个计数组合的 a,b 是常数，因此 SSA 可直接求开窗和关窗时刻。
    在 stop_time 后，判据固定在 stop_time 的取值。
    """
    def coefficients(m, n):
        m, n = np.broadcast_arrays(m, n)
        for route in routes(m, n):
            constraints = []
            for intercept, slope in route:
                a, b, _ = np.broadcast_arrays(intercept, slope, m)
                root = np.divide(-a, b, out=np.zeros(m.shape), where=b != 0)
                constraints.append((a, b, root))
            yield constraints

    def g(t, m, n):
        time = np.minimum(t, stop_time)
        scores = []
        for route in coefficients(m, n):
            values = [np.where(b == 0, a, b*(time-root)) for a, b, root in route]
            scores.append(np.minimum.reduce(values))
        return np.maximum.reduce(scores) / 4

    # def windows(m, n):
    #     bands = []
    #     for route in coefficients(m, n):
    #         start = np.maximum.reduce([np.where(b > 0, root, 0.) for _, b, root in route])
    #         end = np.minimum.reduce([np.where(b < 0, root, np.inf) for _, b, root in route])
    #         valid = (start <= end) & (start <= stop_time)
    #         for a, b, _ in route:
    #             valid &= (b != 0) | (a >= 0)
    #         bands.append((np.where(valid, start, np.inf),
    #                       np.where(valid, np.where(end >= stop_time, np.inf, end), -np.inf)))
    #     return bands

    return FirstEvent(g)

# ✅️
def example_model():
    """三状态双反馈例子。两个求解器的独立演示共用此模型。"""
    def switching(i, m, n):
        activation = (.45 + .55 * n / (3. + n)) / (1. + .3 * m)
        repression = .35 + .4 * m / (2. + m)
        targets = np.eye(3)
        return (targets[(i + 1) % 3] * activation[..., None]
                + targets[(i - 1) % 3] * repression[..., None])

    def transcription(i, m, n):
        base = np.array([.2, .9, 1.7])
        return base[i] * (1. + .35 * n / (3. + n)) / (1. + .12 * m)

    model = TwoStageModel("three_state_feedback", ([.6, .3, .1], 1, 1),
                          switching, transcription, np.array([.5, .75, 1.]),
                          np.array([.7, .8, .9]), np.array([.4, .45, .5]))
    def product_score(m, n):
        return (m + 1) * (n + 1)

    return model, exponential_boundary(product_score, 6., 5., 3.)

# ✅️
def plot_demo(times, survival, filename, *, title, density=None):
    """两个独立示例共用的双栏图。"""
    import matplotlib.pyplot as plt

    fig, (ax_h, ax_fpt) = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    if density is None:
        ax_h.step(times, survival, where="post")
        ax_fpt.stairs(-np.diff(survival) / np.diff(times), times)
    else:
        ax_h.plot(times, survival)
        ax_fpt.plot(times, density)
    ax_h.set(xlabel="Time", ylabel="H(t)", title="Survival")
    ax_fpt.set(xlabel="Time", ylabel="Density", title="First-passage time")
    fig.suptitle(title)
    output = Path(__file__).with_name(filename + ".png")
    fig.savefig(output, dpi=160)
    print(output)
    plt.show()

# ✅️
def demo():
    model, boundary = example_model()
    times = np.linspace(0., 5., 41)
    survival, _ = estimate_batched(
        model.rates, model.states, model.initial,  times,
        paths=12000, seed=2026, first_hit=boundary.first_hit,
    )
    plot_demo(times, survival, "ssa_demo", title="Hard-boundary SSA | 12,000 paths")


# def __getattr__(name):
#     # 旧调用者可从这里访问批跑预设，SSA 独立演示无需加载批跑。
#     if name in ("WINDOW_BOUNDARIES", "SMOOTH_BOUNDARIES", "window_boundary", "smooth_boundary_pair"):
#         import run_models
#         return getattr(run_models, name)
#     raise AttributeError(name)


if __name__ == "__main__":
    demo()
