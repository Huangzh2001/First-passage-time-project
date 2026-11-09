"""式 (8.1) 的有限二项矩系统，在二项反演坐标下求解。"""

from itertools import product
from math import comb

import numpy as np
from scipy.integrate import solve_ivp
from scipy.sparse import diags, lil_matrix


def build_operator(rates, states, initial, L, K):
    """构造不含软杀死的反应矩阵，L 和 K 分别是 mRNA 和蛋白质截断阶数。"""
    grid = list(product(range(states), range(L+1), range(K+1)))
    index = {state: j for j, state in enumerate(grid)}
    Q = lil_matrix((len(grid), len(grid)))
    p0 = np.zeros(len(grid))
    p0[index[initial]] = 1.0

    # 一个越界出生对所有保留矩的贡献，反演成盒内的带符号权重。
    m_weights = [(-1)**(L-a) * comb(L+1, a) for a in range(L+1)]
    n_weights = [(-1)**(K-b) * comb(K+1, b) for b in range(K+1)]
    for col, (i, m, n) in enumerate(grid):
        targets = [(j, m, n) for j in range(states)]
        targets += [(i, m+1, n), (i, m, n+1), (i, m-1, n), (i, m, n-1)]
        for target, rate in zip(targets, rates(i, m, n)):
            if rate == 0:
                continue
            Q[col, col] -= rate
            if target in index:
                Q[index[target], col] += rate
            elif target[1] > L:
                for a, weight in enumerate(m_weights):
                    Q[index[i, a, n], col] += rate * weight
            else:
                for b, weight in enumerate(n_weights):
                    Q[index[i, m, b], col] += rate * weight

    _, mrna, protein = np.asarray(grid).T
    return Q.tocsc(), mrna, protein, p0


def survival(operator, boundary, times, gamma, jumps=(), rate_cap=1e6, *, return_density=False):
    """解软杀死系统并返回 H(t)。boundary(t, m, n) >= 0 表示越界。

    times 从 0 开始且严格递增。jumps 给出边界不连续的时刻。
    rate_cap 是控制刚性的数值上限，设为 np.inf 可关闭。
    return_density=True 时返回 (H(t), -H'(t))，跳点处取右侧杀死率。
    """
    Q, mrna, protein, p = operator
    times = np.asarray(times)
    h = np.empty(len(times))
    density = np.empty(len(times)) if return_density else None
    log_prefactor = np.log(gamma) + (mrna+protein)/gamma
    log_cap = np.log(rate_cap)

    def rate_at(t):
        g = boundary(t, mrna, protein)
        return np.exp(np.minimum(log_prefactor + gamma*g/(1+np.abs(g)), log_cap))

    end = times[-1]
    cuts = (0.0, *sorted({t for t in jumps if 0 < t < end}), end)
    for start, stop in zip(cuts[:-1], cuts[1:]):
        # 分段积分。区间末端取左极限，下一段从新边界开始。
        def killing_rate(t):
            return rate_at(min(t, np.nextafter(stop, start)))

        def rhs(t, y):
            return Q @ y - killing_rate(t)*y

        def jacobian(t, _):
            return Q - diags(killing_rate(t), format="csc")

        selected = (times >= start) & ((times <= stop) if stop == end else (times < stop))
        output_times = np.unique(np.r_[times[selected], stop])
        solution = solve_ivp(rhs, (start, stop), p, t_eval=output_times,
                             jac=jacobian, method="BDF", rtol=2e-9, atol=2e-12, max_step=.1)
        if not solution.success:
            raise RuntimeError(solution.message)
        # 反演坐标求和就是零阶二项矩，也就是软生存概率。
        weights = solution.y[:, :selected.sum()]
        h[selected] = weights.sum(axis=0)
        if return_density:
            # Q 的列和为零，所以 -H'(t) 是软杀死率与当前权重的内积。
            density[selected] = [rate_at(t) @ weight
                                 for t, weight in zip(times[selected], weights.T)]
        p = solution.y[:, -1]
    return (h, density) if return_density else h
