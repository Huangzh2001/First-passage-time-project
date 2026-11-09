"""Gillespie 直接法。这里负责生成原过程轨道和记录硬首达时刻。"""

import numpy as np


def _boundary_lookup(schedule):
    """为每个计数对预计算下一次边界触及它的时刻。"""
    cuts, at_cut, between_cuts, combine = schedule
    thresholds = np.vstack((at_cut, between_cuts))
    finite = np.where(np.isfinite(thresholds), thresholds, 0)
    m_max, n_max = np.maximum(0, np.ceil(finite.max(axis=0))).astype(int)
    m, n = np.ogrid[:m_max+1, :n_max+1]

    def unsafe(levels):
        a, b = levels.T
        return combine(m >= a[:, None, None], n >= b[:, None, None])

    at_cut = unsafe(at_cut)
    between = unsafe(between_cuts)
    next_hit = np.full(at_cut.shape, np.inf)
    next_hit[-1] = np.where(at_cut[-1], cuts[-1], np.inf)
    for k in range(len(cuts)-2, -1, -1):
        next_hit[k] = np.where(at_cut[k] | between[k], cuts[k], next_hit[k+1])
    return cuts, at_cut, between, next_hit


def first_passage_times(rates, states, initial, schedules, horizon, paths, seed):
    """返回形状为 (边界数, 轨道数) 的首达时间，未首达记为 inf。

    rates(i, m, n) 的最后一维为各目标启动子态的切换率，随后依次是
    转录、翻译、mRNA 降解、蛋白质降解。它接受同形数组作为输入。
    schedule 为 (时间断点, 断点阈值, 开区间阈值, OR/AND 合并函数)。
    """
    rng = np.random.default_rng(seed)
    lookups = [_boundary_lookup(schedule) for schedule in schedules]
    hits = np.full((len(schedules), paths), np.inf)
    promoter, mrna, protein = (np.full(paths, value, dtype=int) for value in initial)
    time = np.zeros(paths)
    active = np.arange(paths)

    while active.size:
        i, m, n, now = promoter[active], mrna[active], protein[active], time[active]
        cumulative = rates(i, m, n).cumsum(axis=-1)
        total_rate = cumulative[:, -1]
        scale = np.divide(1.0, total_rate, out=np.full_like(total_rate, np.inf),
                          where=total_rate > 0)
        jump_time = now + rng.exponential(scale)

        # 两次反应之间分子数不变，但时变边界仍可能扫过当前计数。
        for b, (cuts, at_cut, between, next_hit) in enumerate(lookups):
            segment = np.searchsorted(cuts, now, side="right") - 1
            x = m.clip(max=next_hit.shape[1]-1)
            y = n.clip(max=next_hit.shape[2]-1)
            unsafe_now = between[segment, x, y]
            unsafe_now |= (now == cuts[segment]) & at_cut[segment, x, y]
            hit_time = np.where(unsafe_now, now, next_hit[segment+1, x, y])
            new_hit = (hit_time <= horizon) & (hit_time < jump_time)
            new_hit &= np.isinf(hits[b, active])
            hits[b, active[new_hit]] = hit_time[new_hit]

        within_horizon = jump_time < horizon
        uniform = rng.random(within_horizon.sum()) * total_rate[within_horizon]
        event = (cumulative[within_horizon] <= uniform[:, None]).sum(axis=1)
        active = active[within_horizon]
        time[active] = jump_time[within_horizon]
        switch = event < states
        promoter[active[switch]] = event[switch]
        mrna[active] += (event == states).astype(int) - (event == states+2)
        protein[active] += (event == states+1).astype(int) - (event == states+3)

    return hits
