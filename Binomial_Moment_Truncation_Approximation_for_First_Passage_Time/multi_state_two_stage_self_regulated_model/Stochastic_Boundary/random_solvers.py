"""随机阈值的路径模拟与二项反演坐标下的 Galerkin 方程。"""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.sparse import block_diag, diags, eye, kron
from binomial_moments import build_operator


def sample_hits(model, boundary, horizon, paths, dt, seed):
    """分子反应使用 SSA；随机阈值按时间步演化并检查首达。"""
    if horizon <= 0 or paths < 1 or dt <= 0:
        raise ValueError("horizon, paths and dt must be positive")
    rng = np.random.default_rng(seed)
    promoter, m0, n0 = model.initial
    state = (np.full(paths, promoter, dtype=int) if np.ndim(promoter) == 0
             else rng.choice(model.states, size=paths, p=promoter))
    m, n = np.full(paths, m0, dtype=int), np.full(paths, n0, dtype=int)
    x = boundary.initial(rng, paths)
    time, hits = np.zeros(paths), np.full(paths, np.inf)
    hits[boundary.g(m, n, x) >= 0] = 0
    active = np.flatnonzero(np.isinf(hits))
    while active.size:
        rates = np.asarray(model.rates(state[active], m[active], n[active]), dtype=float)
        if rates.shape != (active.size, model.states + 4) or not np.all(np.isfinite(rates) & (rates >= 0)):
            raise ValueError("Invalid molecular rates")
        cumulative = rates.cumsum(axis=1)
        total = cumulative[:, -1]
        scale = np.divide(1., total, out=np.full_like(total, np.inf), where=total > 0)
        waiting = rng.exponential(scale)
        elapsed = np.minimum(np.minimum(waiting, dt), horizon - time[active])
        x[active] = boundary.advance(x[active], elapsed, rng)
        time[active] += elapsed
        reached = boundary.g(m[active], n[active], x[active]) >= 0
        hits[active[reached]] = time[active[reached]]
        reacting = (waiting <= elapsed) & ~reached
        indices = active[reacting]
        uniform = rng.random(indices.size) * total[reacting]
        event = (cumulative[reacting] <= uniform[:, None]).sum(axis=1)
        switch = event < model.states
        state[indices[switch]] = event[switch]
        m[indices] += (event == model.states).astype(int) - (event == model.states + 2)
        n[indices] += (event == model.states + 1).astype(int) - (event == model.states + 3)
        reached = boundary.g(m[indices], n[indices], x[indices]) >= 0
        hits[indices[reached]] = time[indices[reached]]
        active = active[np.isinf(hits[active]) & (time[active] < horizon)]
    return np.sort(hits)


def solve_moments(model, boundary, times, gamma, L, K, N, quadrature=96):
    """R.3 + R.17 的有限基展开；分子顶部修正复用原 build_operator。"""
    if gamma <= 0 or N < 1 or quadrature < N:
        raise ValueError("Require gamma > 0 and quadrature >= N >= 1")
    Q, m, n, initial = build_operator(model.rates, model.states, model.initial, L, K)
    x, weights, phi, eigenvalues = boundary.basis(N, quadrature)
    g = boundary.g(m[:, None], n[:, None], x[None, :])
    potential = gamma * np.exp(gamma * g / (1 + np.abs(g)) + (m[:, None] + n[:, None]) / gamma)
    killing = np.einsum('qa,cq,qb,q->cab', phi, potential, phi, weights)
    operator = (kron(Q, eye(N)) - kron(eye(len(m)), diags(eigenvalues))
                - block_diag(killing)).tocsc()
    z0 = np.zeros((len(m), N))
    z0[:, 0] = initial
    solution = solve_ivp(lambda t, z: operator @ z, (times[0], times[-1]),
                         z0.ravel(), t_eval=times, method='BDF', jac=operator,
                         rtol=2e-8, atol=1e-11)
    if not solution.success:
        raise RuntimeError(solution.message)
    z = solution.y.reshape(len(m), N, len(times))
    survival = z[:, 0].sum(axis=0)
    density = np.einsum('cb,cbt->t', killing[:, 0, :], z)
    valid = bool(np.isfinite(survival).all() and np.isfinite(density).all()
                 and survival.min() >= -1e-5 and survival.max() <= 1 + 1e-5
                 and np.diff(survival).max() <= 1e-5 and density.min() >= -1e-5)
    return survival, density, valid
