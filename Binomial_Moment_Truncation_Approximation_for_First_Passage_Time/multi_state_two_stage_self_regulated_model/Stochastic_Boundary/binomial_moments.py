"""联合二项矩零闭合，在带符号二项反演坐标下求解。"""

from itertools import product
from math import comb, fsum
from time import perf_counter

import numpy as np
from scipy.integrate import BDF
from scipy.sparse import bmat, csc_matrix, diags, lil_matrix
from scipy.sparse.linalg import splu

from ssa import example_model, plot_demo



def build_operator(rates, states, initial, L, K):
    """构造不含软杀死的反应矩阵，L 和 K 分别是 mRNA 和蛋白质截断阶数。"""

    # 1. 给每个坐标编号
    grid = list(product(range(states), range(L+1), range(K+1)))
    index = {state: j for j, state in enumerate(grid)}
    Q = lil_matrix((len(grid), len(grid)))

    # 2. 放入初始状态
    p0 = np.zeros(len(grid))
    promoter, m0, n0 = initial
    if m0 < 0 or n0 < 0 or m0 > L or n0 > K:
        raise ValueError('Initial counts must lie within retained orders')
    if np.ndim(promoter) == 0:
        p0[index[int(promoter), m0, n0]] = 1.0
    else:
        probabilities = np.asarray(promoter, dtype=float)
        if (probabilities.shape != (states,) or not np.isfinite(probabilities).all()
                or np.any(probabilities < 0) or not np.isclose(probabilities.sum(), 1.)):
            raise ValueError('Invalid initial promoter probability vector')
        for state, probability in enumerate(probabilities):
            p0[index[state, m0, n0]] = probability

    # 3. 预先算好顶部修正权重
    # 式 (10.8) 的顶部修正，转录和翻译使用同一个公式。
    def signed_weights(order):
        return [(-1)**(order-j) * comb(order+1, j) for j in range(order+1)]

    m_weights, n_weights = signed_weights(L), signed_weights(K)

    # 4. 把每个坐标的反应速率算出来
    promoters, mrna, protein = np.asarray(grid).T
    propensities = np.asarray(rates(promoters, mrna, protein), dtype=float)
    if (propensities.shape != (len(grid), states + 4)
            or not np.all(np.isfinite(propensities) & (propensities >= 0))
            or np.any(propensities[mrna == 0, states + 2])
            or np.any(propensities[protein == 0, states + 3])):
        raise ValueError("Invalid reaction rates on the moment grid")

    # 5. 逐列把反应写进矩阵
    for col, (i, m, n) in enumerate(grid):
        targets = [(j, m, n) for j in range(states)]
        targets += [(i, m+1, n), (i, m, n+1), (i, m-1, n), (i, m, n-1)]
        for target, rate in zip(targets, propensities[col]):
            if rate == 0:
                continue
            Q[col, col] -= rate
            if target in index:
                Q[index[target], col] += rate
            else:
                mrna_birth = target[1] > L
                weights = m_weights if mrna_birth else n_weights
                for count, weight in enumerate(weights):
                    corrected = (i, count, n) if mrna_birth else (i, m, count)
                    Q[index[corrected], col] += rate * weight

    return Q.tocsc(), mrna, protein, p0


def soft_rate(t, m, n, gamma, boundary):
    """式 (5.1) 的软杀死率，保留原 g 的尺度。"""
    if not np.isfinite(gamma) or gamma <= 0:
        raise ValueError('gamma must be finite and positive')
    g = np.asarray(boundary(t, m, n), dtype=float)
    if not np.isfinite(g).all():
        raise ValueError('g must be finite')
    exponent = np.log(gamma) + gamma*(g/(1+np.abs(g))) + (m+n)/gamma
    if np.max(exponent) > 700:
        raise FloatingPointError('Killing rate exceeds safe float64 range')
    return np.exp(exponent)



# def solve_with_survival(operator, L, K, boundary, times, gamma, *,
#                         scale=1.15, basis="exp", rtol=2e-9, atol=1e-12,
#                         max_step=.035, timeout=180.):
#     """Integrate the original signed moment equations together with H'."""
#     # 1. 第 1 步：读取前面构造好的结果
#     Q, m, n, initial = operator

#     # 2. 第 2 步：引入 weights 缩放
#     weights = scale ** (m + n)
#     if basis == "binomial":
#         weights /= np.array([comb(L + 1, j) for j in range(L + 1)], float)[m]
#         weights /= np.array([comb(K + 1, j) for j in range(K + 1)], float)[n]
#     if not np.isfinite(weights).all() or np.any(weights <= 0):
#         raise ValueError("Scaling overflow/underflow; adjust --scale or moment orders.")
#     entries = Q.tocoo()
#     entries.data *= weights[entries.row] / weights[entries.col]
#     A = entries.tocsc()

#     # 3. 第 3 步：定义软杀死率
#     def killing(t):
#         return soft_rate(t, m, n, gamma, boundary)

#     # 4. 第 4 步：定义真正要求解的微分方程 rhs
#     def rhs(t, state):
#         rates = killing(t)
#         z = state[:-1]
#         return np.r_[A @ z - rates * z, -fsum(rates * (z / weights))]

#     # 5. 第 5 步：定义 Jacobian
#     def jac(t, _state):
#         rates = killing(t)
#         return bmat([[A - diags(rates), None],
#                      [csc_matrix((-rates / weights)[None, :]), csc_matrix((1, 1))]],
#                     format="csc")

#     # 6. 第 6 步：设置初始条件
#     h, density, h_sum = np.full((3, len(times)), np.nan)
#     h[0] = h_sum[0] = fsum(initial)
#     cursor, last = 1, 0.
#     status, message = "finished", ""
#     started = perf_counter()
#     try:
#         density[0] = fsum(killing(0.) * initial)
#         component_atol = atol * weights if basis == "binomial" else np.full(len(weights), atol)

#         # 7. 第 7 步：使用 BDF 求解器
#         solver = BDF(rhs, 0., np.r_[weights * initial, h[0]], times[-1], jac=jac,
#                      rtol=rtol, atol=np.r_[component_atol, rtol * 1e-3], max_step=max_step)

#         def factor(B):
#             # The last variable is H. Its row does not affect the original block.
#             solver.nlu += 1
#             return (splu(B[:-1, :-1].tocsc(), permc_spec="MMD_AT_PLUS_A"),
#                     B[-1, :-1].toarray().ravel(), B[-1, -1])

#         def solve_factor(factorization, b):
#             lu, row, diagonal = factorization
#             z = lu.solve(b[:-1])
#             return np.r_[z, (b[-1] - fsum(row * z)) / diagonal]

#         solver.lu, solver.solve_lu = factor, solve_factor
#         while solver.status == "running":
#             message = solver.step() or ""
#             last = float(solver.t)
#             if solver.status == "failed":
#                 status = "solver_failed"
#                 break
#             stop = np.searchsorted(times, solver.t, side="right")
#             if stop > cursor:
#                 section = slice(cursor, stop)
#                 sample = solver.dense_output()(times[section])
#                 h[section] = sample[-1]
#                 for j in range(cursor, stop):
#                     values = sample[:-1, j-cursor] / weights
#                     h_sum[j] = fsum(values)
#                     density[j] = fsum(killing(times[j]) * values)
#                 # Earlier samples have already passed these checks.
#                 if not np.isfinite(h[section]).all() or not np.isfinite(density[section]).all():
#                     status = "nonfinite"
#                     break
#                 if np.max(np.abs(h[section])) >= 1e4:
#                     status = "diverged"
#                     break
#                 cursor = stop
#             if solver.status == "running" and perf_counter() - started >= timeout:
#                 status = "timeout"
#                 break
#     except (ValueError, FloatingPointError, OverflowError, RuntimeError) as exc:
#         status, message = "solver_failed", str(exc)

#     first_anomaly, first_by_check, finite_samples = curve_diagnostics(times, h, density)
#     return dict(h=h, density=density, h_from_sum=h_sum, status=status, message=message,
#                 last_time=last, first_anomaly=first_anomaly, first_by_check=first_by_check,
#                 finite_samples=finite_samples, seconds=perf_counter() - started)



def solve_with_survival(operator, L, K, boundary, times, gamma, *,
                        scale=1.15, basis="exp", rtol=2e-9, atol=1e-12,
                        max_step=.035, timeout=180.):
    """直接积分未缩放的二项反演方程及生存概率 H(t)。

    保留 scale 和 basis 参数以兼容旧调用；
    此简化版不再使用它们。
    """

    # 1. 读取反应矩阵、坐标和初始条件
    Q, m, n, initial = operator

    # 2. 计算每个坐标对应的软杀死率 V_gamma
    def killing(t):
        return soft_rate(t, m, n, gamma, boundary)

    # 3. 按理论公式直接定义微分方程
    def rhs(t, state):
        x = state[:-1]       # 二项反演坐标向量
        v = killing(t)       # 各坐标的软杀死率

        # dx/dt = Qx - V_gamma*x
        dx = Q @ x - v * x

        # H'(t) = -sum(V_gamma*x)
        dH = -fsum(v * x)

        return np.r_[dx, dH]

    # 4. 初始化需要输出的结果
    h, density, h_sum = np.full((3, len(times)), np.nan)
    # 生存概率的初始值：由理论定义直接得到
    h[0] = 1.0

    # 二项反演坐标初始值的总和：单独计算
    h_sum[0] = fsum(initial)

    cursor, last = 1, 0.
    status, message = "finished", ""
    started = perf_counter()

    try:
        density[0] = fsum(killing(0.) * initial)

        # 5. 直接交给 BDF 求解，不再手动设置 Jacobian 和 LU
        solver = BDF(
            rhs,
            0.,
            np.r_[initial, h[0]],
            times[-1],
            rtol=rtol,
            atol=atol,
            max_step=max_step
        )

        # 6. 逐步进行时间积分
        while solver.status == "running":
            message = solver.step() or ""
            last = float(solver.t)

            if solver.status == "failed":
                status = "solver_failed"
                break

            # 7. 提取指定时间点的计算结果
            stop = np.searchsorted(times, solver.t, side="right")

            if stop > cursor:
                section = slice(cursor, stop)
                sample = solver.dense_output()(times[section])

                # 直接积分得到的生存概率
                h[section] = sample[-1]

                for j in range(cursor, stop):
                    x = sample[:-1, j - cursor]

                    # 通过全部二项反演坐标求和得到 H(t)
                    h_sum[j] = fsum(x)

                    # 首达时间概率密度
                    density[j] = fsum(killing(times[j]) * x)

                # 8. 检查计算是否异常
                if (not np.isfinite(h[section]).all()
                        or not np.isfinite(density[section]).all()):
                    status = "nonfinite"
                    break

                if np.max(np.abs(h[section])) >= 1e4:
                    status = "diverged"
                    break

                cursor = stop

            if (solver.status == "running"
                    and perf_counter() - started >= timeout):
                status = "timeout"
                break

    except (ValueError, FloatingPointError, OverflowError, RuntimeError) as exc:
        status, message = "solver_failed", str(exc)

    # 9. 进行原有的诊断并保持返回格式不变
    first_anomaly, first_by_check, finite_samples = curve_diagnostics(
        times, h, density
    )

    return dict(
        h=h,
        density=density,
        h_from_sum=h_sum,
        status=status,
        message=message,
        last_time=last,
        first_anomaly=first_anomaly,
        first_by_check=first_by_check,
        finite_samples=finite_samples,
        seconds=perf_counter() - started
    )


def curve_diagnostics(times, h, density):
    physical_issues = {
        "survival_out_of_range": (h < -1e-5) | (h > 1 + 1e-5),
        "survival_increased": np.r_[False, np.diff(h) > 1e-5],
        "negative_density": density < -1e-4,
    }
    first_by_check = {}
    for name, mask in physical_issues.items():
        if mask.any():
            first_by_check[name] = float(times[np.flatnonzero(mask)[0]])
    first_anomaly = min(first_by_check.values(), default=np.nan)
    finite = np.isfinite(h) & np.isfinite(density)
    if not finite.all():
        first_by_check["nonfinite_or_missing_sample"] = float(times[np.flatnonzero(~finite)[0]])
    return first_anomaly, first_by_check, int(np.count_nonzero(finite))


def is_valid_curve(run):
    return (run["status"] == "finished" and not np.isfinite(run["first_anomaly"])
            and np.isfinite(run["h"]).all())


def adaptive_solve(model, boundary, times, gamma, *,
                   start=(8, 8), step=(2, 2), cap=(30, 30),
                   order_tol=2e-4, mass_tol=1e-5, on_attempt=None, **solver_options):
    """固定 gamma，连续两次加阶达标后返回曲线、状态和逐阶诊断。"""
    # 1. 检查输入参数是否合法
    if (any(not isinstance(value, (int, np.integer))
            for pair in (start, step, cap) for value in pair)
            or min(start) < 0 or min(step) <= 0
            or any(lo > hi for lo, hi in zip(start, cap))
            or not np.isfinite(order_tol) or order_tol <= 0
            or not np.isfinite(mass_tol) or mass_tol <= 0):
        raise ValueError("Invalid adaptive order settings")
    
    times = np.asarray(times, dtype=float)
    if (times.ndim != 1 or times.size < 2 or times[0] != 0
            or not np.isfinite(times).all() or np.any(np.diff(times) <= 0)):
        raise ValueError("Time grid must increase strictly from zero")
    
    if not np.isfinite(gamma) or gamma <= 0:
        raise ValueError("gamma must be finite and positive")
    
    _, m0, n0 = model.initial
    g0 = float(np.asarray(boundary.g(0., np.array([m0]), np.array([n0]))).item())
    if not np.isfinite(g0) or g0 >= 0:
        raise ValueError("The initial state must satisfy finite g(0,m0,n0) < 0")
    
    L, K = max(start[0], m0), max(start[1], n0)
    if L > cap[0] or K > cap[1]:
        raise ValueError("Moment caps must include initial counts")

    # 2. 初始化自适应计算的记录变量
    previous = None
    consecutive = 0
    attempts = []

    # 3. 进入自适应计算循环
    while True:
        ## （1）
        operator = build_operator(model.rates, model.states, model.initial, L, K)

        ## （2）
        run = solve_with_survival(operator, L, K, boundary.g, times, gamma, **solver_options)

        ## （3）检查生存概率的一致性
        mass_gap = None

        if np.isfinite(run["h"]).all() and np.isfinite(run["h_from_sum"]).all():
            mass_gap = float(np.max(abs(run["h"] - run["h_from_sum"])))

        valid = bool(is_valid_curve(run) and mass_gap is not None and mass_gap <= mass_tol)

        ## （5）检查不同截断阶数之间的收敛性
        change = None
        if valid and previous is not None:
            change = float(np.max(abs(run["h"] - previous)))
        consecutive = consecutive + 1 if change is not None and change <= order_tol else 0

        ## （6）记录每次尝试的诊断信息
        attempt = dict(L=L, K=K, valid=valid, solver_status=run["status"],
                       order_change=change, stable_changes=consecutive, mass_gap=mass_gap,
                       mass_check_failed=mass_gap is None or mass_gap > mass_tol)
        for key in ("first_by_check", "finite_samples", "last_time", "seconds", "message"):
            attempt[key] = run[key]
        first_anomaly = run["first_anomaly"]
        attempt["first_anomaly"] = first_anomaly if np.isfinite(first_anomaly) else None
        attempts.append(attempt)

        ## （7）给当前求解结果添加截断阶数
        run.update(L=L, K=K)

        ## （8）判断是否应该停止
        selection = None
        if consecutive >= 2:
            selection = "converged"
        elif (L, K) == tuple(cap):
            selection = "cap_valid" if valid else "cap_invalid"

        ## （9）打印
        if on_attempt is not None:
            on_attempt(attempt, run, selection)
        if selection is not None:
            return (run if valid else None), selection, attempts

        ## （10）如果还没有停止，就继续增加阶数
        previous = run["h"] if valid else None
        L, K = min(cap[0], L + step[0]), min(cap[1], K + step[1])


def demo():
    model, boundary = example_model()
    times = np.linspace(0., 5., 101)
    L = K = 10
    gamma = 12.
    operator = build_operator(model.rates, model.states, model.initial, L, K)
    run = solve_with_survival(operator, L, K, boundary.g, times, gamma,
                              basis="binomial", atol=1e-11)
    if not is_valid_curve(run):
        raise RuntimeError(f"Binomial moments demo failed: {run['status']}, {run['message']}")
    plot_demo(times, run["h"], "binomial_moments_demo", density=run["density"],
              title=f"Binomial moments | gamma={gamma:g}, L={L}, K={K}")


if __name__ == "__main__":
    demo()
