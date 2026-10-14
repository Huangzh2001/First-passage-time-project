from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid

# ================= 模型输入区 =================
# 仅本区不同；下方核心代码与原 fpt_single.py 完全相同。
THETA = 1.0
SIGMA = 0.8
X0 = 0.5
BETA = 1.0

MU = 1.0            # OU恢复位置，不是常漂移速度
ABSORB_ZERO = False


def drift(x):
    return THETA*(MU-x)


def diffusion(x):
    return SIGMA


def density(t):
    # 由LAPLACE反演生成时间域E1，不用在这里重复反演。
    return ...


def laplace_density(s):
    # 修正后的固定上边界OU首达变换。
    import mpmath as mp
    theta=mp.mpf(str(THETA)); sigma=mp.mpf(str(SIGMA))
    mu=mp.mpf(str(MU))
    za=mp.sqrt(2*theta)/sigma*(mp.mpf(str(X0))-mu)
    zb=mp.sqrt(2*theta)/sigma*(mp.mpf(str(BETA))-mu)
    return (mp.exp((za**2-zb**2)/4)
            * mp.pcfd(-s/theta,-za)/mp.pcfd(-s/theta,-zb))

# def laplace_density(s):
#     import mpmath as mp

#     theta = mp.mpf(str(THETA))
#     sigma = mp.mpf(str(SIGMA))
#     mu = mp.mpf(str(MU))
#     x0 = mp.mpf(str(X0))
#     beta = mp.mpf(str(BETA))

#     # 计算从起点 y 出发时，对应的积分。
#     def integral(y):
#         def integrand(q):
#             return (
#                 q**(s/theta - 1)
#                 / mp.sqrt(1 - q**2)
#                 * mp.exp(
#                     -theta * (beta - mu - (y - mu)*q)**2
#                     / (sigma**2 * (1 - q**2))
#                 )
#             )

#         return mp.quad(integrand, [0, 0.5, 0.9, 0.99, 1])

#     return integral(x0) / integral(beta)


N = 50_000
DT = 0.001
T_MAX = 10.0
SEED = 20261010
SURVIVAL = None
# 本例理论以精确拉普拉斯变换表示，由已有核心反演得到E1。
LAPLACE = laplace_density
INVERSE_MIN_TIME = 0.008
INVERSE_GRID = 120
INVERSE_DEGREE = 32
# ====================================================


def evaluate(func, arg, name):
    result = func(arg)
    if result is Ellipsis:
        raise ValueError(f'请先填写输入区中的 {name}。')
    result = np.broadcast_to(np.asarray(result, dtype=float), arg.shape).copy()
    if not np.all(np.isfinite(result)):
        raise ValueError(f'{name} 返回了无穷大或无效值，请检查公式和模拟步长。')
    return result


def prepare_theory(times):
    """由用户输入的E1，或由用户输入的F(s)反演，生成理论密度。"""
    if LAPLACE is None:
        rho = np.zeros_like(times)
        rho[1:] = evaluate(density, times[1:], 'density / E1')
        return rho, None
    try:
        import mpmath as mp
    except ImportError as exc:
        raise ValueError('反演需要 mpmath，请运行 pip install mpmath。') from exc
    if not callable(LAPLACE):
        raise ValueError('LAPLACE必须是计算F(s)的函数，或设置为None。')
    if not 0 < INVERSE_MIN_TIME < T_MAX or INVERSE_GRID < 20:
        raise ValueError('要求0<INVERSE_MIN_TIME<T_MAX，INVERSE_GRID>=20。')
    mp.mp.dps = 50
    pivot = min(max(1., 2*INVERSE_MIN_TIME), T_MAX)
    grid = np.unique(np.r_[np.geomspace(INVERSE_MIN_TIME, pivot, INVERSE_GRID),
                           np.linspace(pivot, T_MAX, INVERSE_GRID+1)])
    values = []
    for i, t in enumerate(grid):
        print(
            f'开始反演：{i+1}/{len(grid)}，t={t:.6g}',
            flush=True
        )
        val = float(mp.invertlaplace(LAPLACE, float(t), method='dehoog', degree=INVERSE_DEGREE))
        print(f'完成，密度={val:.6g}', flush=True)
        if not np.isfinite(val) or val < -1e-9:
            raise ValueError(f'理论反演产生无效密度，t={t}，rho={val}。')
        values.append(max(val, 0.))
        if (i+1) % 30 == 0 or i+1 == len(grid):
            print(f'理论E1反演：{i+1}/{len(grid)}', flush=True)
    grid = np.r_[0., grid]
    values = np.r_[0., values]
    print('反演网格之前使用插值；需调整最早时间点并加密网格检查误差。')
    return np.interp(times, grid, values), np.column_stack([grid, values])


def run():
    if X0 is Ellipsis or BETA is Ellipsis:
        raise ValueError('请先填写输入区中的 X0、BETA、drift、diffusion 和 density。')
    if not (N >= 2 and DT > 0 and T_MAX > 0 and X0 < BETA):
        raise ValueError('要求 N>=2、DT>0、T_MAX>0、X0<BETA。')
    if ABSORB_ZERO and not 0 < X0 < BETA:
        raise ValueError('零点吸收时要求 0<X0<BETA。')

    # 先检查 E1：缺少理论公式时不开始模拟。
    times = np.linspace(0, T_MAX, 5001)
    rho, inverse_samples = prepare_theory(times)
    if np.any(rho < 0):
        raise ValueError('E1 密度不能为负。')
    if SURVIVAL is None:
        H = 1-cumulative_trapezoid(rho, times, initial=0)
        if np.any(H < -0.005):
            raise ValueError('E1 积分超过1，请检查理论公式或加密 times 网格。')
    else:
        H = np.ones_like(times)
        H[1:] = evaluate(SURVIVAL, times[1:], 'SURVIVAL / E2')
        if np.any((H < -1e-10) | (H > 1+1e-10)) or np.any(np.diff(H) > 1e-8):
            raise ValueError('E2 应在0到1之间且随时间不增加。')

    rng = np.random.default_rng(SEED)
    x = np.full(N, float(X0))
    T = np.full(N, np.inf)
    alive = np.ones(N, dtype=bool)
    zeros = 0
    for k in range(int(np.ceil(T_MAX/DT))):
        ids = np.flatnonzero(alive)
        if not ids.size:
            break
        start = k*DT
        h = min(DT, T_MAX-start)
        if h <= 0:
            break
        old = x[ids]
        f = evaluate(drift, old, 'drift / f(x)')
        g = evaluate(diffusion, old, 'diffusion / g(x)')
        new = old + f*h + g*np.sqrt(h)*rng.standard_normal(ids.size)
        if not np.all(np.isfinite(new)):
            raise ValueError('模拟产生无效位置，请检查模型或减小 DT。')
        hit = new >= BETA
        T[ids[hit]] = start+h
        alive[ids[hit]] = False
        if ABSORB_ZERO:
            at_zero = (new <= 0) & ~hit
            zeros += int(at_zero.sum())
            alive[ids[at_zero]] = False
            new = np.maximum(new, 0)
        x[ids] = new

    finite = np.sort(T[np.isfinite(T)])
    H_mc = 1-np.searchsorted(finite, times, side='right')/N
    edges = np.linspace(0, T_MAX, 101)
    counts, _ = np.histogram(finite, edges)
    plt.rcParams.update({
        'font.family': 'Times New Roman',
        'mathtext.fontset': 'stix',
        'font.size': 9,
        'axes.labelsize': 10,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 8,
        'axes.linewidth': 0.8,
        'pdf.fonttype': 42,
    })

    fig, axes = plt.subplots(
        1, 2, figsize=(7.2, 2.8), constrained_layout=True
    )

    # 左图：理论红色实线，仿真浅蓝色直方图。
    axes[0].plot(
        times[1:], rho[1:],
        color='#b33440', linewidth=1.5,
        label='Theory', zorder=3
    )
    axes[0].bar(
        (edges[:-1]+edges[1:])/2,
        counts/(N*np.diff(edges)),
        width=np.diff(edges), color='#bed5e8',
        edgecolor='none', label='Simulation', zorder=1
    )

    # 右图：理论实线在下层，仿真虚线在上层。
    axes[1].plot(
        times, H,
        color='#b33440', linewidth=1.5, linestyle='-',
        label='Theory', zorder=2
    )
    axes[1].plot(
        times, H_mc,
        color='#24658b', linewidth=1.5, linestyle='--',
        label='Simulation', zorder=3
    )

    for ax, panel, ylabel in zip(
        axes,
        ['(a)', '(b)'],
        [r'Density $\rho(t)$', r'Survival probability $H(t)$']
    ):
        ax.set(xlabel=r'Time $t$', ylabel=ylabel, xlim=(0, T_MAX))
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(direction='out', length=3, width=0.8)
        ax.legend(frameon=False, loc='upper right')
        ax.text(
            0, 1.04, panel, transform=ax.transAxes,
            fontsize=10, fontweight='bold', va='bottom'
        )

    out = Path(__file__).resolve().parent / 'fpt_results_E2'
    out.mkdir(exist_ok=True)
    fig.savefig(out/'comparison.pdf', bbox_inches='tight')
    fig.savefig(out/'comparison.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    np.savetxt(out/'curves.csv', np.column_stack([times, rho, H, H_mc]),
               delimiter=',', header='time,density_theory,survival_theory,survival_simulation', comments='')
    np.save(out/'first_passage_times.npy', T)
    if inverse_samples is not None:
        np.savetxt(out/'inverse_density_samples.csv', inverse_samples, delimiter=',',
                   header='time,density', comments='')
    summary = dict(n=N, dt=DT, t_max=T_MAX, seed=SEED, x0=X0, beta=BETA,
                   method='Euler; endpoint crossing detection', reached=int(finite.size),
                   absorbed_at_zero=zeros, still_running=int(alive.sum()),
                   max_survival_error=float(np.max(np.abs(H-H_mc))),
                   theory_survival_source='input E2' if SURVIVAL is not None else 'numerical integral of E1',
                   density_source='input E1' if LAPLACE is None else 'inverse of input F(s)')
    if finite.size == N:
        summary.update(mean=float(finite.mean()), variance=float(finite.var(ddof=1)))
    (out/'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'Saved: {out}')
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print('减小 DT 检查误差：本程序可能漏掉网格点之间的越界。')
    if ABSORB_ZERO:
        print('零点截断吸收是数值近似，需要检查是否过早吸收。')
    if SURVIVAL is None:
        print('E2 由 E1 数值积分得到；尖锐密度需要加密 times 网格。')
    if finite.size < N:
        print('存在未到达轨迹，不用已到达样本估计整体均值。')


if __name__ == '__main__':
    run()
