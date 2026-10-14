from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid

# ================= 模型输入区：当前为例1 =================
# 换例子时，只替换本区的参数、drift、diffusion、density。
# 模型：dX = drift(X) dt + diffusion(X) dW。
MU = 1.0             # 例1的漂移速度
SIGMA = 0.8          # 例1的噪声系数
X0 = 0.0             # 初始位置
BETA = 1.0           # 固定上边界，必须大于 X0


def drift(x):
    # 漂移函数 f(x)：例1是常数 MU。
    return MU


def diffusion(x):
    # 噪声函数 g(x)：例1是常数 SIGMA，注意不是 SIGMA**2。
    return SIGMA


def density(t):
    # 理论首达密度 E1；用于对比，不用于产生模拟轨迹。
    r = BETA - X0
    return (r / np.sqrt(2*np.pi*SIGMA**2*t**3)
            * np.exp(-(r-MU*t)**2 / (2*SIGMA**2*t)))


# 如果模型规定到零后永远停在零，将下面改成 True。
ABSORB_ZERO = False
N = 50_000           # 模拟轨迹数
DT = 0.001           # 时间步长
T_MAX = 10.0         # 模拟结束时间
SEED = 20261010
# 可选输入 E2：没有则保持 None，由 E1 积分计算生存概率。
SURVIVAL = None      # 也可写成 lambda t: ...
# 若已有时间域 E1，保持 LAPLACE=None，直接填写 density(t)。
# 若理论只有拉普拉斯变换 F(s)，在此定义函数并设置 LAPLACE=该函数。
# 此时 density(t) 可以保持占位符，程序会反演生成 E1。
LAPLACE = None
INVERSE_MIN_TIME = 0.008   # 反演网格最早非零时间，需按模型调整
INVERSE_GRID = 120        # 增大可加密理论曲线
INVERSE_DEGREE = 32       # 数值反演阶数
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
        val = float(mp.invertlaplace(LAPLACE, float(t), method='dehoog', degree=INVERSE_DEGREE))
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

    out = Path(__file__).resolve().parent / 'fpt_results_E1'
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
