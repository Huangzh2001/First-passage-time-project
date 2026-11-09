"""时间依赖边界：仅使用二项矩截断求解器进行五参数敏感性分析。

与 soft_binomial_moment_time_boundary*.py 放在同一文件夹运行。
生成 H(t) 与 f_T(t) 两张 2×3 综合图（各保存 PDF/SVG/PNG）。
达到连续两次截断容差就提前停止；否则采用最后一次成功算出的 N 并警告。
未通过截断稳定性或物理性检查的曲线以虚线表示，仅供诊断。
"""
from pathlib import Path
import importlib.util

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
B, GAMMA = 8, 16.0
BASE = dict(k1=1.0, k2=1.0, mu=5.0, delta=1.0)
PARAMS = {**BASE, "B": B}
FACTORS = [0.5, 1, 2, 4]
INITIAL = dict(n0=0, p_off0=1.0, p_on0=0.0)

T_END = 50.0
T = np.r_[np.linspace(0, 10, 201), np.linspace(10, T_END, 81)[1:]]
FPT_XMAX = 50.0
DOUBLING_TIME = 20.0                 # beta(t) = B0 * 2**(t / DOUBLING_TIME)
SOFT_SCAN_DT = 0.1
N_MIN, N_MAX, N_STEP = 12, 30, 2
TOL = 1e-6


def load(stem):
    files = list(HERE.glob(stem + "*.py"))
    if len(files) != 1:
        raise RuntimeError(f"同一文件夹中应恰好有一个 {stem}*.py 文件；目前找到 {len(files)} 个。")
    spec = importlib.util.spec_from_file_location(stem, files[0])
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def soft_curve(soft, rates, beta):
    """尝试不同 N；优先返回通过判据的解，否则返回最后一次有限的计算结果。"""
    previous, count, last = None, 0, None
    for n in range(N_MIN, N_MAX + 1, N_STEP):
        print(f"      N={n}：开始计算", flush=True)
        try:
            h, f = soft.soft_survival_time_boundary(
                T, beta_function=beta, N=n, gamma=GAMMA,
                boundary_scan_dt=SOFT_SCAN_DT, **rates, **INITIAL
            )
        except Exception as exc:
            print(f"      N={n}：计算失败 ({type(exc).__name__}: {exc})", flush=True)
            previous, count = None, 0
            continue

        if not (np.all(np.isfinite(h)) and np.all(np.isfinite(f))):
            print(f"      N={n}：结果出现 NaN/Inf，跳过", flush=True)
            previous, count = None, 0
            continue

        last = (h, f, n)
        if not soft.is_physical_survival(h):
            print(f"      N={n}：H(t) 未通过物理性检查", flush=True)
            previous, count = None, 0
            continue

        difference = np.inf if previous is None else np.max(np.abs(h - previous))
        count = count + 1 if difference < TOL else 0
        print(f"      N={n}：截断差异={difference:.3e}，连续通过={count}", flush=True)
        if count == 2:
            return h, f, n, True
        previous = h

    if last is None:
        print("      警告：所有 N 都未得到有限结果，本组无法绘制", flush=True)
        return None

    h, f, n = last
    print(f"      警告：未通过完整收敛检查，采用最后的 N={n}；"
          f"物理性检查={'通过' if soft.is_physical_survival(h) else '未通过'}", flush=True)
    return h, f, n, False


def main():
    soft = load("soft_binomial_moment_time_boundary")
    plt.rcParams.update({
        "font.family": "STIXGeneral", "mathtext.fontset": "stix",
        "font.size": 12, "axes.grid": False,
        "axes.spines.top": False, "axes.spines.right": False,
        "svg.fonttype": "none", "pdf.fonttype": 42,
    })

    symbols = [r"k_1", r"k_2", r"\mu", r"\delta", r"B"]
    colors = plt.cm.viridis(np.linspace(0.12, 0.85, len(FACTORS)))
    layout = dict(figsize=(11.5, 6.8), gridspec_kw={"width_ratios": [1, 1, 0.72]})
    fig, axes = plt.subplots(2, 3, sharex=True, sharey=True, **layout)
    fig_f, axes_f = plt.subplots(2, 3, sharex=True, sharey=False, **layout)

    for figure in (fig, fig_f):
        figure.subplots_adjust(left=0.075, right=0.985, top=0.95,
                               bottom=0.10, hspace=0.24, wspace=0.12)

    for ax, ax_f, name, symbol, panel in zip(
        axes.flat, axes_f.flat, PARAMS, symbols, "abcde"
    ):
        for factor, color in zip(FACTORS, colors):
            value = PARAMS[name] * factor
            params = {**PARAMS, name: value}
            b = int(params.pop("B"))
            beta = lambda t, b=b: b * 2.0 ** (t / DOUBLING_TIME)
            print(f"\n[{panel}] {name}={value:g}, B0={b}：开始 Soft 计算", flush=True)

            result = soft_curve(soft, params, beta)
            if result is None:
                continue
            h, f, n, converged = result
            style = "-" if converged else "--"
            ax.plot(T, h, color=color, lw=1.7, ls=style)
            mask = T <= FPT_XMAX
            ax_f.plot(T[mask], f[mask], color=color, lw=1.7, ls=style)
            print(f"[{panel}] {name}={value:g}：N={n}，"
                  f"{'通过截断稳定性检查' if converged else '未通过（虚线）'}", flush=True)

        title = fr"({panel}) ${symbol}$ (baseline: ${PARAMS[name]:g}$)"
        ax.set_title(title, loc="left")
        ax_f.set_title(title, loc="left")
        ax.set(xlim=(0, T[-1]), ylim=(-0.02, 1.02))
        ax_f.set(xlim=(0, FPT_XMAX))

    for axs, ylabel in ((axes, r"$H(t)$"), (axes_f, r"$f_T(t)$")):
        for a in axs[:, 0]:
            a.set_ylabel(ylabel)
        for a in axs[1, :2]:
            a.set_xlabel(r"$t$")

    handles = [
        Line2D([], [], color=c, lw=1.7, label=fr"${v:g}\times$")
        for v, c in zip(FACTORS, colors)
    ]
    handles.append(Line2D([], [], color="0.4", lw=1.7, ls="--",
                          label="Unverified truncation"))
    for axs in (axes, axes_f):
        blank = axs[1, 2]
        blank.set_axis_off()
        blank.legend(handles=handles, loc="center left", bbox_to_anchor=(0.05, 0.5),
                     ncol=1, frameon=False, fontsize=10,
                     labelspacing=0.9, handlelength=2.0)

    print("\n开始保存图片……", flush=True)
    for extension in ("pdf", "svg", "png"):
        fig.savefig(HERE / f"parameter_soft_time_boundary_survival.{extension}",
                    dpi=600, bbox_inches="tight")
        fig_f.savefig(HERE / f"parameter_soft_time_boundary_fpt.{extension}",
                      dpi=600, bbox_inches="tight")
    plt.close(fig)
    plt.close(fig_f)
    print("已保存 H(t) 和 f_T(t) 两张图。", flush=True)


if __name__ == "__main__":
    main()
