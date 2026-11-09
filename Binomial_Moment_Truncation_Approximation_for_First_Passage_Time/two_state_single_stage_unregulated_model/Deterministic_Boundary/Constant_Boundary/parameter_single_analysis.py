from pathlib import Path
import importlib.util
import sys
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from scipy.interpolate import PchipInterpolator

HERE = Path(__file__).resolve().parent
OUT = HERE / "single_parameter_results"
BASE = dict(k1=1.0, k2=1.0, mu=5.0, delta=1.0, B=8)
SYMBOLS = dict(k1=r"k_1", k2=r"k_2", mu=r"\mu", delta=r"\delta", B="B")
FACTORS = (0.5, 1, 2, 4, 8)
GAMMA = 16.0
INITIAL = dict(n0=0, p_off0=1.0, p_on0=0.0)
N_MIN, N_MAX, N_STEP, TOL = 12, 30, 2, 1e-6
T_END, T_STAR, FPT_XMAX = 200.0, 20.0, 50.0

def load(stem):
    files = list(HERE.glob(stem + "*.py"))
    if len(files) != 1:
        raise RuntimeError(f"同文件夹需恰好一个 {stem}*.py，实际有 {len(files)} 个")
    spec = importlib.util.spec_from_file_location(stem, files[0])
    module = importlib.util.module_from_spec(spec)
    sys.modules[stem] = module
    spec.loader.exec_module(module)
    return module, files[0]


def main():
    if not 0 <= T_STAR <= T_END or N_STEP <= 0 or N_MAX < N_MIN + 2 * N_STEP:
        raise ValueError("检查时间范围及 N 的范围（至少测试三个阶数）")
    soft, _ = load("soft_binomial_moment_constant_boundary")
    OUT.mkdir(parents=True, exist_ok=True)
    names, memo = list(BASE), {}
    t = np.unique(np.r_[np.linspace(0, min(10, T_END), 201),
                        np.linspace(min(10, T_END), T_END, 381), T_STAR])

    def compute(params):
        key = tuple(params.values())
        if key in memo:
            return memo[key]
        if int(params["B"]) != params["B"] or params["B"] <= INITIAL["n0"]:
            raise ValueError("B 必须是大于初始分子数的正整数，请检查倍数")
        b = int(params["B"])
        rates = {k: params[k] for k in names if k != "B"}
        print(f"[计算] {params}", flush=True)
        result = dict(h=np.full_like(t, np.nan), f=np.full_like(t, np.nan))
        prev, count = None, 0
        orders = range(max(N_MIN, b), max(N_MAX, b + 10) + 1, N_STEP)
        if np.exp(-GAMMA * (b - 0.5)) == 0:
            print("  Soft 的 q 系数下溢为零，跳过此组曲线。", flush=True)
            orders = ()
        for n in orders:
            try:
                h = np.asarray(soft.soft_survival_truncated(
                    t, B=b, N=n, gamma=GAMMA, **rates, **INITIAL)[0], float)
                if h.shape != t.shape or not np.all(np.isfinite(h)) or not soft.is_physical_survival(h):
                    raise ValueError("H 非有限或未通过物理性检查")
            except Exception as exc:
                print(f"  N={n}: {exc}", flush=True)
                prev, count = None, 0
                continue
            diff = float(np.max(np.abs(h - prev))) if prev is not None else np.inf
            count = count + 1 if diff < TOL else 0
            print(f"  N={n}, 差异={diff:.3e}, 连续通过={count}", flush=True)
            if count == 2:
                result.update(h=h, f=-np.gradient(h, t))
                break
            prev = h
        if not np.all(np.isfinite(result["h"])):
            print("  Soft 未收敛，跳过此组曲线。", flush=True)
        memo[key] = result
        return result

    plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix",
                         "font.size": 11, "axes.grid": False, "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42, "svg.fonttype": "none"})
    colors = plt.get_cmap("viridis")(np.linspace(.1, .85, len(FACTORS)))
    legends = [Line2D([], [], color=c, label=fr"${v:g}\times$") for v, c in zip(FACTORS, colors)]
    legends += [Line2D([], [], color="0.2", label="Interpolation"),
                Line2D([], [], color="0.2", marker="o", mfc="none", ls="none", label="Binomial approximation")]

    # 密集时间网格仅用于插值绘图，不增加求解器调用。
    t_plot = np.unique(np.r_[t, np.linspace(0, T_END, 4001)])
    runs = {name: [compute({**BASE, name: BASE[name] * v})
                   for v in FACTORS] for name in names}
    for quantity, ylabel, xmax in (("h", r"$H(t)$", T_END), ("f", r"$f_T(t)$", FPT_XMAX)):
        fig, axes = plt.subplots(2, 3, figsize=(11.5, 6.8), layout="constrained")
        for idx, name in enumerate(names):
            ax = axes.flat[idx]
            for result, color in zip(runs[name], colors):
                if not np.all(np.isfinite(result[quantity])):
                    continue
                curve = PchipInterpolator(t, result[quantity])(t_plot)
                ax.plot(t_plot, curve, color=color, lw=1.5)
                # 按 Soft 的 H 下降进度选点，同时保留均匀时间点，兼顾快慢曲线。
                levels = [np.argmin(abs(result["h"] - z)) for z in np.linspace(.05, .95, 14)]
                marks = np.unique(np.r_[levels, np.linspace(0, len(t)-1, 18).astype(int)]).astype(int)
                ax.plot(t[marks], result[quantity][marks], "o", color=color, mfc="none", ms=4, mew=.9)
            ax.set(xlim=(0, xmax), xlabel="$t$", ylabel=ylabel)
            ax.set_title(fr"({chr(97+idx)}) {'$' + SYMBOLS[name] + '$'}", loc="left")
            if quantity == "h":
                ax.set_ylim(-.02, 1.02)
        axes.flat[-1].set_axis_off()
        axes.flat[-1].legend(handles=legends, frameon=False, loc="center")
        for ext in ("pdf", "svg", "png"):
            fig.savefig(OUT / f"{'survival' if quantity == 'h' else 'fpt'}.{ext}",
                        dpi=600, bbox_inches="tight")
        plt.close(fig)
    print(f"完成：图片已保存至 {OUT}", flush=True)


if __name__ == "__main__":
    main()
