"""单参数、双参数原有绘图与分类缓存 + 三种多参数图。
多参数 Sobol 使用专门的独立抽样交叉矩阵、覆盖较广的范围，仅计算 H(T_STAR)。
不通过收敛的 Soft 样本不用于 Sobol 估计（不会偷换为其他样本）。
"""
from pathlib import Path
from matplotlib.lines import Line2D
import csv
import hashlib
import importlib.util
from io import BytesIO
import json
import sys
import shutil

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import trapezoid

# ================= 用户设置 =================
HERE = Path(__file__).resolve().parent
BASE = dict(k1=1.0, k2=1.0, mu=5.0, delta=1.0, B=8)
SYMBOLS = dict(k1=r"k_1", k2=r"k_2", mu=r"\mu", delta=r"\delta", B="B")
FACTORS = (0.5, 1, 2, 4, 8)  # 相对各自基准值的倍数
# 要运行的双参数组合：删除或注释不需要的行。5个参数全部配对共10组。
PAIRS = [
    ("k1", "k2"), ("k1", "mu"), ("k1", "delta"), ("k1", "B"),
    ("k2", "mu"), ("k2", "delta"), ("k2", "B"),
    ("mu", "delta"), ("mu", "B"), ("delta", "B"),
]
# 每个参数只有一套取值；任意两两组合自动读取对应网格。
PAIR_GRIDS = dict(
    k1=np.geomspace(0.25, 4, 21),
    k2=np.geomspace(0.25, 4, 21),
    mu=np.linspace(2.5, 40, 41),
    delta=np.geomspace(0.25, 4, 21),
    B=np.arange(4, 21),
)
INTERACTION_LINES = 5      # 从密集数据中选 5 个 B 值，避免曲线过密
GAMMA = 16.0               # 沿用上传脚本的设置
INITIAL = dict(n0=0, p_off0=1.0, p_on0=0.0)
N_MIN, N_MAX, N_STEP, TOL = 12, 30, 2, 1e-6
T_END, T_STAR, FPT_XMAX = 200.0, 20.0, 50.0
OUT = HERE / "parameter_comparison_results"
# ------ 新增的多参数分析设置（控制计算量：2**POWER * (2*5+2)）------
RUN_SINGLE, RUN_DOUBLE, RUN_MULTI = True, True, True
MULTI_POWER = 6  # 2**6=64基础样本，共768个设计求值；正式研究建议8或更大
MULTI_SEED = 2026  # 不变则支持断点复用；增加 POWER 可复用之前已有的参数计算
MULTI_PARALLEL_LINES = 80  # 平行坐标图最多展示80条线，不画全部线
MULTI_RANGES = dict(k1=(0.2, 6.0), k2=(0.2, 6.0), mu=(1.0, 50.0),
                    delta=(0.2, 5.0), B=(4, 20))
MULTI_LOG_PARAMETERS = ("k1", "k2", "delta")
# ============================================


def load(stem):
    files = list(HERE.glob(stem + "*.py"))
    if len(files) != 1:
        raise RuntimeError(f"同文件夹需恰好一个 {stem}*.py，实际有 {len(files)} 个")
    spec = importlib.util.spec_from_file_location(stem, files[0])
    module = importlib.util.module_from_spec(spec)
    sys.modules[stem] = module
    spec.loader.exec_module(module)
    return module, files[0]


def save_fig(fig, folder, name):
    folder.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "svg", "png"):
        buffer = BytesIO()
        fig.savefig(buffer, format=ext, dpi=600, bbox_inches="tight")
        (folder / f"{name}.{ext}").write_bytes(buffer.getvalue())
    plt.close(fig)


def write_csv(path, rows):
    if rows:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def symbol(name):
    return "$" + SYMBOLS[name] + "$"


# ========= 新增：五参数 Sobol + 平行坐标（仅在 RUN_MULTI 时运行） =========
def sobol_design(names, exponent, seed):
    """Saltelli/Sobol 交叉矩阵，每行排列为 A, AB[0:d], BA[0:d], B。"""
    from scipy.stats import qmc
    d = len(names)
    u = qmc.Sobol(d=2*d, scramble=True, seed=seed).random_base2(exponent)
    def transform(z):
        out = np.empty_like(z)
        for j, name in enumerate(names):
            lo, hi = MULTI_RANGES[name]
            if name == "B":
                out[:, j] = lo + np.minimum(np.floor(z[:, j] * (hi-lo+1)), hi-lo)
            elif name in MULTI_LOG_PARAMETERS:
                out[:, j] = np.exp(np.log(lo) + z[:, j] * np.log(hi/lo))
            else:
                out[:, j] = lo + z[:, j] * (hi-lo)
        return out
    a, b = transform(u[:, :d]), transform(u[:, d:])
    design = np.empty((len(a), 2*d+2, d))
    design[:, 0] = a
    design[:, -1] = b
    for j in range(d):
        design[:, 1+j] = a
        design[:, 1+j, j] = b[:, j]
        design[:, 1+d+j] = b
        design[:, 1+d+j, j] = a[:, j]
    return design


def sobol_indices(y):
    """与 SALib Sobol 的 Saltelli 一阶、Jansen 总效应、Saltelli 二阶估计式一致。"""
    y = np.asarray(y, float)
    n, step = y.shape
    d = (step-2)//2
    if np.any(~np.isfinite(y)):
        return None  # 不可以删除失败样本后计算 Sobol；会破坏交叉抽样结构
    s = y.std()
    if s < 1e-13:
        return None
    y = (y-y.mean())/s
    a, b = y[:, 0], y[:, -1]
    ab, ba = y[:, 1:1+d], y[:, 1+d:1+2*d]
    v = np.var(np.r_[a,b])
    if v <= 1e-13:
        return None
    s1 = np.mean(b[:, None] * (ab-a[:, None]), axis=0) / v
    st = np.mean((a[:, None]-ab)**2, axis=0) / (2*v)
    s2 = np.full((d,d), np.nan)
    for j in range(d):
        for k in range(j+1,d):
            s2[j,k] = np.mean(ba[:,j]*ab[:,k]-a*b)/v - s1[j]-s1[k]
    return dict(S1=s1, ST=st, S2=s2)


def plot_multi_sobol(names, hard_si, soft_si, folder):
    """输出两张三列图：一阶/总效应柱状图、二阶交互热力图。"""
    labels = [symbol(k) for k in names]
    diff = None if hard_si is None or soft_si is None else {
        key: soft_si[key]-hard_si[key] for key in ("S1", "ST", "S2")}
    trio = (hard_si, soft_si, diff)
    titles = ("(a) Hard (exact)", "(b) Binomial approximation", "(c) Soft - Hard")

    fig, axs = plt.subplots(1, 3, figsize=(14, 4.5), layout="constrained")
    x = np.arange(len(names))
    for i, (ax, si) in enumerate(zip(axs, trio)):
        ax.set_title(titles[i])
        ax.set_xticks(x, labels)
        if si is None:
            ax.text(.5,.5,"Unavailable: incomplete / constant output", ha="center",va="center",
                    transform=ax.transAxes,fontsize=10,wrap=True)
            continue
        ax.bar(x-.19, si["S1"], width=.37, label=r"$S_1$", color="#3274a1")
        ax.bar(x+.19, si["ST"], width=.37, label=r"$S_T$", color="#e1812c")
        ax.axhline(0,color="0.35",lw=.8)
        ax.set_ylabel("Sobol index difference" if i==2 else "Sobol index estimate")
        if i == 0:
            ax.legend(frameon=False)
    fig.suptitle(fr"Global Sobol sensitivity, $H({T_STAR:g})$ (base N={2**MULTI_POWER})")
    save_fig(fig, folder, "sobol_first_total")

    fig, axs = plt.subplots(1,3,figsize=(14,4.5),layout="constrained")
    mask = ~np.triu(np.ones((len(names),len(names)), dtype=bool), k=1)
    available = [si["S2"][~mask] for si in (hard_si,soft_si) if si is not None]
    common = max((float(np.nanmax(np.abs(v))) for v in available if v.size), default=0.0)
    common = max(common, 1e-8)
    for i, (ax,si) in enumerate(zip(axs,trio)):
        ax.set_title(titles[i])
        ax.set_xticks(range(len(names)), labels)
        ax.set_yticks(range(len(names)), labels)
        if si is None:
            ax.text(.5,.5,"Unavailable: incomplete / constant output",ha="center",va="center",
                    transform=ax.transAxes, fontsize=10,wrap=True)
            continue
        arr=si["S2"]
        clim = common if i < 2 else max(float(np.nanmax(np.abs(arr[~mask]))),1e-8)
        im=ax.imshow(np.ma.masked_array(arr, mask=mask), cmap="coolwarm",
                     vmin=-clim,vmax=clim,interpolation="nearest")
        for j in range(len(names)):
            for k in range(j+1,len(names)):
                ax.text(k,j,f"{arr[j,k]:.2f}",ha="center",va="center",fontsize=8,
                        color="black" if abs(arr[j,k]) < .6*clim else "white")
        fig.colorbar(im,ax=ax,shrink=.72, label=r"$S_2$ difference" if i==2 else r"$S_2$ estimate")
    fig.suptitle(fr"Pairwise Sobol interactions, $H({T_STAR:g})$ (negative estimates = sampling noise)")
    save_fig(fig, folder, "sobol_second_order")


def plot_multi_parallel(names, X, Y, folder):
    """最多画 MULTI_PARALLEL_LINES 条，按 Hard 响应分位数均匀抽取以防视觉过密。"""
    good = np.where(np.isfinite(Y[:,0]) & np.isfinite(Y[:,1]))[0]
    if len(good):
        ordered = good[np.argsort(Y[good,0])]
        chosen = ordered[np.unique(np.linspace(0,len(ordered)-1,
                              min(MULTI_PARALLEL_LINES,len(ordered))).round().astype(int))]
    else:
        chosen = np.array([],dtype=int)
    xx = X[chosen]
    resp = np.c_[Y[chosen,0],Y[chosen,1],Y[chosen,1]-Y[chosen,0]]
    d=len(names)
    scaled=np.zeros_like(xx)
    for i,k in enumerate(names):
        lo,hi=MULTI_RANGES[k]
        if k in MULTI_LOG_PARAMETERS:
            scaled[:,i] = np.log(xx[:,i]/lo)/np.log(hi/lo)
        else:
            scaled[:,i] = (xx[:,i]-lo)/(hi-lo)
    titles=("(a) Hard (exact)","(b) Binomial approximation","(c) Soft - Hard")
    fig,axs=plt.subplots(1,3,figsize=(16,5),layout="constrained")
    for col,ax in enumerate(axs):
        ax.set_title(titles[col])
        ax.set_xlim(-.1,d+.1)
        ax.set_ylim(-.08,1.08)
        ax.set_xticks(range(d+1),[symbol(k) for k in names]+[r"$H$" if col<2 else r"$\Delta H$"])
        ax.set_yticks([0,.5,1],["low","middle","high"])
        for k in range(d+1):
            ax.axvline(k,color="0.75",lw=.8,zorder=0)
            if k<d:
                low,high=MULTI_RANGES[names[k]]
            else:
                low,high=(-1,1) if col==2 else (0,1)
            ax.text(k,1.015,f"{high:g}",ha="center",va="bottom",fontsize=8,color="0.35")
            ax.text(k,-.015,f"{low:g}",ha="center",va="top",fontsize=8,color="0.35")
        if not len(chosen):
            ax.text(.5,.5,"Unavailable: no paired valid Soft results",ha="center",va="center",transform=ax.transAxes)
            continue
        z=resp[:,col]
        clim=max(np.max(np.abs(z)),1e-12) if col==2 else 1
        last=(z/clim+1)/2 if col==2 else np.clip(z,0,1)
        cmap=plt.get_cmap("coolwarm" if col==2 else "viridis")
        from matplotlib.colors import Normalize
        norm=Normalize(vmin=-clim if col==2 else 0,vmax=clim if col==2 else 1)
        for row,val in zip(np.column_stack([scaled,last]),z):
            ax.plot(range(d+1),row,color=cmap(norm(val)),lw=.9,alpha=.30)
        from matplotlib.cm import ScalarMappable
        fig.colorbar(ScalarMappable(norm=norm,cmap=cmap),ax=ax,shrink=.65,
                     label=r"$\Delta H$" if col==2 else r"$H$")
    fig.suptitle(fr"Parallel coordinates: {len(chosen)} representative samples (axes normalized)")
    save_fig(fig, folder, "parallel_coordinates")


def run_multi(hard, soft, hard_path, soft_path, names, full_memo, settings, out):
    """仅计算 H(T_STAR)，图像都复用同一个 Sobol-Saltelli 五维样本设计。"""
    from scipy.linalg import expm
    folder=out / "multi_parameter"
    folder.mkdir(parents=True,exist_ok=True)
    cdir=out / "calculation_cache" / "multi_parameter" / "sobol"
    cdir.mkdir(parents=True,exist_ok=True)
    time_probe=np.linspace(0,T_STAR,41)  # Soft 检查 [0,T_STAR] 的物理性；无需算到 T_END
    fingerprint=hashlib.sha256(json.dumps(dict(settings,probe=time_probe.tolist(),
        policy="sobol_scalar_at_tstar_v1"),sort_keys=True).encode()+
        hard_path.read_bytes()+soft_path.read_bytes()).hexdigest()[:12]
    compact_memo={}
    def eval_one(p):
        key=tuple(float(p[k]) for k in names)
        if key in compact_memo:
            return compact_memo[key]
        filename="__".join(f"{k}={int(p[k]) if k=='B' else repr(float(p[k]))}" for k in names)+".npz"
        file=cdir / filename
        result=None
        if file.exists():
            try:
                with np.load(file,allow_pickle=False) as f:
                    if str(f["_signature"].item())==fingerprint:
                        result={k:f[k].item() for k in ("hard_star","soft_star","n","status","diff")}
            except (OSError,KeyError,ValueError,EOFError):
                pass
        if result is None:
            if key in full_memo:
                old=full_memo[key]
                result=dict(hard_star=float(old["hard_star"]),soft_star=float(old["soft_star"]),
                            n=int(old["n"]),status=str(old["status"]),diff=float(old["diff"]))
            else:
                rates={k:float(p[k]) for k in names if k!='B'}
                b=int(p["B"])
                try:
                    q=hard.build_Q_B(b,**rates)
                    hard.check_Q_B(q,b,rates["mu"])
                    p0=hard.initial_probability(b,**INITIAL)
                    hs=float(np.sum(expm(T_STAR*q) @ p0))
                except Exception as e:
                    result=dict(hard_star=np.nan,soft_star=np.nan,n=-1,status="hard_failed",diff=np.nan)
                    print(f"  Hard error: {e}",flush=True)
                else:
                    result=dict(hard_star=hs,soft_star=np.nan,n=-1,status="not_converged",diff=np.nan)
                    if np.exp(-GAMMA*(b-.5))==0:
                        result["status"]="killing_underflow"
                    else:
                        prev,count=None,0
                        for n in range(max(N_MIN,b),max(N_MAX,b+10)+1,N_STEP):
                            try:
                                h=np.asarray(soft.soft_survival_truncated(time_probe,B=b,N=n,
                                      gamma=GAMMA,**rates,**INITIAL)[0],float)
                                if (h.shape!=time_probe.shape or
                                    not soft.is_physical_survival(h)):
                                    raise ValueError("nonphysical survival")
                            except Exception as e:
                                print(f"  Soft N={n}: {e}",flush=True)
                                prev,count=None,0
                                continue
                            dif=float(np.max(np.abs(h-prev))) if prev is not None else np.inf
                            count=count+1 if dif<TOL else 0
                            result.update(n=n,diff=dif)
                            if count>=2:
                                result.update(soft_star=float(h[-1]),status="converged")
                                break
                            prev=h
            temp=file.with_suffix(".tmp.npz")
            np.savez_compressed(temp,**result,_signature=fingerprint)
            temp.replace(file)
        compact_memo[key]=result
        return result

    design=sobol_design(names,MULTI_POWER,MULTI_SEED)
    m,step,d=design.shape
    total=m*step
    y=np.full((m,step,2),np.nan)
    rows=[]
    kinds=["A"]+[f"AB_{n}" for n in names]+[f"BA_{n}" for n in names]+["B"]
    for i in range(m):
        for j in range(step):
            p={name:(int(design[i,j,k]) if name=="B" else float(design[i,j,k]))
               for k,name in enumerate(names)}
            r=eval_one(p)
            y[i,j]=[r["hard_star"],r["soft_star"]]
            rows.append({"sample":i,"type":kinds[j],**p,"hard_H_star":r["hard_star"],
                "soft_H_star":r["soft_star"],"difference":r["soft_star"]-r["hard_star"],
                "N":r["n"],"status":r["status"],"last_N_difference":r["diff"]})
            done=i*step+j+1
            if done%12==0 or done==total:
                print(f"[多参数 {done}/{total}] 独立配置数={len(compact_memo)}",flush=True)
                write_csv(folder/"sample_results.csv",rows)  # 中断前保留已计算清单
    hsi=sobol_indices(y[:,:,0]); ssi=sobol_indices(y[:,:,1])
    index_rows=[]
    for i,name in enumerate(names):
        for order in ("S1","ST"):
            hv=float(hsi[order][i]) if hsi is not None else np.nan
            sv=float(ssi[order][i]) if ssi is not None else np.nan
            index_rows.append({"index":order,"parameter_1":name,"parameter_2":"",
                               "hard":hv,"soft":sv,"soft_minus_hard":sv-hv})
        for j in range(i+1,len(names)):
            hv=float(hsi["S2"][i,j]) if hsi is not None else np.nan
            sv=float(ssi["S2"][i,j]) if ssi is not None else np.nan
            index_rows.append({"index":"S2","parameter_1":name,"parameter_2":names[j],
                               "hard":hv,"soft":sv,"soft_minus_hard":sv-hv})
    write_csv(folder/"sobol_indices.csv",index_rows)
    plot_multi_sobol(names,hsi,ssi,folder)
    plot_multi_parallel(names,design.reshape(-1,d),y.reshape(-1,2),folder)
    (folder/"sampling_settings.json").write_text(json.dumps({
        "base_sample_size":m,"design_evaluations":total,"seed":MULTI_SEED,
        "input_ranges":MULTI_RANGES,"log_parameters":list(MULTI_LOG_PARAMETERS),
        "B_distribution":"independent discrete uniform integer", "response_time":T_STAR,
        "valid_hard":int(np.isfinite(y[:,:,0]).sum()),
        "valid_soft":int(np.isfinite(y[:,:,1]).sum()),
        "soft_sobol_available":ssi is not None,
        "warning":"Sobol estimates are exploratory at low N; no missing-data imputation.",
        "cache_signature":fingerprint},ensure_ascii=False,indent=2),encoding="utf-8")
    print(f"多参数完成：{total} 个设计求值，{len(compact_memo)} 个不同配置，"
          f"Soft 成功 {np.isfinite(y[:,:,1]).sum()}/{total}。",flush=True)
# ========= 新增结束 =========


def main():
    if not 0 <= T_STAR <= T_END or N_STEP <= 0 or N_MAX < N_MIN + 2 * N_STEP:
        raise ValueError("检查时间范围及 N 的范围（至少测试三个阶数）")
    hard, hard_path = load("hard_exact_constant_boundary")
    soft, soft_path = load("soft_binomial_moment_constant_boundary")
    names = list(BASE)
    single = OUT / "single_parameter"
    single.mkdir(parents=True, exist_ok=True)
    if RUN_DOUBLE:
        if not PAIRS or len(PAIRS) != len(set(tuple(p) for p in PAIRS)):
            raise ValueError("PAIRS 不能为空，也不能包含重复的组合")
        for first, second in PAIRS:
            if first == second or first not in BASE or second not in BASE:
                raise ValueError(f"无效的双参数组合: {(first, second)}")
            for name in (first, second):
                if name not in PAIR_GRIDS:
                    raise ValueError(f"PAIR_GRIDS 缺少参数: {name}")
                values = np.asarray(PAIR_GRIDS[name], float)
                if len(values) < 2 or not np.all(np.isfinite(values)) or np.any(values <= 0) or np.any(np.diff(values) <= 0):
                    raise ValueError(f"{name} 的扫描取值应为至少两个递增的正数")
                if name == "B" and np.any(values != np.floor(values)):
                    raise ValueError("B 是整数阈值，不能使用非整数取值")
    t = np.unique(np.r_[np.linspace(0, min(10, T_END), 201),
                        np.linspace(min(10, T_END), T_END, 381), T_STAR])
    settings = dict(gamma=GAMMA, n=[N_MIN, N_MAX, N_STEP], tol=TOL, t=t.tolist(), init=INITIAL,
                    policy="start_at_B_two_consecutive_v3")
    signature = hashlib.sha256(json.dumps(settings, sort_keys=True).encode() +
                               hard_path.read_bytes() + soft_path.read_bytes()).hexdigest()[:12]
    cache_root = OUT / "calculation_cache"
    # 按名称索引已有文件，便于跨单参数/双参数文件夹复用同一组计算。
    cache_index = {}
    for path in cache_root.rglob("*.npz") if cache_root.exists() else ():
        cache_index.setdefault(path.name, []).append(path)
    memo, memo_file, journal = {}, {}, []

    def compute(params, group):
        key = tuple(float(params[k]) for k in names)
        if int(params["B"]) != params["B"] or params["B"] <= INITIAL["n0"]:
            raise ValueError("B 必须是大于初始分子数的正整数，请检查倍数")
        rates = {k: params[k] for k in names if k != "B"}
        b = int(params["B"])
        # 文件夹标注分析类型；文件名保留五个参数的实际值，避免基准值变化造成覆盖。
        folder = cache_root / group
        filename = "__".join(f"{k}={b if k == 'B' else repr(float(params[k]))}" for k in names) + ".npz"
        file = folder / filename
        if key in memo:
            if memo_file[key] != file:
                folder.mkdir(parents=True, exist_ok=True)
                shutil.copy2(memo_file[key], file)
                cache_index.setdefault(filename, []).append(file)
            return memo[key]
        result = None
        for candidate in [file] + cache_index.get(filename, []):
            if not candidate.exists():
                continue
            try:
                with np.load(candidate, allow_pickle=False) as data:
                    if "_signature" not in data.files or str(data["_signature"].item()) != signature:
                        continue
                    result = {k: data[k].item() if data[k].ndim == 0 else data[k]
                              for k in data.files if k != "_signature"}
            except (OSError, ValueError, KeyError):
                continue
            if candidate != file:
                folder.mkdir(parents=True, exist_ok=True)
                shutil.copy2(candidate, file)
                cache_index.setdefault(filename, []).append(file)
            print(f"[缓存] {params}: {result['status']}", flush=True)
            break
        if result is None:
            print(f"[计算] {params}", flush=True)
            hh, hf, _ = hard.hard_survival_exact(t, B=b, **rates, **INITIAL)
            result = dict(hard_h=hh, hard_f=hf, h=np.full_like(t, np.nan),
                          f=np.full_like(t, np.nan), n=-1, status="not_converged", diff=np.nan)
            prev, count = None, 0
            orders = range(max(N_MIN, b), max(N_MAX, b + 10) + 1, N_STEP)
            # 原求解器分别计算 q 和 c；q 下溢为零会给出虚假的“无杀死”结果。
            if np.exp(-GAMMA * (b - 0.5)) == 0:
                result["status"] = "killing_underflow"
                print("  Soft 的 q 系数下溢为零，此组仅保留 Hard。", flush=True)
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
                result.update(n=n, diff=diff)
                print(f"  N={n}, 差异={diff:.3e}, 连续通过={count}", flush=True)
                if count == 2:
                    # 沿用原参数脚本：两者的 f 均由同一输出网格上的 H 数值求导。
                    result.update(h=h, f=-np.gradient(h, t), status="converged")
                    break
                prev = h
            folder.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(file, **result, _signature=signature)
            cache_index.setdefault(filename, []).append(file)
        memo_file[key] = file
        result.update(hard_star=float(np.interp(T_STAR, t, result["hard_h"])),
                      soft_star=float(np.interp(T_STAR, t, result["h"])))
        journal.append({**params, "N": result["n"], "status": result["status"],
                        "last_N_difference": result["diff"], "hard_H_star": result["hard_star"],
                        "soft_H_star": result["soft_star"],
                        "max_H_error": float(np.max(np.abs(result["h"] - result["hard_h"]))),
                        "hard_restricted_mean": float(trapezoid(result["hard_h"], t)),
                        "soft_restricted_mean": float(trapezoid(result["h"], t))})
        memo[key] = result
        write_csv(OUT / "all_run_status.csv", journal)
        return result

    plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix",
                         "font.size": 11, "axes.grid": False, "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42, "svg.fonttype": "none"})
    colors = plt.get_cmap("viridis")(np.linspace(.1, .85, len(FACTORS)))
    legends = [Line2D([], [], color=c, label=fr"${v:g}\times$") for v, c in zip(FACTORS, colors)]
    legends += [Line2D([], [], color="0.2", label="Hard (exact)"),
                Line2D([], [], color="0.2", marker="o", mfc="none", ls="none", label="Binomial approximation")]

    # 1. 单因素：五个参数合在一张图里；H、f 各一张。
    if RUN_SINGLE:
        runs = {name: [compute({**BASE, name: BASE[name] * v}, f"single_parameter/{name}")
                       for v in FACTORS] for name in names}
        for quantity, ylabel, xmax in (("h", r"$H(t)$", T_END), ("f", r"$f_T(t)$", FPT_XMAX)):
            fig, axes = plt.subplots(2, 3, figsize=(11.5, 6.8), layout="constrained")
            for idx, name in enumerate(names):
                ax = axes.flat[idx]
                for result, color in zip(runs[name], colors):
                    ax.plot(t, result["hard_" + quantity], color=color, lw=1.5)
                    # 按 H 的下降进度选点，同时保留均匀时间点，兼顾快慢曲线。
                    levels = [np.argmin(abs(result["hard_h"] - z)) for z in np.linspace(.05, .95, 14)]
                    marks = np.unique(np.r_[levels, np.linspace(0, len(t)-1, 18).astype(int)]).astype(int)
                    ax.plot(t[marks], result[quantity][marks], "o", color=color, mfc="none", ms=4, mew=.9)
                ax.set(xlim=(0, xmax), xlabel="$t$", ylabel=ylabel)
                ax.set_title(fr"({chr(97+idx)}) {symbol(name)} (baseline {BASE[name]:g})", loc="left")
                if quantity == "h":
                    ax.set_ylim(-.02, 1.02)
            axes.flat[-1].set_axis_off()
            axes.flat[-1].legend(handles=legends, frameon=False, loc="center")
            save_fig(fig, single, "survival" if quantity == "h" else "fpt")
    # 双参数：逐组计算；单组仍共用同一密集网格画三种图，缓存跨组复用。
    if RUN_DOUBLE:
        for pair_index, (first, second) in enumerate(PAIRS, 1):
            xs, ys = (np.asarray(PAIR_GRIDS[name], float) for name in (first, second))
            pair = OUT / "two_parameter" / f"{first}_{second}"
            pair.mkdir(parents=True, exist_ok=True)
            print(f"[双参数组合 {pair_index}/{len(PAIRS)}] {first} × {second}，{len(xs)*len(ys)} 组", flush=True)
            hr, sr = np.empty((len(ys), len(xs))), np.empty((len(ys), len(xs)))
            rows = []
            for i, y in enumerate(ys):
                for j, x in enumerate(xs):
                    print(f"[双参数 {i*len(xs)+j+1}/{hr.size}]", flush=True)
                    r = compute({**BASE, first: x, second: y}, f"two_parameter/{first}_{second}")
                    hr[i,j], sr[i,j] = r["hard_star"], r["soft_star"]
                    rows.append({first: x, second: y, "hard_H_star": hr[i,j], "soft_H_star": sr[i,j],
                                 "difference": sr[i,j]-hr[i,j], "N": r["n"], "status": r["status"]})
            write_csv(pair / "responses.csv", rows)
            xx, yy = np.meshgrid(xs, ys)
            tables = (hr, sr, sr-hr)
            finite_error = abs(tables[2][np.isfinite(tables[2])])
            limit = max(float(finite_error.max()) if finite_error.size else 0., 1e-12)
            titles = ("(a) Hard (exact)", "(b) Binomial approximation", "(c) Soft − Hard")
            response = fr"$H({T_STAR:g})$"
            selected = np.unique(np.linspace(0, len(ys)-1, min(INTERACTION_LINES, len(ys))).astype(int))
            line_colors = plt.get_cmap("viridis")(np.linspace(.1, .85, len(selected)))
            for kind in ("surfaces", "heatmaps", "interaction"):
                fig = plt.figure(figsize=(13, 4.3), layout="constrained")
                for k, table in enumerate(tables):
                    ax = fig.add_subplot(1, 3, k+1, projection="3d" if kind == "surfaces" else None)
                    lo, hi = (-limit, limit) if k == 2 else (0, 1)
                    cmap = "coolwarm" if k == 2 else "viridis"
                    label = fr"$\Delta H({T_STAR:g})$" if k == 2 else response
                    if kind == "surfaces":
                        image = ax.plot_surface(xx, yy, table, cmap=cmap, vmin=lo, vmax=hi,
                                                rcount=len(ys), ccount=len(xs), linewidth=0, antialiased=True)
                        ax.set_zlim(lo, hi)
                        ax.view_init(elev=25, azim=-130)
                        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
                            axis.pane.fill = False
                    elif kind == "heatmaps":
                        image = ax.pcolormesh(xs, ys, np.ma.masked_invalid(table), shading="nearest",
                                              cmap=cmap, vmin=lo, vmax=hi, rasterized=False)
                    else:
                        for row, color in zip(selected, line_colors):
                            ax.plot(xs, table[row], color=color, lw=1.6,
                                    label=fr"{symbol(second)}$={ys[row]:g}$")
                        ax.set(ylim=(lo, hi), ylabel=label)
                        if k == 0:
                            ax.legend(frameon=False, fontsize=9)
                        if k == 2:
                            ax.axhline(0, color="0.5", lw=.6)
                    ax.set(xlabel=symbol(first), title=titles[k])
                    ax.grid(False)
                    if kind != "interaction":
                        ax.set_ylabel(symbol(second))
                        fig.colorbar(image, ax=ax, shrink=.65, pad=.08, label=label)
                    if not np.any(np.isfinite(table)):
                        put_text = ax.text2D if kind == "surfaces" else ax.text
                        put_text(.5, .5, "Unavailable: see run status", transform=ax.transAxes, ha="center")
                fig.suptitle(fr"Response: {response}; $\gamma={GAMMA:g}$")
                save_fig(fig, pair, kind)
    (OUT / "settings.json").write_text(json.dumps(dict(settings, baseline=BASE, factors=FACTORS,
        pairs=PAIRS, pair_grids={k: np.asarray(v, float).tolist() for k, v in PAIR_GRIDS.items()},
        t_star=T_STAR), indent=2), encoding="utf-8")
    if RUN_MULTI:
        if T_STAR <= 0:
            raise ValueError("多参数分析需要 T_STAR > 0")
        for key,(lo,hi) in MULTI_RANGES.items():
            if key not in names or lo<=0 or hi<=lo:
                raise ValueError(f"无效的多参数取值范围: {key} = {(lo,hi)}")
        lo,hi=MULTI_RANGES["B"]
        if int(lo)!=lo or int(hi)!=hi:
            raise ValueError("B 必须采用整数范围")
        run_multi(hard,soft,hard_path,soft_path,names,memo,settings,OUT)
    print(f"完成：{len(journal)} 个不同参数组合（单/双参数）。结果：{OUT}", flush=True)


if __name__ == "__main__":
    main()
