from __future__ import annotations
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from scipy.special import roots_jacobi, roots_legendre, eval_jacobi, gammaln, betaln
from scipy.sparse import lil_matrix, diags
from scipy.sparse.linalg import splu
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ===== 可修改参数 =====
GAMMAS = [2., 8., 64.]
M = 12                           # 二项矩截断阶数，不是分子数上限
K = 64                           # Jacobi 基函数个数
ODE_DT = 0.00125
MC_DT = 0.001
MC_PATHS = 120000
BATCH_SIZE = 5000
OBSERVATION_DT = 0.05
HISTOGRAM_BINS = 40
SEED = 20263009
COLORS = ["#4C78A8", "#59A14F", "#E15759"]
HERE = Path(__file__).resolve().parent

plt.rcParams.update({
    "font.family": "STIXGeneral", "mathtext.fontset": "stix",
    "font.size": 8, "axes.labelsize": 9, "axes.titlesize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.fontsize": 7, "legend.frameon": False,
    "axes.linewidth": 0.7, "lines.linewidth": 1.5,
    "xtick.direction": "in", "ytick.direction": "in",
    "axes.grid": False, "pdf.fonttype": 42, "ps.fonttype": 42,
    "svg.fonttype": "path",
})

@dataclass
class Model:
    beta_minus: float; beta_plus: float; kappa: float; theta: float; sigma: float
    k1: float; k2: float; mu: float; delta: float; N0: int; p_on0: float; T: float
    initial_kind: str; initial_lo: float | None; initial_hi: float | None
    @property
    def a(self): return 2*self.kappa*self.theta/self.sigma**2
    @property
    def b(self): return 2*self.kappa*(1-self.theta)/self.sigma**2

def observation_times(p,s):
    count=max(1,int(round(p.T/s["observation_dt"])))
    return np.linspace(0,p.T,count+1)

def aligned_step(obs,desired):
    stride=max(1,int(math.ceil((obs[1]-obs[0])/desired-1e-12)))
    return (obs[1]-obs[0])/stride,stride

def phi_matrix(p,z,K):
    """Normalized Jacobi P_n^(b-1,a-1) in L2(Beta(a,b))."""
    z=np.asarray(z); n=np.arange(1,K,dtype=float)
    norm=np.ones(K)
    norm[1:]=np.exp(gammaln(n+p.a)+gammaln(n+p.b)-np.log(2*n+p.a+p.b-1)
                   -gammaln(n+1)-gammaln(n+p.a+p.b-1)-betaln(p.a,p.b))
    if not np.all(np.isfinite(norm)) or np.any(norm<=0): raise FloatingPointError("Jacobi基底归一化超出当前浮点精度")
    return np.column_stack([eval_jacobi(i,p.b-1,p.a-1,z)/np.sqrt(norm[i]) for i in range(K)])

def initial_coefficients(p,K):
    if p.initial_kind=="stationary":
        u=np.zeros(K);u[0]=1;return u
    z,w=roots_legendre(max(128,2*K))
    x=(p.initial_lo+p.initial_hi)/2+(p.initial_hi-p.initial_lo)*z/2
    y=2*(x-p.beta_minus)/(p.beta_plus-p.beta_minus)-1
    return phi_matrix(p,y,K).T@(w/2)

def initial_vector(p,M,K):
    """Exact finite transform of binomial(N0,m) initial moments, also if M<N0."""
    u=initial_coefficients(p,K);z=np.zeros((2,M+1,K))
    for n in range(M+1):
        coefficient=sum((-1)**(m-n)*math.comb(m,n)*math.comb(p.N0,m)
                        for m in range(n,min(M,p.N0)+1))
        z[0,n]=coefficient*(1-p.p_on0)*u;z[1,n]=coefficient*p.p_on0*u
    return z.ravel()

def scaled_matrix(p,gamma,M,K,nq=None):
    nq=int(nq or max(128,2*K+24))
    nodes,w=roots_jacobi(nq,p.b-1,p.a-1);w=w/w.sum()
    x=(p.beta_minus+p.beta_plus)/2+(p.beta_plus-p.beta_minus)*nodes/2
    phi=phi_matrix(p,nodes,K)
    Q=phi.T@((w*gamma*np.exp(-gamma*(x-p.beta_minus)))[:,None]*phi)
    Q=(Q+Q.T)/2
    lam=.5*p.sigma**2*np.arange(K)*(np.arange(K)+p.a+p.b-1)
    exponent=gamma*(np.arange(M+1)-p.beta_minus)
    if np.max(exponent)>700: raise FloatingPointError("gamma*(M-beta_minus)>700：当前逐行缩放的双精度范围不足")
    dc=np.exp(-np.maximum(0,exponent));d=np.tile(np.repeat(dc,K),2)
    J=lil_matrix((len(d),len(d)));I=np.eye(K)
    def sl(i,n): return slice((i*(M+1)+n)*K,(i*(M+1)+n+1)*K)
    for i in [0,1]:
        for n in range(M+1):
            dn=dc[n];factor=np.exp(min(0.,exponent[n]))
            J[sl(i,n),sl(i,n)]=-factor*Q-np.diag(dn*(lam+(p.k2 if i==0 else p.k1)+p.delta*n+(p.mu if i==1 else 0)))
            J[sl(i,n),sl(1-i,n)]=dn*(p.k1 if i==0 else p.k2)*I
            if n<M: J[sl(i,n),sl(i,n+1)]=dn*p.delta*(n+1)*I
            if i==1:
                if n>0: J[sl(i,n),sl(i,n-1)]=dn*p.mu*I
                correction=-p.mu*(-1)**(M+1-n)*math.comb(M+1,n)*dn
                J[sl(i,n),sl(i,M)]=J[sl(i,n),sl(i,M)]+correction*I
    return J.tocsc(),d,dict(quadrature=nq,gram_error=float(np.max(np.abs(phi.T@(w[:,None]*phi)-np.eye(K)))),
                           Q_min_eigenvalue=float(np.linalg.eigvalsh(Q)[0]))

# def spectral_solve(p,s,g,M,K,step,nq=None):
#     J,d,meta=scaled_matrix(p,g,M,K,nq)
#     obs=observation_times(p,s);h,stride=aligned_step(obs,step)
#     alpha=1-1/np.sqrt(2);tau=2-np.sqrt(2)
#     A=1/(tau*(2-tau));B=(1-tau)**2/(tau*(2-tau))
#     lhs=diags(d)-alpha*h*J;lu=splu(lhs.tocsc())
#     z=initial_vector(p,M,K);H=[float(z.reshape(2,M+1,K)[:,:,0].sum())]
#     residual=0.;maximum=float(np.max(np.abs(z)))
#     for ti in range(1,len(obs)):
#         for j in range(stride):
#             stage=lu.solve(d*z+alpha*h*(J@z));rhs=d*(A*stage-B*z);z=lu.solve(rhs)
#             if j==0 and ti in [1,len(obs)-1]:
#                 residual=max(residual,float(np.linalg.norm(lhs@z-rhs,np.inf)/(np.linalg.norm(rhs,np.inf)+1e-300)))
#         H.append(float(z.reshape(2,M+1,K)[:,:,0].sum()));maximum=max(maximum,float(np.max(np.abs(z))))
#         if not np.all(np.isfinite(z)): raise FloatingPointError("系数出现NaN/inf；不丢弃、不替换该结果")
#     meta.update(dt=h,linear_relative_residual=residual,max_abs_coefficient=maximum)
#     return dict(t=obs,H=np.asarray(H),meta=np.array(json.dumps(meta)))

from scipy.integrate import solve_ivp

def spectral_solve(p, s, g, M, K, step, nq=None):
    # 取得你的方程：D z' = J z
    J, d, meta = scaled_matrix(p, g, M, K, nq)

    # 取得已知初值
    z0 = initial_vector(p, M, K)

    # 需要输出结果的时刻
    obs = observation_times(p, s)

    # 直接定义微分方程的导数
    def derivative(t, z):
        return (J @ z) / d

    # 导数对 z 的系数矩阵，提供给求解器以提高效率
    jacobian = (diags(1.0 / d) @ J).tocsc()

    # 将微分方程、初值和时间区间交给现成求解器
    result = solve_ivp(
        fun=derivative,
        t_span=(float(obs[0]), float(obs[-1])),
        y0=z0,
        t_eval=obs,
        method="BDF",
        jac=jacobian,
        max_step=step,
        rtol=1e-7,
        atol=1e-10,
    )

    if not result.success:
        raise RuntimeError("微分方程求解失败：" + result.message)

    if not np.all(np.isfinite(result.y)):
        raise FloatingPointError("求解结果出现 NaN/inf")

    # 每个时刻提取生存概率
    coefficients = result.y.reshape(
        2, M + 1, K, len(result.t)
    )
    H = coefficients[:, :, 0, :].sum(axis=(0, 1))

    meta.update(
        solver="BDF",
        max_step=step,
        rtol=1e-7,
        atol=1e-10,
        max_abs_coefficient=float(np.max(np.abs(result.y))),
    )

    return dict(
        t=result.t,
        H=H,
        meta=np.array(json.dumps(meta)),
    )

# def jacobi_step(p,y,h,rng):
#     # if np.isclose(p.a,1.,rtol=1e-12,atol=1e-12) and np.isclose(p.b,1.,rtol=1e-12,atol=1e-12):
#     #     z=2*y-1;g1=rng.standard_normal(len(y));g2=rng.standard_normal(len(y))
#     #     r=p.sigma*np.sqrt(h)*np.sqrt(g1*g1+g2*g2)
#     #     znew=z*np.cos(r)+np.sqrt(np.maximum(0,1-z*z))*p.sigma*np.sqrt(h)*g1*np.sinc(r/np.pi)
#     #     return (np.clip(znew,-1,1)+1)/2
#     # General a,b>0: bounded weak generator-consistent drift/diffusion splitting.
#     # Pure diffusion has mean y and variance y(1-y)*(1-exp(-sigma^2 h)).
#     # A Beta draw matches those two moments. This is NOT exact transition sampling.
#     half=np.exp(-p.kappa*h/2)
#     q=p.theta+(y-p.theta)*half
#     concentration=1/np.expm1(p.sigma**2*h)
#     aa=q*concentration;bb=(1-q)*concentration
#     if np.any(aa<=0) or np.any(bb<=0): raise FloatingPointError("Jacobi split Beta形状参数落入浮点端点；需调整时间步/精度")
#     q=rng.beta(aa,bb)
#     return p.theta+(q-p.theta)*half

def jacobi_step(p, y, h, rng):
    # 布朗运动增量：ΔW ~ N(0, h)
    dW = np.sqrt(h) * rng.standard_normal(len(y))

    # 漂移增量：κ(θ-y)h
    drift = p.kappa * (p.theta - y) * h

    # 随机扩散增量：σ√[y(1-y)]ΔW
    diffusion = (
        p.sigma
        * np.sqrt(np.maximum(y * (1 - y), 0.0))
        * dW
    )

    # 两种增量直接加到当前位置
    y_new = y + drift + diffusion

    # 防止离散计算使边界位置跑出 [0,1]
    return np.clip(y_new, 0.0, 1.0)

def safe_wait(rate,rng):
    wait=np.full(len(rate),np.inf);positive=rate>0
    wait[positive]=rng.exponential(1/rate[positive]);return wait



# def mc_chunk(p, s, paths, desired_dt, seed_words):
#     rng = np.random.default_rng(np.random.SeedSequence(seed_words))
#     obs = observation_times(p, s)
#     h, stride = aligned_step(obs, desired_dt)
#     width = p.beta_plus - p.beta_minus

#     # 初始边界、分子数、启动子状态
#     if p.initial_kind == "stationary":
#         y = rng.beta(p.a, p.b, paths)
#     else:
#         y = (
#             rng.uniform(p.initial_lo, p.initial_hi, paths)
#             - p.beta_minus
#         ) / width

#     N = np.full(paths, p.N0, dtype=np.int64)
#     S = (rng.random(paths) < p.p_on0).astype(np.int8)

#     def total_rate():
#         return (
#             np.where(S == 1, p.k1, p.k2)
#             + p.mu * S
#             + p.delta * N
#         )

#     event = safe_wait(total_rate(), rng)

#     # 只记录 hard 生存权重和首次命中时间
#     hard = np.ones(paths)
#     hit = np.full(paths, np.inf)

#     initial_hit = N >= p.beta_minus + width * y
#     hard[initial_hit] = 0
#     hit[initial_hit] = 0

#     hard_sums = [float(hard.sum())]
#     allids = np.arange(paths)

#     # 处理分子数不变的子区间，保留原来的桥跨越近似
#     def segment(ids, start, end, b0, b1, n):
#         if len(ids) == 0:
#             return

#         duration = end - start
#         crossed = (n >= b0) | (n >= b1)
#         prob = crossed.astype(float)

#         variance = (
#             p.sigma**2
#             * (n - p.beta_minus)
#             * (p.beta_plus - n)
#         )
#         eligible = (~crossed) & (variance > 0) & (duration > 0)

#         prob[eligible] = np.exp(
#             -2 * (b0[eligible] - n[eligible])
#             * (b1[eligible] - n[eligible])
#             / (variance[eligible] * duration[eligible])
#         )

#         hard[ids] *= 1 - prob

#         candidates = np.flatnonzero(~np.isfinite(hit[ids]))
#         selected = candidates[
#             rng.random(len(candidates)) < prob[candidates]
#         ]
#         hit[ids[selected]] = (start[selected] + end[selected]) / 2

#     # 模拟边界和基因反应
#     for step in range((len(obs) - 1) * stride):
#         t0 = step * h
#         t1 = (step + 1) * h

#         ynew = jacobi_step(p, y, h, rng)
#         bstart = p.beta_minus + width * y
#         bend = p.beta_minus + width * ynew
#         local = np.full(paths, t0)

#         while True:
#             ids = np.flatnonzero(event < t1)
#             if len(ids) == 0:
#                 break

#             e = event[ids]

#             b0 = (
#                 bstart[ids]
#                 + (bend[ids] - bstart[ids]) * (local[ids] - t0) / h
#             )
#             b1 = (
#                 bstart[ids]
#                 + (bend[ids] - bstart[ids]) * (e - t0) / h
#             )
#             segment(ids, local[ids], e, b0, b1, N[ids])

#             # SSA：选择并执行反应
#             switch = np.where(S[ids] == 1, p.k1, p.k2)
#             birth = p.mu * S[ids]
#             death = p.delta * N[ids]

#             choice = rng.random(len(ids)) * (switch + birth + death)
#             sw = choice < switch
#             bi = (choice >= switch) & (choice < switch + birth)
#             de = choice >= switch + birth

#             S[ids[sw]] = 1 - S[ids[sw]]
#             N[ids[bi]] += 1
#             N[ids[de]] -= 1

#             # 检查反应跳跃是否造成命中
#             crossing = N[ids] >= b1
#             new = crossing & ~np.isfinite(hit[ids])
#             hit[ids[new]] = e[new]
#             hard[ids[crossing]] = 0

#             local[ids] = e
#             event[ids] = e + safe_wait(total_rate()[ids], rng)

#         # 处理本步剩余的无反应区间
#         b0 = bstart + (bend - bstart) * (local - t0) / h
#         segment(allids, local, np.full(paths, t1), b0, bend, N)
#         y = ynew

#         if (step + 1) % stride == 0:
#             hard_sums.append(float(hard.sum()))

#     return dict(
#         paths=np.array(paths),
#         t=obs,
#         dt=np.array(h),
#         sum_hard=np.asarray(hard_sums),
#         first_hit=hit,
#     )

def mc_chunk(p, s, paths, desired_dt, seed_words):
    rng = np.random.default_rng(np.random.SeedSequence(seed_words))

    obs = observation_times(p, s)
    h, stride = aligned_step(obs, desired_dt)
    width = p.beta_plus - p.beta_minus

    # 1. 初始边界
    if p.initial_kind == "stationary":
        y = rng.beta(p.a, p.b, paths)
    else:
        y = (
            rng.uniform(p.initial_lo, p.initial_hi, paths)
            - p.beta_minus
        ) / width

    # 2. 初始分子数和启动子状态
    N = np.full(paths, p.N0, dtype=np.int64)
    S = (rng.random(paths) < p.p_on0).astype(np.int8)

    def total_rate():
        return (
            np.where(S == 1, p.k1, p.k2)
            + p.mu * S
            + p.delta * N
        )

    # 3. 首次反应时间、存活标记、首次命中时间
    event = safe_wait(total_rate(), rng)
    hard = np.ones(paths, dtype=np.int8)
    hit = np.full(paths, np.inf)

    # 检查初始时刻是否已经越界
    crossing = N >= p.beta_minus + width * y
    hard[crossing] = 0
    hit[crossing] = 0.0

    hard_sums = [int(hard.sum())]

    # 4. 逐步模拟
    for step in range((len(obs) - 1) * stride):
        t0 = step * h
        t1 = (step + 1) * h

        ynew = jacobi_step(p, y, h, rng)
        bstart = p.beta_minus + width * y
        bend = p.beta_minus + width * ynew

        # 处理本时间步内的全部反应
        while True:
            ids = np.flatnonzero(event < t1)
            if len(ids) == 0:
                break

            e = event[ids]

            # 线性插值得到反应时刻的边界
            beta = (
                bstart[ids]
                + (bend[ids] - bstart[ids]) * (e - t0) / h
            )

            # 检查反应前是否已经越界
            new = (hard[ids] == 1) & (N[ids] >= beta)
            hit[ids[new]] = e[new]
            hard[ids[new]] = 0

            # SSA：计算三种反应的速率
            switch = np.where(S[ids] == 1, p.k1, p.k2)
            birth = p.mu * S[ids]
            death = p.delta * N[ids]

            # 按速率比例选择反应
            choice = rng.random(len(ids)) * (switch + birth + death)
            sw = choice < switch
            bi = (choice >= switch) & (choice < switch + birth)
            de = choice >= switch + birth

            # 执行反应
            S[ids[sw]] = 1 - S[ids[sw]]
            N[ids[bi]] += 1
            N[ids[de]] -= 1

            # 检查反应后是否越界
            new = (hard[ids] == 1) & (N[ids] >= beta)
            hit[ids[new]] = e[new]
            hard[ids[new]] = 0

            # 采样下一次反应时间
            event[ids] = e + safe_wait(total_rate()[ids], rng)

        # 检查本时间步终点是否越界
        new = (hard == 1) & (N >= bend)
        hit[new] = t1
        hard[new] = 0

        y = ynew

        # 到达输出时刻，记录仍存活的轨迹数量
        if (step + 1) % stride == 0:
            hard_sums.append(int(hard.sum()))

    return dict(
        paths=np.array(paths),
        t=obs,
        dt=np.array(h),
        sum_hard=np.asarray(hard_sums),
        first_hit=hit,
    )


def draw_figure(curves, times, hard_H, hits):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6))
    axes[0].plot(times, hard_H, color="0.45", linestyle=":", linewidth=2,
                 label="Hard boundary (MC)")
    edges = np.linspace(times[0], times[-1], HISTOGRAM_BINS + 1)
    counts, _ = np.histogram(hits[np.isfinite(hits)], bins=edges)
    # 所有轨迹均计入分母，包括观察期内没有越界的轨迹。
    density = counts / (len(hits) * np.diff(edges))
    axes[1].stairs(density, edges, baseline=None, color="0.45",
                   linestyle=":", linewidth=2, label="Hard boundary (MC)")
    for (gamma, result), color in zip(curves.items(), COLORS):
        t, H = result["t"], result["H"]
        # 区间平均密度，在区间中点绘制；不平滑、不截零。
        f = -np.diff(H) / np.diff(t)
        mid = (t[:-1] + t[1:]) / 2
        axes[0].plot(t, H, color=color, label=rf"$\gamma={gamma:g}$")
        axes[1].plot(mid, f, color=color, label=rf"$\gamma={gamma:g}$")
        print(f"gamma={gamma:g}: 与硬MC最大差={np.max(np.abs(H-hard_H)):.6g}", flush=True)
        if np.any(f < -1e-8) or H.min() < -1e-8 or H.max() > 1+1e-8:
            print("注意：这条曲线出现负密度或超出生存概率范围，请增加分辨率。", flush=True)
    axes[0].set_ylabel(r"$H(t)$")
    axes[1].set_ylabel(r"$f_{\mathrm{FPT}}(t)$")
    axes[0].set_ylim(0, 1)
    axes[1].set_ylim(bottom=0)
    for panel, ax in zip([
        "(a) Binomial moments survival probability",
        "(b) Binomial moment FPT density",
    ], axes):
        ax.set_title(panel, loc="left", x=-0.06, y=1.04, fontsize=11)
        ax.set_xlabel(r"$t$")
        ax.set_xlim(times[0], times[-1])
        ax.tick_params(which="both", top=False, right=False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[1].legend(loc="upper right")
    fig.tight_layout()
    name = "jacobi_fast_boundary"
    for extension in ["svg", "pdf", "png"]:
        path = HERE / f"{name}.{extension}"
        fig.savefig(path, dpi=900, bbox_inches="tight")
        print("已保存：", path, flush=True)
    plt.close(fig)


def main():
    p = Model(beta_minus=2.75, beta_plus=3.25, kappa=1., theta=.5, sigma=1.,
              k1=.5, k2=1.2, mu=5., delta=.8, N0=0, p_on0=1., T=4.,
              initial_kind="stationary", initial_lo=None, initial_hi=None)
    m, k, ode_dt, mc_dt, paths = M, K, ODE_DT, MC_DT, MC_PATHS
    if len(GAMMAS) > len(COLORS):
        raise ValueError("请为每个 gamma 配置一种 COLORS 颜色。")
    s = dict(gammas=GAMMAS, observation_dt=OBSERVATION_DT)
    print(f"二项矩计算：M={m}, K={k}, 时间步={ode_dt}", flush=True)
    curves = {}
    for gamma in GAMMAS:
        curves[gamma] = spectral_solve(p, s, gamma, m, k, ode_dt)
        print(f"已完成 gamma={gamma:g}", flush=True)
    times = observation_times(p, s)
    hard_sum = np.zeros(len(times))
    
    hit_chunks = []
    print(f"开始硬边界及软边界MC：{paths}条轨迹，时间步={mc_dt}", flush=True)
    for chunk, start in enumerate(range(0, paths, BATCH_SIZE)):
        count = min(BATCH_SIZE, paths-start)
        # 与原批量程序 fine 阶段相同的随机种子规则。
        result = mc_chunk(p, s, count, mc_dt, [SEED, 1, chunk])
        hard_sum += result["sum_hard"]
        
        hit_chunks.append(result["first_hit"])
        print(f"MC进度：{start+count}/{paths}", flush=True)
    hard_H = hard_sum / paths

    draw_figure(curves, times, hard_H, np.concatenate(hit_chunks))


if __name__ == "__main__":
    main()
