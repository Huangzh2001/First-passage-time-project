"""Jacobi 三层近似批量实验；单文件、参数可添加、原子断点、现场计算图片。

运行: python jacobi_batch_resume.py
快速流程检查: python jacobi_batch_resume.py --quick
列出参数: python jacobi_batch_resume.py --list
自定义JSON: python jacobi_batch_resume.py --config my_cases.json

K=Jacobi基函数数；M=原文二项矩截断，绝不是分子数硬上限。
有限实验给出数值诊断，不证明无穷极限或所有合法参数下的定理。
"""
from __future__ import annotations
import os
os.environ["OPENBLAS_NUM_THREADS"]="1"
os.environ["OMP_NUM_THREADS"]="1"
os.environ["MKL_NUM_THREADS"]="1"
import argparse
import ast
import contextlib
import datetime as dtm
import hashlib
import json
import math
import sys
import time
import traceback
import uuid
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np
import pandas as pd
import scipy
from scipy.special import roots_jacobi, roots_legendre, eval_jacobi, gammaln, betaln
from scipy.sparse import lil_matrix, csc_matrix, diags
from scipy.sparse.linalg import splu
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ===== 用户主要修改这里：可直接在 PARAMETER_SETS 中追加字典 =====
# 稳定输出根目录：重启时仍使用这里，才会找到断点。
OUTPUT_ROOT = Path(__file__).resolve().parent / "jacobi_batch_results"
# Windows也可以改为 Path(r"D:\Downloads\OneDrive\桌面\hhhhhhhhhhh\batch_results")

MODEL_DEFAULTS = dict(beta_minus=2.75,beta_plus=3.25,kappa=1.,theta=.5,sigma=1.,
                      k1=.7,k2=.8,mu=2.,delta=.5,N0=0,p_on0=1.,T=6.,
                      initial_kind="stationary",initial_lo=None,initial_hi=None)
PARAMETER_SETS = [
    dict(id="P01_baseline",label="原参数、平稳初值",seed=20261009),
    dict(id="P02_slow_boundary",label="较慢的Jacobi边界",seed=20262009,
         kappa=.16,sigma=.4,T=6.),
    dict(id="P03_fast_reactions",label="较快反应",seed=20263009,
         mu=5.,delta=.8,k1=.5,k2=1.2,T=4.),
    dict(id="P04_asymmetric",label="不对称Jacobi边界",seed=20264009,
         kappa=1.2,theta=.7,sigma=.8,T=6.),
    dict(id="P05_near_hit_peak",label="接近越界的非平稳集中初值",seed=20265009,
         N0=3,T=2.,initial_kind="uniform_window",initial_lo=3.005,initial_hi=3.015,
         M_values=[3,4,6,8,12]),
]
RUN_SETTINGS = dict(gammas=[2.,4.,8.,16.,32.,64.],K_values=[4,8,12,20,32,48,64],
                    M_values=[2,3,4,6,8,10,12],observation_dt=.05,ode_dt=.0025,
                    mc_coarse_dt=.002,mc_fine_dt=.001,coarse_paths=60000,
                    fine_paths=120000,mc_batch_size=5000)
# 这些是有限实验的诊断阈值；不是数学定理的假设或严格误差上界。
TOLERANCES = dict(K_step=1e-5,M_step=1e-5,ODE_step=1e-6,quadrature=1e-6,
                  soft_hard=.005,soft_MC_excess=.001,MC_step=.005,range=1e-5)
QUICK_SETTINGS = dict(gammas=[2.,8.,32.],K_values=[4,8,16],M_values=[3,6,8],
                      ode_dt=.005,mc_coarse_dt=.005,mc_fine_dt=.0025,
                      coarse_paths=1000,fine_paths=2000,mc_batch_size=1000)
# ===== 以下是计算、保存和汇总代码 =====

@dataclass
class Model:
    beta_minus: float; beta_plus: float; kappa: float; theta: float; sigma: float
    k1: float; k2: float; mu: float; delta: float; N0: int; p_on0: float; T: float
    initial_kind: str; initial_lo: float | None; initial_hi: float | None
    @property
    def a(self): return 2*self.kappa*self.theta/self.sigma**2
    @property
    def b(self): return 2*self.kappa*(1-self.theta)/self.sigma**2

def utc(): return dtm.datetime.now(dtm.timezone.utc).isoformat()
def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def json_hash(obj): return hashlib.sha256(json.dumps(obj,sort_keys=True,ensure_ascii=False,allow_nan=False).encode()).hexdigest()
def atomic_json(path,obj):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+"."+uuid.uuid4().hex+".tmp")
    with temp.open("w",encoding="utf-8") as f:
        json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
    os.replace(temp,path)
def atomic_csv(path,frame):
    path=Path(path);temp=path.with_name(path.name+"."+uuid.uuid4().hex+".tmp")
    frame.to_csv(temp,index=False,encoding="utf-8-sig");os.replace(temp,path)
def atomic_npz(path,data):
    path=Path(path);temp=path.with_name(path.name+"."+uuid.uuid4().hex+".tmp")
    with temp.open("wb") as f:
        np.savez_compressed(f,**data);f.flush();os.fsync(f.fileno())
    os.replace(temp,path)

@contextlib.contextmanager
def process_lock(root):
    """OS releases this lock after a crash; no stale-PID guessing on Windows."""
    f=(root/"run.lock").open("a+b")
    if f.seek(0,2)==0: f.write(b"0");f.flush()
    f.seek(0)
    try:
        if os.name=="nt":
            import msvcrt
            msvcrt.locking(f.fileno(),msvcrt.LK_NBLCK,1)
        else:
            import fcntl
            fcntl.flock(f.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    except OSError:
        f.close();raise RuntimeError("该输出目录已有程序运行。请结束另一进程或指定不同输出目录。")
    try: yield
    finally:
        f.seek(0)
        if os.name=="nt":
            import msvcrt
            msvcrt.locking(f.fileno(),msvcrt.LK_UNLCK,1)
        else:
            import fcntl
            fcntl.flock(f.fileno(),fcntl.LOCK_UN)
        f.close()

class Cache:
    def __init__(self,folder):
        self.folder=Path(folder);self.folder.mkdir(parents=True,exist_ok=True)
        self.new=0;self.hits=0;self.failures=[]
    def task(self,kind,key,compute):
        token=kind+"_"+json_hash(key)[:20]
        path=self.folder/(token+".npz");meta_path=self.folder/(token+".json")
        try:
            meta=json.loads(meta_path.read_text(encoding="utf-8"))
            if meta["status"]=="complete" and meta["key"]==key and digest(path)==meta["sha256"]:
                with np.load(path,allow_pickle=False) as z: data={k:z[k].copy() for k in z.files}
                self.hits+=1;return data
        except (OSError,ValueError,KeyError): pass
        try:
            atomic_json(meta_path,dict(status="running",key=key,started=utc()))
            data=compute()
            atomic_npz(path,data)
            atomic_json(meta_path,dict(status="complete",key=key,finished=utc(),sha256=digest(path)))
            self.new+=1;return data
        except KeyboardInterrupt:
            atomic_json(meta_path,dict(status="interrupted",key=key,time=utc()))
            raise
        except Exception as error:
            message=f"{type(error).__name__}: {error}"
            atomic_json(meta_path,dict(status="failed",key=key,time=utc(),error=message,traceback=traceback.format_exc()))
            self.failures.append(dict(kind=kind,key=key,error=message))
            print("任务失败，已保存并继续：",kind,key,message,flush=True)
            return None

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

def spectral_solve(p,s,g,M,K,step,nq=None):
    J,d,meta=scaled_matrix(p,g,M,K,nq)
    obs=observation_times(p,s);h,stride=aligned_step(obs,step)
    alpha=1-1/np.sqrt(2);tau=2-np.sqrt(2)
    A=1/(tau*(2-tau));B=(1-tau)**2/(tau*(2-tau))
    lhs=diags(d)-alpha*h*J;lu=splu(lhs.tocsc())
    z=initial_vector(p,M,K);H=[float(z.reshape(2,M+1,K)[:,:,0].sum())]
    residual=0.;maximum=float(np.max(np.abs(z)))
    for ti in range(1,len(obs)):
        for j in range(stride):
            stage=lu.solve(d*z+alpha*h*(J@z));rhs=d*(A*stage-B*z);z=lu.solve(rhs)
            if j==0 and ti in [1,len(obs)-1]:
                residual=max(residual,float(np.linalg.norm(lhs@z-rhs,np.inf)/(np.linalg.norm(rhs,np.inf)+1e-300)))
        H.append(float(z.reshape(2,M+1,K)[:,:,0].sum()));maximum=max(maximum,float(np.max(np.abs(z))))
        if not np.all(np.isfinite(z)): raise FloatingPointError("系数出现NaN/inf；不丢弃、不替换该结果")
    meta.update(dt=h,linear_relative_residual=residual,max_abs_coefficient=maximum)
    return dict(t=obs,H=np.asarray(H),meta=np.array(json.dumps(meta)))

def original_matrix_check(p):
    """Independent assembly of the manuscript's original binomial moment matrix."""
    g=2.;M=5;K=4
    J,d,_=scaled_matrix(p,g,M,K)
    nodes,w=roots_jacobi(128,p.b-1,p.a-1);w=w/w.sum();phi=phi_matrix(p,nodes,K)
    x=(p.beta_minus+p.beta_plus)/2+(p.beta_plus-p.beta_minus)*nodes/2
    Q=phi.T@((w*g*np.exp(-g*x))[:,None]*phi)
    modes=np.arange(K);L=-np.diag(.5*p.sigma**2*modes*(modes+p.a+p.b-1));I=np.eye(K)
    A=np.zeros(J.shape)
    def sl(i,m): return slice((i*(M+1)+m)*K,(i*(M+1)+m+1)*K)
    for i in [0,1]:
        for m in range(M+1):
            A[sl(i,m),sl(i,m)]=L-((p.k2 if i==0 else p.k1)+p.delta*m)*I
            A[sl(i,m),sl(1-i,m)]=(p.k1 if i==0 else p.k2)*I
            if i==1 and m>0: A[sl(i,m),sl(i,m-1)]=p.mu*I
            for r in range(M-m+1): A[sl(i,m),sl(i,m+r)]-=np.exp(g*m)*math.comb(m+r,m)*np.expm1(g)**r*Q
    T=np.zeros((M+1,M+1));inv=np.zeros_like(T)
    for n in range(M+1):
        for m in range(n,M+1): T[n,m]=(-1)**(m-n)*math.comb(m,n);inv[n,m]=math.comb(m,n)
    T=np.kron(np.eye(2),np.kron(T,I));inv=np.kron(np.eye(2),np.kron(inv,I))
    ref=d[:,None]*(T@A@inv);relative=float(np.max(np.abs(ref-J.toarray()))/max(1,np.max(np.abs(ref))))
    if relative>1e-9: raise AssertionError(f"原方程与计算矩阵不等价，relative={relative}；该组不继续宣称验证收敛")
    return dict(relative=np.array(relative))

def jacobi_step(p,y,h,rng):
    if np.isclose(p.a,1.,rtol=1e-12,atol=1e-12) and np.isclose(p.b,1.,rtol=1e-12,atol=1e-12):
        z=2*y-1;g1=rng.standard_normal(len(y));g2=rng.standard_normal(len(y))
        r=p.sigma*np.sqrt(h)*np.sqrt(g1*g1+g2*g2)
        znew=z*np.cos(r)+np.sqrt(np.maximum(0,1-z*z))*p.sigma*np.sqrt(h)*g1*np.sinc(r/np.pi)
        return (np.clip(znew,-1,1)+1)/2
    # General a,b>0: bounded weak generator-consistent drift/diffusion splitting.
    # Pure diffusion has mean y and variance y(1-y)*(1-exp(-sigma^2 h)).
    # A Beta draw matches those two moments. This is NOT exact transition sampling.
    half=np.exp(-p.kappa*h/2)
    q=p.theta+(y-p.theta)*half
    concentration=1/np.expm1(p.sigma**2*h)
    aa=q*concentration;bb=(1-q)*concentration
    if np.any(aa<=0) or np.any(bb<=0): raise FloatingPointError("Jacobi split Beta形状参数落入浮点端点；需调整时间步/精度")
    q=rng.beta(aa,bb)
    return p.theta+(q-p.theta)*half

def safe_wait(rate,rng):
    wait=np.full(len(rate),np.inf);positive=rate>0
    wait[positive]=rng.exponential(1/rate[positive]);return wait

def mc_chunk(p,s,paths,desired_dt,seed_words):
    rng=np.random.default_rng(np.random.SeedSequence(seed_words))
    obs=observation_times(p,s);h,stride=aligned_step(obs,desired_dt);G=np.asarray(s["gammas"])
    if p.initial_kind=="stationary": y=rng.beta(p.a,p.b,paths)
    else: y=(rng.uniform(p.initial_lo,p.initial_hi,paths)-p.beta_minus)/(p.beta_plus-p.beta_minus)
    initial_y=y.copy();N=np.full(paths,p.N0,dtype=np.int64);S=(rng.random(paths)<p.p_on0).astype(np.int8)
    event=safe_wait(np.where(S==1,p.k1,p.k2)+p.mu*S+p.delta*N,rng)
    hazard=np.zeros((len(G),paths));hard=np.ones(paths)
    hit=np.full(paths,np.inf);hit_endpoint=np.full(paths,np.inf)
    sums_soft=[];squares_soft=[];sums_hard=[];squares_hard=[];sums_bias=[];squares_bias=[]
    counts_hit=[];counts_endpoint=[];sample_y=[y[:32].copy()];sample_N=[N[:32].copy()]
    events=0;allids=np.arange(paths)
    def segment(ids,t0,t1,b0,b1,n):
        duration=t1-t0
        if len(ids)==0:return
        middle=(b0+b1)/2
        for gi,g in enumerate(G):
            r=.5*g*(b1-b0);factor=np.ones(len(ids));mask=np.abs(r)>1e-6
            factor[mask]=np.sinh(r[mask])/r[mask]
            with np.errstate(divide="ignore",over="ignore"):
                log_added=np.log(g)+np.log(duration)+g*(n-middle)+np.log(factor)
                added=np.exp(np.minimum(np.log(1e6),log_added))
            hazard[gi,ids]=np.minimum(1e6,hazard[gi,ids]+added)
        end_hit=(n>=b0)|(n>=b1)
        new=end_hit & ~np.isfinite(hit_endpoint[ids]);hit_endpoint[ids[new]]=np.where(n[new]>=b0[new],t0[new],t1[new])
        prob=np.zeros(len(ids));prob[end_hit]=1
        variance=p.sigma**2*(n-p.beta_minus)*(p.beta_plus-n)
        eligible=~end_hit & (variance>0)&(duration>0)
        prob[eligible]=np.exp(-2*(b0[eligible]-n[eligible])*(b1[eligible]-n[eligible])/(variance[eligible]*duration[eligible]))
        hard[ids]*=(1-prob)
        candidates=np.flatnonzero(~np.isfinite(hit[ids]))
        if len(candidates):
            select=candidates[rng.random(len(candidates))<prob[candidates]]
            hit[ids[select]]=(t0[select]+t1[select])/2
    def record(t):
        soft=np.exp(-hazard);bias=soft-hard
        sums_soft.append(soft.sum(axis=1));squares_soft.append((soft*soft).sum(axis=1))
        sums_hard.append(hard.sum());squares_hard.append(np.dot(hard,hard))
        sums_bias.append(bias.sum(axis=1));squares_bias.append((bias*bias).sum(axis=1))
        counts_hit.append(np.sum(hit>t));counts_endpoint.append(np.sum(hit_endpoint>t))
    record(0.)
    for step in range((len(obs)-1)*stride):
        t0=step*h;t1=(step+1)*h;ynew=jacobi_step(p,y,h,rng)
        bstart=p.beta_minus+(p.beta_plus-p.beta_minus)*y;bend=p.beta_minus+(p.beta_plus-p.beta_minus)*ynew
        local=np.full(paths,t0)
        while True:
            ids=np.flatnonzero(event<t1)
            if not len(ids):break
            e=event[ids]
            b0=bstart[ids]+(bend[ids]-bstart[ids])*(local[ids]-t0)/h
            b1=bstart[ids]+(bend[ids]-bstart[ids])*(e-t0)/h
            segment(ids,local[ids],e,b0,b1,N[ids])
            switch=np.where(S[ids]==1,p.k1,p.k2);birth=p.mu*S[ids];death=p.delta*N[ids]
            choice=rng.random(len(ids))*(switch+birth+death)
            sw=choice<switch;bi=(choice>=switch)&(choice<switch+birth);de=choice>=switch+birth
            S[ids[sw]]=1-S[ids[sw]];N[ids[bi]]+=1;N[ids[de]]-=1
            crossing=N[ids]>=b1
            new=crossing & ~np.isfinite(hit[ids]);hit[ids[new]]=e[new]
            new=crossing & ~np.isfinite(hit_endpoint[ids]);hit_endpoint[ids[new]]=e[new]
            hard[ids[crossing]]=0;local[ids]=e
            event[ids]=e+safe_wait(np.where(S[ids]==1,p.k1,p.k2)+p.mu*S[ids]+p.delta*N[ids],rng)
            events+=len(ids)
        b0=bstart+(bend-bstart)*(local-t0)/h
        segment(allids,local,np.full(paths,t1),b0,bend,N);y=ynew
        if (step+1)%stride==0:
            record(obs[(step+1)//stride]);sample_y.append(y[:32].copy());sample_N.append(N[:32].copy())
    return dict(paths=np.array(paths),t=obs,dt=np.array(h),sum_soft=np.asarray(sums_soft).T,
                square_soft=np.asarray(squares_soft).T,sum_hard=np.asarray(sums_hard),square_hard=np.asarray(squares_hard),
                sum_bias=np.asarray(sums_bias).T,square_bias=np.asarray(squares_bias).T,
                count_hit=np.asarray(counts_hit),count_endpoint=np.asarray(counts_endpoint),
                first_hit=hit,first_hit_endpoint=hit_endpoint,final_soft=np.exp(-hazard),final_hard=hard,
                initial_y=initial_y,final_y=y,final_N=N,final_S=S,sample_y=np.asarray(sample_y),sample_N=np.asarray(sample_N),
                reaction_events=np.array(events),seed_words=np.asarray(seed_words))

def aggregate_mc(chunks,gammas):
    if not chunks or any(c is None for c in chunks): return None
    total=sum(int(c["paths"]) for c in chunks);t=chunks[0]["t"]
    def stats(key):
        sm=sum(c["sum_"+key] for c in chunks);sq=sum(c["square_"+key] for c in chunks)
        mean=sm/total;var=np.maximum(0,(sq-sm*sm/total)/(total-1));return mean,np.sqrt(var/total)
    soft,se_soft=stats("soft");hard,se_hard=stats("hard");bias,se_bias=stats("bias")
    endpoint=sum(c["count_endpoint"] for c in chunks)/total;hit=sum(c["count_hit"] for c in chunks)/total
    rows=[]
    for gi,g in enumerate(gammas):
        for ti,x in enumerate(t): rows.append(dict(gamma=g,t=x,H_soft=soft[gi,ti],SE_soft=se_soft[gi,ti],
            H_hard=hard[ti],SE_hard=se_hard[ti],soft_minus_hard=bias[gi,ti],SE_bias=se_bias[gi,ti],
            H_hard_sampled=hit[ti],H_hard_endpoints=endpoint[ti],paths=total,dt=float(chunks[0]["dt"])))
    return pd.DataFrame(rows)

def legal_case(raw,settings):
    allowed=set(MODEL_DEFAULTS)|{"id","label","seed","gammas","K_values","M_values","settings"}
    extra=set(raw)-allowed
    if extra: raise ValueError("未知参数名："+str(sorted(extra)))
    cid=str(raw.get("id",""))
    if not cid or any(ch not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for ch in cid):
        raise ValueError("id须为非空英文字母/数字/下划线/横线")
    data={**MODEL_DEFAULTS,**{k:v for k,v in raw.items() if k in MODEL_DEFAULTS}}
    p=Model(**data);s={**settings,**raw.get("settings",{})}
    for key in ["gammas","K_values","M_values"]:
        if key in raw: s[key]=raw[key]
    for key in ["beta_minus","beta_plus","kappa","theta","sigma","k1","k2","mu","delta","p_on0","T"]:
        if not np.isfinite(getattr(p,key)): raise ValueError(key+"必须有限")
    if not (p.beta_minus<p.beta_plus and p.kappa>0 and p.sigma>0 and 0<p.theta<1 and p.T>0):
        raise ValueError("要求beta_minus<beta_plus，kappa,sigma,T>0，0<theta<1")
    if min(p.k1,p.k2,p.mu,p.delta)<0: raise ValueError("反应率不能为负数")
    if not isinstance(p.N0,int) or isinstance(p.N0,bool) or p.N0<0: raise ValueError("N0须为非负整数")
    if not 0<=p.p_on0<=1: raise ValueError("p_on0必须在[0,1]")
    if p.initial_kind=="stationary":
        if p.N0>p.beta_minus: raise ValueError("平稳初值须保证P(N0<beta0)=1：本例要求N0<=beta_minus")
    elif p.initial_kind=="uniform_window":
        if p.initial_lo is None or p.initial_hi is None or not (p.beta_minus<p.initial_lo<p.initial_hi<p.beta_plus):
            raise ValueError("uniform_window须严格位于边界区间内部，以保证相对pi的L2密度")
        if p.N0>p.initial_lo: raise ValueError("窗口初值不满足P(N0<beta0)=1")
    else: raise ValueError("initial_kind只能是stationary或uniform_window；不使用delta初值")
    # This detector handles interior level crossings. Endpoint-only contact needs
    # a different first-hit implementation; rejecting it is software scope, not
    # a claim that integer-endpoint Jacobi models are mathematically illegal.
    if abs(p.beta_minus-round(p.beta_minus))<1e-10 or abs(p.beta_plus-round(p.beta_plus))<1e-10:
        raise ValueError("本版未实现整数端点的纯接触首次时间检测，请选非整数区间端点")
    for key in ["gammas","K_values","M_values"]:
        values=s[key]
        if len(values)<2 or any(not np.isfinite(x) for x in values) or any(a>=b for a,b in zip(values,values[1:])):
            raise ValueError(key+"须有至少两个严格递增的有限值")
        if key=="gammas" and min(values)<=0: raise ValueError("gamma必须>0")
        if key!="gammas" and (any(not isinstance(x,int) for x in values) or min(values)<(1 if key=="K_values" else 0)):
            raise ValueError(key+"须为合法整数")
    for key in ["observation_dt","ode_dt","mc_coarse_dt","mc_fine_dt"]:
        if not np.isfinite(s[key]) or s[key]<=0: raise ValueError(key+"必须为正数")
    for key in ["coarse_paths","fine_paths","mc_batch_size"]:
        if not isinstance(s[key],int) or s[key]<2: raise ValueError(key+"须为>=2整数")
    if s["mc_fine_dt"]>=s["mc_coarse_dt"]: raise ValueError("fine MC时间步必须小于coarse")
    seed=raw.get("seed",20261009)
    if not isinstance(seed,int) or isinstance(seed,bool) or seed<0:
        raise ValueError("seed须为非负整数")
    return cid,str(raw.get("label",cid)),seed,p,s

NUMERICAL_NAMES={"Model","observation_times","aligned_step","phi_matrix","initial_coefficients","initial_vector",
                 "scaled_matrix","spectral_solve","original_matrix_check","jacobi_step","safe_wait","mc_chunk","aggregate_mc"}
def algorithm_hash():
    tree=ast.parse(Path(__file__).read_text(encoding="utf-8"))
    nodes=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in NUMERICAL_NAMES]
    return hashlib.sha256(ast.dump(ast.Module(body=nodes,type_ignores=[]),include_attributes=False).encode()).hexdigest()

def plot_case(folder,p,s,curves,refs,fine,convergence,per_gamma):
    if fine is None: return
    lastg=s["gammas"][-1];hard=fine[fine.gamma==lastg].sort_values("t")
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    for axis,cdf in zip(axes,[False,True]):
        y=hard.H_hard.to_numpy();se=hard.SE_hard.to_numpy();t=hard.t.to_numpy()
        value=1-y if cdf else y
        axis.plot(t,value,"k",lw=2.5,label="MC hard first hit")
        axis.fill_between(t,value-1.96*se,value+1.96*se,color="k",alpha=.12)
        for g in s["gammas"]:
            if g in refs:axis.plot(t,1-refs[g] if cdf else refs[g],label=f"gamma={g:g}")
        axis.set(xlabel="time",ylabel="first-hit CDF" if cdf else "survival");axis.grid(alpha=.25)
    axes[0].legend(fontsize=8);fig.tight_layout();fig.savefig(folder/"survival_and_first_hit.png",dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(14,4))
    for layer,axis in zip(["K","M"],axes[:2]):
        for g in s["gammas"]:
            v=convergence[(convergence.layer==layer)&(convergence.gamma==g)]
            if len(v):axis.semilogy(v.current,np.maximum(v.sup_difference,1e-16),"o-",label=f"g={g:g}")
        axis.set(xlabel=layer,ylabel="successive difference",title=layer+"-layer");axis.grid(alpha=.25)
    axes[0].legend(fontsize=7)
    if len(per_gamma):
        axes[2].loglog(per_gamma.gamma,per_gamma.sup_vs_MC_hard,"o-",label="spectral vs hard MC")
        axes[2].loglog(per_gamma.gamma,np.maximum(per_gamma.sup_vs_MC_soft,1e-16),"s--",label="spectral vs soft MC")
    axes[2].set(xlabel="gamma",ylabel="difference",title="gamma-layer");axes[2].legend(fontsize=7);axes[2].grid(alpha=.25)
    fig.tight_layout();fig.savefig(folder/"three_layer_convergence.png",dpi=160);plt.close(fig)

def analyze_case(cid,label,p,s,folder,curves,refs,validation,fine,coarse,failures,tol):
    rows=[]
    for g in s["gammas"]:
        for layer,grid in [("K",s["K_values"]),("M",s["M_values"])]:
            for a,b in zip(grid[:-1],grid[1:]):
                ka=(g,max(s["M_values"]),a) if layer=="K" else (g,a,max(s["K_values"]))
                kb=(g,max(s["M_values"]),b) if layer=="K" else (g,b,max(s["K_values"]))
                if ka in curves and kb in curves:rows.append(dict(layer=layer,gamma=g,previous=a,current=b,sup_difference=float(np.max(np.abs(curves[ka]-curves[kb])))))
    conv=pd.DataFrame(rows,columns=["layer","gamma","previous","current","sup_difference"])
    atomic_csv(folder/"layer_convergence.csv",conv)
    pg=[]
    if fine is not None:
        for g in s["gammas"]:
            if g not in refs:continue
            m=fine[fine.gamma==g].sort_values("t");H=refs[g]
            cm=coarse[coarse.gamma==g].sort_values("t") if coarse is not None else None
            pg.append(dict(gamma=g,H_T=float(H[-1]),H_MC_hard_T=float(m.H_hard.iloc[-1]),SE_hard_T=float(m.SE_hard.iloc[-1]),
                sup_vs_MC_hard=float(np.max(np.abs(H-m.H_hard.to_numpy()))),sup_vs_MC_soft=float(np.max(np.abs(H-m.H_soft.to_numpy()))),
                MC_soft_excess_4SE=float(np.max(np.maximum(0,np.abs(H-m.H_soft.to_numpy())-4*m.SE_soft.to_numpy()))),
                MC_refinement_sup=None if cm is None else float(np.max(np.abs(m.H_hard.to_numpy()-cm.H_hard.to_numpy()))),
                min_H=float(np.min(H)),max_H=float(np.max(H))))
    per=pd.DataFrame(pg);atomic_csv(folder/"per_gamma_summary.csv",per)
    lastK=conv[conv.layer=="K"].groupby("gamma").tail(1)
    lastM=conv[conv.layer=="M"].groupby("gamma").tail(1)
    kstep=None if len(lastK)!=len(s["gammas"]) else float(lastK.sup_difference.max())
    mstep=None if len(lastM)!=len(s["gammas"]) else float(lastM.sup_difference.max())
    od=[r["sup_difference"] for r in validation if r["check"]=="ODE_halfstep"]
    qu=[r["sup_difference"] for r in validation if r["check"]=="quadrature_doubled"]
    ode=None if len(od)!=len(s["gammas"]) else max(od);quad=None if len(qu)!=len(s["gammas"]) else max(qu)
    last=None if not len(per) else per[per.gamma==s["gammas"][-1]]
    last=None if last is None or not len(last) else last.iloc[0].to_dict()
    flags=[]
    if failures:flags.append("NUMERICAL_TASK_FAILED")
    if kstep is None or kstep>tol["K_step"]:flags.append("NEED_LARGER_K")
    if mstep is None or mstep>tol["M_step"]:flags.append("NEED_LARGER_M")
    if ode is None or ode>tol["ODE_step"]:flags.append("NEED_SMALLER_ODE_DT")
    if quad is None or quad>tol["quadrature"]:flags.append("QUADRATURE_OR_ROUNDOFF_LIMIT")
    if fine is None or coarse is None:flags.append("MC_INCOMPLETE")
    if len(per) and per.MC_soft_excess_4SE.max()>tol["soft_MC_excess"]:flags.append("SOFT_SOLVER_MC_DISAGREEMENT")
    if last is None or last["sup_vs_MC_hard"]>tol["soft_hard"]:flags.append("SOFT_HARD_NOT_YET_CLOSE")
    if last is not None and last["MC_refinement_sup"] is not None and last["MC_refinement_sup"]>tol["MC_step"]:flags.append("MC_STEP_OR_SAMPLING_UNCERTAINTY")
    if len(per) and (per.min_H.min() < -tol["range"] or per.max_H.max()>1+tol["range"]):flags.append("FINITE_APPROXIMATION_RANGE_WARNING")
    result=dict(id=cid,label=label,status="NUMERICAL_SUPPORT" if not flags else "NEEDS_MORE_RESOLUTION_OR_CHECKS",
                flags=flags,shape_a=p.a,shape_b=p.b,K_last_step_max=kstep,M_last_step_max=mstep,
                ODE_halfstep_max=ode,quadrature_max=quad,final_gamma=s["gammas"][-1],
                final_K=max(s["K_values"]),final_M=max(s["M_values"]),last=last,
                note="有限分辨率诊断，不是无穷极限证明；4SE不是同时置信带；MC粗细差包含统计波动")
    atomic_json(folder/"case_summary.json",result)
    plot_case(folder,p,s,curves,refs,fine,conv,per)
    return result

def run_case(raw,settings,tol,root,ah,source_hash,retry_failed=False):
    cid,label,seed,p,s=legal_case(raw,settings)
    fingerprint=json_hash(dict(model=asdict(p),settings=s,seed=seed,algorithm=ah,numpy=np.__version__,scipy=scipy.__version__))
    folder=root/(cid+"__"+fingerprint[:12]);folder.mkdir(parents=True,exist_ok=True)
    done=folder/"COMPLETE.json";analysis_key=json_hash(dict(tolerances=tol,report_version=2))
    try:
        mark=json.loads(done.read_text(encoding="utf-8"))
        if mark["fingerprint"]==fingerprint and mark["analysis_key"]==analysis_key and not (retry_failed and mark["had_failures"]):
            if all(digest(folder/name)==value for name,value in mark["checksums"].items()):
                print("跳过已完成参数组：",cid,flush=True)
                return json.loads((folder/"case_summary.json").read_text(encoding="utf-8")),folder
    except (OSError,ValueError,KeyError):pass
    atomic_json(folder/"configuration.json",dict(model=asdict(p),settings=s,label=label,seed=seed,
                shape_a=p.a,shape_b=p.b,fingerprint=fingerprint,algorithm_hash=ah,source_hash=source_hash,
                numerical_environment=dict(numpy=np.__version__,scipy=scipy.__version__),started=utc()))
    cache=Cache(folder/"checkpoints");print("开始/续跑参数组：",cid,label,flush=True)
    check=cache.task("algebra",{},lambda:original_matrix_check(p))
    if check is None:
        result=dict(id=cid,label=label,status="FORMULA_CHECK_FAILED",flags=["ORIGINAL_MATRIX_MISMATCH"])
        atomic_json(folder/"case_summary.json",result);return result,folder
    curves={};refs={};validation=[];curve_rows=[]
    jobs={(g,max(s["M_values"]),K) for g in s["gammas"] for K in s["K_values"]}
    jobs|={(g,M,max(s["K_values"])) for g in s["gammas"] for M in s["M_values"]}
    for i,(g,M,K) in enumerate(sorted(jobs)):
        key=dict(gamma=g,M=M,K=K,dt=s["ode_dt"],quadrature=0)
        data=cache.task("spectral",key,lambda g=g,M=M,K=K:spectral_solve(p,s,g,M,K,s["ode_dt"]))
        if data is not None:
            curves[g,M,K]=data["H"]
            for t,H in zip(data["t"],data["H"]):curve_rows.append(dict(gamma=g,M=M,K=K,t=t,H=H))
        print(f"{cid} spectral {i+1}/{len(jobs)} g={g:g},M={M},K={K}; 已保存/复用",flush=True)
    atomic_csv(folder/"spectral_curves.csv",pd.DataFrame(curve_rows))
    for g in s["gammas"]:
        M=max(s["M_values"]);K=max(s["K_values"]);half=s["ode_dt"]/2
        key=dict(gamma=g,M=M,K=K,dt=half,quadrature=0)
        fine=cache.task("spectral",key,lambda g=g:spectral_solve(p,s,g,M,K,half))
        keyq=dict(gamma=g,M=M,K=K,dt=half,quadrature=max(256,4*K+48))
        quad=cache.task("spectral",keyq,lambda g=g:spectral_solve(p,s,g,M,K,half,keyq["quadrature"]))
        if fine is not None:
            refs[g]=fine["H"]
            if (g,M,K) in curves:validation.append(dict(gamma=g,check="ODE_halfstep",sup_difference=float(np.max(np.abs(fine["H"]-curves[g,M,K])))))
            if quad is not None:validation.append(dict(gamma=g,check="quadrature_doubled",sup_difference=float(np.max(np.abs(fine["H"]-quad["H"])))))
    atomic_csv(folder/"numerical_validation.csv",pd.DataFrame(validation))
    ref_rows=[dict(gamma=g,t=t,H=h) for g,H in refs.items() for t,h in zip(observation_times(p,s),H)]
    atomic_csv(folder/"reference_curves.csv",pd.DataFrame(ref_rows))
    frames={};stage_chunks={}
    for stage,count_key,step_key in [(0,"coarse_paths","mc_coarse_dt"),(1,"fine_paths","mc_fine_dt")]:
        name="coarse" if stage==0 else "fine";chunks=[];total=s[count_key];batch=s["mc_batch_size"]
        for ci,start in enumerate(range(0,total,batch)):
            size=min(batch,total-start);words=[seed,stage,ci]
            key=dict(stage=name,chunk=ci,paths=size,dt=s[step_key],seed_words=words)
            data=cache.task("mc",key,lambda size=size,words=words:mc_chunk(p,s,size,s[step_key],words))
            chunks.append(data)
            print(f"{cid} MC {name}: batch {ci+1}/{math.ceil(total/batch)}；断点已保存/复用",flush=True)
        frames[name]=aggregate_mc(chunks,s["gammas"])
        if frames[name] is not None:atomic_csv(folder/("mc_"+name+".csv"),frames[name])
        stage_chunks[name]=chunks
    # Diagnostics for boundary simulation: exact mean and second moment.
    boundary_checks=[]
    for stage in ["coarse","fine"]:
        available=stage_chunks[stage]
        if any(c is None for c in available):continue
        y=np.concatenate([c["final_y"] for c in available])
        if p.initial_kind=="stationary":m10=p.theta;m20=p.a*(p.a+1)/((p.a+p.b)*(p.a+p.b+1))
        else:
            lo=(p.initial_lo-p.beta_minus)/(p.beta_plus-p.beta_minus);hi=(p.initial_hi-p.beta_minus)/(p.beta_plus-p.beta_minus)
            m10=(lo+hi)/2;m20=(lo*lo+lo*hi+hi*hi)/3
        lam2=2*p.kappa+p.sigma**2;A=2*p.kappa*p.theta+p.sigma**2
        m1=p.theta+(m10-p.theta)*np.exp(-p.kappa*p.T)
        m2=np.exp(-lam2*p.T)*m20+A*p.theta/lam2*(1-np.exp(-lam2*p.T))+A*(m10-p.theta)*(np.exp(-p.kappa*p.T)-np.exp(-lam2*p.T))/(lam2-p.kappa)
        boundary_checks.append(dict(stage=stage,paths=len(y),mean_MC=float(y.mean()),mean_exact=float(m1),
            mean_SE=float(y.std(ddof=1)/np.sqrt(len(y))),second_MC=float(np.mean(y*y)),second_exact=float(m2),
            second_SE=float((y*y).std(ddof=1)/np.sqrt(len(y)))))
    atomic_csv(folder/"boundary_MC_checks.csv",pd.DataFrame(boundary_checks))
    atomic_json(folder/"failed_tasks.json",cache.failures)
    result=analyze_case(cid,label,p,s,folder,curves,refs,validation,frames.get("fine"),frames.get("coarse"),cache.failures,tol)
    checksums={str(path.relative_to(folder)):digest(path) for path in folder.rglob("*") if path.is_file() and path.name!="COMPLETE.json" and not path.name.endswith(".tmp")}
    atomic_json(done,dict(fingerprint=fingerprint,analysis_key=analysis_key,had_failures=bool(cache.failures),
                         finished=utc(),checksums=checksums,new_tasks=cache.new,reused_tasks=cache.hits))
    print(cid,"完成：",result["status"],result["flags"],"新算",cache.new,"复用",cache.hits,flush=True)
    return result,folder

def write_overall(root,records):
    translations={"NUMERICAL_SUPPORT":"本次有限精度下支持逼近", "NEEDS_MORE_RESOLUTION_OR_CHECKS":"当前精度不足或仍需检查",
                  "FORMULA_CHECK_FAILED":"公式等价性检查失败","INVALID_PARAMETERS":"参数/软件适用范围检查失败","CASE_ERROR":"该组执行异常"}
    check_text={"NUMERICAL_TASK_FAILED":"存在计算失败的任务", "NEED_LARGER_K":"需增加K", "NEED_LARGER_M":"需增加M",
                "NEED_SMALLER_ODE_DT":"需减小谱方程时间步", "QUADRATURE_OR_ROUNDOFF_LIMIT":"求积或舍入精度待检查",
                "MC_INCOMPLETE":"蒙特卡洛未完成", "SOFT_SOLVER_MC_DISAGREEMENT":"软边界谱解与蒙特卡洛仍有差异",
                "SOFT_HARD_NOT_YET_CLOSE":"当前gamma下与首次越界仍不够接近",
                "MC_STEP_OR_SAMPLING_UNCERTAINTY":"蒙特卡洛时间步或样本量需检查",
                "FINITE_APPROXIMATION_RANGE_WARNING":"有限近似超出生存概率范围"}
    rows=[]
    for result,folder in records:
        last=result.get("last") or {}
        rows.append(dict(id=result["id"],label=result.get("label",""),status=result["status"],
            judgement_zh=translations.get(result["status"],result["status"]),
            checks_needed_zh="；".join(check_text.get(f,f) for f in result.get("flags",[])) or "无预设标记",
            flags=";".join(result.get("flags",[])),K_last_step_max=result.get("K_last_step_max"),
            M_last_step_max=result.get("M_last_step_max"),ODE_halfstep_max=result.get("ODE_halfstep_max"),
            final_gamma=result.get("final_gamma"),final_K=result.get("final_K"),final_M=result.get("final_M"),
            H_T=last.get("H_T"),MC_hard_T=last.get("H_MC_hard_T"),sup_vs_MC_hard=last.get("sup_vs_MC_hard"),
            results_folder=str(folder)))
    frame=pd.DataFrame(rows);atomic_csv(root/"overall_summary.csv",frame)
    atomic_json(root/"overall_summary.json",[r for r,_ in records])
    lines=["# Jacobi批量实验汇总","","这是有限参数的数值诊断，不是对所有参数或无穷极限的证明。",
           "误差非单调不自动判失败；MC粗细差含有统计波动，4SE也不是同时置信带。","",
           "| 参数组 | 判断 | gamma | K | M | 与首次越界MC最大差 | 待检查项 |","|---|---|---:|---:|---:|---:|---|"]
    for r in rows:
        gap="—" if r["sup_vs_MC_hard"] is None else f'{r["sup_vs_MC_hard"]:.6g}'
        lines.append(f'| {r["id"]} | {r["judgement_zh"]} | {r["final_gamma"]} | {r["final_K"]} | {r["final_M"]} | {gap} | {r["checks_needed_zh"]} |')
    (root/"REPORT_zh.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    html="<!doctype html><html><meta charset='utf-8'><title>Jacobi batch summary</title><style>body{max-width:1300px;margin:25px auto;font:17px sans-serif}img{width:100%}table{border-collapse:collapse}td,th{padding:5px;border:1px solid #ccc}</style><h1>Jacobi batch: computed results</h1><p>Finite-resolution diagnostics, not a proof of the infinite limit.</p>"+frame.to_html(index=False)
    for result,folder in records:
        html+="<h2>"+result["id"]+"</h2>"
        for name in ["survival_and_first_hit.png","three_layer_convergence.png"]:
            if (folder/name).exists():html+="<img src='"+(folder/name).relative_to(root).as_posix()+"'>"
    (root/"results.html").write_text(html+"</html>",encoding="utf-8")

def self_test():
    import tempfile
    from scipy.linalg import expm
    p=Model(**MODEL_DEFAULTS);s={**RUN_SETTINGS,**QUICK_SETTINGS}
    checks=[]
    for kappa,theta,sigma in [(1.,.5,1.),(1.2,.7,.8),(.3,.3,1.2)]:
        p.kappa=kappa;p.theta=theta;p.sigma=sigma
        check=original_matrix_check(p);J,d,meta=scaled_matrix(p,2.,5,12)
        assert float(check["relative"])<1e-9 and meta["gram_error"]<1e-10
        checks.append(dict(a=p.a,b=p.b,relative=float(check["relative"]),gram_error=meta["gram_error"]))
    p=Model(**MODEL_DEFAULTS);p.T=.2;s["observation_dt"]=.05
    J,d,_=scaled_matrix(p,2.,4,8);z=initial_vector(p,4,8)
    exact=expm(p.T*(J.toarray()/d[:,None]))@z;expected=float(exact.reshape(2,5,8)[:,:,0].sum())
    observed=spectral_solve(p,s,2.,4,8,.0005)["H"][-1]
    assert abs(observed-expected)<1e-7
    with tempfile.TemporaryDirectory() as temp:
        cache=Cache(Path(temp));count=[0]
        def compute():count[0]+=1;return dict(value=np.array([7]))
        assert cache.task("test",{"x":1},compute)["value"][0]==7
        assert cache.task("test",{"x":1},compute)["value"][0]==7 and count[0]==1
        datafile=next(Path(temp).glob("*.npz"));datafile.write_bytes(b"corrupted")
        assert cache.task("test",{"x":1},compute)["value"][0]==7 and count[0]==2
        failed=cache.task("test",{"x":2},lambda:(_ for _ in ()).throw(RuntimeError("intentional interrupted-task test")))
        assert failed is None
        assert cache.task("test",{"x":2},compute)["value"][0]==7
    print("SELF TEST PASSED:",checks,"matrix exponential difference",float(abs(observed-expected)))

def main():
    parser=argparse.ArgumentParser(description="Jacobi multi-parameter experiments with crash-safe checkpoints")
    parser.add_argument("--config",type=Path,help="JSON：列表，或含cases/settings/tolerances的对象")
    parser.add_argument("--output-root",type=Path,default=OUTPUT_ROOT)
    parser.add_argument("--quick",action="store_true",help="仅低成本流程检查；不用于严格收敛结论")
    parser.add_argument("--list",action="store_true",help="只检查并列出参数，不进行仿真")
    parser.add_argument("--case",action="append",help="只运行指定id；可以多次提供")
    parser.add_argument("--retry-failed",action="store_true",help="重试完成组里失败的子任务；成功断点仍复用")
    parser.add_argument("--self-test",action="store_true")
    args=parser.parse_args()
    if args.self_test:self_test();return
    cases=PARAMETER_SETS;settings=RUN_SETTINGS.copy();tol=TOLERANCES.copy()
    if args.config:
        payload=json.loads(args.config.read_text(encoding="utf-8-sig"))
        if isinstance(payload,list):cases=payload
        else:cases=payload["cases"];settings.update(payload.get("settings",{}));tol.update(payload.get("tolerances",{}))
    if args.quick:settings.update(QUICK_SETTINGS)
    if len({r.get("id") for r in cases})!=len(cases):parser.error("参数组id不能重复")
    if args.case:cases=[r for r in cases if r.get("id") in args.case]
    if not cases:parser.error("没有选中参数组")
    if args.list:
        for raw in cases:
            try:
                cid,label,seed,p,s=legal_case(raw,settings);print(cid,label,"a=",p.a,"b=",p.b,asdict(p))
            except Exception as e:print("INVALID",raw.get("id"),e)
        return
    root=args.output_root.resolve();root.mkdir(parents=True,exist_ok=True)
    ah=algorithm_hash();source_hash=digest(__file__);records=[]
    with process_lock(root):
        # Both config and execution provenance are saved before the first simulation.
        atomic_json(root/"last_run_plan.json",dict(cases=cases,settings=settings,tolerances=tol,
                     quick=args.quick,algorithm_hash=ah,source_hash=source_hash,started=utc()))
        print("输出目录：",root,"；直接再次运行即可续跑。",flush=True)
        if args.quick:print("QUICK模式：只验证程序流程，结果精度较低。",flush=True)
        for raw in cases:
            try:
                result,folder=run_case(raw,settings,tol,root,ah,source_hash,args.retry_failed)
            except KeyboardInterrupt:
                write_overall(root,records);print("已中断。已有断点保存，下次重新运行会续跑。",flush=True);return
            except ValueError as error:
                result=dict(id=str(raw.get("id","unknown")),label=raw.get("label",""),status="INVALID_PARAMETERS",flags=[str(error)])
                folder=root;print("参数检查失败，保留说明并继续：",result,flush=True)
            except Exception as error:
                result=dict(id=str(raw.get("id","unknown")),label=raw.get("label",""),status="CASE_ERROR",flags=[f"{type(error).__name__}: {error}"])
                folder=root;atomic_json(root/(result["id"]+"_exception.json"),dict(result=result,traceback=traceback.format_exc()))
                print("参数组异常，已保存，继续其他组：",result,flush=True)
            records.append((result,folder));write_overall(root,records)
        print("全部所选参数组处理完成。汇总：",root/"overall_summary.csv",flush=True)
        print("图片页面：",root/"results.html",flush=True)
        print("中文说明：",root/"REPORT_zh.md",flush=True)

if __name__=="__main__":main()
