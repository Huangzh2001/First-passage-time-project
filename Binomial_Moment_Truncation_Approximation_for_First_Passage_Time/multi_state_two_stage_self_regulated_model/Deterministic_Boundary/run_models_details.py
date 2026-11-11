"""在 VS Code 直接运行本文件。改顶部设置选择模型、边界与计算规模。"""

import csv
import hashlib
import inspect
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from binomial_moments import adaptive_solve
from boundary_definitions import (
    COUPLED_BOUNDARIES, NEW_BOUNDARIES, SMOOTH_BOUNDARIES, WINDOW_BOUNDARIES,
    comparison_boundary, coupled_boundary, smooth_boundary_pair, smooth_presets,
    window_boundary,
)
from model_definitions import network
from ssa import affine_time_boundary, estimate_batched, interval_first_hit


# 给网络或边界加 # 可跳过相应组合。
NETWORKS = (
    0,  # 蛋白促进，3 个启动子状态
    1,  # 蛋白抑制，4 个启动子状态
    2,  # mRNA 与蛋白协同，5 个启动子状态
    3,  # 竞争调控，6 个启动子状态
)
BOUNDARIES = (
    "product",           # 乘积阈值
    "weighted",          # 加权阈值
    "saturating",        # 饱和项阈值
    "curved",            # 二次项阈值

    # A. 固定门槛，测试不同的计数触发逻辑
    "either_count",      # 任一计数达标
    "joint_or_high",     # 两种计数联合达标，或总分单独达标

    # B. 门槛随时间放宽，放宽后保持
    "delayed_ramp",      # 延迟后逐渐放宽
    "step_release",      # 在固定时刻一次性放宽

    # C. 门槛临时放宽，高分状态始终可触发
    "guarded_pulse",         # 放宽一次
    "guarded_double_pulse",  # 放宽两次，中间收紧

    # D. 计数落入指定范围可触发，高总分也始终可触发
    "protein_band",      # 蛋白进入中间区间，包括降解时从上方进入
    "balance_band",      # 两种计数足够接近，包括降解后变得接近

    # E. 比例与总量共同决定是否触发
    "shifting_ratio",    # 目标比例由 2:1 移到 1:2，高总分也可触发
    # "widening_ratio",    # 总量越高、时间越晚，允许的比例偏差越大

    # F. 两条触发路线随时间变化
    "merging_routes",    # 两条路线之间的中间区域逐渐开放
    # "swapping_routes",   # 一条路线逐渐变难，另一条逐渐变容易

    # G. 蛋白下穿门槛随 mRNA 和时间改变
    # "coupled_downcross", # mRNA 越高，所需的蛋白降幅越大，后期放宽
    "moving_downcross",  # 最容易下穿的 mRNA 区间随时间移动

    # 原来的六个窗口，取消注释即可重新运行
    # "ratio_window",      # 比例决定的时间窗
    # "tilted_ellipse",    # 移动椭圆
    # "parabolic_double",  # 抛物线双窗口
    # "periodic_windows",  # 周期窗口
    # "moving_ring",       # 移动环形
    # "opposing_bands",    # 双向扫过的条带
)
GAMMAS = (8., 16., 32., 64.)
ORDER_START = (8, 8)
ORDER_STEP = (2, 2)
ORDER_CAP = (30, 30)
ORDER_TOL = 2e-4  # H 采样网格上的相邻阶最大绝对差
MASS_TOL = 1e-5  # 独立积分的 H 与反演坐标求和的最大允许差
PATHS = 50000
SEED = 6188
HORIZON = 8.
BINS = 100
POINTS_PER_BIN = 10
SOLVER_OPTIONS = dict(basis="binomial", rtol=2e-9, atol=1e-12,
                      max_step=.035, timeout=25.)
RESULTS = Path(__file__).resolve().parent / "results"
OUTPUT_FILES = (
    "comparison.png", "comparison.svg", "comparison.pdf",
    "survival.csv", "fpt_density.csv", "fpt_bins.csv", "diagnostics.json",
)
PREVIOUS_CODE = "a42714aa0f888b9d10398afdddde385824176f721c554928b00923763e0de10e"
DIAGNOSTICS_ONLY_UPDATE_CODE = "4d86e66a164f37438d392efec7db3b79237c1e8e3f7c9426b5699a3cc9335c13"
# 对已逐项验证的重构，沿用旧签名，避免重新计算已有结果。
EQUIVALENT_SIGNATURES = {
    "7e1e3d38e0a7c1b4ae51e92ddb9ddd87b271d4dd4a246114d8ed5b8eac5adfb6":
        "985cf45a3826e94ac638e6c7fee3cc41462a6111faa4fb549d85f7061f25f83b",
    "4669f320ced63bd326d48512e1f7cb25fdbea3e3df090cd4ea6a14e0b331b2cf":
        "30c825dd986c1b5277adad18ac4f13dc3ac04f4fea1b8105ae23bc7c219f1d5b",
    "64bd846f59fcb91bf7954fa0f028ccebc95727727b8c4e82cb1353833ee9a694":
        "7c0912512b2d69074054b91be613aff38164d5f26a9fa65fe99b79418e1d995e",
}


def boundary_fingerprint(*definitions):
    source = "".join(inspect.getsource(definition) for definition in definitions)
    digest = hashlib.sha256(source.encode("utf-8")).hexdigest()
    return EQUIVALENT_SIGNATURES.get(digest, digest)


def experiment_settings(gammas):
    code = hashlib.sha256()
    for definition in (network, window_boundary, smooth_presets):
        code.update(inspect.getsource(definition).encode("utf-8"))
    for name in ("binomial_moments.py", "ssa.py"):
        code.update(Path(__file__).with_name(name).read_bytes())
    digest = code.hexdigest()
    return dict(
        format_version=1, model_and_solver_code=EQUIVALENT_SIGNATURES.get(digest, digest),
        gammas=list(gammas),
        paths=PATHS, seed=SEED, horizon=HORIZON, bins=BINS, points_per_bin=POINTS_PER_BIN,
        start=list(ORDER_START), step=list(ORDER_STEP), cap=list(ORDER_CAP),
        order_tolerance=ORDER_TOL, mass_tolerance=MASS_TOL, solver=SOLVER_OPTIONS,
    )


def already_completed(folder, title, settings):
    if not all((folder / name).is_file() and (folder / name).stat().st_size > 0
               for name in OUTPUT_FILES):
        return False
    try:
        report = json.loads((folder / "diagnostics.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    if not isinstance(report, dict) or report.get("model") != title:
        return False
    for run in report.get("runs", []):
        for attempt in run.get("attempts", []):
            if attempt.get("valid", True):
                continue
            samples = attempt.get("failed_samples")
            if attempt is run["attempts"][-1]:
                samples = samples or run.get("failed_samples")
            if not samples or not (folder / samples).is_file():
                return False
    saved_settings = report.get("completed_settings")
    if saved_settings == settings:
        return True
    return (settings["model_and_solver_code"] == DIAGNOSTICS_ONLY_UPDATE_CODE
            and saved_settings == dict(settings, model_and_solver_code=PREVIOUS_CODE)
            and len(report.get("runs", [])) == len(settings["gammas"])
            and all(run.get("selection") in ("converged", "cap_valid") for run in report["runs"]))


def write_diagnostics(folder, report):
    path = folder / "diagnostics.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def write_csv(path, headers, rows):
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(headers)
        writer.writerows(rows)
    temporary.replace(path)


def save_gamma_samples(folder, gamma, times, run, *, failed=False):
    if failed:
        stem = f"failed_gamma_{gamma:.17g}_L{run['L']}_K{run['K']}"
    else:
        stem = f"partial_gamma_{gamma:.17g}"
    path = folder / f"{stem}.csv"
    rows = []
    for sample in zip(times, run["h"], run["density"], run["h_from_sum"]):
        if np.isfinite(sample[1:]).any():
            rows.append(sample)
    write_csv(path, ("time", "H", "fpt_density", "H_from_moment_sum"), rows)
    return path


def save_comparison(folder, title, times, hard, hits, selected, diagnostics, settings):
    edges = times[::POINTS_PER_BIN]
    widths = np.diff(edges)
    hard_fpt = -np.diff(hard[::POINTS_PER_BIN]) / widths
    dkw = float(np.sqrt(np.log(2 / (1 - .95)) / (2 * len(hits))))
    unique, counts = np.unique(hits[np.isfinite(hits)], return_counts=True)
    repeated = counts > 1
    fig, (ax_h, ax_fpt) = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    ax_h.fill_between(times, np.maximum(0, hard - dkw), np.minimum(1, hard + dkw),
                      step="post", color=".8", alpha=.45, label="SSA 95% DKW band")
    ax_h.step(times, hard, where="post", color="black", lw=1.4, label="Hard SSA")
    ax_fpt.stairs(hard_fpt, edges, baseline=None, color="black", lw=1.4, label="Hard SSA")
    survival = {"time": times, "hard_ssa_H": hard}
    density = {"time": times}
    bins = {"bin_start": edges[:-1], "bin_end": edges[1:], "hard_ssa_fpt_per_unit_time": hard_fpt}
    colors = plt.get_cmap("viridis")(np.linspace(.12, .86, max(1, len(diagnostics))))
    for color, record in zip(colors, diagnostics):
        gamma = record["gamma"]
        if gamma not in selected:
            continue
        run = selected[gamma]
        h = run["h"]
        fpt = -np.diff(h[::POINTS_PER_BIN]) / widths
        converged = record["selection"] == "converged"
        style = "-" if converged else "--"
        label = f"γ={gamma:g}, ({run['L']}, {run['K']})"
        label += "" if converged else " · cap"
        ax_h.plot(times, h, style, color=color, lw=1.4, label=label)
        ax_fpt.plot(times, run["density"], style, color=color, lw=1.4, label=label)
        survival[f"H_gamma_{gamma:g}"] = h
        density[f"theory_density_gamma_{gamma:g}"] = run["density"]
        bins[f"fpt_per_unit_time_gamma_{gamma:g}"] = fpt
        gap = h - hard
        record["selected_order"] = [run["L"], run["K"]]
        record["max_abs_H_gap_on_grid"] = float(np.max(abs(gap)))
        record["mean_abs_H_gap_on_grid"] = float(np.sum((abs(gap[:-1]) + abs(gap[1:])) * np.diff(times))
                                                       / (2 * times[-1]))
        # 包含观察期末仍未首达的概率质量。
        record["fpt_bin_tv_with_tail"] = float(
            .5 * (np.sum(np.abs(np.diff(gap[::POINTS_PER_BIN]))) + abs(gap[-1]))
        )
    fig.suptitle(title.replace("_", " "))
    ax_h.set(title="Survival", ylabel="H(t)", ylim=(-.015, 1.015))
    ax_fpt.set(title="First passage time (SSA binned)", ylabel="Probability per unit time")
    for axis in (ax_h, ax_fpt):
        axis.set(xlabel="Time", xlim=(0, times[-1]))
        axis.grid(alpha=.22)
    ax_h.legend(fontsize=8)
    ax_fpt.legend(fontsize=8)
    for ext in ("png", "svg", "pdf"):
        fig.savefig(folder / f"comparison.{ext}", dpi=180)
    plt.close(fig)
    for name, columns in (("survival", survival), ("fpt_density", density), ("fpt_bins", bins)):
        write_csv(folder / f"{name}.csv", columns, zip(*columns.values()))
    report = dict(
        model=title, stage="complete", completed_settings=settings,
        paths=len(hits), horizon=float(times[-1]), bins=BINS, seed=SEED,
        points_per_bin=POINTS_PER_BIN, start=ORDER_START, step=ORDER_STEP, cap=ORDER_CAP,
        order_tolerance=ORDER_TOL, mass_tolerance=MASS_TOL, solver=SOLVER_OPTIONS,
        hard_ssa_dkw_95_half_width=dkw, runs=diagnostics,
        hard_repeated_hit_times=[
            {"time": float(time), "empirical_mass": int(count) / len(hits)}
            for time, count in zip(unique[repeated], counts[repeated])
        ],
        comparison_note=("Theoretical FPT curves show the sampled ODE killing flux. "
                         "FPT bins are (left, right]. Repeated hard-hit times flag possible "
                         "atoms. H gaps are sampled diagnostics and may include atoms. "
                         "Order convergence is a numerical criterion, not an error bound."),
    )
    write_diagnostics(folder, report)


def clear_partial_curves(folder):
    for path in folder.glob("partial_gamma_*.csv"):
        path.unlink(missing_ok=True)


def main():
    gammas = sorted(set(GAMMAS))
    variants = tuple(dict.fromkeys(NETWORKS))
    boundary_names = tuple(dict.fromkeys(BOUNDARIES))
    allowed = set(SMOOTH_BOUNDARIES) | set(NEW_BOUNDARIES) | set(COUPLED_BOUNDARIES) | set(WINDOW_BOUNDARIES)
    if (not gammas or any(not np.isfinite(g) or g <= 0 for g in gammas)
            or not variants or any(v not in range(4) for v in variants)
            or not boundary_names or not allowed.issuperset(boundary_names)
            or not np.isfinite(HORIZON) or HORIZON <= 0
            or min(PATHS, BINS, POINTS_PER_BIN) < 1):
        raise ValueError("Check the experiment settings at the top of run_models.py")
    base_settings = experiment_settings(gammas)
    new_boundary_code = boundary_fingerprint(comparison_boundary)
    coupled_boundary_code = boundary_fingerprint(coupled_boundary, affine_time_boundary)
    times = np.linspace(0., HORIZON, BINS * POINTS_PER_BIN + 1)
    for variant in variants:
        model = network(variant)
        for name in boundary_names:
            settings = base_settings
            if name in SMOOTH_BOUNDARIES:
                boundary = smooth_boundary_pair(name)
            elif name in NEW_BOUNDARIES:
                boundary = comparison_boundary(name)
                settings = dict(base_settings, boundary_code=new_boundary_code)
            elif name in COUPLED_BOUNDARIES:
                boundary = coupled_boundary(name)
                settings = dict(base_settings, boundary_code=coupled_boundary_code)
            else:
                boundary = window_boundary(WINDOW_BOUNDARIES.index(name), variant)
            title = f"{model.name}__{name}"
            folder = RESULTS / title
            if already_completed(folder, title, settings):
                print(f"{title}: already completed, skipping", flush=True)
                continue
            folder.mkdir(parents=True, exist_ok=True)
            clear_partial_curves(folder)
            selected, diagnostics, failure_files = {}, [], set()
            progress = dict(model=title, stage="hard_ssa", settings=settings, runs=diagnostics)
            write_diagnostics(folder, progress)
            print(f"{title}: Hard SSA ({PATHS:,} paths)", flush=True)
            hard, hits = estimate_batched(
                model.rates, model.states, model.initial, None, times,
                paths=PATHS, seed=SEED, first_hit=boundary.first_hit,
            )
            ssa_checkpoint = folder / "partial_ssa.csv"
            np.savetxt(ssa_checkpoint, np.column_stack((times, hard)), delimiter=",",
                       header="time,hard_ssa_H", comments="")

            progress["stage"] = "theory"
            for gamma in gammas:
                record = dict(gamma=gamma, selection="running", attempts=[])
                diagnostics.append(record)
                write_diagnostics(folder, progress)

                def record_attempt(attempt, trial, selection):
                    record["attempts"].append(attempt)
                    if not attempt["valid"]:
                        path = save_gamma_samples(folder, gamma, times, trial, failed=True)
                        attempt["failed_samples"] = path.name
                        failure_files.add(path)
                    if selection:
                        record["selection"] = selection
                        if selection == "cap_invalid":
                            record["failed_samples"] = attempt["failed_samples"]
                        else:
                            save_gamma_samples(folder, gamma, times, trial)
                    write_diagnostics(folder, progress)
                    issues = ", ".join(attempt["first_by_check"])
                    extra = f" | {issues}" if issues else ""
                    if "failed_samples" in attempt:
                        extra += f" | saved {attempt['failed_samples']}"
                    print(f"  γ={gamma:g} ({attempt['L']}, {attempt['K']}): "
                          f"{attempt['solver_status']} at t={attempt['last_time']:.6g}{extra}", flush=True)

                try:
                    run, status, attempts = adaptive_solve(
                        model, boundary, times, gamma, start=ORDER_START, step=ORDER_STEP,
                        cap=ORDER_CAP, order_tol=ORDER_TOL, mass_tol=MASS_TOL,
                        on_attempt=record_attempt, **SOLVER_OPTIONS,
                    )
                except (Exception, KeyboardInterrupt) as exc:
                    progress.update(stage="interrupted", error=f"{type(exc).__name__}: {exc}")
                    write_diagnostics(folder, progress)
                    raise
                if run is not None:
                    selected[gamma] = run
                last = attempts[-1]
                print(f"  γ={gamma:g} ({last['L']}, {last['K']}): {status}", flush=True)
            # 先写完整套文件，再发布。最后发布 diagnostics.json 作为完成标记。
            with TemporaryDirectory(prefix=".pending-", dir=RESULTS) as pending:
                staging = Path(pending)
                save_comparison(staging, title, times, hard, hits, selected, diagnostics, settings)
                for name in OUTPUT_FILES:
                    (staging / name).replace(folder / name)
            ssa_checkpoint.unlink(missing_ok=True)
            clear_partial_curves(folder)
            for stale in folder.glob("failed_gamma_*.csv"):
                if stale not in failure_files:
                    stale.unlink()
    print(f"Results: {RESULTS}", flush=True)


if __name__ == "__main__":
    main()
