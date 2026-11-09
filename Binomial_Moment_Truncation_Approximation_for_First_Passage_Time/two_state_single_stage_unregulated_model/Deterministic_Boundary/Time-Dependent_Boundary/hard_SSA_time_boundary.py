"""SSA benchmark for hard first passage through a time-dependent boundary.

Model
-----
    ON  --k1--> OFF
    OFF --k2--> ON
    ON  --mu--> ON + X
    X   --delta--> empty

For a continuous deterministic threshold beta(t), the integer hard boundary is

    B(t) = ceil(beta(t)),

and the first-passage time is

    T = inf{t >= 0 : N(t) >= B(t)}.

Because N(t) is integer, the hit condition is equivalently N(t) >= beta(t).
The implementation checks both reaction jumps and boundary crossings between
reaction times.  The latter is essential for decreasing or oscillatory
boundaries.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from math import ceil, pi
from pathlib import Path
from statistics import NormalDist
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq


BoundaryFunction = Callable[[float], float]


@dataclass(frozen=True)
class SSAResult:
    """Output of the hard first-passage SSA benchmark."""

    t_grid: np.ndarray
    survival: np.ndarray
    ci_lower: np.ndarray
    ci_upper: np.ndarray
    first_passage_times: np.ndarray
    n_trajectories: int
    confidence_level: float


def sinusoidal_threshold(
    t: float,
    B0: float = 8.0,
    amplitude: float = 2.0,
    period: float = 10.0,
) -> float:
    """Default test threshold beta(t) = B0 + A sin(2*pi*t/period)."""
    if period <= 0.0:
        raise ValueError("period must be positive.")
    return B0 + amplitude * np.sin(2.0 * pi * t / period)


def cell_growth_threshold(
    t: float,
    B0: float = 6.0,
    doubling_time: float = 20.0,
) -> float:
    """Threshold proportional to an exponentially growing cell volume."""
    if B0 <= 0.0:
        raise ValueError("B0 must be positive.")

    if doubling_time <= 0.0:
        raise ValueError("doubling_time must be positive.")

    return B0 * 2.0 ** (t / doubling_time)


def integer_boundary(t: float, beta_function: BoundaryFunction) -> int:
    """Return B(t) = ceil(beta(t))."""
    value = float(beta_function(float(t)))
    if not np.isfinite(value):
        raise ValueError(f"beta(t) must be finite; got beta({t})={value}.")
    return int(ceil(value))


def _first_boundary_hit_in_interval(
    n: int,
    t_left: float,
    t_right: float,
    beta_function: BoundaryFunction,
    boundary_scan_dt: float,
    hit_tolerance: float,
) -> float | None:
    """Find the first boundary-caused hit while N(t)=n on [t_left,t_right].

    A general callable beta(t) can oscillate inside a reaction waiting time.
    We bracket its first downward crossing of level n on a grid no coarser than
    ``boundary_scan_dt`` and refine the crossing with Brent's method.

    Consequently, ``boundary_scan_dt`` must resolve the fastest feature of the
    chosen boundary.  For known breakpoints, include sufficiently small scan
    spacing around them.
    """
    if t_right < t_left:
        raise ValueError("Need t_right >= t_left.")

    def gap(t: float) -> float:
        return float(beta_function(float(t))) - n

    g_left = gap(t_left)
    if not np.isfinite(g_left):
        raise ValueError("beta_function returned a non-finite value.")
    if g_left <= hit_tolerance:
        return float(t_left)
    if t_right == t_left:
        return None

    n_steps = max(1, int(np.ceil((t_right - t_left) / boundary_scan_dt)))
    previous_t = float(t_left)
    previous_gap = g_left

    for j in range(1, n_steps + 1):
        current_t = t_left + (t_right - t_left) * j / n_steps
        current_gap = gap(current_t)
        if not np.isfinite(current_gap):
            raise ValueError("beta_function returned a non-finite value.")

        if current_gap <= hit_tolerance:
            # A small positive gap is accepted as a numerical hit, but it
            # cannot be passed to brentq because no sign change is present.
            if current_gap >= 0.0:
                return float(current_t)
            if previous_gap > 0.0:
                root = brentq(
                    lambda s: float(beta_function(float(s))) - n,
                    previous_t,
                    current_t,
                    xtol=1e-12,
                    rtol=1e-12,
                )
                return float(root)
            return float(current_t)

        previous_t = float(current_t)
        previous_gap = current_gap

    return None


def simulate_first_passage_time(
    rng: np.random.Generator,
    beta_function: BoundaryFunction,
    t_max: float,
    k1: float,
    k2: float,
    mu: float,
    delta: float,
    n0: int = 0,
    p_off0: float = 1.0,
    p_on0: float = 0.0,
    boundary_scan_dt: float = 0.01,
    hit_tolerance: float = 1e-12,
) -> float:
    """Simulate one exact CTMC path and return its hard first-passage time.

    ``np.inf`` denotes right-censoring at ``t_max``.
    """
    if t_max <= 0.0:
        raise ValueError("t_max must be positive.")
    if n0 < 0:
        raise ValueError("n0 must be nonnegative.")
    if min(k1, k2, mu, delta) < 0.0:
        raise ValueError("Reaction rates must be nonnegative.")
    if boundary_scan_dt <= 0.0:
        raise ValueError("boundary_scan_dt must be positive.")
    if min(p_off0, p_on0) < 0.0 or not np.isclose(p_off0 + p_on0, 1.0):
        raise ValueError("Need p_off0,p_on0 >= 0 and p_off0 + p_on0 = 1.")

    # promoter_state: 0 = OFF, 1 = ON
    promoter_state = int(rng.random() >= p_off0)
    n = int(n0)
    t = 0.0

    if n >= integer_boundary(t, beta_function):
        return 0.0

    while t < t_max:
        if promoter_state == 0:
            switch_rate = k2       # OFF -> ON
            birth_rate = 0.0
        else:
            switch_rate = k1       # ON -> OFF
            birth_rate = mu
        death_rate = delta * n
        total_rate = switch_rate + birth_rate + death_rate

        if total_rate > 0.0:
            reaction_time = t + rng.exponential(1.0 / total_rate)
        else:
            reaction_time = np.inf

        interval_end = min(reaction_time, t_max)

        # A falling boundary can hit the unchanged state before the next
        # chemical reaction. Checking only reaction times would miss this.
        boundary_hit = _first_boundary_hit_in_interval(
            n=n,
            t_left=t,
            t_right=interval_end,
            beta_function=beta_function,
            boundary_scan_dt=boundary_scan_dt,
            hit_tolerance=hit_tolerance,
        )
        if boundary_hit is not None:
            return boundary_hit

        if reaction_time > t_max:
            return np.inf

        t = float(reaction_time)

        draw = rng.random() * total_rate
        if draw < switch_rate:
            promoter_state = 1 - promoter_state
        elif draw < switch_rate + birth_rate:
            n += 1
        else:
            # This branch has positive probability only when n > 0.
            n -= 1

        # Detect a jump of N(t) into the absorbing region.
        if n >= integer_boundary(t, beta_function):
            return t

    return np.inf


def _wilson_interval(
    successes: np.ndarray,
    trials: int,
    confidence_level: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Pointwise Wilson intervals for binomial survival probabilities."""
    if not (0.0 < confidence_level < 1.0):
        raise ValueError("confidence_level must lie in (0,1).")
    z = NormalDist().inv_cdf(0.5 + confidence_level / 2.0)
    p = np.asarray(successes, dtype=float) / trials
    denominator = 1.0 + z * z / trials
    center = (p + z * z / (2.0 * trials)) / denominator
    radius = (
        z
        * np.sqrt(p * (1.0 - p) / trials + z * z / (4.0 * trials * trials))
        / denominator
    )
    return np.maximum(0.0, center - radius), np.minimum(1.0, center + radius)


def hard_survival_ssa(
    t_grid: np.ndarray,
    beta_function: BoundaryFunction,
    n_trajectories: int,
    k1: float,
    k2: float,
    mu: float,
    delta: float,
    n0: int = 0,
    p_off0: float = 1.0,
    p_on0: float = 0.0,
    seed: int = 20260908,
    boundary_scan_dt: float = 0.01,
    confidence_level: float = 0.95,
    show_progress: bool = False,
) -> SSAResult:
    """Estimate H_hard(t)=P(T>t) using independent SSA trajectories."""
    t_grid = np.asarray(t_grid, dtype=float)
    if t_grid.ndim != 1 or len(t_grid) == 0:
        raise ValueError("t_grid must be a nonempty one-dimensional array.")
    if np.any(~np.isfinite(t_grid)) or np.any(t_grid < 0.0):
        raise ValueError("t_grid must contain finite nonnegative times.")
    if np.any(np.diff(t_grid) <= 0.0):
        raise ValueError("t_grid must be strictly increasing.")
    if n_trajectories <= 0:
        raise ValueError("n_trajectories must be positive.")

    rng = np.random.default_rng(seed)
    first_passage_times = np.empty(n_trajectories, dtype=float)
    t_max = float(t_grid[-1])
    progress_step = max(1, n_trajectories // 20)

    for trajectory in range(n_trajectories):
        first_passage_times[trajectory] = simulate_first_passage_time(
            rng=rng,
            beta_function=beta_function,
            t_max=t_max,
            k1=k1,
            k2=k2,
            mu=mu,
            delta=delta,
            n0=n0,
            p_off0=p_off0,
            p_on0=p_on0,
            boundary_scan_dt=boundary_scan_dt,
        )
        completed = trajectory + 1
        if show_progress and (
            completed % progress_step == 0 or completed == n_trajectories
        ):
            percent = 100.0 * completed / n_trajectories
            print(
                f"SSA progress: {completed}/{n_trajectories} "
                f"({percent:.0f}%)",
                flush=True,
            )

    sorted_times = np.sort(first_passage_times)
    hits_by_t = np.searchsorted(sorted_times, t_grid, side="right")
    survivors = n_trajectories - hits_by_t
    survival = survivors / n_trajectories
    ci_lower, ci_upper = _wilson_interval(
        survivors, n_trajectories, confidence_level
    )

    return SSAResult(
        t_grid=t_grid,
        survival=survival,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        first_passage_times=first_passage_times,
        n_trajectories=n_trajectories,
        confidence_level=confidence_level,
    )


def save_result(result: SSAResult, output_dir: Path) -> tuple[Path, Path]:
    """Save the survival table and the raw first-passage samples."""
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "hard_SSA_time_boundary_survival.csv"
    sample_path = output_dir / "hard_SSA_time_boundary_fpt_samples.npy"

    table = np.column_stack(
        (result.t_grid, result.survival, result.ci_lower, result.ci_upper)
    )
    np.savetxt(
        csv_path,
        table,
        delimiter=",",
        header="t,H_hard_SSA,CI_lower,CI_upper",
        comments="",
    )
    np.save(sample_path, result.first_passage_times)
    return csv_path, sample_path



def main() -> None:
    # ============================================================
    # Parameters that can be changed directly
    # ============================================================

    # SSA numerical parameters
    n_trajectories = 100_000
    seed = 20260908
    t_max = 20.0
    n_time_points = 2001
    boundary_scan_dt = 0.01
    confidence_level = 0.95
    show_plot = True

    # Output folder: a subfolder beside this Python script
    script_dir = Path(__file__).resolve().parent
    output_dir = script_dir / "hard_SSA_time_boundary_results"

    # Same physical parameters as the constant-boundary comparison script.
    k1 = 1.0
    k2 = 1.0
    mu = 5.0
    delta = 1.0
    n0 = 0
    p_off0 = 1.0
    p_on0 = 0.0

    # First time-dependent benchmark. Replace only this callable to test a
    # different deterministic threshold beta(t). The hard boundary used by
    # the simulation is B(t)=ceil(beta(t)).
    # B0 = 8.0
    # amplitude = 2.0
    # period = 10.0
    # beta_function = lambda t: sinusoidal_threshold(t, B0, amplitude, period)

    # Boundary 2: threshold proportional to cell growth
    B0 = 6.0
    doubling_time = 20.0

    beta_function = lambda t: cell_growth_threshold(
        t,
        B0,
        doubling_time,
    )

    t_grid = np.linspace(0.0, t_max, n_time_points)
    print("Starting hard first-passage SSA...", flush=True)
    print(f"trajectories = {n_trajectories}", flush=True)
    print(f"output directory = {output_dir}", flush=True)
    result = hard_survival_ssa(
        t_grid=t_grid,
        beta_function=beta_function,
        n_trajectories=n_trajectories,
        k1=k1,
        k2=k2,
        mu=mu,
        delta=delta,
        n0=n0,
        p_off0=p_off0,
        p_on0=p_on0,
        seed=seed,
        boundary_scan_dt=boundary_scan_dt,
        confidence_level=confidence_level,
        show_progress=True,
    )

    # Finite first-passage samples observed before t_max
    finite_fpt = result.first_passage_times[
        np.isfinite(result.first_passage_times)
    ]

    # Use t_grid as the histogram bin edges
    bin_edges = result.t_grid
    bin_widths = np.diff(bin_edges)

    # Number of first-passage events in each time interval
    fpt_counts, _ = np.histogram(
        finite_fpt,
        bins=bin_edges,
    )

    # Unconditional first-passage density estimate
    fpt_density = (
        fpt_counts
        / result.n_trajectories
        / bin_widths
    )

    csv_path, sample_path = save_result(result, output_dir)
    n_hit = int(np.count_nonzero(np.isfinite(result.first_passage_times)))
    print(f"trajectories = {result.n_trajectories}")
    print(f"hits by t_max = {n_hit}")
    print(f"right-censored at t_max = {result.n_trajectories - n_hit}")
    print(f"survival CSV = {csv_path}")
    print(f"FPT samples = {sample_path}")

    if show_plot:
        confidence_percent = (
            100.0 * result.confidence_level
        )

        # ========================================================
        # Estimate the first-passage density from raw SSA samples
        # ========================================================
        finite_fpt = result.first_passage_times[
            np.isfinite(result.first_passage_times)
        ]

        bin_edges = result.t_grid
        bin_widths = np.diff(bin_edges)

        fpt_counts, _ = np.histogram(
            finite_fpt,
            bins=bin_edges,
        )

        fpt_density = (
            fpt_counts
            / result.n_trajectories
            / bin_widths
        )

        # ========================================================
        # Create two vertically arranged panels
        # ========================================================
        fig, (ax_survival, ax_density) = plt.subplots(
            2,
            1,
            figsize=(8, 8),
            sharex=True,
        )

        # Upper panel: survival probability
        ax_survival.step(
            result.t_grid,
            result.survival,
            where="post",
            color="black",
            linewidth=2.0,
            label=r"$H_{\mathrm{hard}}^{\mathrm{SSA}}(t)$",
        )

        ax_survival.fill_between(
            result.t_grid,
            result.ci_lower,
            result.ci_upper,
            step="post",
            color="gray",
            alpha=0.3,
            label=f"pointwise {confidence_percent:.0f}% CI",
        )

        ax_survival.set_ylabel("Survival probability")
        ax_survival.set_ylim(0.0, 1.02)

        ax_survival.set_title(
            "Hard first-passage SSA, "
            r"$B(t)=\lceil\beta(t)\rceil$"
        )

        ax_survival.grid(True, alpha=0.3)
        ax_survival.legend()

        # Lower panel: first-passage density
        ax_density.stairs(
            fpt_density,
            bin_edges,
            fill=True,
            color="tab:blue",
            alpha=0.5,
            linewidth=1.5,
            label=r"$\widehat f_T^{\mathrm{SSA}}(t)$",
        )

        ax_density.set_xlabel("t")
        ax_density.set_ylabel("First-passage density")
        ax_density.set_ylim(bottom=0.0)
        ax_density.grid(True, alpha=0.3)
        ax_density.legend()

        fig.tight_layout()

        figure_path = (
            output_dir
            / "hard_SSA_time_boundary_survival_and_density.png"
        )

        fig.savefig(
            figure_path,
            dpi=300,
            bbox_inches="tight",
        )

        print(
            f"survival and density figure = {figure_path}"
        )

        plt.show()


if __name__ == "__main__":
    main()
