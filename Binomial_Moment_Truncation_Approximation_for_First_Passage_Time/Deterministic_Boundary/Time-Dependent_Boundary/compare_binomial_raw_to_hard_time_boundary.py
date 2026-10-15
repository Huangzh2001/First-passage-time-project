from pathlib import Path
import importlib.util

import numpy as np
import matplotlib.pyplot as plt



# ============================================================
# Load the three existing calculation files from this folder
# ============================================================

import numpy as np
import matplotlib.pyplot as plt

import soft_binomial_moment_time_boundary as binomial
import soft_raw_moment_time_boundary as raw
import hard_SSA_time_boundary as hard

# ============================================================
# Common parameters used by all three methods
# ============================================================

theta = 0.5
N_low = 8

# Physical parameters
k1 = 1.0
k2 = 1.0
mu = 5.0
delta = 1.0

n0 = 0
p_off0 = 1.0
p_on0 = 0.0

# Time-dependent boundary
B0 = 6.0
doubling_time = 20.0

beta_function = lambda t: hard.cell_growth_threshold(
    t,
    B0,
    doubling_time,
)

boundary_scan_dt = 0.01

# SSA parameters
n_trajectories = 100_000
seed = 20260908
confidence_level = 0.95

# Common output time grid
t_grid = np.linspace(
    0.0,
    20.0,
    2001,
)

# Soft-killing parameters
gamma_values = [
    0.5,
    1.0,
    2.0,
    4.0,
    6.0,
    8.0,
]

# Truncation orders
N_values = list(
    range(
        N_low,
        30,
        2,
    )
)

# N-convergence tolerance
tol_N = 1e-6
# ============================================================
# Existing solver wrappers
# ============================================================

def hard_curve():
    result = hard.hard_survival_ssa(
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

    return result


def binomial_curve(gamma, N):
    H, f = binomial.soft_survival_time_boundary(
        t_grid=t_grid,
        beta_function=beta_function,
        N=N,
        gamma=gamma,
        k1=k1,
        k2=k2,
        mu=mu,
        delta=delta,
        theta=theta,
        n0=n0,
        p_off0=p_off0,
        p_on0=p_on0,
        boundary_scan_dt=boundary_scan_dt,
    )

    return H, f


def raw_curve(gamma, N):
    H, f = raw.soft_raw_survival_time_boundary(
        t_grid=t_grid,
        beta_function=beta_function,
        N=N,
        gamma=gamma,
        k1=k1,
        k2=k2,
        mu=mu,
        delta=delta,
        theta=theta,
        n0=n0,
        p_off0=p_off0,
        p_on0=p_on0,
        boundary_scan_dt=boundary_scan_dt,
    )

    return H, f


def choose_curve_for_gamma(
    method_name,
    solver,
    physical_check,
    gamma,
):
    """
    For one fixed gamma, increase N until two consecutive
    N-errors are below tol_N.

    If reliable N-convergence is not obtained, use the
    first-N curve as fallback.
    """
    H_previous = None

    H_converged = None
    f_converged = None
    N_converged = None

    N_first = N_values[0]
    H_first = None
    f_first = None

    consecutive_convergence = 0

    for N in N_values:

        try:
            H_gamma_N, f_gamma_N = solver(
                gamma,
                N,
            )

        except Exception as exc:

            print(
                f"{method_name}: "
                f"gamma={gamma}, N={N}: "
                "solver failed"
            )

            print(f"    {exc}")

            break

        # Save the first-N curve as fallback
        if N == N_first:
            H_first = H_gamma_N.copy()
            f_first = f_gamma_N.copy()

        # Reject nonphysical survival curves
        if not physical_check(H_gamma_N):

            print(
                f"{method_name}: "
                f"gamma={gamma}, N={N}: "
                "nonphysical curve -> "
                f"use first N={N_first}"
            )

            break

        # Compare with the previous truncation order
        if H_previous is not None:

            error_N = float(
                np.max(
                    np.abs(
                        H_gamma_N
                        - H_previous
                    )
                )
            )

            print(
                f"{method_name}: "
                f"gamma={gamma}, N={N}, "
                f"N-error={error_N:.3e}"
            )

            if error_N < tol_N:
                consecutive_convergence += 1
            else:
                consecutive_convergence = 0

            # Require two consecutive agreements
            if consecutive_convergence >= 2:

                H_converged = (
                    H_gamma_N.copy()
                )

                f_converged = (
                    f_gamma_N.copy()
                )

                N_converged = N

                break

        H_previous = H_gamma_N.copy()

    # No result was obtained even at the first N
    if H_first is None or f_first is None:
        raise RuntimeError(
            f"{method_name}: no curve was obtained "
            f"for gamma={gamma}."
        )

    # Case 1: reliable N-convergence was found
    if H_converged is not None:

        return {
            "H": H_converged,
            "f": f_converged,
            "N": N_converged,
            "converged": True,
        }

    # Case 2: use the first-N curve as fallback
    return {
        "H": H_first,
        "f": f_first,
        "N": N_first,
        "converged": False,
    }


# ============================================================
# Main comparison
# ============================================================

if __name__ == "__main__":

    # ========================================================
    # 1. Re-run hard first-passage SSA
    # ========================================================

    print("\n" + "=" * 72)
    print("RUNNING HARD FIRST-PASSAGE SSA")
    print("=" * 72)

    ssa_result = hard_curve()

    H_hard = ssa_result.survival
    ci_lower = ssa_result.ci_lower
    ci_upper = ssa_result.ci_upper

    # ========================================================
    # 2. Estimate the hard-SSA FPT density
    #    from the raw first-passage samples
    # ========================================================

    finite_fpt = (
        ssa_result.first_passage_times[
            np.isfinite(
                ssa_result.first_passage_times
            )
        ]
    )

    # Use a coarser grid for the density histogram.
    # This avoids an excessively noisy density plot.
    n_density_bins = 100

    density_bin_edges = np.linspace(
        float(t_grid[0]),
        float(t_grid[-1]),
        n_density_bins + 1,
    )

    density_bin_widths = np.diff(
        density_bin_edges
    )

    fpt_counts, _ = np.histogram(
        finite_fpt,
        bins=density_bin_edges,
    )

    # Unconditional SSA first-passage density estimate
    f_hard = (
        fpt_counts
        / ssa_result.n_trajectories
        / density_bin_widths
    )

    density_bin_centers = 0.5 * (
        density_bin_edges[:-1]
        + density_bin_edges[1:]
    )

    # Check consistency between the histogram mass
    # and the final survival probability.
    histogram_mass = float(
        np.sum(
            f_hard * density_bin_widths
        )
    )

    print(
        "SSA density histogram mass = "
        f"{histogram_mass:.6f}"
    )

    print(
        "1 - H_SSA(t_max) = "
        f"{1.0 - H_hard[-1]:.6f}"
    )

    # ========================================================
    # 3. Re-run the two moment-truncation methods
    # ========================================================

    binomial_results = {}
    raw_results = {}

    for gamma in gamma_values:

        print("\n" + "=" * 72)
        print(f"gamma = {gamma}")
        print("=" * 72)

        # Binomial-moment truncation
        binomial_results[gamma] = (
            choose_curve_for_gamma(
                method_name="BINOMIAL",
                solver=binomial_curve,
                physical_check=(
                    binomial.is_physical_survival
                ),
                gamma=gamma,
            )
        )

        # Raw-moment truncation
        raw_results[gamma] = (
            choose_curve_for_gamma(
                method_name="RAW",
                solver=raw_curve,
                physical_check=(
                    raw.is_physical_survival
                ),
                gamma=gamma,
            )
        )

    # ========================================================
    # 4. Print survival errors against hard SSA
    # ========================================================

    print("\n" + "=" * 90)
    print("COMPARISON WITH HARD SSA")
    print("=" * 90)

    for gamma in gamma_values:

        binomial_result = (
            binomial_results[gamma]
        )

        raw_result = raw_results[gamma]

        error_binomial_H = float(
            np.max(
                np.abs(
                    binomial_result["H"]
                    - H_hard
                )
            )
        )

        error_raw_H = float(
            np.max(
                np.abs(
                    raw_result["H"]
                    - H_hard
                )
            )
        )

        binomial_tag = (
            f"N*={binomial_result['N']}"
            if binomial_result["converged"]
            else (
                "fallback "
                f"N={binomial_result['N']}"
            )
        )

        raw_tag = (
            f"N*={raw_result['N']}"
            if raw_result["converged"]
            else (
                "fallback "
                f"N={raw_result['N']}"
            )
        )

        print(
            f"gamma={gamma:>4}: "
            f"BINOMIAL {binomial_tag:>14}, "
            f"H-error={error_binomial_H:.3e} | "
            f"RAW {raw_tag:>14}, "
            f"H-error={error_raw_H:.3e}"
        )

    # ========================================================
    # 5. Plot the four comparison panels
    # ========================================================

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(14, 10),
    )

    # ========================================================
    # Upper-left: binomial survival versus hard SSA
    # ========================================================

    ax = axes[0, 0]

    ax.fill_between(
        t_grid,
        ci_lower,
        ci_upper,
        step="post",
        color="gray",
        alpha=0.3,
        label="SSA pointwise 95% CI",
    )

    ax.step(
        t_grid,
        H_hard,
        where="post",
        color="black",
        linewidth=2.5,
        label=r"$H_{\mathrm{hard}}^{\mathrm{SSA}}(t)$",
    )

    for gamma in gamma_values:

        result = binomial_results[gamma]

        linestyle = (
            "-"
            if result["converged"]
            else "--"
        )

        if result["converged"]:

            label = (
                fr"$\gamma={gamma},\ "
                fr"N^*={result['N']}$"
            )

        else:

            label = (
                fr"$\gamma={gamma},\ "
                fr"\mathrm{{fallback}}\ "
                fr"N={result['N']}$"
            )

        ax.plot(
            t_grid,
            result["H"],
            linestyle=linestyle,
            linewidth=1.5,
            label=label,
        )

    ax.set_title(
        "Binomial moments vs hard SSA"
    )

    ax.set_xlabel("t")
    ax.set_ylabel("Survival probability")
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    # ========================================================
    # Upper-right: raw survival versus hard SSA
    # ========================================================

    ax = axes[0, 1]

    ax.fill_between(
        t_grid,
        ci_lower,
        ci_upper,
        step="post",
        color="gray",
        alpha=0.3,
        label="SSA pointwise 95% CI",
    )

    ax.step(
        t_grid,
        H_hard,
        where="post",
        color="black",
        linewidth=2.5,
        label=r"$H_{\mathrm{hard}}^{\mathrm{SSA}}(t)$",
    )

    for gamma in gamma_values:

        result = raw_results[gamma]

        linestyle = (
            "-"
            if result["converged"]
            else "--"
        )

        if result["converged"]:

            label = (
                fr"$\gamma={gamma},\ "
                fr"N^*={result['N']}$"
            )

        else:

            label = (
                fr"$\gamma={gamma},\ "
                fr"\mathrm{{fallback}}\ "
                fr"N={result['N']}$"
            )

        ax.plot(
            t_grid,
            result["H"],
            linestyle=linestyle,
            linewidth=1.5,
            label=label,
        )

    ax.set_title(
        "Raw moments vs hard SSA"
    )

    ax.set_xlabel("t")
    ax.set_ylabel("Survival probability")
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    # ========================================================
    # Lower-left: binomial soft density and SSA histogram
    # ========================================================

    ax = axes[1, 0]

    ax.stairs(
        f_hard,
        density_bin_edges,
        color="black",
        linewidth=2.0,
        fill=False,
        label="Hard SSA histogram density",
    )

    for gamma in gamma_values:

        result = binomial_results[gamma]

        linestyle = (
            "-"
            if result["converged"]
            else "--"
        )

        if result["converged"]:

            label = (
                fr"$\gamma={gamma},\ "
                fr"N^*={result['N']}$"
            )

        else:

            label = (
                fr"$\gamma={gamma},\ "
                fr"\mathrm{{fallback}}\ "
                fr"N={result['N']}$"
            )

        ax.plot(
            t_grid,
            result["f"],
            linestyle=linestyle,
            linewidth=1.5,
            label=label,
        )

    ax.set_title(
        "Binomial soft FPT density"
    )

    ax.set_xlabel("t")
    ax.set_ylabel("Density")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    # ========================================================
    # Lower-right: raw soft density and SSA histogram
    # ========================================================

    ax = axes[1, 1]

    ax.stairs(
        f_hard,
        density_bin_edges,
        color="black",
        linewidth=2.0,
        fill=False,
        label="Hard SSA histogram density",
    )

    for gamma in gamma_values:

        result = raw_results[gamma]

        linestyle = (
            "-"
            if result["converged"]
            else "--"
        )

        if result["converged"]:

            label = (
                fr"$\gamma={gamma},\ "
                fr"N^*={result['N']}$"
            )

        else:

            label = (
                fr"$\gamma={gamma},\ "
                fr"\mathrm{{fallback}}\ "
                fr"N={result['N']}$"
            )

        ax.plot(
            t_grid,
            result["f"],
            linestyle=linestyle,
            linewidth=1.5,
            label=label,
        )

    ax.set_title(
        "Raw soft FPT density"
    )

    ax.set_xlabel("t")
    ax.set_ylabel("Density")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    # ========================================================
    # 6. Overall title and layout
    # ========================================================

    fig.suptitle(
        "Time-dependent boundary: "
        "binomial and raw moments vs hard SSA\n"
        r"$B(t)=\lceil\beta(t)\rceil$",
        fontsize=14,
    )

    fig.tight_layout(
        rect=(0.0, 0.0, 1.0, 0.95)
    )

    plt.show()