from pathlib import Path
import importlib.util

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({
    "font.family": "STIXGeneral",
    "mathtext.fontset": "stix",
    "font.size": 8,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 7,
    "legend.frameon": False,
    "axes.linewidth": 0.7,
    "lines.linewidth": 1.2,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "axes.grid": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})
plt.rcParams["axes.prop_cycle"] = plt.cycler(
    color=["#4C78A8", "#59A14F", "#E15759"]
)

# ============================================================
# Load the three existing calculation files from this folder
# ============================================================

HERE = Path(__file__).resolve().parent

BINOMIAL_FILE = "soft_binomial_moment_constant_boundary.py"
RAW_FILE = "soft_raw_moment_constant_boundary.py"
HARD_FILE = "hard_exact_constant_boundary.py"


def load_module(filename, module_name):
    path = HERE / filename

    if not path.exists():
        raise FileNotFoundError(
            f"Cannot find {filename}\n"
            f"Put this file and the three calculation files in the same folder."
        )

    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


binomial = load_module(BINOMIAL_FILE, "binomial_module")
raw = load_module(RAW_FILE, "raw_module")
hard = load_module(HARD_FILE, "hard_module")


# ============================================================
# Fixed physical parameters
# ============================================================

B = 8
theta = 0.5

k1 = 1.0
k2 = 1.0
mu = 5.0
delta = 1.0

n0 = 0
p_off0 = 1.0
p_on0 = 0.0

t_grid = np.linspace(0.0, 200.0, 2001)

# gamma keeps increasing
gamma_values = [0.5, 2.0, 8.0]

# For each fixed gamma, increase N
N_values = list(range(B, 90, 2))

# Require two consecutive N-errors below this threshold
tol_N = 1e-6


# ============================================================
# Existing solver wrappers
# ============================================================

def hard_curve():
    H, f, _ = hard.hard_survival_exact(
        t_grid,
        B=B,
        k1=k1,
        k2=k2,
        mu=mu,
        delta=delta,
        n0=n0,
        p_off0=p_off0,
        p_on0=p_on0,
    )

    return H, f


def binomial_curve(gamma, N):
    H, f, _  = binomial.soft_survival_truncated(
        t_grid=t_grid,
        B=B,
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
    )

    return H, f


def raw_curve(gamma, N):
    H, f = raw.soft_raw_survival_truncated(
        t_grid=t_grid,
        B=B,
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
    )

    return H, f


# ============================================================
# Same adaptive-N logic as your original plotting code
#
# For each fixed gamma:
#   N = B, B+2, B+4, ...
#
# Compare H_{gamma,N} with the previous N.
# If N-error < tol_N twice consecutively:
#       accept the current N and stop.
#
# If convergence is not found, or a nonphysical curve appears:
#       use the first N = B as fallback.
# ============================================================

def choose_curve_for_gamma(
    method_name,
    solver,
    physical_check,
    gamma,
):

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
            H_gamma_N, f_gamma_N = solver(gamma,N)
        except Exception as exc:

            print(
                f"{method_name}: gamma={gamma}, N={N}: "
                f"solver failed -> use first N={N_first}"
            )

            print(f"    {exc}")

            break

        # Save the first-N curve
        if N == N_first:
            H_first = H_gamma_N.copy()
            f_first = f_gamma_N.copy()

        # Reject nonphysical curve
        if not physical_check(H_gamma_N):

            print(
                f"{method_name}: gamma={gamma}, N={N}: "
                f"nonphysical -> use first N={N_first}"
            )

            break

        # Compare with previous N
        if H_previous is not None:

            error_N = np.max(
                np.abs(H_gamma_N - H_previous)
            )

            print(
                f"{method_name}: gamma={gamma}, N={N}, "
                f"N-error={error_N:.3e}"
            )

            if error_N < tol_N:
                consecutive_convergence += 1
            else:
                consecutive_convergence = 0

            # Require two consecutive agreements
            if consecutive_convergence >= 2:

                H_converged = H_gamma_N.copy()
                f_converged = f_gamma_N.copy()
                N_converged = N

                break

        H_previous = H_gamma_N.copy()

    # Case 1: N-convergence found
    if H_converged is not None:

        return {
            "H": H_converged,
            "f": f_converged,
            "N": N_converged,
            "converged": True,
        }

    # Case 2: no convergence / instability -> first N
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

    H_hard, f_hard = hard_curve()

    binomial_results = {}
    raw_results = {}

    # --------------------------------------------------------
    # gamma increases one by one
    # --------------------------------------------------------

    for gamma in gamma_values:

        print("\n" + "=" * 68)
        print(f"gamma = {gamma}")
        print("=" * 68)

        # Binomial moments
        binomial_results[gamma] = choose_curve_for_gamma(
            method_name="BINOMIAL",
            solver=binomial_curve,
            physical_check=binomial.is_physical_survival,
            gamma=gamma,
        )

        # Raw moments
        raw_results[gamma] = choose_curve_for_gamma(
            method_name="RAW",
            solver=raw_curve,
            physical_check=raw.is_physical_survival,
            gamma=gamma,
        )

    # ========================================================
    # Print errors against Hard exact
    # ========================================================

    print("\n" + "=" * 90)
    print("COMPARISON WITH HARD EXACT")
    print("=" * 90)

    for gamma in gamma_values:

        b = binomial_results[gamma]
        r = raw_results[gamma]

        err_bin = np.max(
            np.abs(b["H"] - H_hard)
        )

        err_raw = np.max(
            np.abs(r["H"] - H_hard)
        )

        err_bin_f = np.max(
            np.abs(b["f"] - f_hard)
        )

        err_raw_f = np.max(
            np.abs(r["f"] - f_hard)
        )

        b_tag = (
            f"N*={b['N']}"
            if b["converged"]
            else f"fallback N={b['N']}"
        )

        r_tag = (
            f"N*={r['N']}"
            if r["converged"]
            else f"fallback N={r['N']}"
        )

        print(
            f"gamma={gamma:>4}: "
            f"BINOMIAL {b_tag:>14}, "
            f"H-error={err_bin:.3e}, "
            f"f-error={err_bin_f:.3e} | "
            f"RAW {r_tag:>14}, "
            f"H-error={err_raw:.3e}, "
            f"f-error={err_raw_f:.3e}"
        )

    # ========================================================
    # Plot:
    # left  = binomial vs Hard
    # right = raw vs Hard
    # ========================================================

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(7.2, 5.2),
    )

    # ---------------- Binomial ----------------

    ax = axes[0,0]

    ax.plot(
        t_grid,
        H_hard,
        color="0.5",
linestyle=":",
linewidth=2.0,
        label="Hard exact",
    )

    for gamma in gamma_values:

        result = binomial_results[gamma]

        linestyle = "-"

        if result["converged"]:
            label = fr"$\gamma={gamma:g}$"
        else:
            label = fr"$\gamma={gamma:g}$"

        ax.plot(
            t_grid,
            result["H"],
            linestyle,
            label=label,
        )

    ax.set_title(
    "(a) Binomial moments survival probability",
    loc="left", x=-0.06, y=1.04, fontsize=11
)
    ax.set_xlabel("t")
    ax.set_ylabel("Survival H(t)")
    ax.grid(False)
    

    # ---------------- Raw ----------------

    ax = axes[1,0]

    ax.plot(
        t_grid,
        H_hard,
        color="0.5",
linestyle=":",
linewidth=2.0,
        label="Hard exact",
    )

    for gamma in gamma_values:

        result = raw_results[gamma]

        linestyle = "-"

        if result["converged"]:
            label = fr"$\gamma={gamma:g}$"
        else:
            label = fr"$\gamma={gamma:g}$"

        ax.plot(
            t_grid,
            result["H"],
            linestyle,
            label=label,
        )

    ax.set_title(
        "(c) Raw moments survival probability",
        loc="left", x=-0.06, y=1.04, fontsize=11
    )
    ax.set_xlabel("t")
    ax.grid(False)
    

    # ---------------- Binomial density ----------------

    ax = axes[0,1]

    for gamma in gamma_values:

        result = binomial_results[gamma]

        linestyle = "-"

        label = (
            fr"$\gamma={gamma:g}$"
            if result["converged"]
            else
            fr"$\gamma={gamma:g}$"
        )

        ax.plot(
            t_grid,
            result["f"],
            linestyle,
            label=label,
        )

    ax.plot(
                t_grid,
                f_hard,
                color="0.5",
linestyle=":",
linewidth=2.0,
                label="Hard exact",
            )    

    ax.set_title(
        "(b) Binomial moment FPT density",
        loc="left", x=-0.06, y=1.04, fontsize=11
    )

    ax.set_xlabel("t")

    ax.set_ylabel("f(t)")

    ax.grid(False)

    

    # ---------------- Raw density ----------------

    ax = axes[1,1]

    for gamma in gamma_values:

        result = raw_results[gamma]

        linestyle = "-"

        label = (
            fr"$\gamma={gamma:g}$"
            if result["converged"]
            else
            fr"$\gamma={gamma:g}$"
        )

        ax.plot(
            t_grid,
            result["f"],
            linestyle,
            label=label,
        )

    ax.plot(
                t_grid,
                f_hard,
                color="0.5",
linestyle=":",
linewidth=2.0,
                label="Hard exact",
            )    

    ax.set_title(
        "(d) Raw moments FPT density",
        loc="left", x=-0.06, y=1.04, fontsize=11
    )

    ax.set_xlabel("t")

    ax.set_ylabel("f(t)")

    ax.grid(False)

    
    for row in range(2):
        axes[row, 0].set_ylabel(r"$H(t)$")
        axes[row, 1].set_ylabel(r"$f_T(t)$")

        axes[row, 0].set_ylim(0, 1)
        axes[row, 1].set_ylim(0, 0.10)

        for col in range(2):
            axes[row, col].set_xlabel(r"$t$")
            axes[row, col].tick_params(
                which="both", top=False, right=False
            )
            axes[row, col].spines["top"].set_visible(False)
            axes[row, col].spines["right"].set_visible(False)

    handles, labels = axes[0, 1].get_legend_handles_labels()

    axes[0, 1].legend(
        [handles[-1]] + handles[:-1],
        [labels[-1]] + labels[:-1],
        loc="upper right",
        frameon=False,
        fontsize=7,
    )

    fig.tight_layout()
    fig.savefig(
        HERE / "constant_boundary_comparison.png",
        dpi=900,
        bbox_inches="tight",
    )
    fig.savefig(
        HERE / "constant_boundary_comparison.pdf",
        bbox_inches="tight",
    )
    plt.show()
