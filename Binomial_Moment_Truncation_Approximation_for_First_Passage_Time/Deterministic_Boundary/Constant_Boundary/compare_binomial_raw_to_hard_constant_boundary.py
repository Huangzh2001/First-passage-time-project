from pathlib import Path
import importlib.util

import numpy as np
import matplotlib.pyplot as plt



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
gamma_values = [0.5, 1.0, 2.0, 4.0, 6.0, 8.0]

# For each fixed gamma, increase N
N_values = list(range(B, 30, 2))

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
        figsize=(14, 10),
    )

    # ---------------- Binomial ----------------

    ax = axes[0,0]

    ax.plot(
        t_grid,
        H_hard,
        "k-",
        linewidth=2.5,
        label="Hard exact",
    )

    for gamma in gamma_values:

        result = binomial_results[gamma]

        linestyle = "-" if result["converged"] else "--"

        if result["converged"]:
            label = fr"$\gamma={gamma},\ N^*={result['N']}$"
        else:
            label = fr"$\gamma={gamma},\ N={result['N']}$"

        ax.plot(
            t_grid,
            result["H"],
            linestyle,
            label=label,
        )

    ax.set_title("Binomial moments vs Hard exact")
    ax.set_xlabel("t")
    ax.set_ylabel("Survival H(t)")
    ax.grid(True)
    ax.legend(fontsize=8)

    # ---------------- Raw ----------------

    ax = axes[0,1]

    ax.plot(
        t_grid,
        H_hard,
        "k-",
        linewidth=2.5,
        label="Hard exact",
    )

    for gamma in gamma_values:

        result = raw_results[gamma]

        linestyle = "-" if result["converged"] else "--"

        if result["converged"]:
            label = fr"$\gamma={gamma},\ N^*={result['N']}$"
        else:
            label = fr"$\gamma={gamma},\ N={result['N']}$"

        ax.plot(
            t_grid,
            result["H"],
            linestyle,
            label=label,
        )

    ax.set_title("Raw moments vs Hard exact")
    ax.set_xlabel("t")
    ax.grid(True)
    ax.legend(fontsize=8)

    # ---------------- Binomial density ----------------

    ax = axes[1,0]

    for gamma in gamma_values:

        result = binomial_results[gamma]

        linestyle = "-" if result["converged"] else "--"

        label = (
            fr"$\gamma={gamma},\ N^*={result['N']}$"
            if result["converged"]
            else
            fr"$\gamma={gamma},\ N={result['N']}$"
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
                "k-",
                linewidth=2.5,
                label="Hard exact",
            )    

    ax.set_title("Binomial FPT density")

    ax.set_xlabel("t")

    ax.set_ylabel("f(t)")

    ax.grid(True)

    ax.legend(fontsize=8)

    # ---------------- Raw density ----------------

    ax = axes[1,1]

    for gamma in gamma_values:

        result = raw_results[gamma]

        linestyle = "-" if result["converged"] else "--"

        label = (
            fr"$\gamma={gamma},\ N^*={result['N']}$"
            if result["converged"]
            else
            fr"$\gamma={gamma},\ N={result['N']}$"
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
                "k-",
                linewidth=2.5,
                label="Hard exact",
            )    

    ax.set_title("Raw FPT density")

    ax.set_xlabel("t")

    ax.set_ylabel("f(t)")

    ax.grid(True)

    ax.legend(fontsize=8)

    fig.suptitle(
        "Constant-boundary comparison: "
        "gamma increases, adaptive N for each gamma"
    )

    fig.tight_layout()

    plt.show()
