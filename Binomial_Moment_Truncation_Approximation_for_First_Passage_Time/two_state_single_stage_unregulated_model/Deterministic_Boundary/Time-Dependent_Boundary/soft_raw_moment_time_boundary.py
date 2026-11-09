import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm
from math import ceil, pi
from typing import Callable
from scipy.optimize import brentq


BoundaryFunction = Callable[[float], float]


def sinusoidal_threshold(
    t: float,
    B0: float = 8.0,
    amplitude: float = 2.0,
    period: float = 10.0,
) -> float:
    if period <= 0.0:
        raise ValueError("period must be positive.")

    return B0 + amplitude * np.sin(
        2.0 * pi * t / period
    )

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

def find_boundary_switch_times(
    beta_function: BoundaryFunction,
    t_start: float,
    t_end: float,
    scan_dt: float = 0.01,
) -> np.ndarray:
    """
    Find all times at which B(t)=ceil(beta(t)) changes.

    The function first scans beta(t) on a fine time grid. When an
    integer-boundary change is detected, Brent's method is used to
    locate the corresponding integer-level crossing accurately.
    """
    if t_start < 0.0:
        raise ValueError("t_start must be nonnegative.")

    if t_end < t_start:
        raise ValueError("Need t_end >= t_start.")

    if scan_dt <= 0.0:
        raise ValueError("scan_dt must be positive.")

    if t_end == t_start:
        return np.empty(0, dtype=float)

    n_steps = max(
        1,
        int(np.ceil(
            (t_end - t_start) / scan_dt
        )),
    )

    scan_times = np.linspace(
        t_start,
        t_end,
        n_steps + 1,
    )

    beta_values = np.asarray(
        [
            beta_function(t)
            for t in scan_times
        ],
        dtype=float,
    )

    if np.any(~np.isfinite(beta_values)):
        raise ValueError(
            "beta_function returned a non-finite value."
        )

    switch_times = []

    for (
        t_left,
        t_right,
        beta_left,
        beta_right,
    ) in zip(
        scan_times[:-1],
        scan_times[1:],
        beta_values[:-1],
        beta_values[1:],
    ):
        B_left = int(ceil(beta_left))
        B_right = int(ceil(beta_right))

        # No integer-boundary change in this scan interval
        if B_left == B_right:
            continue

        # There may be more than one integer level crossed
        for level in range(
            min(B_left, B_right),
            max(B_left, B_right),
        ):
            gap_left = beta_left - level
            gap_right = beta_right - level

            if gap_left == 0.0:
                root = float(t_left)

            elif gap_right == 0.0:
                root = float(t_right)

            elif gap_left * gap_right < 0.0:
                root = float(
                    brentq(
                        lambda t, k=level: (
                            float(beta_function(t)) - k
                        ),
                        float(t_left),
                        float(t_right),
                        xtol=1e-12,
                        rtol=1e-12,
                    )
                )

            else:
                continue

            if t_start < root < t_end:
                switch_times.append(root)

    if len(switch_times) == 0:
        return np.empty(0, dtype=float)

    switch_times.sort()

    # Remove duplicated roots caused by adjacent scan intervals
    unique_switch_times = [switch_times[0]]

    for switch_time in switch_times[1:]:
        if (
            abs(
                switch_time
                - unique_switch_times[-1]
            )
            > 1e-10
        ):
            unique_switch_times.append(
                switch_time
            )

    return np.asarray(
        unique_switch_times,
        dtype=float,
    )

def build_L_raw(B, N, gamma, k1, k2, mu, delta, theta=0.5):
    """Build the constant-boundary soft RAW-moment truncation matrix."""

    q = math.exp(gamma * (theta - B))
    L = np.zeros((2 * (N + 1), 2 * (N + 1)), dtype=float)

    def idx(k, state):
        return 2 * k + state

    # gamma^r / r!
    tay = np.empty(N + 1)
    tay[0] = 1.0
    for r in range(1, N + 1):
        tay[r] = tay[r - 1] * gamma / r

    for k in range(N + 1):
        off = idx(k, 0)
        on = idx(k, 1)

        # promoter switching
        L[off, idx(k, 0)] += -k2
        L[off, idx(k, 1)] += k1
        L[on, idx(k, 0)] += k2
        L[on, idx(k, 1)] += -k1

        # production in ON: mu * [(n+1)^k - n^k]
        for ell in range(1, k + 1):
            L[on, idx(k - ell, 1)] += mu * math.comb(k, ell)

        # degradation: delta*n * [(n-1)^k - n^k]
        for ell in range(1, k + 1):
            j = k - ell + 1
            coeff = delta * math.comb(k, ell) * ((-1.0) ** ell)
            L[off, idx(j, 0)] += coeff
            L[on, idx(j, 1)] += coeff

        # soft killing:
        # -q * sum_{r=0}^{N-k} gamma^r/r! * u_{k+r}
        for r in range(N - k + 1):
            coeff = -q * tay[r]
            L[off, idx(k + r, 0)] += coeff
            L[on, idx(k + r, 1)] += coeff

    return L


def initial_raw(N, n0, p_off0=1.0, p_on0=0.0):
    """Initial raw moments: u_k(0) = promoter probability * n0^k."""

    y0 = np.zeros(2 * (N + 1), dtype=float)

    for k in range(N + 1):
        value = 1.0 if k == 0 else float(n0 ** k)
        y0[2 * k] = p_off0 * value
        y0[2 * k + 1] = p_on0 * value

    return y0



def soft_raw_survival_time_boundary(
    t_grid,
    beta_function,
    N,
    gamma,
    k1,
    k2,
    mu,
    delta,
    theta=0.5,
    n0=0,
    p_off0=1.0,
    p_on0=0.0,
    boundary_scan_dt=0.01,
):
    """
    Compute H_{gamma,N}(t) and f_{gamma,N}(t) for

        B(t) = ceil(beta(t)).

    The original constant-boundary matrix is used separately
    on every time interval where B(t) remains constant.
    """
    B_initial = integer_boundary(
        0.0,
        beta_function,
    )

    if not (0 <= n0 < B_initial):
        raise ValueError(
            "For the benchmark, need 0 <= n0 < B(0)."
        )

    t_grid = np.asarray(t_grid, dtype=float)

    if t_grid.ndim != 1:
        raise ValueError("t_grid must be one-dimensional.")
    if np.any(t_grid < 0):
        raise ValueError("All times must be nonnegative.")

    if len(t_grid) == 0:
        raise ValueError(
            "t_grid must not be empty."
        )

    if np.any(np.diff(t_grid) <= 0.0):
        raise ValueError(
            "t_grid must be strictly increasing."
        )

    a0 = initial_raw(N, n0, p_off0, p_on0)

    switch_times = find_boundary_switch_times(
        beta_function=beta_function,
        t_start=0.0,
        t_end=float(t_grid[-1]),
        scan_dt=boundary_scan_dt,
    )

    a_t = a0.copy()

    current_time = 0.0
    switch_index = 0

    # Compute exp(t L) @ a0 directly, without forming exp(t L)

    H = np.empty(len(t_grid), dtype=float)
    f = np.empty(len(t_grid), dtype=float)

    for j, target_time in enumerate(t_grid):

        target_time = float(target_time)

        while (
            switch_index < len(switch_times)
            and switch_times[switch_index] <= target_time
        ):
            switch_time = float(
                switch_times[switch_index]
            )

            if switch_time > current_time:

                midpoint = 0.5 * (
                    current_time + switch_time
                )

                B_current = integer_boundary(
                    midpoint,
                    beta_function,
                )

                L_current = build_L_raw(
                    B_current,
                    N,
                    gamma,
                    k1,
                    k2,
                    mu,
                    delta,
                    theta,
                )

                dt = switch_time - current_time

                a_t = expm(
                    dt * L_current
                ) @ a_t

                current_time = switch_time

            switch_index += 1

        if target_time > current_time:
            midpoint = 0.5 * (
                    current_time + target_time
            )

            B_current = integer_boundary(
                midpoint,
                beta_function,
            )

            L_current = build_L_raw(
                B_current,
                N,
                gamma,
                k1,
                k2,
                mu,
                delta,
                theta,
            )

            dt = target_time - current_time

            a_t = expm(
                dt * L_current
            ) @ a_t

            current_time = target_time

        # a_t = expm(t * L) @ a0

        H[j] = a_t[0] + a_t[1]

        # da_t = L @ a_t

        B_now = integer_boundary(
            target_time,
            beta_function,
        )

        L_now = build_L_raw(
            B_now,
            N,
            gamma,
            k1,
            k2,
            mu,
            delta,
            theta,
        )

        da_t = L_now @ a_t

        f[j] = -(da_t[0] + da_t[1])

    return H, f


def is_physical_survival(H, tol=1e-8):
    """
    Check whether H behaves like a survival probability.
    """
    return (
        np.all(np.isfinite(H))
        and np.min(H) >= -tol
        and np.max(H) <= 1.0 + tol
        and np.all(np.diff(H) <= tol)
    )


if __name__ == "__main__":

    # =========================================================
    # Model parameters
    # =========================================================


    theta = 0.5

    # ========================================================
    # Time-dependent boundary
    # ========================================================

    # Sinusoidal boundary: currently disabled
    # B0 = 8.0
    # amplitude = 2.0
    # period = 10.0
    #
    # beta_function = lambda t: sinusoidal_threshold(
    #     t,
    #     B0,
    #     amplitude,
    #     period,
    # )

    # Cell-growth boundary: currently enabled
    B0 = 6.0
    doubling_time = 20.0

    beta_function = lambda t: cell_growth_threshold(
        t,
        B0,
        doubling_time,
    )

    boundary_scan_dt = 0.01

    k1 = 1.0
    k2 = 1.0
    mu = 5.0
    delta = 1.0

    n0 = 0
    p_off0 = 1.0
    p_on0 = 0.0

    # =========================================================
    # Time grid
    # =========================================================

    t_grid = np.linspace(0.0, 20.0, 2001)

    # =========================================================
    # gamma and N scan
    # =========================================================

    gamma_values = [0.5, 1.0, 2.0, 4.0, 6.0, 8.0]

    # For each fixed gamma, increase N
    N_min = 8
    N_max = 28
    N_step = 2

    N_values = list(
        range(
            N_min,
            N_max + 1,
            N_step,
        )
    )

    # N-convergence tolerance
    tol_N = 1e-6

    plt.figure()

    # =========================================================
    # For each gamma, first search for N-convergence
    # =========================================================

    for gamma in gamma_values:

        H_previous = None

        H_converged = None
        N_converged = None

        # Save the first-N curve as fallback
        N_first = N_values[0]
        H_first = None

        consecutive_convergence = 0

        # -----------------------------------------------------
        # Increase N for fixed gamma
        # -----------------------------------------------------

        for N in N_values:

            H_gamma_N, f_gamma_N = (
                soft_raw_survival_time_boundary(
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
            )

            # Save the first-N result
            if N == N_first:
                H_first = H_gamma_N.copy()

            # -------------------------------------------------
            # Check whether this curve is physical
            # -------------------------------------------------

            if not is_physical_survival(H_gamma_N):

                print(
                    f"gamma={gamma}, N={N}: "
                    f"nonphysical -> stop N scan, "
                    f"use first N={N_first}."
                )

                break

            # -------------------------------------------------
            # Compare with previous N
            # -------------------------------------------------

            if H_previous is not None:

                error_N = np.max(
                    np.abs(H_gamma_N - H_previous)
                )

                print(
                    f"gamma={gamma}, N={N}, "
                    f"N-error={error_N:.3e}"
                )

                # One successful N-agreement
                if error_N < tol_N:
                    consecutive_convergence += 1

                else:
                    consecutive_convergence = 0

                # Require two consecutive agreements
                if consecutive_convergence >= 2:

                    H_converged = H_gamma_N.copy()
                    N_converged = N

                    break

            # Current N becomes previous N
            H_previous = H_gamma_N.copy()

        # =====================================================
        # Plot only ONE curve for this gamma
        # =====================================================

        # -----------------------------------------------------
        # Case 1:
        # N-convergence was found
        # -----------------------------------------------------

        if H_converged is not None:

            plt.plot(
                t_grid,
                H_converged,
                label=fr"$\gamma={gamma},\ N^*={N_converged}$",
            )

            print(
                f"gamma={gamma}: "
                f"converged at N={N_converged}"
            )

        # -----------------------------------------------------
        # Case 2:
        # no reliable convergence / instability
        # -> use first N as reference
        # -----------------------------------------------------

        else:

            plt.plot(
                t_grid,
                H_first,
                "--",
                label=fr"$\gamma={gamma},\ N={N_first}$",
            )

            print(
                f"gamma={gamma}: "
                f"no reliable N-convergence -> "
                f"use first N={N_first}"
            )

    # =========================================================
    # Plot
    # =========================================================

    plt.xlabel("t")

    plt.ylabel(
        r"$H_{\gamma,N}^{raw}(t)$"
    )

    plt.title(
        "Soft raw-moment survival"
    )

    plt.grid(True)

    plt.legend()

    plt.show()