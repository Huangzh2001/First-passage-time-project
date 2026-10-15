import numpy as np
from math import ceil, comb, pi
import matplotlib.pyplot as plt
from scipy.sparse.linalg import expm_multiply
import multiprocessing as mp
from scipy.sparse import csr_matrix
from scipy.linalg import expm
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

def build_L_N(B, N, gamma, k1, k2, mu, delta, theta=0.5):
    """
    Build the constant matrix L_N for the truncated unnormalized
    binomial-moment system

        a_N'(t) = L_N a_N(t)

    using the ordering

        [a_0^(0), a_0^(1), a_1^(0), a_1^(1), ..., a_N^(0), a_N^(1)]^T.

    The corrected soft-killing rate is

        V_gamma(n) = exp(gamma * (n - B + theta)),

    so

        c = exp(gamma),
        q_theta = exp(-gamma * (B - theta)),

    and V_gamma(n) = q_theta * c^n.
    """
    if B <= 0:
        raise ValueError("B must be a positive integer.")
    if N < 0:
        raise ValueError("N must be a nonnegative integer.")
    if gamma <= 0:
        raise ValueError("gamma must be positive.")
    if not (0.0 < theta < 1.0):
        raise ValueError("theta must satisfy 0 < theta < 1.")
    if min(k1, k2, mu, delta) < 0:
        raise ValueError("k1, k2, mu, delta must be nonnegative.")

    c = np.exp(gamma)
    q_theta = np.exp(-gamma * (B - theta))

    dim = 2 * (N + 1)
    L = np.zeros((dim, dim), dtype=float)

    def idx(m, promoter_state):
        # promoter_state = 0 for OFF, 1 for ON
        return 2 * m + promoter_state

    for m in range(N + 1):
        row_off = idx(m, 0)
        row_on = idx(m, 1)

        # ------------------------------------------------------------
        # OFF equation:
        #
        # da_m^(0)/dt
        # = -k2 a_m^(0) + k1 a_m^(1)
        #   - delta*m*a_m^(0)
        #   - q_theta*c^m * sum_{r=0}^{N-m}
        #       C(m+r,m)(c-1)^r a_{m+r}^(0)
        # ------------------------------------------------------------
        L[row_off, idx(m, 0)] += -k2 - delta * m
        L[row_off, idx(m, 1)] += k1

        # ------------------------------------------------------------
        # ON equation:
        #
        # da_m^(1)/dt
        # = k2 a_m^(0) - k1 a_m^(1)
        #   + mu a_{m-1}^(1)
        #   - delta*m*a_m^(1)
        #   - q_theta*c^m * sum_{r=0}^{N-m}
        #       C(m+r,m)(c-1)^r a_{m+r}^(1)
        # ------------------------------------------------------------
        L[row_on, idx(m, 0)] += k2
        L[row_on, idx(m, 1)] += -k1 - delta * m

        if m >= 1:
            L[row_on, idx(m - 1, 1)] += mu

        # Soft-killing terms, separately for OFF and ON.
        prefactor = q_theta * (c ** m)

        for r in range(N - m + 1):
            j = m + r
            coeff = -prefactor * comb(m + r, m) * ((c - 1.0) ** r)

            L[row_off, idx(j, 0)] += coeff
            L[row_on, idx(j, 1)] += coeff

    return L


def initial_a_N(N, n0, p_off0=1.0, p_on0=0.0):
    """
    Initial vector for the unnormalized binomial moments:

        a_{m,N}^{(0)}(0) = p_off0 * C(n0,m)
        a_{m,N}^{(1)}(0) = p_on0  * C(n0,m)
    """
    if n0 < 0:
        raise ValueError("n0 must be nonnegative.")
    if not np.isclose(p_off0 + p_on0, 1.0):
        raise ValueError("Need p_off0 + p_on0 = 1.")
    if p_off0 < 0 or p_on0 < 0:
        raise ValueError("Initial promoter probabilities must be nonnegative.")

    a0 = np.zeros(2 * (N + 1), dtype=float)

    for m in range(N + 1):
        binom = comb(n0, m) if m <= n0 else 0
        a0[2 * m] = p_off0 * binom
        a0[2 * m + 1] = p_on0 * binom

    return a0


def soft_survival_time_boundary(
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

    a0 = initial_a_N(N, n0, p_off0, p_on0)

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

                L_current = build_L_N(
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

            L_current = build_L_N(
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

        L_now = build_L_N(
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

    t_grid = np.linspace(0.0, 20.0, 2001)

    gamma_values = [0.5, 1.0, 2.0, 4.0, 6.0, 8.0]

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

    tol_N = 1e-6

    plt.figure()

    for gamma in gamma_values:

        H_previous = None
        H_converged = None
        N_converged = None

        # Save the curve corresponding to the first N
        H_first = None
        N_first = N_values[0]

        consecutive_convergence = 0

        for N in N_values:

            H_gamma_N, f_N = soft_survival_time_boundary(
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

            # Save the first-N result
            if N == N_first:
                H_first = H_gamma_N.copy()

            # Reject numerical instability
            if not is_physical_survival(H_gamma_N):

                print(
                    f"gamma={gamma}, N={N}: "
                    f"numerically unstable -> use first N={N_first}."
                )

                break

            # Compare with previous N
            if H_previous is not None:

                error_N = np.max(
                    np.abs(H_gamma_N - H_previous)
                )

                print(
                    f"gamma={gamma}, N={N}, "
                    f"N-error={error_N:.3e}"
                )

                if error_N < tol_N:
                    consecutive_convergence += 1
                else:
                    consecutive_convergence = 0

                # Require two consecutive agreements
                if consecutive_convergence >= 2:

                    H_converged = H_gamma_N
                    N_converged = N

                    break

            H_previous = H_gamma_N

        # --------------------------------------------------------
        # Case 1: N-convergence was found
        # --------------------------------------------------------

        if H_converged is not None:

            plt.plot(
                t_grid,
                H_converged,
                label=fr"$\gamma={gamma},\ N^*={N_converged}$"
            )

            print(
                f"gamma={gamma}: converged at N={N_converged}"
            )

        # --------------------------------------------------------
        # Case 2: no convergence / instability -> use first N
        # --------------------------------------------------------

        else:

            plt.plot(
                t_grid,
                H_first,
                "--",
                label=fr"$\gamma={gamma},\ N={N_first}$"
            )

            print(
                f"gamma={gamma}: use first N={N_first}"
            )

    plt.xlabel("t")
    plt.ylabel(r"$H_{\gamma,N}(t)$")
    plt.title(
        "Soft binomial-moment survival"
    )
    plt.grid(True)
    plt.legend()
    plt.show()