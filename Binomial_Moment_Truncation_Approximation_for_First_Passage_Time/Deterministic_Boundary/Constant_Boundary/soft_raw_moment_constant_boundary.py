import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm


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


def soft_raw_survival_truncated(
    t_grid,
    B,
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
):
    """Compute H_raw(t) = u_0^OFF(t) + u_0^ON(t)."""

    L = build_L_raw(B, N, gamma, k1, k2, mu, delta, theta)
    y0 = initial_raw(N, n0, p_off0, p_on0)

    H = np.empty(len(t_grid), dtype=float)
    f = np.empty(len(t_grid), dtype=float)

    for j, t in enumerate(t_grid):
        y_t = expm(t * L) @ y0
        H[j] = y_t[0] + y_t[1]

        dy_t = L @ y_t

        f[j] = -(dy_t[0] + dy_t[1])

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

    B = 8
    theta = 0.5

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

    t_grid = np.linspace(0.0, 20.0, 201)

    # =========================================================
    # gamma and N scan
    # =========================================================

    gamma_values = [0.5, 1.0, 2.0, 4.0, 6.0, 8.0]

    # For each fixed gamma, increase N
    N_values = list(range(B, 30, 2))

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

            H_gamma_N, f_gamma_N = soft_raw_survival_truncated(
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
        "Soft raw-moment survival, constant boundary"
    )

    plt.grid(True)

    plt.legend()

    plt.show()