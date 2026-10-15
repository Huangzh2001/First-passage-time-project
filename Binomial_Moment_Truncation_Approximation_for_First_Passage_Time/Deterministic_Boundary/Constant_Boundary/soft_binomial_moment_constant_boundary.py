import numpy as np
from math import comb
import matplotlib.pyplot as plt
from scipy.sparse.linalg import expm_multiply
import multiprocessing as mp
from scipy.sparse import csr_matrix
from scipy.linalg import expm
from scipy.integrate import solve_ivp

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


def soft_survival_truncated(
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
    """
    Compute

        H_{gamma,N}(t)
        = a_{0,N}^{(0)}(t) + a_{0,N}^{(1)}(t)
        = u^T exp(t L_N) a_N(0)

    for a constant integer boundary B.
    """
    if not (0 <= n0 < B):
        raise ValueError("For the benchmark, need 0 <= n0 < B.")

    t_grid = np.asarray(t_grid, dtype=float)

    if t_grid.ndim != 1:
        raise ValueError("t_grid must be one-dimensional.")
    if np.any(t_grid < 0):
        raise ValueError("All times must be nonnegative.")

    L = build_L_N(B, N, gamma, k1, k2, mu, delta, theta)
    a0 = initial_a_N(N, n0, p_off0, p_on0)

    # Compute exp(t L) @ a0 directly, without forming exp(t L)

    H = np.empty(len(t_grid), dtype=float)
    f = np.empty(len(t_grid), dtype=float)

    sol = solve_ivp(
        lambda t, a: L @ a,
        (0.0, float(t_grid[-1])),
        a0,
        method="Radau",
        jac=L,
        t_eval=t_grid,
        rtol=1e-9,
        atol=1e-12,
    )
    if not sol.success:
        raise RuntimeError(sol.message)

    for j, t in enumerate(t_grid):

        # a_t = expm(t * L) @ a0
        a_t = sol.y[:, j]

        H[j] = a_t[0] + a_t[1]

        da_t = L @ a_t

        f[j] = -(da_t[0] + da_t[1])

    return H, f, L


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

    B = 8
    theta = 0.5

    k1 = 1.0
    k2 = 1.0
    mu = 5.0
    delta = 1.0

    n0 = 0
    p_off0 = 1.0
    p_on0 = 0.0

    t_grid = np.linspace(0.0, 20.0, 201)

    gamma_values = [0.5, 1.0, 2.0, 4.0, 6.0, 8.0]

    N_values = list(range(B, 30, 2))

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

            H_gamma_N, f_N, L_N = soft_survival_truncated(
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
    plt.title("Soft binomial-moment survival")
    plt.grid(True)
    plt.legend()
    plt.show()