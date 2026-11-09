import numpy as np
from scipy.linalg import expm
import matplotlib.pyplot as plt
import os

def build_Q_B(B, k1, k2, mu, delta):
    """
    Build the 2B x 2B transient generator Q_B for

        ON  --k1--> OFF
        OFF --k2--> ON
        ON  --mu--> ON + X
        X   --delta--> empty

    State ordering:
        [OFF, n=0,...,B-1,  ON, n=0,...,B-1]

    Column-vector convention:
        p'(t) = Q_B p(t)
    """
    Q = np.zeros((2 * B, 2 * B), dtype=float)

    def off(n):
        return n

    def on(n):
        return B + n

    for n in range(B):
        # ----- source state: (OFF, n) -----
        j = off(n)

        # OFF -> ON
        Q[on(n), j] += k2
        Q[j, j] -= k2

        # degradation: n -> n-1
        if n > 0:
            Q[off(n - 1), j] += delta * n
            Q[j, j] -= delta * n

        # ----- source state: (ON, n) -----
        j = on(n)

        # ON -> OFF
        Q[off(n), j] += k1
        Q[j, j] -= k1

        # degradation: n -> n-1
        if n > 0:
            Q[on(n - 1), j] += delta * n
            Q[j, j] -= delta * n

        # birth: n -> n+1
        # If n = B-1, the birth crosses the hard boundary and leaves
        # the transient state space, so only the loss term remains.
        Q[j, j] -= mu
        if n < B - 1:
            Q[on(n + 1), j] += mu

    return Q


def check_Q_B(Q, B, mu, atol=1e-12):
    """Sanity check for the transient generator in column convention."""
    expected = np.zeros(2 * B)
    expected[2 * B - 1] = -mu   # source state (ON, B-1) loses mass across the boundary

    column_sums = Q.sum(axis=0)
    if not np.allclose(column_sums, expected, atol=atol):
        raise ValueError(
            "Q_B failed the column-sum check. "
            "Expected all columns to sum to 0 except (ON, B-1), which should sum to -mu."
        )

    return column_sums


def initial_probability(B, n0, p_off0=1.0, p_on0=0.0):
    """Initial transient probability vector p(0)."""
    if not (0 <= n0 < B):
        raise ValueError("Need 0 <= n0 < B.")
    if not np.isclose(p_off0 + p_on0, 1.0):
        raise ValueError("Need p_off0 + p_on0 = 1.")

    p0 = np.zeros(2 * B, dtype=float)
    p0[n0] = p_off0
    p0[B + n0] = p_on0
    return p0


def hard_survival_exact(t_grid, B, k1, k2, mu, delta,
                        n0=0, p_off0=1.0, p_on0=0.0):
    """
    Exact hard-boundary survival:
        H_hard(t) = 1^T exp(t Q_B) p0
    """
    Q = build_Q_B(B, k1, k2, mu, delta)
    check_Q_B(Q, B, mu)
    p0 = initial_probability(B, n0, p_off0, p_on0)

    H = np.empty(len(t_grid), dtype=float)

    for j, t in enumerate(t_grid):
        p_t = expm(t * Q) @ p0
        H[j] = np.sum(p_t)

    f = fpt_density_from_survival(t_grid, H)

    return H, f, Q

def fpt_density_from_survival(t_grid, H):
    """
    Compute the FPT density from the survival function:

        f_T(t) = -dH(t)/dt
    """
    return -np.gradient(H, t_grid)


if __name__ == "__main__":
    # Example parameters only; replace them with the benchmark parameters.
    B = 4

    k1 = 0.01
    k2 = 0.03
    mu = 0.5
    delta = 1.0

    n0 = 3
    p_off0 = 1.0
    p_on0 = 0.0

    t_grid = np.linspace(0.0, 200.0, 20001)

    H_hard, f, Q_B = hard_survival_exact(
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

    print("Q_B shape =", Q_B.shape)
    print("column sums =", Q_B.sum(axis=0))
    
    plt.plot(t_grid, H_hard)
    plt.xlabel("t")
    plt.ylabel("H(t)")
    plt.title("Hard-boundary survival probability")
    plt.grid(True)
    plt.show()

    script_dir = os.path.dirname(os.path.abspath(__file__))

    filename = os.path.join(
        script_dir,
        f"H_hard_exact_B{B}_k1{k1}_k2{k2}_mu{mu}_delta{delta}.csv"
    )

    np.savetxt(
        filename,
        np.column_stack((t_grid, H_hard)),
        delimiter=",",
        header="t,H_hard",
        comments=""
    )

    print(f"Saved H(t) to: {filename}")

    f_T = fpt_density_from_survival(t_grid, H_hard)

    plt.figure()

    plt.plot(t_grid, f_T)

    plt.xlabel("t")
    plt.ylabel("f_T(t)")
    plt.title("First-passage-time density")

    plt.grid(True)

    plt.show()



    
