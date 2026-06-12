"""
Delay-time model with opportunistic maintenance for a series system.

Implements the same equations as the original optimization script, but with
grid-based numerical integration (vectorized trapezoidal rule) instead of
nested scipy.integrate.quad calls, so the optimization is fast enough for
interactive use in Streamlit.

Equations:
    fz_j(t) = ∫0^t fx_j(x) · fh_j(t-x) dx          (convolution Z_j = X_j + H_j)
    Fz_j(t) = ∫0^t fz_j(u) du,  Sz_j = 1 - Fz_j
    fs(t)   = Σ_j fz_j(t) · Π_{k≠j} Sz_k(t)         (series-system failure density)
    M(T)    = (1/n) ∫0^T fs(u) du                    (approx. expected failures per component)
    FW(τ)   = exp(-μτ)                               (prob. of no opportunity up to τ)
    EC      = FW·(Ci + failure_cost) + (1-FW)·(Co + failure_cost)
    EV      = FW·T + (1-FW)·(T - τ/2)
    C∞(T,τ) = EC_cycle / EV_cycle
"""

import numpy as np
from scipy.optimize import minimize

# np.trapz was renamed to np.trapezoid in NumPy 2.x
_trapz = getattr(np, "trapezoid", None) or np.trapz


def weibull_pdf(h, beta, eta):
    """Weibull density (shape beta, scale eta), vectorized, 0 for h <= 0."""
    h = np.asarray(h, dtype=float)
    out = np.zeros_like(h)
    pos = h > 0
    hp = h[pos] / eta
    out[pos] = (beta / eta) * hp ** (beta - 1.0) * np.exp(-(hp ** beta))
    return out


def build_grid(lambda_j, beta_j, eta_j, t_max, n_t=600, n_x=240):
    """Precompute fz_j, Sz_j, fs and M on a time grid [0, t_max].

    Returns a dictionary with the grid "t" and the corresponding arrays.
    """
    lambda_j = np.asarray(lambda_j, dtype=float)
    beta_j = np.asarray(beta_j, dtype=float)
    eta_j = np.asarray(eta_j, dtype=float)
    n = len(lambda_j)

    t = np.linspace(0.0, float(t_max), int(n_t))
    fz = np.zeros((n, len(t)))

    for j in range(n):
        lam, beta, eta = lambda_j[j], beta_j[j], eta_j[j]
        for i in range(1, len(t)):
            x = np.linspace(0.0, t[i], int(n_x))
            integrand = lam * np.exp(-lam * x) * weibull_pdf(t[i] - x, beta, eta)
            fz[j, i] = _trapz(integrand, x)

    # Fz by cumulative trapezoidal rule, clipped to [0, 1]
    dt = np.diff(t)
    Fz = np.zeros_like(fz)
    Fz[:, 1:] = np.cumsum(0.5 * (fz[:, 1:] + fz[:, :-1]) * dt, axis=1)
    Fz = np.clip(Fz, 0.0, 1.0)
    Sz = 1.0 - Fz

    fs = np.zeros(len(t))
    for j in range(n):
        prod = np.ones(len(t))
        for k in range(n):
            if k != j:
                prod *= Sz[k]
        fs += fz[j] * prod

    M = np.zeros(len(t))
    M[1:] = np.cumsum(0.5 * (fs[1:] + fs[:-1]) * dt) / n

    return {"t": t, "fz": fz, "Fz": Fz, "Sz": Sz, "fs": fs, "M": M}


def expected_failure_cost(T, grid, Cef, Cf):
    """Σ_j M_j(T)·(Cf + Cef_j), with M_j(T) = M(T) for every j (as in the original)."""
    M_T = np.interp(T, grid["t"], grid["M"])
    Cef = np.asarray(Cef, dtype=float)
    return M_T * float(np.sum(Cf + Cef))


def C_inf(T, tau, grid, Cef, Ci, Co, Cf, mu):
    """Long-run cost rate C∞(T, τ). Penalizes the infeasible region."""
    if tau <= 0 or tau >= T:
        return 1e12

    failure_cost = expected_failure_cost(T, grid, Cef, Cf)
    FW = np.exp(-mu * tau)

    # ECi and ECo kept separate for fidelity to the original script
    ECi = FW * (Ci + failure_cost) + (1.0 - FW) * (Co + failure_cost)
    ECo = FW * (Ci + failure_cost) + (1.0 - FW) * (Co + failure_cost)
    EVi = FW * T + (1.0 - FW) * (T - tau / 2.0)
    EVo = FW * T + (1.0 - FW) * (T - tau / 2.0)

    pi_i = FW
    pi_o = 1.0 - FW

    EC_cycle = pi_i * ECi + pi_o * ECo
    EV_cycle = pi_i * EVi + pi_o * EVo

    return EC_cycle / EV_cycle


def optimize(grid, Cef, Ci, Co, Cf, mu, bounds=((50, 5000), (1, 4999)), x0=(250.0, 120.0)):
    """Minimizes C∞(T, τ) with L-BFGS-B, as in the original script."""
    result = minimize(
        lambda x: C_inf(x[0], x[1], grid, Cef, Ci, Co, Cf, mu),
        x0=np.asarray(x0, dtype=float),
        bounds=bounds,
        method="L-BFGS-B",
    )
    return {
        "T_star": float(result.x[0]),
        "tau_star": float(result.x[1]),
        "C_inf": float(result.fun),
        "success": bool(result.success),
        "message": str(result.message),
    }


if __name__ == "__main__":
    # Quick test with the parameters from Table 1
    lambda_j = [0.0015, 0.0010, 0.0007, 0.0005]
    beta_j = [2.0, 2.5, 3.0, 2.2]
    eta_j = [180.0, 250.0, 350.0, 450.0]
    Cef = [500, 800, 1000, 1200]

    grid = build_grid(lambda_j, beta_j, eta_j, t_max=5000.0)
    res = optimize(grid, Cef, Ci=500.0, Co=300.0, Cf=1500.0, mu=0.001)
    print("T*    =", res["T_star"])
    print("tau*  =", res["tau_star"])
    print("C_inf =", res["C_inf"])
    print("success:", res["success"], "-", res["message"])
