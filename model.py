
import numpy as np
from scipy.optimize import minimize

_trapz = getattr(np, "trapezoid", None) or np.trapz

# =====================================================
# INTERNAL OPTIMIZATION SETTINGS (HIDDEN FROM USER)
# =====================================================

OPTIMIZATION_CONFIG = {
    "T_MIN": 50.0,
    "T_MAX": 5000.0,
    "T0": 250.0,
    "TAU0": 120.0,
    "N_T": 600,
    "N_X": 240,
}

# =====================================================
# WEIBULL PDF
# =====================================================

def weibull_pdf(h, beta, eta):
    h = np.asarray(h, dtype=float)
    out = np.zeros_like(h)
    pos = h > 0
    hp = h[pos] / eta
    out[pos] = (beta / eta) * hp ** (beta - 1.0) * np.exp(-(hp ** beta))
    return out

# =====================================================
# GRID BUILDER
# =====================================================

def build_grid(lambda_j, beta_j, eta_j, t_max):
    n_t = OPTIMIZATION_CONFIG["N_T"]
    n_x = OPTIMIZATION_CONFIG["N_X"]

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

    dt = np.diff(t)

    Fz = np.zeros_like(fz)
    Fz[:, 1:] = np.cumsum(0.5 * (fz[:, 1:] + fz[:, :-1]) * dt, axis=1)
    Fz = np.clip(Fz, 0.0, 1.0)

    Sz = 1.0 - Fz

    # ✅ FAST VECTORIZED SYSTEM FAILURE
    all_survival = np.prod(Sz, axis=0)

    fs = np.zeros(len(t))
    for j in range(n):
        fs += fz[j] * (all_survival / Sz[j])

    M = np.zeros(len(t))
    M[1:] = np.cumsum(0.5 * (fs[1:] + fs[:-1]) * dt) / n

    return {"t": t, "fz": fz, "Sz": Sz, "fs": fs, "M": M}

# =====================================================
# COST FUNCTIONS
# =====================================================

def expected_failure_cost(T, grid, Cef, Cf):
    M_T = np.interp(T, grid["t"], grid["M"])
    Cef = np.asarray(Cef, dtype=float)
    return M_T * np.sum(Cf + Cef)

def C_inf(T, tau, grid, Cef, Ci, Co, Cf, mu):
    if tau <= 0 or tau >= T:
        return 1e12

    failure_cost = expected_failure_cost(T, grid, Cef, Cf)

    FW = np.exp(-mu * tau)

    EC = FW * (Ci + failure_cost) + (1 - FW) * (Co + failure_cost)
    EV = FW * T + (1 - FW) * (T - tau / 2)

    return EC / EV

# =====================================================
# OPTIMIZATION
# =====================================================

def optimize(grid, Cef, Ci, Co, Cf, mu):

    bounds = (
        (OPTIMIZATION_CONFIG["T_MIN"], OPTIMIZATION_CONFIG["T_MAX"]),
        (1, OPTIMIZATION_CONFIG["T_MAX"] - 1),
    )

    x0 = (
        OPTIMIZATION_CONFIG["T0"],
        OPTIMIZATION_CONFIG["TAU0"],
    )

    result = minimize(
        lambda x: C_inf(x[0], x[1], grid, Cef, Ci, Co, Cf, mu),
        x0=np.asarray(x0),
        bounds=bounds,
        method="L-BFGS-B",
    )

    return {
        "T_star": float(result.x[0]),
        "tau_star": float(result.x[1]),
        "C_inf": float(result.fun),
        "success": result.success,
        "message": result.message,
    }
