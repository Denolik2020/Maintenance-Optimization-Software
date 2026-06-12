import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize
from scipy.stats import weibull_min

# =====================================================
# PARAMETERS FROM TABLE 1
# =====================================================

lambda_j = np.array([
    0.0015,
    0.0010,
    0.0007,
    0.0005
])

beta_j = np.array([
    2.0,
    2.5,
    3.0,
    2.2
])

eta_j = np.array([
    180.0,
    250.0,
    350.0,
    450.0
])

Cef = np.array([
    500,
    800,
    1000,
    1200
])

# =====================================================
# COSTS
# =====================================================

Ci = 500.0
Co = 300.0
Cf = 1500.0

# Opportunity arrival rate
mu = 0.001

# =====================================================
# BASIC DISTRIBUTIONS
# =====================================================

def fx(x, lam):
    return lam * np.exp(-lam*x)

def Fx(x, lam):
    return 1 - np.exp(-lam*x)

def sx(x, lam):
    return np.exp(-lam*x)

def fh(h, beta, eta):
    return weibull_min.pdf(h, beta, scale=eta)

def Fh(h, beta, eta):
    return weibull_min.cdf(h, beta, scale=eta)

def Sh(h, beta, eta):
    return 1 - Fh(h, beta, eta)

# =====================================================
# CONVOLUTION
# Zj = Xj + Hj
# =====================================================

def fz(t, lam, beta, eta):

    integrand = lambda x: (
        fx(x, lam)
        *
        fh(t-x, beta, eta)
    )

    val, _ = quad(
        integrand,
        0,
        t
    )

    return val

def Fz(t, lam, beta, eta):

    val, _ = quad(
        lambda u: fz(u, lam, beta, eta),
        0,
        t
    )

    return val

def Sz(t, lam, beta, eta):
    return 1 - Fz(t, lam, beta, eta)

# =====================================================
# DEFECTIVE PROBABILITY
# G_j(t)
# =====================================================

def G(t, lam, beta, eta):

    integrand = lambda x: (
        fx(x, lam)
        *
        Sh(t-x, beta, eta)
    )

    val, _ = quad(
        integrand,
        0,
        t
    )

    return val

# =====================================================
# SERIES SYSTEM FAILURE DENSITY
# =====================================================

def fs(t):

    total = 0.0

    for j in range(4):

        term = fz(
            t,
            lambda_j[j],
            beta_j[j],
            eta_j[j]
        )

        prod = 1.0

        for k in range(4):

            if k != j:

                prod *= Sz(
                    t,
                    lambda_j[k],
                    beta_j[k],
                    eta_j[k]
                )

        total += term * prod

    return total

# =====================================================
# EXPECTED FAILURES OF COMPONENT j
# APPROXIMATION
# =====================================================

def Mj(T, j):

    val, _ = quad(
        lambda t: fs(t),
        0,
        T
    )

    return val / 4.0

# =====================================================
# OPPORTUNITY DISTRIBUTION
# =====================================================

def FW(tau):
    return np.exp(-mu*tau)

def fW(w):
    return mu*np.exp(-mu*w)

# =====================================================
# EXPECTED COST
# =====================================================

def ECi(T, tau):

    cost_fail = 0

    for j in range(4):

        cost_fail += (
            Mj(T, j)
            *
            (Cf + Cef[j])
        )

    no_opportunity = FW(tau)

    opportunity = (
        1 - FW(tau)
    )

    return (
        no_opportunity
        *
        (Ci + cost_fail)
        +
        opportunity
        *
        (Co + cost_fail)
    )

def ECo(T, tau):

    cost_fail = 0

    for j in range(4):

        cost_fail += (
            Mj(T, j)
            *
            (Cf + Cef[j])
        )

    return (
        FW(tau)
        *
        (Ci + cost_fail)
        +
        (1 - FW(tau))
        *
        (Co + cost_fail)
    )

# =====================================================
# EXPECTED DURATIONS
# =====================================================

def EVi(T, tau):

    return (
        FW(tau)*T
        +
        (1-FW(tau))
        *
        (T - tau/2)
    )

def EVo(T, tau):

    return (
        FW(tau)*T
        +
        (1-FW(tau))
        *
        (T - tau/2)
    )

# =====================================================
# STATIONARY PROBABILITIES
# =====================================================

def pi_i(tau):
    return FW(tau)

def pi_o(tau):
    return 1 - FW(tau)

# =====================================================
# GENERIC CYCLE
# =====================================================

def EC_cycle(T, tau):

    return (
        pi_i(tau)*ECi(T, tau)
        +
        pi_o(tau)*ECo(T, tau)
    )

def EV_cycle(T, tau):

    return (
        pi_i(tau)*EVi(T, tau)
        +
        pi_o(tau)*EVo(T, tau)
    )

# =====================================================
# OBJECTIVE FUNCTION
# C∞
# =====================================================

def C_inf(x):

    T = x[0]
    tau = x[1]

    if tau <= 0:
        return 1e12

    if tau >= T:
        return 1e12

    return (
        EC_cycle(T, tau)
        /
        EV_cycle(T, tau)
    )

# =====================================================
# OPTIMIZATION
# =====================================================

bounds = [
    (50, 5000),
    (1, 4999)
]
print("Run optimization")
result = minimize(
    C_inf,
    x0=[250,120],
    bounds=bounds,
    method="L-BFGS-B"
)

T_star = result.x[0]
tau_star = result.x[1]

print("\nOPTIMAL POLICY")
print("------------------")
print("T*    =", T_star)
print("tau*  =", tau_star)
print("C_inf =", result.fun)