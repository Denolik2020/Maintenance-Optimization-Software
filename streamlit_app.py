




import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import weibull_min
from scipy.optimize import minimize

# =====================================================
# PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="Maintenance Optimization Software",
    layout="wide"
)

# =====================================================
# CUSTOM STYLE
# =====================================================

st.markdown("""
<style>
.main-title {
    font-size: 40px;
    font-weight: bold;
}

.subtitle {
    font-size: 18px;
    color: gray;
}

.metric-card {
    padding: 15px;
    border-radius: 10px;
    background-color: #f2f2f2;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# =====================================================
# LANDING PAGE
# =====================================================

st.markdown(
    '<div class="main-title">Maintenance Optimization Software</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Production Engineering - UFPE | Random Lab</div>',
    unsafe_allow_html=True
)

st.write("""
This software optimizes maintenance and inspection policies for
multi-component systems using reliability and cost models.

### Main Features
- Flexible number of components
- Weibull reliability modelling
- Cost optimization
- Interactive visualization
- Dashboard results
""")

st.divider()

# =====================================================
# INPUT DATA
# =====================================================

st.header("Input Data")

n = st.number_input(
    "Number of Component Types",
    min_value=1,
    value=1,
    step=1
)

components = []

st.subheader("Component Parameters")

for i in range(int(n)):

    with st.expander(f"Component {i+1}", expanded=True):

        col1, col2 = st.columns(2)

        with col1:
            quantity = st.number_input(
                f"Quantity n{i+1}",
                min_value=1,
                value=1,
                key=f"qty{i}"
            )

            lam = st.number_input(
                f"Failure Rate λ{i+1}",
                min_value=0.000001,
                value=0.001,
                key=f"lam{i}"
            )

            beta = st.number_input(
                f"Weibull β{i+1}",
                min_value=0.1,
                value=2.0,
                key=f"beta{i}"
            )

        with col2:
            eta = st.number_input(
                f"Weibull η{i+1}",
                min_value=1.0,
                value=200.0,
                key=f"eta{i}"
            )

            cef = st.number_input(
                f"Cost per Failure Cef{i+1}",
                min_value=0.0,
                value=500.0,
                key=f"cef{i}"
            )

        components.append(
            {
                "n": quantity,
                "lambda": lam,
                "beta": beta,
                "eta": eta,
                "cef": cef
            }
        )

# =====================================================
# GLOBAL PARAMETERS
# =====================================================

st.subheader("Global Cost Parameters")

col1, col2 = st.columns(2)

with col1:
    Ci = st.number_input(
        "Inspection Cost (Ci)",
        min_value=0.0,
        value=500.0
    )

    Co = st.number_input(
        "Opportunity Cost (Co)",
        min_value=0.0,
        value=300.0
    )

with col2:
    Cf = st.number_input(
        "Failure Cost (Cf)",
        min_value=0.0,
        value=1500.0
    )

    mu = st.number_input(
        "Opportunity Arrival Rate (μ)",
        min_value=0.000001,
        value=0.001
    )

# =====================================================
# MODEL PARAMETERS
# =====================================================

T_max = 5000
N_grid = 1000

t_grid = np.linspace(0, T_max, N_grid)
dt = t_grid[1] - t_grid[0]

# =====================================================
# COMPONENT FUNCTIONS
# =====================================================

def compute_component_functions(comp):

    lam = comp["lambda"]
    beta = comp["beta"]
    eta = comp["eta"]

    fx = lam * np.exp(-lam * t_grid)

    fh = weibull_min.pdf(
        t_grid,
        beta,
        scale=eta
    )

    fz = np.convolve(fx, fh)[:len(t_grid)] * dt

    Sz = 1 - np.cumsum(fz) * dt
    Sz = np.maximum(Sz, 0)

    return fz, Sz

component_data = []

for comp in components:
    fz, Sz = compute_component_functions(comp)
    component_data.append((fz, Sz, comp))

# =====================================================
# SYSTEM FAILURE DENSITY
# =====================================================

def system_failure_density():

    fs = np.zeros_like(t_grid)

    for j, (fz_j, Sz_j, comp_j) in enumerate(component_data):

        term = comp_j["n"] * fz_j

        prod = np.ones_like(t_grid)

        for k, (fz_k, Sz_k, comp_k) in enumerate(component_data):

            if k != j:
                prod *= np.power(
                    np.maximum(Sz_k, 1e-12),
                    comp_k["n"]
                )

        fs += term * prod

    return fs

fs_vals = system_failure_density()

# =====================================================
# EXPECTED FAILURES
# =====================================================

def expected_failures(T):

    idx = int(T / T_max * N_grid)

    idx = max(2, min(idx, N_grid))

    return np.trapz(
        fs_vals[:idx],
        t_grid[:idx]
    )

# =====================================================
# COST FUNCTION
# =====================================================

def cost_rate(x):

    T, tau = x

    if tau <= 0:
        return 1e12

    if tau >= T:
        return 1e12

    M = expected_failures(T)

    failure_cost = 0

    for comp in components:
        failure_cost += (
            comp["n"]
            * M
            * (Cf + comp["cef"])
        )

    FW = np.exp(-mu * tau)

    EC = (
        FW * (Ci + failure_cost)
        + (1 - FW) * (Co + failure_cost)
    )

    EV = (
        FW * T
        + (1 - FW) * (T - tau / 2)
    )

    return EC / EV

# =====================================================
# OPTIMIZATION
# =====================================================

st.divider()

if st.button("Run Optimization"):

    result = minimize(
        cost_rate,
        x0=[200, 50],
        bounds=[(10, 5000), (1, 4999)],
        method="L-BFGS-B"
    )

    T_star = result.x[0]
    tau_star = result.x[1]

    st.success("Optimization completed successfully.")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Optimal T*",
            f"{T_star:.2f}"
        )

    with col2:
        st.metric(
            "Optimal τ*",
            f"{tau_star:.2f}"
        )

    with col3:
        st.metric(
            "Cost Rate",
            f"{result.fun:.4f}"
        )

    st.subheader("System Failure Density")

    fig, ax = plt.subplots(figsize=(8, 4))

    ax.plot(t_grid, fs_vals)

    ax.set_xlabel("Time")
    ax.set_ylabel("Failure Density")
    ax.set_title("System Failure Behaviour")

    st.pyplot(fig)

# =====================================================
# EMAIL SECTION
# =====================================================

st.divider()

st.header("Send Results")

email = st.text_input("Recipient Email")

if st.button("Send Results"):

    if email.strip() == "":
        st.warning("Please enter an email address.")

    else:
        st.success(
            f"Results would be sent to: {email}"
        )
