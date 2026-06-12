"""
Streamlit frontend for the delay-time model with opportunistic maintenance
(series system, optimization of T* and τ* minimizing C∞).

Run with:  streamlit run app.py
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from model import C_inf, build_grid, optimize

st.set_page_config(
    page_title="Opportunistic Maintenance Policy",
    page_icon="🔧",
    layout="wide",
)

st.title("🔧 Opportunistic Maintenance Policy Optimization")
st.markdown(
    "**Delay-time** model with opportunities for a **series system**. "
    "The policy is defined by the inspection age **T** and the opportunity "
    "window **τ**; the goal is to minimize the long-run cost rate **C∞(T, τ)**."
)

# =====================================================
# SIDEBAR — PARAMETERS
# =====================================================

st.sidebar.header("Component parameters")
st.sidebar.caption(
    "λ: defect arrival rate (exponential) · β, η: shape and scale of the "
    "delay-time (Weibull) · Cef: extra failure cost of the component"
)

df_default = pd.DataFrame(
    {
        "λ (lambda)": [0.0015, 0.0010, 0.0007, 0.0005],
        "β (beta)": [2.0, 2.5, 3.0, 2.2],
        "η (eta)": [180.0, 250.0, 350.0, 450.0],
        "Cef": [500.0, 800.0, 1000.0, 1200.0],
    },
    index=[f"Comp. {j + 1}" for j in range(4)],
)

df_comp = st.sidebar.data_editor(
    df_default,
    num_rows="dynamic",
    use_container_width=True,
    key="component_table",
)

st.sidebar.header("Costs and opportunities")
Ci = st.sidebar.number_input("Ci — scheduled inspection cost", min_value=0.0, value=500.0, step=50.0)
Co = st.sidebar.number_input("Co — opportunistic intervention cost", min_value=0.0, value=300.0, step=50.0)
Cf = st.sidebar.number_input("Cf — base failure cost", min_value=0.0, value=1500.0, step=100.0)
mu = st.sidebar.number_input(
    "μ — opportunity arrival rate",
    min_value=0.0,
    value=0.001,
    step=0.0005,
    format="%.5f",
)

st.sidebar.header("Optimization")
T_min, T_max = st.sidebar.slider(
    "Bounds for T", min_value=1.0, max_value=10000.0, value=(50.0, 5000.0), step=1.0
)
T0 = st.sidebar.number_input("Initial guess T₀", min_value=1.0, value=250.0, step=10.0)
tau0 = st.sidebar.number_input("Initial guess τ₀", min_value=0.1, value=120.0, step=10.0)

with st.sidebar.expander("Numerical grid (advanced)"):
    n_t = st.number_input("Time grid points", min_value=100, max_value=3000, value=600, step=100)
    n_x = st.number_input("Points per convolution", min_value=50, max_value=1000, value=240, step=50)

# =====================================================
# MODEL CONSTRUCTION (cached)
# =====================================================


@st.cache_data(show_spinner=False)
def cached_grid(lambda_j, beta_j, eta_j, t_max, n_t, n_x):
    return build_grid(lambda_j, beta_j, eta_j, t_max=t_max, n_t=n_t, n_x=n_x)


df_valid = df_comp.dropna()
if len(df_valid) == 0:
    st.error("Define at least one component in the sidebar table.")
    st.stop()

lambda_j = tuple(float(v) for v in df_valid["λ (lambda)"])
beta_j = tuple(float(v) for v in df_valid["β (beta)"])
eta_j = tuple(float(v) for v in df_valid["η (eta)"])
Cef = tuple(float(v) for v in df_valid["Cef"])

if any(v <= 0 for v in lambda_j + beta_j + eta_j):
    st.error("λ, β and η must be strictly positive for every component.")
    st.stop()

run = st.button("▶️ Run optimization", type="primary")

if run:
    with st.spinner("Building the model and optimizing…"):
        grid = cached_grid(lambda_j, beta_j, eta_j, T_max, int(n_t), int(n_x))
        result = optimize(
            grid,
            Cef,
            Ci=Ci,
            Co=Co,
            Cf=Cf,
            mu=mu,
            bounds=((T_min, T_max), (1.0, T_max - 1.0)),
            x0=(T0, tau0),
        )
    st.session_state["grid"] = grid
    st.session_state["result"] = result
    st.session_state["params"] = {"Cef": Cef, "Ci": Ci, "Co": Co, "Cf": Cf, "mu": mu,
                                  "T_min": T_min, "T_max": T_max}

# =====================================================
# RESULTS
# =====================================================

if "result" not in st.session_state:
    st.info("Adjust the parameters in the sidebar and click **Run optimization**.")
    st.stop()

grid = st.session_state["grid"]
result = st.session_state["result"]
params = st.session_state["params"]

if not result["success"]:
    st.warning(f"The optimizer did not report convergence: {result['message']}")

st.subheader("Optimal policy")
c1, c2, c3 = st.columns(3)
c1.metric("T* — inspection age", f"{result['T_star']:.1f}")
c2.metric("τ* — opportunity window", f"{result['tau_star']:.1f}")
c3.metric("C∞ — minimum cost rate", f"{result['C_inf']:.4f}")

cost_tab, model_tab = st.tabs(["📉 Cost surface", "📈 Model curves"])

# -----------------------------------------------------
# Tab 1 — cost
# -----------------------------------------------------
with cost_tab:
    T_star, tau_star = result["T_star"], result["tau_star"]
    kw = dict(Cef=params["Cef"], Ci=params["Ci"], Co=params["Co"],
              Cf=params["Cf"], mu=params["mu"])

    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown(f"**C∞ as a function of T** (τ fixed at τ* = {tau_star:.1f})")
        T_vals = np.linspace(max(params["T_min"], tau_star + 1.0), params["T_max"], 300)
        curve = [C_inf(T, tau_star, grid, **kw) for T in T_vals]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(T_vals, curve, lw=2)
        ax.axvline(T_star, color="crimson", ls="--", label=f"T* = {T_star:.1f}")
        ax.set_xlabel("T")
        ax.set_ylabel("C∞")
        ax.legend()
        ax.grid(alpha=0.3)
        st.pyplot(fig)
        plt.close(fig)

    with col_right:
        st.markdown("**Map of C∞(T, τ)** over the feasible region (τ < T)")
        T_grid = np.linspace(max(params["T_min"], 2.0), params["T_max"], 60)
        tau_grid = np.linspace(1.0, params["T_max"] - 1.0, 60)
        Z = np.full((len(tau_grid), len(T_grid)), np.nan)
        for a, tau_v in enumerate(tau_grid):
            for b, T_v in enumerate(T_grid):
                if tau_v < T_v:
                    Z[a, b] = C_inf(T_v, tau_v, grid, **kw)
        fig, ax = plt.subplots(figsize=(6, 4))
        cont = ax.contourf(T_grid, tau_grid, Z, levels=30, cmap="viridis")
        fig.colorbar(cont, ax=ax, label="C∞")
        ax.plot(T_star, tau_star, "r*", ms=14, label="optimum")
        ax.set_xlabel("T")
        ax.set_ylabel("τ")
        ax.legend(loc="upper left")
        st.pyplot(fig)
        plt.close(fig)

# -----------------------------------------------------
# Tab 2 — model curves
# -----------------------------------------------------
with model_tab:
    t = grid["t"]
    n_comp = grid["fz"].shape[0]

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("**Density of Z_j = X_j + H_j (defect arrival + delay-time)**")
        fig, ax = plt.subplots(figsize=(6, 4))
        for j in range(n_comp):
            ax.plot(t, grid["fz"][j], label=f"Comp. {j + 1}")
        ax.set_xlabel("t")
        ax.set_ylabel("f_Z(t)")
        ax.legend()
        ax.grid(alpha=0.3)
        st.pyplot(fig)
        plt.close(fig)

        st.markdown("**Series-system failure density, f_s(t)**")
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(t, grid["fs"], color="darkorange", lw=2)
        ax.set_xlabel("t")
        ax.set_ylabel("f_s(t)")
        ax.grid(alpha=0.3)
        st.pyplot(fig)
        plt.close(fig)

    with col_b:
        st.markdown("**Reliability S_Z(t) per component**")
        fig, ax = plt.subplots(figsize=(6, 4))
        for j in range(n_comp):
            ax.plot(t, grid["Sz"][j], label=f"Comp. {j + 1}")
        ax.set_xlabel("t")
        ax.set_ylabel("S_Z(t)")
        ax.legend()
        ax.grid(alpha=0.3)
        st.pyplot(fig)
        plt.close(fig)

        st.markdown("**Expected failures per component, M(T)**")
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(t, grid["M"], color="seagreen", lw=2)
        ax.axvline(result["T_star"], color="crimson", ls="--",
                   label=f"T* = {result['T_star']:.1f}")
        ax.set_xlabel("T")
        ax.set_ylabel("M(T)")
        ax.legend()
        ax.grid(alpha=0.3)
        st.pyplot(fig)
        plt.close(fig)

st.caption(
    "Note: the nested scipy.integrate.quad integrals of the original script "
    "were replaced by grid-based integration (trapezoidal rule) to make the "
    "optimization interactive. The model equations are the same."
)
