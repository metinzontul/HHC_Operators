# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 12:28:03 2026

@author: sedak
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from matplotlib import cm

# ============================================================
# SETTINGS
# ============================================================
SAVE_FIGURES = True
DPI = 600

t = 1.0
z = 2.0
ns = [5, 10, 20, 40]

# Domain Omega = [-1,1]^2
Omega_min, Omega_max = -1.0, 1.0
Nu = 61

# Truncation window for w-integrals: w in [-L,L]^2
L = 12.0
Nw = 241

# Delta_n parameters for Q_n^s
gamma1 = 2
gamma2 = 2
lam = 1.0 / ((gamma1 + 1) * (gamma2 + 1))   # = 1/9

# Visualization choices
# All computations: n = 5,10,20,40
# All figures: n = 20
n_show = 20
n_compare = 20

# Slice-plot parameters
u2_slices = [-0.8, -0.4, 0.0, 0.4, 0.8]
eps = 0.015
offset = 0.03

# Optional matplotlib defaults
plt.rcParams["savefig.dpi"] = DPI
plt.rcParams["figure.dpi"] = 600  # screen only
plt.rcParams["figure.figsize"] = (7.2, 5.6)

# ============================================================
# ACTIVATION FUNCTION AND DENSITIES
# ============================================================
def chi(u):
    """
    Adjustable half-hyperbolic tangent activation:
    chi_{t,z}(u) = (1 - t e^{-zu}) / (1 + t e^{-zu})
    """
    return (1.0 - t * np.exp(-z * u)) / (1.0 + t * np.exp(-z * u))

def D_base(u, t_local):
    """
    Base density:
    D_{t,z}(u) = 1/4 [chi_{t,z}(u+1) - chi_{t,z}(u-1)]
    """
    return 0.25 * (
        (1.0 - t_local * np.exp(-z * (u + 1.0))) / (1.0 + t_local * np.exp(-z * (u + 1.0)))
        - (1.0 - t_local * np.exp(-z * (u - 1.0))) / (1.0 + t_local * np.exp(-z * (u - 1.0)))
    )

def D(u):
    return D_base(u, t)

def Ds(u):
    """
    Symmetrized density:
    D^s_{t,z}(u) = (D_{t,z}(u) + D_{1/t,z}(u)) / 2
    """
    return 0.5 * (D_base(u, t) + D_base(u, 1.0 / t))

# ============================================================
# TEST FUNCTION
# ============================================================
def f(u1, u2):
    return u1**2 + u2**2 + 0.5 * np.sin(2 * np.pi * u1) * np.cos(2 * np.pi * u2)

# ============================================================
# UTILITY: 2D TRAPEZOIDAL INTEGRAL ON TENSOR GRID
# ============================================================
def integrate2D(F, grid):
    return float(np.trapezoid(np.trapezoid(F, grid, axis=1), grid, axis=0))

# ============================================================
# BUILD GRIDS
# ============================================================
u1 = np.linspace(Omega_min, Omega_max, Nu)
u2 = np.linspace(Omega_min, Omega_max, Nu)
U1, U2 = np.meshgrid(u1, u2, indexing="xy")

w = np.linspace(-L, L, Nw)
W1, W2 = np.meshgrid(w, w, indexing="xy")

# ============================================================
# PRODUCT KERNEL ON w-GRID AND NUMERICAL NORMALIZATION
# ============================================================
K_w = Ds(W1) * Ds(W2)
K_sum = integrate2D(K_w, w)
K_w = K_w / K_sum

# ============================================================
# OPERATORS  (v = n u - w  =>  v/n = u - w/n)
# ============================================================
def Cn_approx(n):
    """C_n^s(f)(u) = ∬ f(u - w/n) K(w) dw."""
    out = np.zeros_like(U1)
    for i in range(Nu):
        for j in range(Nu):
            X = U1[i, j] - W1 / n
            Y = U2[i, j] - W2 / n
            integrand = f(X, Y) * K_w
            out[i, j] = integrate2D(integrand, w)
    return out

def Kn_approx(n):
    """
    K_n^s(f)(u): midpoint approximation of the cell average
    over [v1/n,(v1+1)/n] x [v2/n,(v2+1)/n].
    """
    out = np.zeros_like(U1)
    shift = 0.5 / n
    for i in range(Nu):
        for j in range(Nu):
            X = U1[i, j] - W1 / n + shift
            Y = U2[i, j] - W2 / n + shift
            integrand = f(X, Y) * K_w
            out[i, j] = integrate2D(integrand, w)
    return out

def Delta_vals_on_w(u1_val, u2_val, n):
    """
    Δ_n(f)(v) with v = n u - w:
    Δ_n(f)(v) = Σ_{r1=0..γ1} Σ_{r2=0..γ2}
                λ f(u - w/n + r/(nγ))
    """
    baseX = u1_val - W1 / n
    baseY = u2_val - W2 / n
    acc = np.zeros_like(W1, dtype=float)

    for r1 in range(gamma1 + 1):
        for r2 in range(gamma2 + 1):
            acc += lam * f(
                baseX + r1 / (n * gamma1),
                baseY + r2 / (n * gamma2)
            )
    return acc

def Qn_approx(n):
    """Q_n^s(f)(u) = ∬ Δ_n(f)(v) K(w) dw."""
    out = np.zeros_like(U1)
    for i in range(Nu):
        for j in range(Nu):
            Delta_vals = Delta_vals_on_w(
                float(U1[i, j]),
                float(U2[i, j]),
                n
            )
            integrand = Delta_vals * K_w
            out[i, j] = integrate2D(integrand, w)
    return out

# ============================================================
# ERROR METRICS
# ============================================================
def RMSE(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def MAE(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def MAXERR(y_true, y_pred):
    return np.max(np.abs(y_true - y_pred))

# ============================================================
# HELPERS FOR SLICE PLOT
# ============================================================
def idx_closest(arr, val):
    return int(np.argmin(np.abs(arr - val)))

def slab(ax, xgrid, y0, zvals, color, alpha=0.8):
    X = np.vstack([xgrid, xgrid])
    Y = np.vstack([
        np.full_like(xgrid, y0 - eps),
        np.full_like(xgrid, y0 + eps)
    ])
    Z = np.vstack([zvals, zvals])
    ax.plot_surface(X, Y, Z, color=color, alpha=alpha, linewidth=0)

# ============================================================
# COMPUTE APPROXIMATIONS + ERRORS FOR ALL n
# ============================================================
Ftrue = f(U1, U2)

Einf_C, Einf_K, Einf_Q = [], [], []
C_store, K_store, Q_store = {}, {}, {}
rows = []

for n in ns:
    print(f"Computing for n = {n} ...")

    Cn = Cn_approx(n)
    Kn = Kn_approx(n)
    Qn = Qn_approx(n)

    C_store[n] = Cn
    K_store[n] = Kn
    Q_store[n] = Qn

    errC = np.abs(Cn - Ftrue)
    errK = np.abs(Kn - Ftrue)
    errQ = np.abs(Qn - Ftrue)

    Einf_C.append(np.max(errC))
    Einf_K.append(np.max(errK))
    Einf_Q.append(np.max(errQ))

    rows.append([r"$C_n^s$", n, RMSE(Ftrue, Cn), MAE(Ftrue, Cn), MAXERR(Ftrue, Cn)])
    rows.append([r"$K_n^s$", n, RMSE(Ftrue, Kn), MAE(Ftrue, Kn), MAXERR(Ftrue, Kn)])
    rows.append([r"$Q_n^s$", n, RMSE(Ftrue, Qn), MAE(Ftrue, Qn), MAXERR(Ftrue, Qn)])

    print(
        f"  E_inf(C) = {Einf_C[-1]:.6e}, "
        f"E_inf(K) = {Einf_K[-1]:.6e}, "
        f"E_inf(Q) = {Einf_Q[-1]:.6e}"
    )

# ============================================================
# ERROR METRICS TABLE
# ============================================================
df_metrics = pd.DataFrame(rows, columns=["Operator", "n", "RMSE", "MAE", "Max Error"])
pd.set_option("display.float_format", lambda x: f"{x:.6f}")

print("\nError metrics table:\n")
print(df_metrics)

print("\nLaTeX rows:\n")
for _, row in df_metrics.iterrows():
    print(
        f"{row['Operator']} & {int(row['n'])} & "
        f"{row['RMSE']:.6f} & {row['MAE']:.6f} & {row['Max Error']:.6f} \\\\"
    )

print("\nDiscrete uniform errors E_infty(n):\n")
for n in ns:
    E_C = MAXERR(Ftrue, C_store[n])
    E_K = MAXERR(Ftrue, K_store[n])
    E_Q = MAXERR(Ftrue, Q_store[n])

    print(
        f"n = {n:2d}  |  "
        f"E_inf(C_n^s) = {E_C:.6f}   "
        f"E_inf(K_n^s) = {E_K:.6f}   "
        f"E_inf(Q_n^s) = {E_Q:.6f}"
    )

# ============================================================
# FIGURES FOR n = 20
# ============================================================
ErrC = np.abs(C_store[n_show] - Ftrue)
ErrK = np.abs(K_store[n_show] - Ftrue)
ErrQ = np.abs(Q_store[n_show] - Ftrue)

Cn_show = C_store[n_compare]
Kn_show = K_store[n_compare]
Qn_show = Q_store[n_compare]

# ------------------------------------------------------------
# Figure 1a: Heatmap C
# ------------------------------------------------------------
plt.figure(figsize=(7.2, 5.6))
plt.imshow(
    ErrC,
    extent=[Omega_min, Omega_max, Omega_min, Omega_max],
    origin="lower",
    aspect="auto"
)
plt.colorbar()
plt.title(rf"$|C_{{{n_show}}}^s(f)-f|$ on $\Omega=[-1,1]^2$")
plt.xlabel(r"$u_1$")
plt.ylabel(r"$u_2$")
plt.tight_layout()
if SAVE_FIGURES:
    plt.savefig("Figure1a.png", dpi=600, bbox_inches="tight")

# ------------------------------------------------------------
# Figure 1b: Heatmap K
# ------------------------------------------------------------
plt.figure(figsize=(7.2, 5.6))
plt.imshow(
    ErrK,
    extent=[Omega_min, Omega_max, Omega_min, Omega_max],
    origin="lower",
    aspect="auto"
)
plt.colorbar()
plt.title(rf"$|K_{{{n_show}}}^s(f)-f|$ on $\Omega=[-1,1]^2$")
plt.xlabel(r"$u_1$")
plt.ylabel(r"$u_2$")
plt.tight_layout()
if SAVE_FIGURES:
    plt.savefig("Figure1b.png", dpi=600, bbox_inches="tight")

# ------------------------------------------------------------
# Figure 1c: Heatmap Q
# ------------------------------------------------------------
plt.figure(figsize=(7.2, 5.6))
plt.imshow(
    ErrQ,
    extent=[Omega_min, Omega_max, Omega_min, Omega_max],
    origin="lower",
    aspect="auto"
)
plt.colorbar()
plt.title(rf"$|Q_{{{n_show}}}^s(f)-f|$ on $\Omega=[-1,1]^2$")
plt.xlabel(r"$u_1$")
plt.ylabel(r"$u_2$")
plt.tight_layout()
if SAVE_FIGURES:
    plt.savefig("Figure1c.png", dpi=600, bbox_inches="tight")

# ------------------------------------------------------------
# Figure 2: Convergence plot E_infty(n)
# ------------------------------------------------------------
plt.figure(figsize=(7.2, 5.2))
plt.plot(ns, Einf_C, marker="o", label=r"$E_\infty(C_n^s)$")
plt.plot(ns, Einf_K, marker="o", label=r"$E_\infty(K_n^s)$")
plt.plot(ns, Einf_Q, marker="o", label=r"$E_\infty(Q_n^s)$")
plt.yscale("log")
plt.xlabel(r"$n$")
plt.ylabel(r"$E_\infty(n)$")
plt.title(r"Uniform error $E_\infty(n)$ on $\Omega=[-1,1]^2$")
plt.legend()
plt.tight_layout()
if SAVE_FIGURES:
    plt.savefig("Figure2.png", dpi=600, bbox_inches="tight")

# ------------------------------------------------------------
# Figure 3: 3D overlay
# ------------------------------------------------------------
fig = plt.figure(figsize=(9.0, 7.0))
ax = fig.add_subplot(111, projection="3d")

ax.plot_wireframe(U1, U2, Ftrue, rstride=3, cstride=3,
                  color="black", linewidth=1.0)

ax.plot_surface(U1, U2, Cn_show, cmap="viridis", alpha=0.45,
                linewidth=0, antialiased=True)
ax.plot_surface(U1, U2, Kn_show, cmap="plasma", alpha=0.45,
                linewidth=0, antialiased=True)
ax.plot_surface(U1, U2, Qn_show, cmap="cividis", alpha=0.45,
                linewidth=0, antialiased=True)

ax.set_title(
    rf"Approximation of $f$ by $C_{{{n_compare}}}^s$, "
    rf"$K_{{{n_compare}}}^s$, $Q_{{{n_compare}}}^s$ (black: $f$)"
)
ax.set_xlabel(r"$u_1$")
ax.set_ylabel(r"$u_2$")
ax.set_zlabel("value")

plt.tight_layout()
if SAVE_FIGURES:
    plt.savefig("Figure3.png", dpi=600, bbox_inches="tight")

# ------------------------------------------------------------
# Figure 4: 3D error surfaces
# ------------------------------------------------------------
fig = plt.figure(figsize=(15, 4.5))
for idx, (mat, cl, ttl) in enumerate([
    (ErrC, "Reds",   rf"$|C_{{{n_show}}}^s(f)-f|$"),
    (ErrK, "Blues",  rf"$|K_{{{n_show}}}^s(f)-f|$"),
    (ErrQ, "Greens", rf"$|Q_{{{n_show}}}^s(f)-f|$")
]):
    ax = fig.add_subplot(1, 3, idx + 1, projection="3d")
    ax.plot_surface(U1, U2, mat, cmap=cl, linewidth=0, antialiased=True)
    ax.set_title(ttl)
    ax.set_xlabel(r"$u_1$")
    ax.set_ylabel(r"$u_2$")
    ax.set_zlabel("error")

plt.tight_layout()
if SAVE_FIGURES:
    plt.savefig("Figure4.png", dpi=600, bbox_inches="tight")

# ------------------------------------------------------------
# Figure 5: Cake-slice view
# ------------------------------------------------------------
color_C = "blue"
color_K = "orange"
color_Q = "green"

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection="3d")

ax.plot_wireframe(U1, U2, Ftrue, color="black", linewidth=0.8)

for c in u2_slices:
    j = idx_closest(u2, c)

    slab(ax, u1, u2[j] - offset, Cn_show[j, :], color_C, alpha=0.80)
    slab(ax, u1, u2[j],          Kn_show[j, :], color_K, alpha=0.80)
    slab(ax, u1, u2[j] + offset, Qn_show[j, :], color_Q, alpha=0.80)

ax.set_title(
    rf"Cake-slice view of $C_{{{n_compare}}}^s$, "
    rf"$K_{{{n_compare}}}^s$, $Q_{{{n_compare}}}^s$ with $f$"
)
ax.set_xlabel(r"$u_1$")
ax.set_ylabel(r"$u_2$")
ax.set_zlabel("value")

plt.tight_layout()
if SAVE_FIGURES:
    plt.savefig("Figure5.png", dpi=600, bbox_inches="tight")

# ------------------------------------------------------------
# Figure 6: 3D convergence behavior
# ------------------------------------------------------------
fig = plt.figure(figsize=(12, 10))
gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.1])

colors_C = cm.turbo(np.linspace(0.20, 0.95, len(ns)))
colors_K = cm.magma(np.linspace(0.20, 0.95, len(ns)))
colors_Q = cm.viridis(np.linspace(0.20, 0.95, len(ns)))

# Top-left: C_n^s
ax1 = fig.add_subplot(gs[0, 0], projection="3d")
ax1.plot_wireframe(U1, U2, Ftrue, rstride=3, cstride=3,
                   color="black", linewidth=0.9)
for idx, n in enumerate(ns):
    ax1.plot_surface(
        U1, U2, C_store[n],
        color=colors_C[idx],
        alpha=0.35 + 0.10 * idx,
        linewidth=0,
        antialiased=True
    )
ax1.set_title(r"Convergence of $C_n^s$ toward $f$")
ax1.set_xlabel(r"$u_1$")
ax1.set_ylabel(r"$u_2$")
ax1.set_zlabel("value")

# Top-right: K_n^s
ax2 = fig.add_subplot(gs[0, 1], projection="3d")
ax2.plot_wireframe(U1, U2, Ftrue, rstride=3, cstride=3,
                   color="black", linewidth=0.9)
for idx, n in enumerate(ns):
    ax2.plot_surface(
        U1, U2, K_store[n],
        color=colors_K[idx],
        alpha=0.35 + 0.10 * idx,
        linewidth=0,
        antialiased=True
    )
ax2.set_title(r"Convergence of $K_n^s$ toward $f$")
ax2.set_xlabel(r"$u_1$")
ax2.set_ylabel(r"$u_2$")
ax2.set_zlabel("value")

# Bottom: Q_n^s
ax3 = fig.add_subplot(gs[1, :], projection="3d")
ax3.plot_wireframe(U1, U2, Ftrue, rstride=3, cstride=3,
                   color="black", linewidth=0.9)
for idx, n in enumerate(ns):
    ax3.plot_surface(
        U1, U2, Q_store[n],
        color=colors_Q[idx],
        alpha=0.35 + 0.10 * idx,
        linewidth=0,
        antialiased=True
    )
ax3.set_title(r"Convergence of $Q_n^s$ toward $f$")
ax3.set_xlabel(r"$u_1$")
ax3.set_ylabel(r"$u_2$")
ax3.set_zlabel("value")

fig.suptitle(
    r"3D convergence behavior of $C_n^s$, $K_n^s$, and $Q_n^s$ toward $f$ on $\Omega=[-1,1]^2$",
    fontsize=14,
    y=0.98
)

plt.tight_layout()
if SAVE_FIGURES:
    plt.savefig("Figure6.png", dpi=600, bbox_inches="tight")

# ============================================================
# SHOW ALL FIGURES
# ============================================================
plt.show()

# ============================================================
# OPTIONAL: SAVE METRICS TO CSV
# ============================================================
df_metrics.to_csv("error_metrics_symmetry_enhanced.csv", index=False)
print("\nCSV file saved as: error_metrics_symmetry_enhanced.csv")