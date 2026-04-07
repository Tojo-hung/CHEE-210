"""
visualizer.py
Matplotlib plotting functions for the Brayton cycle project.

All public functions accept pre-computed result dictionaries from engine.py
and return the created Figure object so callers can save or further customise.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import fluid_properties as fp

# ── Constants for consistent figure style ────────────────────────────────────
_FONT  = 40   # axis-label / title font size for Phase 1 single-panel plots
_LINE  = 3    # default line width
_CMAP  = "viridis"


# ── Shared helpers ────────────────────────────────────────────────────────────

def _vline(ax: plt.Axes, x: float, color: str, ls: str, label: str) -> None:
    """Draw a labelled vertical dashed line on ax."""
    ax.axvline(x, color=color, lw=1.4, ls=ls, alpha=0.85, label=label)


def _mark_rp(
    ax: plt.Axes,
    rp: float,
    y_val: float,
    color: str = "crimson",
    label: str | None = None,
) -> None:
    """Draw a vertical marker line and scatter point at (rp, y_val)."""
    lbl = label or f"$r_p$ = {rp:.1f}"
    _vline(ax, rp, color, "--", lbl)
    ax.scatter([rp], [y_val], color=color, zorder=5, s=60)


# ── Phase 1 plots ─────────────────────────────────────────────────────────────

def plot_phase1_net_power(
    phase1_results: dict,
    design_point: dict,
) -> plt.Figure:
    """Net power output vs pressure ratio (Phase 1).

    Args:
        phase1_results: Dictionary returned by BraytonCycle.solve_phase1.
        design_point:   Dictionary returned by BraytonCycle.get_design_point
                        (criterion='max_power').

    Returns:
        matplotlib Figure.
    """
    rp_vals   = phase1_results["rp_vals"]
    W_dot_MW  = phase1_results["W_dot_net"] / 1000.0
    rp_best   = design_point["rp"]
    idx_best  = design_point["idx"]

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.plot(rp_vals, W_dot_MW, color="steelblue", lw=_LINE)
    _mark_rp(ax, rp_best, W_dot_MW[idx_best],
             label=f"Max $\\dot{{W}}_{{net}}$: $r_p$={rp_best:.1f}")
    ax.set_xlabel(r"Pressure Ratio $r_p$", fontsize=_FONT)
    ax.set_ylabel(r"$\dot{W}_{net}$ (MW)", fontsize=_FONT)
    ax.set_title("Net Power Output", fontsize=_FONT)
    ax.legend(fontsize=_FONT)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_phase1_efficiency(
    phase1_results: dict,
    design_point: dict,
) -> plt.Figure:
    """Thermal efficiency vs pressure ratio (Phase 1).

    Args:
        phase1_results: Dictionary returned by BraytonCycle.solve_phase1.
        design_point:   Dictionary returned by BraytonCycle.get_design_point
                        (criterion='max_power').

    Returns:
        matplotlib Figure.
    """
    rp_vals  = phase1_results["rp_vals"]
    eta_pct  = phase1_results["eta_th"] * 100.0
    rp_best  = design_point["rp"]
    idx_best = design_point["idx"]

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.plot(rp_vals, eta_pct, color="darkorange", lw=_LINE)
    _mark_rp(ax, rp_best, eta_pct[idx_best],
             label=f"Max $\\dot{{W}}_{{net}}$: $r_p$={rp_best:.1f}")
    ax.set_xlabel(r"Pressure Ratio $r_p$", fontsize=_FONT)
    ax.set_ylabel(r"$\eta_{th}$ (%)", fontsize=_FONT)
    ax.set_title("Thermal Efficiency", fontsize=_FONT)
    ax.legend(fontsize=_FONT)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_phase1_bwr(
    phase1_results: dict,
    design_point: dict,
) -> plt.Figure:
    """Back work ratio vs pressure ratio (Phase 1).

    Args:
        phase1_results: Dictionary returned by BraytonCycle.solve_phase1.
        design_point:   Dictionary returned by BraytonCycle.get_design_point
                        (criterion='max_power').

    Returns:
        matplotlib Figure.
    """
    rp_vals  = phase1_results["rp_vals"]
    bwr_pct  = phase1_results["bwr"] * 100.0
    rp_best  = design_point["rp"]
    idx_best = design_point["idx"]

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.plot(rp_vals, bwr_pct, color="seagreen", lw=_LINE)
    _mark_rp(ax, rp_best, bwr_pct[idx_best],
             label=f"Max $\\dot{{W}}_{{net}}$: $r_p$={rp_best:.1f}")
    ax.set_xlabel(r"Pressure Ratio $r_p$", fontsize=_FONT)
    ax.set_ylabel(r"BWR = $\dot{W}_{in}/\dot{W}_{out}$ (%)", fontsize=_FONT)
    ax.set_title("Back Work Ratio", fontsize=_FONT)
    ax.legend(fontsize=_FONT)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_phase1_summary_table(
    design_best: dict,
    design_mmp: dict,
    P1: float,
    Q_dot_in: float,
) -> plt.Figure:
    """Summary comparison table for the two Phase 1 optimal design points.

    Args:
        design_best: Design point dict from get_design_point('max_power').
        design_mmp:  Design point dict from get_design_point('max_specific_work').
        P1:          Compressor inlet pressure in kPa (for pressure display).
        Q_dot_in:    Heat input rate in kW.

    Returns:
        matplotlib Figure.
    """
    db = design_best
    dm = design_mmp

    summary = (
        f"{'Criterion':<14} {'Max Ẇ_net':>10} {'Max w/kg':>10}\n"
        f"{'─' * 36}\n"
        f"{'r_p':<14} {db['rp']:>10.1f} {dm['rp']:>10.1f}\n"
        f"{'T2 (K)':<14} {db['T2']:>10.1f} {dm['T2']:>10.1f}\n"
        f"{'T4 (K)':<14} {db['T4']:>10.1f} {dm['T4']:>10.1f}\n"
        f"{'P2 (MPa)':<14} {P1*db['rp']/1000:>10.3f} {P1*dm['rp']/1000:>10.3f}\n"
        f"{'ṁ (kg/s)':<14} {db['m_dot']:>10.3f} {dm['m_dot']:>10.3f}\n"
        f"{'Ẇ_net (MW)':<14} {db['W_dot_net']/1000:>10.3f} {dm['W_dot_net']/1000:>10.3f}\n"
        f"{'η_th (%)':<14} {db['eta_th']*100:>10.2f} {dm['eta_th']*100:>10.2f}\n"
        f"{'BWR (%)':<14} {db['bwr']*100:>10.2f} {dm['bwr']*100:>10.2f}\n"
        f"{'w_net(kJ/kg)':<14} {db['w_net']:>10.2f} {dm['w_net']:>10.2f}\n"
        f"\n● = max Ẇ_net   ◆ = max w_net/kg"
    )

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.axis("off")
    ax.text(
        0.03, 0.97, summary,
        transform=ax.transAxes,
        fontsize=9.5, va="top", fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow",
                  edgecolor="gray", alpha=0.9),
    )
    plt.tight_layout()
    return fig


def plot_phase1_optimization(
    phase1_results: dict,
) -> plt.Figure:
    """Optimization Model 1 score (1/rp)·w_net·η_th·(1/BWR) vs rp (Phase 1).

    Args:
        phase1_results: Dictionary returned by BraytonCycle.solve_phase1.

    Returns:
        matplotlib Figure.
    """
    rp_vals    = phase1_results["rp_vals"]
    w_net_spec = phase1_results["w_net_spec"]
    eta_th     = phase1_results["eta_th"]
    bwr        = phase1_results["bwr"]

    a = 0.75
    b = 0.4
    c = 0.5

    with np.errstate(invalid="ignore", divide="ignore"):
        metric = ((w_net_spec/np.max(w_net_spec)) + ((eta_th/np.max(eta_th)))) - (bwr/np.max(bwr)) - (rp_vals/np.max(rp_vals))

    idx_peak = int(np.nanargmax(metric))
    rp_peak  = rp_vals[idx_peak]

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.plot(rp_vals, metric, color="teal", lw=_LINE)
    ax.axvline(rp_peak, color="crimson", ls="--", lw=1.4,
               label=f"Peak: $r_p$ = {rp_peak:.1f}")
    ax.scatter([rp_peak], [metric[idx_peak]], color="crimson", zorder=5, s=60)
    ax.set_xlabel(r"Pressure Ratio $r_p$", fontsize=_FONT)
    ax.set_ylabel(
        r"Linear combination",  
        fontsize=_FONT,
    )
    ax.set_title(
        r"Linear combination of normalized metrics: $w_{net}$, $\eta_{th}$, BWR, and $r_p$",
        fontsize=_FONT,
    )
    ax.legend(fontsize=_FONT)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig

def plot_phase1_optimization_old(
    phase1_results: dict,
) -> plt.Figure:
    """Optimization Model 1 score (1/rp)·w_net·η_th·(1/BWR) vs rp (Phase 1).

    Args:
        phase1_results: Dictionary returned by BraytonCycle.solve_phase1.

    Returns:
        matplotlib Figure.
    """
    rp_vals    = phase1_results["rp_vals"]
    w_net_spec = phase1_results["w_net_spec"]
    eta_th     = phase1_results["eta_th"]
    bwr        = phase1_results["bwr"]

    a = 0.75
    b = 0.4
    c = 0.5

    with np.errstate(invalid="ignore", divide="ignore"):
        metric = (1/(rp_vals/np.max(rp_vals))) * (w_net_spec/np.max(w_net_spec)) * (eta_th/np.max(eta_th)) * ((1/bwr/np.max(bwr)))

    idx_peak = int(np.nanargmax(metric))
    rp_peak  = rp_vals[idx_peak]

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.plot(rp_vals, metric, color="teal", lw=_LINE)
    ax.axvline(rp_peak, color="crimson", ls="--", lw=1.4,
               label=f"Peak: $r_p$ = {rp_peak:.1f}")
    ax.scatter([rp_peak], [metric[idx_peak]], color="crimson", zorder=5, s=60)
    ax.set_xlabel(r"Pressure Ratio $r_p$", fontsize=_FONT)
    ax.set_ylabel(
        r"Function Model",
        fontsize=_FONT,
    )
    ax.set_title(
        r"Optimization Model Old",
        fontsize=_FONT,
    )
    ax.legend(fontsize=_FONT)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig



# ── Phase 2 contour maps ──────────────────────────────────────────────────────

def _phase2_contour_panel(
    fig: plt.Figure,
    ax: plt.Axes,
    eta_vals: np.ndarray,
    data: np.ndarray,
    title: str,
    cbar_label: str,
    cmap: str = _CMAP,
    levels: int = 500,
) -> None:
    """Draw one contourf panel for a Phase 2 figure."""
    cf = ax.contourf(eta_vals, eta_vals, data, levels=levels, cmap=cmap)
    cb = fig.colorbar(cf, ax=ax, pad=0.02)
    cb.set_label(cbar_label, fontsize=10)
    ax.set_xlabel(r"$\eta_{C,II}$", fontsize=11)
    ax.set_ylabel(r"$\eta_{T,II}$", fontsize=11)
    ax.set_title(title, fontsize=11)


def plot_phase2_contours(
    eta_vals: np.ndarray,
    contour_results: dict,
    rp: float,
    Q_dot_in_MW: float,
    T_max_C: float,
) -> plt.Figure:
    """Four-panel Phase 2 contour maps over (η_C, η_T) space.

    Args:
        eta_vals:        1-D array of efficiency values (axes of the grid).
        contour_results: Dictionary returned by BraytonCycle.solve_phase2_contour.
        rp:              Pressure ratio used for this contour set.
        Q_dot_in_MW:     Heat input rate in MW (for the suptitle).
        T_max_C:         Turbine inlet temperature in °C (for the suptitle).
        eta_C_sample:    Compressor efficiency of the sample point marker.
        eta_T_sample:    Turbine efficiency of the sample point marker.

    Returns:
        matplotlib Figure.
    """
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    fig.suptitle(
        rf"Phase 2 - Non-ideal Brayton Cycle on $CO_2$",
        fontsize=13, fontweight="bold", y=0.99,
    )

    kw = dict(
        fig=fig,
        eta_vals=eta_vals,
    )

    _phase2_contour_panel(ax=axes[0, 0],
                          data=contour_results["m_dot_grid"],
                          title=r"Mass Flow Rate $\dot{m}$",
                          cbar_label=r"$\dot{m}$ (kg/s)",
                          **kw)

    _phase2_contour_panel(ax=axes[0, 1],
                          data=contour_results["W_dot_grid"],
                          title=r"Net Power $\dot{W}_{net}$",
                          cbar_label=r"$\dot{W}_{net}$ (MW)",
                          cmap="plasma",
                          **kw)

    _phase2_contour_panel(ax=axes[1, 0],
                          data=contour_results["eta_th_grid"],
                          title=r"Thermal Efficiency $\eta_{th}$",
                          cbar_label=r"$\eta_{th}$ (%)",
                          cmap="RdYlGn",
                          **kw)

    _phase2_contour_panel(ax=axes[1, 1],
                          data=contour_results["bwr_grid"],
                          title=r"Back Work Ratio $\dot{W}_{in}/\dot{W}_{out}$",
                          cbar_label="BWR (%)",
                          cmap="RdYlBu_r",
                          **kw)

    plt.tight_layout()
    return fig


# ── Phase 2 pressure-ratio optimization plot ─────────────────────────────────

def plot_phase2_mean_power_search(optimization_results: dict) -> plt.Figure:
    """Single-panel plot of mean net power vs pressure ratio.

    Args:
        optimization_results: Dictionary returned by
            BraytonCycle.solve_phase2_mean_power_search.

    Returns:
        matplotlib Figure.
    """
    rp_search = optimization_results["rp_search"]
    mean_wdot = optimization_results["mean_Wdot"]
    rp_opt    = optimization_results["rp_optimal"]
    idx_opt   = optimization_results["idx_optimal"]

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.plot(rp_search, mean_wdot / 1000.0, color="steelblue", lw=2)
    ax.axvline(
        rp_opt,
        color="crimson",
        ls="--",
        lw=1.4,
        label=rf"Optimal $r_p$ = {rp_opt:.1f}",
    )
    ax.scatter([rp_opt], [mean_wdot[idx_opt] / 1000.0], color="crimson", zorder=5, s=60)
    ax.set_xlabel(r"Pressure Ratio $r_p$", fontsize=_FONT)
    ax.set_ylabel(r"Mean $\dot{W}_{net}$ over efficiency grid (MW)", fontsize=_FONT)
    ax.set_title("Pressure-Ratio Optimization by Mean Net Power", fontsize=_FONT)
    ax.legend(fontsize=_FONT)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


# ── Phase 3 comparison plots ──────────────────────────────────────────────────

def plot_phase3_comparison(
    phase1_point: dict,
    phase3_results: dict,
    rp: float,
    Q_dot_in_MW: float,
) -> plt.Figure:
    """T-s state-point diagram and performance bar chart (Phase 3 vs Phase 1).

    Args:
        phase1_point:   dict from BraytonCycle.solve_point for the same rp.
                        Must include keys: T2, T4, q_in (Phase 1 values).
        phase3_results: dict from BraytonCycle.solve_phase3.
        rp:             Pressure ratio used (for suptitle).
        Q_dot_in_MW:    Heat input rate in MW (for suptitle).

    Returns:
        matplotlib Figure.
    """
    p1  = phase1_point
    p3  = phase3_results
    T1_ig  = p1.get("T1", 298.15)   # compressor inlet temperature
    T3_val = p3.get("T3", 1073.15)  # turbine inlet temperature
    T2_ig  = p1["T2"]
    T4_ig  = p1["T4"]
    T2_rf  = p3["T2"]
    T4_rf  = p3["T4"]

    s1_val = p3["s1"]
    s3_val = p3["s3"]

    fig, (ax_ts, ax_bar) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        rf"Phase 3 - Ideal Brayton Cycle: Ideal Gas vs Real $CO_2$  "
        rf"($r_p={rp:.2f}$, \dot{{Q}}_{{in}}={Q_dot_in_MW:.0f}\,\mathrm{{MW}}$)",
        fontsize=13, fontweight="bold",
    )

    # T–s diagram (straight segments connect the four comparison state points)
    ax_ts.plot([s1_val, s1_val], [T1_ig, T2_ig], "r-", lw=1.8, label="Ideal gas")
    ax_ts.plot([s1_val, s3_val], [T2_ig, T3_val], "r-", lw=1.8)
    ax_ts.plot([s3_val, s3_val], [T4_ig, T3_val], "r-", lw=1.8)
    ax_ts.plot([s3_val, s1_val], [T4_ig, T1_ig], "r-", lw=1.8)

    ax_ts.plot([s1_val, s1_val], [T1_ig, T2_rf], "b:", lw=2.2, label="Real fluid")
    ax_ts.plot([s1_val, s3_val], [T2_rf, T3_val], "b:", lw=2.2)
    ax_ts.plot([s3_val, s3_val], [T4_rf, T3_val], "b:", lw=2.2)
    ax_ts.plot([s3_val, s1_val], [T4_rf, T1_ig], "b:", lw=2.2)

    for s_, T_, lbl, col, dy in [
        (s1_val, T1_ig,  "1",         "steelblue", (6,   4)),
        (s1_val, T2_ig,  "2s\n(IG)",  "steelblue", (6,   4)),
        (s3_val, T3_val, "3",         "steelblue", (6,   4)),
        (s3_val, T4_ig,  "4s\n(IG)",  "steelblue", (6,   4)),
    ]:
        ax_ts.scatter([s_], [T_], color=col, zorder=5, s=50)
        ax_ts.annotate(lbl, (s_, T_), textcoords="offset points",
                       xytext=dy, fontsize=8, color=col)

    for s_, T_, lbl, col, dy in [
        (s1_val, T2_rf,  "2s\n(RF)", "tomato", (6, -16)),
        (s3_val, T4_rf,  "4s\n(RF)", "tomato", (6, -16)),
    ]:
        ax_ts.scatter([s_], [T_], color=col, zorder=5, s=50)
        ax_ts.annotate(lbl, (s_, T_), textcoords="offset points",
                       xytext=dy, fontsize=8, color=col)

    ax_ts.set_xlabel(r"$s$ (kJ/kgK)", fontsize=11)
    ax_ts.set_ylabel("T (K)", fontsize=11)
    ax_ts.set_title("Isentropic State Points on T-s Diagram", fontsize=11)
    ax_ts.legend(fontsize=9)
    ax_ts.grid(True, alpha=0.3)

    # Bar chart comparison
    metrics  = [r"$\eta_{th}$ (%)", r"BWR (%)", r"$\dot{W}_{net}$ (MW)", r"$\dot{m}$ (kg/s)"]
    vals_ig  = [p1["eta_th"] * 100, p1["bwr"] * 100, p1["W_dot_net"] / 1000, p1["m_dot"]]
    vals_rf  = [p3["eta_th"] * 100, p3["bwr"] * 100, p3["W_dot_net"] / 1000, p3["m_dot"]]

    x     = np.arange(len(metrics))
    w_bar = 0.35
    bars1 = ax_bar.bar(x - w_bar / 2, vals_ig, w_bar,
                       label="Ideal Gas (Ph.1)", color="steelblue", alpha=0.85)
    bars2 = ax_bar.bar(x + w_bar / 2, vals_rf, w_bar,
                       label="Real Fluid (Ph.3)", color="tomato", alpha=0.85)

    top = max(vals_ig + vals_rf) * 0.01
    for bar in list(bars1) + list(bars2):
        ax_bar.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + top,
            f"{bar.get_height():.3f}",
            ha="center", va="bottom", fontsize=8,
        )

    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(metrics, fontsize=10)
    ax_bar.set_title("Performance: Ideal Gas vs Real Fluid", fontsize=11)
    ax_bar.legend(fontsize=9)
    ax_bar.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    return fig


def plot_phase4_eta_vs_regen(phase4) -> plt.Figure:
    """Plot thermal efficiency against regenerator heat-transfer rate."""
    fig, ax = plt.subplots(figsize=(12, 7))
    colors = ("steelblue", "darkorange", "seagreen", "crimson")

    for idx, case in enumerate(phase4.cases):
        sweep = case.sweep
        feasible = sweep["feasible"]
        q_regen_mw = sweep["Q_dot_regen"][feasible] / 1000.0
        eta_pct = sweep["eta_th"][feasible] * 100.0
        color = colors[idx % len(colors)]
        order = np.argsort(q_regen_mw)

        ax.plot(
            q_regen_mw[order],
            eta_pct[order],
            color=color,
            lw=2,
            label=rf"{case.rp_case.label}: $r_p$ = {case.rp_case.rp:.2f}",
        )
        ax.scatter(
            [0.0],
            [case.baseline_phase3["eta_th"] * 100.0],
            color=color,
            marker="x",
            s=90,
            zorder=5,
        )

    ax.set_xlabel(r"$\dot{Q}_{regen}$ (MW)", fontsize=_FONT)
    ax.set_ylabel(r"$\eta_{th}$ (%)", fontsize=_FONT)
    ax.set_title("Phase 4 Regenerator Study: Thermal Efficiency", fontsize=_FONT)
    ax.legend(fontsize=max(12, _FONT // 2))
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_phase4_power_vs_regen(phase4) -> plt.Figure:
    """Plot net power against regenerator heat-transfer rate."""
    fig, ax = plt.subplots(figsize=(12, 7))
    colors = ("steelblue", "darkorange", "seagreen", "crimson")

    for idx, case in enumerate(phase4.cases):
        sweep = case.sweep
        feasible = sweep["feasible"]
        q_regen_mw = sweep["Q_dot_regen"][feasible] / 1000.0
        wdot_mw = sweep["W_dot_net"][feasible] / 1000.0
        color = colors[idx % len(colors)]
        order = np.argsort(q_regen_mw)

        ax.plot(
            q_regen_mw[order],
            wdot_mw[order],
            color=color,
            lw=2,
            label=rf"{case.rp_case.label}: $r_p$ = {case.rp_case.rp:.2f}",
        )
        ax.scatter(
            [0.0],
            [case.baseline_phase3["W_dot_net"] / 1000.0],
            color=color,
            marker="x",
            s=90,
            zorder=5,
        )

    ax.set_xlabel(r"$\dot{Q}_{regen}$ (MW)", fontsize=_FONT)
    ax.set_ylabel(r"$\dot{W}_{net}$ (MW)", fontsize=_FONT)
    ax.set_title("Phase 4 Regenerator Study: Net Power", fontsize=_FONT)
    ax.legend(fontsize=max(12, _FONT // 2))
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_phase4_rp_sweep_metric(
    rp_vals: np.ndarray,
    sweep_results: list[dict],
    metric_key: str,
    ylabel: str,
    title: str,
    cmap: str = "viridis",
) -> plt.Figure:
    """Plot a Phase 4 metric against Q_dot_regen for many pressure ratios."""
    fig, ax = plt.subplots(figsize=(12, 7))
    norm = plt.Normalize(vmin=float(np.min(rp_vals)), vmax=float(np.max(rp_vals)))
    colors = plt.cm.get_cmap(cmap)

    for rp, sweep in zip(rp_vals, sweep_results):
        feasible = sweep["feasible"]
        if np.count_nonzero(feasible) < 2:
            continue

        q_regen_mw = sweep["Q_dot_regen"][feasible] / 1000.0
        metric_vals = sweep[metric_key][feasible]
        if metric_key in {"eta_th"}:
            metric_vals = metric_vals * 100.0
        elif metric_key in {"W_dot_net"}:
            metric_vals = metric_vals / 1000.0

        order = np.argsort(q_regen_mw)
        ax.plot(
            q_regen_mw[order],
            metric_vals[order],
            color=colors(norm(rp)),
            lw=2,
            alpha=0.95,
        )

    sm = plt.cm.ScalarMappable(norm=norm, cmap=colors)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label(r"Pressure Ratio $r_p$", fontsize=12)

    ax.set_xlabel(r"$\dot{Q}_{regen}$ (MW)", fontsize=_FONT)
    ax.set_ylabel(ylabel, fontsize=_FONT)
    ax.set_title(title, fontsize=max(18, _FONT // 2))
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_phase4_ts_diagram(phase5_case_artifact) -> plt.Figure:
    """Plot the real-fluid Phase 4 regenerated Brayton cycle on a T-s diagram."""
    selected = phase5_case_artifact.selected_phase4_case
    rp_case = phase5_case_artifact.rp_case
    fluid = getattr(phase5_case_artifact, "fluid", "CO2")

    segments = [
        fp.coolprop_isentropic_ts_path(
            selected["P1"],
            selected["P2"],
            selected["s1"],
            fluid=fluid,
        ),
        fp.coolprop_isobaric_ts_path(
            selected["P2"],
            selected["T2"],
            selected["T3"],
            fluid=fluid,
        ),
        fp.coolprop_isobaric_ts_path(
            selected["P3"],
            selected["T3"],
            selected["T4"],
            fluid=fluid,
        ),
        fp.coolprop_isentropic_ts_path(
            selected["P4"],
            selected["P5"],
            selected["s4"],
            fluid=fluid,
        ),
        fp.coolprop_isobaric_ts_path(
            selected["P5"],
            selected["T5"],
            selected["T6"],
            fluid=fluid,
        ),
        fp.coolprop_isobaric_ts_path(
            selected["P6"],
            selected["T6"],
            selected["T1"],
            fluid=fluid,
        ),
    ]

    s_paths = [segments[0]["s"]]
    t_paths = [segments[0]["T"]]
    for segment in segments[1:]:
        s_paths.append(segment["s"][1:])
        t_paths.append(segment["T"][1:])
    s_vals = np.concatenate(s_paths)
    t_vals = np.concatenate(t_paths)

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.plot(s_vals, t_vals, color="steelblue", lw=2.5)
    ax.scatter(
        [selected[f"s{idx}"] for idx in range(1, 7)],
        [selected[f"T{idx}"] for idx in range(1, 7)],
        color="crimson",
        s=70,
        zorder=5,
    )

    offsets = {
        1: (8, -14),
        2: (8, 6),
        3: (8, 6),
        4: (8, 6),
        5: (8, -14),
        6: (8, -14),
    }
    for idx in range(1, 7):
        ax.annotate(
            str(idx),
            (selected[f"s{idx}"], selected[f"T{idx}"]),
            textcoords="offset points",
            xytext=offsets[idx],
            fontsize=12,
            color="black",
        )

    ax.set_xlabel(r"$s$ (kJ/kg$\cdot$K)", fontsize=_FONT)
    ax.set_ylabel(r"$T$ (K)", fontsize=_FONT)
    ax.set_title(
        (
            f"Phase 4 Regenerated Brayton Cycle T-s Diagram\n"
            f"{rp_case.label}: $r_p$ = {rp_case.rp:.2f}, "
            f"$\\Delta T_{{approach}}$ = {selected['delta_T_approach']:.1f} K"
        ),
        fontsize=max(18, _FONT // 2),
    )
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig
