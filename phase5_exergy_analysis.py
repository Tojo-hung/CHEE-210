#!/usr/bin/env python3
"""
phase5_exergy_analysis.py
Exergy analysis of the ideal Brayton cycle (ideal gas, CO2)
across the full pressure-ratio range.

Produces:
  * Specific exergy at each state vs pressure ratio
  * Exergetic efficiency vs pressure ratio
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from constants import (
    CO2_A,
    CO2_B,
    CO2_C,
    CO2_D,
    M_CO2,
    T_INLET,
    T_MAX,
    R_CO2,
)
from engine import BraytonCycle
from fluid_properties import delta_h_analytic
from phase_workflows import apply_plot_style, build_default_cycle

ResultDict = dict[str, Any]

_FIGURE_DPI = 200


def delta_s0_analytic(T_low: float | np.ndarray, T_high: float | np.ndarray) -> float | np.ndarray:
    """Analytic ideal-gas entropy-function change using the A-2c cp polynomial."""

    def S0(T: float | np.ndarray) -> float | np.ndarray:
        return (
            CO2_A * np.log(T)
            + CO2_B * T
            + CO2_C * T**2 / 2.0
            + CO2_D * T**3 / 3.0
        ) / M_CO2

    return S0(T_high) - S0(T_low)


def compute_ideal_gas_exergy_sweep(cycle: BraytonCycle, rp_vals: np.ndarray) -> ResultDict:
    """
    Compute ideal-gas exergy at all four state points across a range of rp.

    Parameters
    ----------
    cycle : BraytonCycle
        Configured cycle instance.
    rp_vals : ndarray
        Pressure ratio values to sweep.

    Returns
    -------
    dict
        Exergy and performance results across the pressure-ratio sweep.
    """
    phase1 = cycle.solve_phase1(rp_vals)

    rp_vals = np.asarray(phase1["rp_vals"], dtype=float)
    T2 = np.asarray(phase1["T2"], dtype=float)
    T4 = np.asarray(phase1["T4"], dtype=float)
    m_dot = np.asarray(phase1["m_dot"], dtype=float)
    W_dot_net = np.asarray(phase1["W_dot_net"], dtype=float)
    eta_th = np.asarray(phase1["eta_th"], dtype=float)

    T0 = cycle.T1
    P0 = cycle.P1
    T1 = np.full_like(rp_vals, cycle.T1, dtype=float)
    T3 = np.full_like(rp_vals, cycle.T3, dtype=float)
    P1 = np.full_like(rp_vals, cycle.P1, dtype=float)
    P2 = cycle.P1 * rp_vals
    P3 = cycle.P1 * rp_vals
    P4 = np.full_like(rp_vals, cycle.P1, dtype=float)

    h1_minus_h0 = np.zeros_like(rp_vals)
    h2_minus_h0 = np.asarray(delta_h_analytic(T0, T2), dtype=float)
    h3_minus_h0 = np.asarray(delta_h_analytic(T0, T3), dtype=float)
    h4_minus_h0 = np.asarray(delta_h_analytic(T0, T4), dtype=float)

    s1_minus_s0 = np.zeros_like(rp_vals)
    with np.errstate(invalid="ignore", divide="ignore"):
        s2_minus_s0 = np.asarray(delta_s0_analytic(T0, T2), dtype=float) - R_CO2 * np.log(P2 / P0)
        s3_minus_s0 = np.asarray(delta_s0_analytic(T0, T3), dtype=float) - R_CO2 * np.log(P3 / P0)
        s4_minus_s0 = np.asarray(delta_s0_analytic(T0, T4), dtype=float) - R_CO2 * np.log(P4 / P0)

    x1 = h1_minus_h0 - T0 * s1_minus_s0
    x2 = h2_minus_h0 - T0 * s2_minus_s0
    x3 = h3_minus_h0 - T0 * s3_minus_s0
    x4 = h4_minus_h0 - T0 * s4_minus_s0

    delta_x_comp = x2 - x1
    delta_x_heater = x3 - x2
    delta_x_turbine = x3 - x4
    delta_x_cooler = x4 - x1

    X_dot_1 = m_dot * x1
    X_dot_2 = m_dot * x2
    X_dot_3 = m_dot * x3
    X_dot_4 = m_dot * x4

    X_dot_comp = m_dot * delta_x_comp
    X_dot_heater = m_dot * delta_x_heater
    X_dot_turbine = m_dot * delta_x_turbine
    X_dot_cooler = m_dot * delta_x_cooler

    X_dot_in_source = cycle.Q_dot_in * (1.0 - T0 / cycle.T3)
    with np.errstate(invalid="ignore", divide="ignore"):
        eta_exergy = W_dot_net / X_dot_heater
        eta_exergy_source = W_dot_net / X_dot_in_source

    return {
        "rp_vals": rp_vals,
        "T1": T1,
        "T2": T2,
        "T3": T3,
        "T4": T4,
        "P1": P1,
        "P2": P2,
        "P3": P3,
        "P4": P4,
        "m_dot": m_dot,
        "W_dot_net": W_dot_net,
        "eta_th": eta_th,
        "x1": x1,
        "x2": x2,
        "x3": x3,
        "x4": x4,
        "delta_x_comp": delta_x_comp,
        "delta_x_heater": delta_x_heater,
        "delta_x_turbine": delta_x_turbine,
        "delta_x_cooler": delta_x_cooler,
        "X_dot_1": X_dot_1,
        "X_dot_2": X_dot_2,
        "X_dot_3": X_dot_3,
        "X_dot_4": X_dot_4,
        "X_dot_comp": X_dot_comp,
        "X_dot_heater": X_dot_heater,
        "X_dot_turbine": X_dot_turbine,
        "X_dot_cooler": X_dot_cooler,
        "eta_exergy": eta_exergy,
        "eta_exergy_source": eta_exergy_source,
        "X_dot_in_source": X_dot_in_source,
    }


def find_curve_intersection(
    x_vals: np.ndarray,
    y_a: np.ndarray,
    y_b: np.ndarray,
) -> tuple[float, float] | None:
    """Return the first interpolated intersection of two curves, if one exists."""
    diff = np.asarray(y_a, dtype=float) - np.asarray(y_b, dtype=float)
    finite = np.isfinite(x_vals) & np.isfinite(diff)
    x = np.asarray(x_vals, dtype=float)[finite]
    d = diff[finite]

    if len(x) < 2:
        return None

    exact_hits = np.where(np.isclose(d, 0.0, atol=1e-10))[0]
    if exact_hits.size:
        idx = int(exact_hits[0])
        y_val = 0.5 * (np.asarray(y_a, dtype=float)[finite][idx] + np.asarray(y_b, dtype=float)[finite][idx])
        return float(x[idx]), float(y_val)

    sign_changes = np.where(d[:-1] * d[1:] < 0.0)[0]
    if sign_changes.size == 0:
        return None

    idx = int(sign_changes[0])
    x0, x1 = x[idx], x[idx + 1]
    d0, d1 = d[idx], d[idx + 1]
    rp_cross = x0 - d0 * (x1 - x0) / (d1 - d0)

    y_a_finite = np.asarray(y_a, dtype=float)[finite]
    y_b_finite = np.asarray(y_b, dtype=float)[finite]
    y0 = 0.5 * (y_a_finite[idx] + y_b_finite[idx])
    y1 = 0.5 * (y_a_finite[idx + 1] + y_b_finite[idx + 1])
    y_cross = y0 + (y1 - y0) * (rp_cross - x0) / (x1 - x0)
    return float(rp_cross), float(y_cross)


def plot_exergy_vs_rp(exergy_results: ResultDict) -> plt.Figure:
    """Specific exergy at each state vs pressure ratio."""
    rp_vals = exergy_results["rp_vals"]
    x2 = exergy_results["x2"]
    x4 = exergy_results["x4"]
    intersection = find_curve_intersection(rp_vals, x2, x4)

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.plot(rp_vals, exergy_results["x1"], lw=2.5, color="black", label="State 1")
    ax.plot(rp_vals, x2, lw=2.5, color="steelblue", label="State 2")
    ax.plot(rp_vals, exergy_results["x3"], lw=2.5, color="darkorange", label="State 3")
    ax.plot(rp_vals, x4, lw=2.5, color="seagreen", label="State 4")
    if intersection is not None:
        rp_cross, x_cross = intersection
        ax.axvline(
            rp_cross,
            color="red",
            lw=1.2,
            ls="--",
            alpha=0.9,
            label=fr"State 2 = State 4 at $r_p={rp_cross:.2f}$",
        )
        ax.scatter([rp_cross], [x_cross], color="red", s=60, zorder=6)
        ax.text(
            rp_cross,
            x_cross,
            f"  intersection\n  $r_p={rp_cross:.2f}$",
            color="red",
            va="bottom",
            ha="left",
        )
    ax.set_xlabel(r"Pressure Ratio $r_p$")
    ax.set_ylabel(r"Specific Exergy $x$ (kJ/kg)")
    ax.set_title("Ideal-Brayton Specific Exergy by State")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    return fig


def plot_exergetic_efficiency_vs_rp(exergy_results: ResultDict) -> plt.Figure:
    """Heater-based exergetic efficiency vs pressure ratio."""
    rp_vals = exergy_results["rp_vals"]
    eta_ex_pct = exergy_results["eta_exergy"] * 100.0
    eta_th_pct = exergy_results["eta_th"] * 100.0

    idx_peak = int(np.nanargmax(exergy_results["eta_exergy"]))
    rp_peak = rp_vals[idx_peak]
    eta_peak = eta_ex_pct[idx_peak]

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.plot(rp_vals, eta_ex_pct, lw=2.7, color="crimson", label=r"Exergetic efficiency $\eta_{ex}$")
    ax.plot(
        rp_vals,
        eta_th_pct,
        lw=2.2,
        ls="--",
        color="dimgray",
        label=r"Thermal efficiency $\eta_{th}$",
    )
    ax.axvline(rp_peak, color="crimson", lw=1.2, ls=":", alpha=0.9)
    ax.scatter([rp_peak], [eta_peak], color="crimson", s=55, zorder=5)
    ax.text(
        rp_peak,
        eta_peak,
        f"  max at $r_p={rp_peak:.2f}$",
        color="crimson",
        va="bottom",
        ha="left",
    )
    ax.set_xlabel(r"Pressure Ratio $r_p$")
    ax.set_ylabel("Efficiency (%)")
    ax.set_title("Ideal-Brayton Exergetic Efficiency")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    return fig


def print_exergy_summary(
    cycle: BraytonCycle,
    sample_rp_vals: tuple[float, ...] = (5.0, 10.0, 20.0, 40.0, 60.0),
) -> None:
    """Print representative pressure-ratio summary rows and the efficiency optimum."""
    sample_rp_array = np.asarray(sample_rp_vals, dtype=float)
    sample_rp_array = sample_rp_array[(sample_rp_array >= 1.0) & (sample_rp_array <= cycle.rp_max)]
    results = compute_ideal_gas_exergy_sweep(cycle, sample_rp_array)

    print("\nIdeal-gas exergy summary at representative pressure ratios")
    print(
        f"{'rp':>7} {'T2 (K)':>10} {'T4 (K)':>10} {'x1':>10} {'x2':>10} "
        f"{'x3':>10} {'x4':>10} {'eta_th (%)':>12} {'eta_ex (%)':>12} {'W_net (MW)':>12}"
    )
    print("-" * 107)
    for idx in range(len(results["rp_vals"])):
        print(
            f"{results['rp_vals'][idx]:7.2f} "
            f"{results['T2'][idx]:10.2f} "
            f"{results['T4'][idx]:10.2f} "
            f"{results['x1'][idx]:10.2f} "
            f"{results['x2'][idx]:10.2f} "
            f"{results['x3'][idx]:10.2f} "
            f"{results['x4'][idx]:10.2f} "
            f"{results['eta_th'][idx] * 100.0:12.2f} "
            f"{results['eta_exergy'][idx] * 100.0:12.2f} "
            f"{results['W_dot_net'][idx] / 1000.0:12.3f}"
        )

    idx_peak = int(np.nanargmax(results["eta_exergy"]))
    print(
        "\nMaximum exergetic efficiency in summary sample: "
        f"{results['eta_exergy'][idx_peak] * 100.0:.2f}% "
        f"at rp = {results['rp_vals'][idx_peak]:.2f}"
    )
    print(
        "Heat-source exergy input: "
        f"{results['X_dot_in_source']:.2f} kW "
        f"(using T0 = {T_INLET:.2f} K and Th = {T_MAX:.2f} K)"
    )


def save_figures(figures: list[tuple[plt.Figure, str]], output_dir: Path) -> None:
    """Save figures to PNG in the chosen output directory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for fig, filename in figures:
        fig.savefig(output_dir / filename, dpi=_FIGURE_DPI, bbox_inches="tight")


def main(show_plots: bool = True, save_output: bool = True, output_dir: str = ".") -> ResultDict:
    """Run the ideal-gas exergy sweep, print summary results, and create figures."""
    apply_plot_style()
    cycle = build_default_cycle()
    rp_vals = np.linspace(1.1, cycle.rp_max, 500)

    results = compute_ideal_gas_exergy_sweep(cycle, rp_vals)
    print_exergy_summary(cycle)

    idx_peak = int(np.nanargmax(results["eta_exergy"]))
    print(
        "Sweep maximum exergetic efficiency: "
        f"{results['eta_exergy'][idx_peak] * 100.0:.2f}% "
        f"at rp = {results['rp_vals'][idx_peak]:.2f}"
    )

    fig_exergy = plot_exergy_vs_rp(results)
    fig_eta = plot_exergetic_efficiency_vs_rp(results)

    if save_output:
        save_figures(
            [
                (fig_exergy, "phase5_ideal_exergy_states_vs_rp.png"),
                (fig_eta, "phase5_ideal_exergetic_efficiency_vs_rp.png"),
            ],
            Path(output_dir),
        )
        print(f"Saved figures to '{Path(output_dir).resolve()}'")

    if show_plots:
        plt.show()
    else:
        plt.close("all")

    return results


if __name__ == "__main__":
    main()
