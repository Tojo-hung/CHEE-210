"""
reporting.py
Console-report helpers for the phase runner scripts.

These functions keep printing/report formatting separate from the solver logic
so each runner can decide what to show without duplicating text formatting.
"""

from __future__ import annotations

import re
from pathlib import Path

from phase_workflows import (
    Phase1Artifacts,
    Phase2Artifacts,
    Phase3Artifacts,
    Phase4Artifacts,
    Phase5Artifacts,
)
from output_manager import write_text_outputs


def sanitize_label_slug(text: str) -> str:
    """Convert a free-form branch label/key into a LaTeX-safe label slug."""
    slug = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    slug = re.sub(r"_+", "_", slug)
    return slug or "case"


def _sanitize_latex_label(text: str) -> str:
    """Backward-compatible alias for existing slug-generation call sites."""
    return sanitize_label_slug(text)


def _build_latex_table(
    caption: str,
    label: str,
    headers: list[str],
    rows: list[list[str]],
    column_spec: str | None = None,
) -> str:
    """Build a compact LaTeX table snippet from headers and row strings."""
    column_spec = column_spec or ("l" + "c" * (len(headers) - 1))
    lines = [
        r"\begin{table}[htbp]",
        r"    \centering",
        f"    \\caption{{{caption}}}",
        f"    \\label{{{label}}}",
        f"    \\begin{{tabular}}{{{column_spec}}}",
        r"        \hline",
        "        " + " & ".join(headers) + r" \\",
        r"        \hline",
    ]
    lines.extend("        " + " & ".join(row) + r" \\" for row in rows)
    lines.extend([
        r"        \hline",
        r"    \end{tabular}",
        r"\end{table}",
    ])
    return "\n".join(lines)


def _print_design(
    label: str,
    design_point: dict,
    T1: float,
    T3: float,
    P1: float,
) -> None:
    """Print a formatted design-point summary."""
    print(f"\n  {label}")
    print(f"  {'-' * 50}")
    print(f"    r_p    = {design_point['rp']:.2f}")
    print(f"    T1 = {T1:.2f} K   T2 = {design_point['T2']:.2f} K")
    print(f"    T3 = {T3:.2f} K   T4 = {design_point['T4']:.2f} K")
    print(
        f"    P1=P4 = {P1 / 1000:.3f} MPa   "
        f"P2=P3 = {P1 * design_point['rp'] / 1000:.3f} MPa"
    )
    print(
        f"    w_c   = {design_point['w_c']:.2f} kJ/kg   "
        f"w_t  = {design_point['w_t']:.2f} kJ/kg"
    )
    print(
        f"    w_net = {design_point['w_net']:.2f} kJ/kg   "
        f"q_in = {design_point['q_in']:.2f} kJ/kg"
    )
    print(f"    m_dot = {design_point['m_dot']:.3f} kg/s")
    print(
        f"    W_dot = {design_point['W_dot_net'] / 1000:.3f} MW   "
        f"eta_th = {design_point['eta_th'] * 100:.2f}%   "
        f"BWR = {design_point['bwr'] * 100:.2f}%"
    )


def print_phase1_report(phase1: Phase1Artifacts) -> None:
    """Print the standard Phase 1 summary and verification points."""
    print("\n" + "=" * 57)
    print("  CHEE 210 - Phase 1: Ideal Brayton Cycle on Ideal Gas")
    print("=" * 57)
    _print_design(
        "Criterion 1 - Max total net power",
        phase1.design_max_power,
        phase1.T1,
        phase1.T3,
        phase1.P1,
    )
    _print_design(
        "Criterion 2 - Max specific net work",
        phase1.design_max_specific_work,
        phase1.T1,
        phase1.T3,
        phase1.P1,
    )
    print(
        f"\n  Delta r_p between criteria: "
        f"{abs(phase1.design_max_power['rp'] - phase1.design_max_specific_work['rp']):.2f}"
    )
    print("=" * 57)

    print("\nPhase 1 verification points:")
    for rp_chk, point in phase1.verification_points.items():
        print(f"\n{'=' * 52}")
        print(f"  VERIFICATION: rp = {rp_chk:.2f}")
        print(f"{'=' * 52}")
        print(f"  T1 = {phase1.T1:.2f} K   T2 = {point['T2']:.2f} K")
        print(f"  T3 = {phase1.T3:.2f} K   T4 = {point['T4']:.2f} K")
        print(
            f"  P1=P4 = {phase1.P1 / 1000:.3f} MPa   "
            f"P2=P3 = {phase1.P1 * rp_chk / 1000:.3f} MPa"
        )
        print(f"  w_c   = {point['w_c']:.2f} kJ/kg")
        print(f"  w_t   = {point['w_t']:.2f} kJ/kg")
        print(f"  w_net = {point['w_net']:.2f} kJ/kg")
        print(f"  q_in  = {point['q_in']:.2f} kJ/kg")
        print(f"  eta_th = {point['eta_th'] * 100:.2f} %")
        print(f"  BWR   = {point['bwr'] * 100:.2f} %")
        print(f"  m_dot = {point['m_dot']:.3f} kg/s")
        print(f"  W_dot = {point['W_dot_net'] / 1000:.3f} MW")
    print("=" * 52)


def print_phase2_report(phase2: Phase2Artifacts) -> None:
    """Print the standard Phase 2 summary."""
    eta_c_sample, eta_t_sample = phase2.sample_efficiencies
    optimization = phase2.optimization_search

    print("\n" + "=" * 57)
    print("  CHEE 210 - Phase 2: Non-ideal Brayton Cycle on Ideal Gas")
    print("=" * 57)
    print(f"  Pressure ratio used for contour plots: rp = {phase2.selected_rp:.2f}")
    print(f"  rp source: {phase2.rp_source}")

    if phase2.phase1_design is not None:
        print(
            f"  Phase 1 design point carried into Phase 2: "
            f"rp = {phase2.phase1_design['rp']:.2f}"
        )

    print(
        f"\n  Sample calculation at eta_C = {eta_c_sample:.2f}, "
        f"eta_T = {eta_t_sample:.2f}"
    )
    print(f"  {'-' * 50}")
    print(f"    T2s     = {phase2.sample['T2s']:.2f} K")
    print(f"    T4s     = {phase2.sample['T4s']:.2f} K")
    print(f"    w_c,s   = {phase2.sample['w_c_s']:.2f} kJ/kg")
    print(f"    w_t,s   = {phase2.sample['w_t_s']:.2f} kJ/kg")
    print(f"    w_c,act = {phase2.sample['w_c_act']:.2f} kJ/kg")
    print(f"    w_t,act = {phase2.sample['w_t_act']:.2f} kJ/kg")
    print(f"    T2_act  = {phase2.sample['T2_act']:.2f} K")
    print(f"    q_in    = {phase2.sample['q_in']:.2f} kJ/kg")
    print(f"    w_net   = {phase2.sample['w_net']:.2f} kJ/kg")
    print(f"    m_dot   = {phase2.sample['m_dot']:.3f} kg/s")
    print(f"    W_dot   = {phase2.sample['W_dot_net'] / 1000:.3f} MW")
    print(f"    eta_th  = {phase2.sample['eta_th'] * 100:.2f} %")
    print(f"    BWR     = {phase2.sample['bwr'] * 100:.2f} %")

    print("\n  Mean-power pressure-ratio optimization")
    print(f"  {'-' * 50}")
    print(f"    Optimal rp               = {optimization['rp_optimal']:.2f}")
    print(
        f"    Mean W_dot at optimum    = "
        f"{optimization['mean_Wdot'][optimization['idx_optimal']] / 1000:.3f} MW"
    )
    print("=" * 57)


def print_phase3_report(phase3: Phase3Artifacts) -> None:
    """Print the standard Phase 3 real-fluid comparison."""
    ideal = phase3.ideal_reference
    real = phase3.real_fluid

    print("\n" + "=" * 65)
    print("  CHEE 210 - Phase 3: Ideal Brayton Cycle on Real Fluid")
    print("=" * 65)
    print(f"  Pressure ratio used: rp = {phase3.selected_rp:.2f}")
    print(f"  rp source: {phase3.rp_source}")
    print(f"  Fluid model: {phase3.fluid}")
    print(f"  T1 = {phase3.T1:.2f} K,   P1 = {phase3.P1 / 1000:.3f} MPa")

    if phase3.rp_candidates:
        print("\n  Optimization-based pressure-ratio candidates:")
        for label, value in phase3.rp_candidates.items():
            print(f"    {label:<26} = {value:.2f}")

    print("\n  State conditions (real fluid):")
    print(
        f"    State 1:  T = {phase3.T1:.2f} K,   h = {real['h1']:.2f} kJ/kg,  "
        f"s = {real['s1']:.4f} kJ/(kg.K)"
    )
    print(f"    State 2s: T = {real['T2']:.2f} K,   h = {real['h2']:.2f} kJ/kg")
    print(
        f"    State 3:  T = {phase3.T3:.2f} K,   h = {real['h3']:.2f} kJ/kg,  "
        f"s = {real['s3']:.4f} kJ/(kg.K)"
    )
    print(f"    State 4s: T = {real['T4']:.2f} K,   h = {real['h4']:.2f} kJ/kg")

    print("\n  Specific work / heat:")
    print(f"    w_c   = {real['w_c']:.2f} kJ/kg")
    print(f"    w_t   = {real['w_t']:.2f} kJ/kg")
    print(f"    w_net = {real['w_net']:.2f} kJ/kg")
    print(f"    q_in  = {real['q_in']:.2f} kJ/kg")

    print("\n  Performance (real fluid, isentropic machines):")
    print(f"    m_dot   = {real['m_dot']:.4f} kg/s")
    print(f"    W_dot   = {real['W_dot_net'] / 1000:.4f} MW")
    print(f"    eta_th  = {real['eta_th'] * 100:.2f} %")
    print(f"    BWR     = {real['bwr'] * 100:.2f} %")

    print(
        f"\n  {'Quantity':<22} {'Phase 1 (ideal gas)':>22} "
        f"{'Phase 3 (real fluid)':>22} {'Delta':>10}"
    )
    print(f"  {'-' * 80}")
    comparisons = [
        ("T2 [K]", ideal["T2"], real["T2"]),
        ("T4 [K]", ideal["T4"], real["T4"]),
        ("w_c [kJ/kg]", ideal["w_c"], real["w_c"]),
        ("w_t [kJ/kg]", ideal["w_t"], real["w_t"]),
        ("w_net [kJ/kg]", ideal["w_net"], real["w_net"]),
        ("q_in [kJ/kg]", ideal["q_in"], real["q_in"]),
        ("eta_th [%]", ideal["eta_th"] * 100, real["eta_th"] * 100),
        ("BWR [%]", ideal["bwr"] * 100, real["bwr"] * 100),
        ("m_dot [kg/s]", ideal["m_dot"], real["m_dot"]),
        ("W_dot [MW]", ideal["W_dot_net"] / 1000, real["W_dot_net"] / 1000),
    ]
    for label, ideal_value, real_value in comparisons:
        deviation = (real_value - ideal_value) / abs(ideal_value) * 100
        print(
            f"  {label:<22} {ideal_value:>22.3f} "
            f"{real_value:>22.3f} {deviation:>+9.2f}%"
        )
    print("=" * 65)


def print_phase4_report(phase4: Phase4Artifacts) -> None:
    """Print the standard Phase 4 regenerator-study summary."""
    print("\n" + "=" * 68)
    print("  CHEE 210 - Phase 4: Regenerator Study on Real-Fluid CO2")
    print("=" * 68)
    print(f"  Fluid model: {phase4.fluid}")
    print(
        f"  Fixed conditions: T1 = {phase4.T1:.2f} K, "
        f"T3 = {phase4.T3:.2f} K, P1 = {phase4.P1 / 1000:.3f} MPa, "
        f"Q_dot_in = {phase4.Q_dot_in / 1000:.3f} MW"
    )

    for case in phase4.cases:
        sweep = case.sweep
        feasible_idx = [
            idx for idx, feasible in enumerate(sweep["feasible"])
            if feasible
        ]

        print(f"\n  {case.rp_case.label}")
        print(f"  {'-' * 56}")
        print(
            f"    source = {case.rp_case.source_phase}   "
            f"r_p = {case.rp_case.rp:.2f}"
        )
        print(
            f"    Phase 3 baseline: W_dot = {case.baseline_phase3['W_dot_net'] / 1000:.3f} MW   "
            f"eta_th = {case.baseline_phase3['eta_th'] * 100:.2f}%"
        )

        if not feasible_idx:
            print("    No feasible regenerator cases for the current delta-T sweep.")
            continue

        best_idx = max(feasible_idx, key=lambda idx: sweep["eta_th"][idx])
        best = sweep["cases"][best_idx]
        delta_eta_pct_pts = (best["eta_th"] - case.baseline_phase3["eta_th"]) * 100.0

        print(
            f"    Best feasible delta-T approach = {best['delta_T_approach']:.1f} K"
        )
        print(
            f"    T2 = {best['T2']:.2f} K   T3 = {best['T3']:.2f} K   "
            f"T5 = {best['T5']:.2f} K   T6 = {best['T6']:.2f} K"
        )
        print(
            f"    P1=P5=P6 = {best['P1'] / 1000:.3f} MPa   "
            f"P2=P3=P4 = {best['P2'] / 1000:.3f} MPa"
        )
        print(
            f"    q_regen = {best['q_regen']:.2f} kJ/kg   "
            f"Q_dot_regen = {best['Q_dot_regen'] / 1000:.3f} MW"
        )
        print(
            f"    m_dot = {best['m_dot']:.3f} kg/s   "
            f"W_dot = {best['W_dot_net'] / 1000:.3f} MW   "
            f"eta_th = {best['eta_th'] * 100:.2f}%"
        )
        print(
            f"    Improvement vs Phase 3 baseline: "
            f"{delta_eta_pct_pts:+.2f} percentage points in eta_th"
        )
        first = sweep["cases"][feasible_idx[0]]
        last = sweep["cases"][feasible_idx[-1]]
        print(
            f"    Feasible delta-T range: "
            f"{first['delta_T_approach']:.1f} K to {last['delta_T_approach']:.1f} K "
            f"({len(feasible_idx)} points)"
        )

    print("=" * 68)


def print_phase5_report(phase5: Phase5Artifacts) -> None:
    """Print the standard Phase 5 exergy-analysis summary."""
    print("\n" + "=" * 72)
    print("  CHEE 210 - Phase 5: Exergy Analysis of Selected Regenerated Cases")
    print("=" * 72)
    print(
        f"  Dead state: T0 = {phase5.T_dead:.2f} K, "
        f"P0 = {phase5.P_dead / 1000:.3f} MPa"
    )
    print(f"  Selection policy: {phase5.selection_policy}")

    for case in phase5.case_artifacts:
        selected = case.selected_phase4_case
        component_delta_x = case.component_summary["delta_x"]

        print(f"\n  {case.rp_case.label}")
        print(f"  {'-' * 60}")
        print(
            f"    source = {case.rp_case.source_phase}   "
            f"r_p = {case.rp_case.rp:.2f}"
        )
        print(
            f"    Selected Phase 4 case: delta_T_approach = {selected['delta_T_approach']:.1f} K   "
            f"Q_dot_regen = {selected['Q_dot_regen'] / 1000:.3f} MW"
        )
        print(
            f"    eta_th = {selected['eta_th'] * 100:.2f}%   "
            f"W_dot = {selected['W_dot_net'] / 1000:.3f} MW   "
            f"m_dot = {selected['m_dot']:.3f} kg/s"
        )

        print("\n    Stream exergy table:")
        print(f"    {'State':<8} {'x [kJ/kg]':>14} {'X_dot [kW]':>16}")
        print(f"    {'-' * 40}")
        for idx in range(1, 7):
            print(
                f"    {idx:<8} "
                f"{case.stream_exergy[f'x{idx}']:>14.3f} "
                f"{case.stream_exergy_rate[f'X_dot{idx}']:>16.3f}"
            )

        print("\n    Component exergy changes:")
        for label, value in component_delta_x.items():
            print(f"    {label:<18} = {value:>9.3f} kJ/kg")

        print("\n    Notes:")
        for note in case.notes:
            print(f"    - {note}")

    print("=" * 72)


def build_phase1_latex_tables(phase1: Phase1Artifacts) -> dict[str, str]:
    """Build the standard Phase 1 LaTeX tables."""
    design_rows = [
        ["$r_p$", f"{phase1.design_max_power['rp']:.2f}", f"{phase1.design_max_specific_work['rp']:.2f}"],
        ["$T_1$ (K)", f"{phase1.T1:.2f}", f"{phase1.T1:.2f}"],
        ["$T_2$ (K)", f"{phase1.design_max_power['T2']:.2f}", f"{phase1.design_max_specific_work['T2']:.2f}"],
        ["$T_3$ (K)", f"{phase1.T3:.2f}", f"{phase1.T3:.2f}"],
        ["$T_4$ (K)", f"{phase1.design_max_power['T4']:.2f}", f"{phase1.design_max_specific_work['T4']:.2f}"],
        [
            "$P_2=P_3$ (MPa)",
            f"{phase1.P1 * phase1.design_max_power['rp'] / 1000.0:.3f}",
            f"{phase1.P1 * phase1.design_max_specific_work['rp'] / 1000.0:.3f}",
        ],
        ["$w_c$ (kJ/kg)", f"{phase1.design_max_power['w_c']:.2f}", f"{phase1.design_max_specific_work['w_c']:.2f}"],
        ["$w_t$ (kJ/kg)", f"{phase1.design_max_power['w_t']:.2f}", f"{phase1.design_max_specific_work['w_t']:.2f}"],
        ["$w_{net}$ (kJ/kg)", f"{phase1.design_max_power['w_net']:.2f}", f"{phase1.design_max_specific_work['w_net']:.2f}"],
        ["$q_{in}$ (kJ/kg)", f"{phase1.design_max_power['q_in']:.2f}", f"{phase1.design_max_specific_work['q_in']:.2f}"],
        ["$\\eta_{th}$ (\\%)", f"{phase1.design_max_power['eta_th'] * 100:.2f}", f"{phase1.design_max_specific_work['eta_th'] * 100:.2f}"],
        ["BWR (\\%)", f"{phase1.design_max_power['bwr'] * 100:.2f}", f"{phase1.design_max_specific_work['bwr'] * 100:.2f}"],
        ["$\\dot{m}$ (kg/s)", f"{phase1.design_max_power['m_dot']:.3f}", f"{phase1.design_max_specific_work['m_dot']:.3f}"],
        ["$\\dot{W}_{net}$ (MW)", f"{phase1.design_max_power['W_dot_net'] / 1000.0:.3f}", f"{phase1.design_max_specific_work['W_dot_net'] / 1000.0:.3f}"],
    ]
    verification_rows: list[list[str]] = []
    for rp_chk, point in phase1.verification_points.items():
        verification_rows.append([
            f"{rp_chk:.2f}",
            f"{point['T2']:.2f}",
            f"{point['T4']:.2f}",
            f"{phase1.P1 * rp_chk / 1000.0:.3f}",
            f"{point['w_c']:.2f}",
            f"{point['w_t']:.2f}",
            f"{point['w_net']:.2f}",
            f"{point['q_in']:.2f}",
            f"{point['eta_th'] * 100:.2f}",
            f"{point['bwr'] * 100:.2f}",
            f"{point['m_dot']:.3f}",
            f"{point['W_dot_net'] / 1000.0:.3f}",
        ])

    return {
        "design_point_comparison.tex": _build_latex_table(
            caption="Phase 1 comparison of the maximum net-power and maximum specific-work design points.",
            label="tab:phase1_design_point_comparison",
            headers=["Quantity", "Max net power", "Max specific work"],
            rows=design_rows,
        ),
        "verification_points.tex": _build_latex_table(
            caption="Phase 1 verification-point results across selected pressure ratios.",
            label="tab:phase1_verification_points",
            headers=[
                "$r_p$",
                "$T_2$ (K)",
                "$T_4$ (K)",
                "$P_2=P_3$ (MPa)",
                "$w_c$",
                "$w_t$",
                "$w_{net}$",
                "$q_{in}$",
                "$\\eta_{th}$ (\\%)",
                "BWR (\\%)",
                "$\\dot{m}$",
                "$\\dot{W}_{net}$ (MW)",
            ],
            rows=verification_rows,
            column_spec="c" * 12,
        ),
    }


def write_phase1_latex_tables(phase1: Phase1Artifacts, output_dir: str | Path) -> None:
    """Write the standard Phase 1 LaTeX tables to disk."""
    write_text_outputs(build_phase1_latex_tables(phase1), output_dir)


def build_phase2_latex_tables(phase2: Phase2Artifacts) -> dict[str, str]:
    """Build the standard Phase 2 LaTeX tables."""
    sample = phase2.sample
    optimization = phase2.optimization_search
    sample_rows = [
        ["$r_p$", f"{sample['rp']:.2f}"],
        ["$\\eta_C$", f"{phase2.sample_efficiencies[0]:.2f}"],
        ["$\\eta_T$", f"{phase2.sample_efficiencies[1]:.2f}"],
        ["$T_{2s}$ (K)", f"{sample['T2s']:.2f}"],
        ["$T_{4s}$ (K)", f"{sample['T4s']:.2f}"],
        ["$w_{c,s}$ (kJ/kg)", f"{sample['w_c_s']:.2f}"],
        ["$w_{t,s}$ (kJ/kg)", f"{sample['w_t_s']:.2f}"],
        ["$w_{c,act}$ (kJ/kg)", f"{sample['w_c_act']:.2f}"],
        ["$w_{t,act}$ (kJ/kg)", f"{sample['w_t_act']:.2f}"],
        ["$T_{2,act}$ (K)", f"{sample['T2_act']:.2f}"],
        ["$q_{in}$ (kJ/kg)", f"{sample['q_in']:.2f}"],
        ["$w_{net}$ (kJ/kg)", f"{sample['w_net']:.2f}"],
        ["$\\dot{m}$ (kg/s)", f"{sample['m_dot']:.3f}"],
        ["$\\dot{W}_{net}$ (MW)", f"{sample['W_dot_net'] / 1000.0:.3f}"],
        ["$\\eta_{th}$ (\\%)", f"{sample['eta_th'] * 100:.2f}"],
        ["BWR (\\%)", f"{sample['bwr'] * 100:.2f}"],
    ]
    optimization_rows = [
        ["Contour-plot $r_p$", f"{phase2.selected_rp:.2f}"],
        ["Contour-plot source", phase2.rp_source],
        ["Optimal $r_p$", f"{optimization['rp_optimal']:.2f}"],
        ["Mean $\\dot{W}_{net}$ at optimum (MW)", f"{optimization['mean_Wdot'][optimization['idx_optimal']] / 1000.0:.3f}"],
    ]
    return {
        "sample_calculation.tex": _build_latex_table(
            caption="Phase 2 sample non-ideal Brayton-cycle calculation at the selected efficiency pair.",
            label="tab:phase2_sample_calculation",
            headers=["Quantity", "Value"],
            rows=sample_rows,
        ),
        "mean_power_optimization.tex": _build_latex_table(
            caption="Phase 2 pressure-ratio optimization based on mean net power over the efficiency grid.",
            label="tab:phase2_mean_power_optimization",
            headers=["Quantity", "Value"],
            rows=optimization_rows,
        ),
    }


def write_phase2_latex_tables(phase2: Phase2Artifacts, output_dir: str | Path) -> None:
    """Write the standard Phase 2 LaTeX tables to disk."""
    write_text_outputs(build_phase2_latex_tables(phase2), output_dir)


def build_phase3_latex_tables(phase3_cases: list[Phase3Artifacts]) -> dict[str, str]:
    """Build the standard Phase 3 LaTeX tables for all optimization cases."""
    if not phase3_cases:
        return {}

    candidate_rows: list[list[str]] = []
    seen_candidates: set[str] = set()
    for case in phase3_cases:
        for key, value in case.rp_candidates.items():
            if key in seen_candidates:
                continue
            seen_candidates.add(key)
            candidate_rows.append([key, f"{value:.2f}"])

    tables = {
        "pressure_ratio_candidates.tex": _build_latex_table(
            caption="Optimization-derived pressure-ratio candidates considered in Phase 3.",
            label="tab:phase3_pressure_ratio_candidates",
            headers=["Candidate", "$r_p$"],
            rows=candidate_rows,
        )
    }

    for case in phase3_cases:
        slug = sanitize_label_slug(case.rp_case.key)
        real = case.real_fluid
        ideal = case.ideal_reference
        state_rows = [
            ["1", f"{case.T1:.2f}", f"{real['h1']:.2f}", f"{real['s1']:.4f}"],
            ["2s", f"{real['T2']:.2f}", f"{real['h2']:.2f}", "--"],
            ["3", f"{case.T3:.2f}", f"{real['h3']:.2f}", f"{real['s3']:.4f}"],
            ["4s", f"{real['T4']:.2f}", f"{real['h4']:.2f}", "--"],
        ]
        comparison_rows = []
        comparisons = [
            ("$T_2$ (K)", ideal["T2"], real["T2"]),
            ("$T_4$ (K)", ideal["T4"], real["T4"]),
            ("$w_c$ (kJ/kg)", ideal["w_c"], real["w_c"]),
            ("$w_t$ (kJ/kg)", ideal["w_t"], real["w_t"]),
            ("$w_{net}$ (kJ/kg)", ideal["w_net"], real["w_net"]),
            ("$q_{in}$ (kJ/kg)", ideal["q_in"], real["q_in"]),
            ("$\\eta_{th}$ (\\%)", ideal["eta_th"] * 100.0, real["eta_th"] * 100.0),
            ("BWR (\\%)", ideal["bwr"] * 100.0, real["bwr"] * 100.0),
            ("$\\dot{m}$ (kg/s)", ideal["m_dot"], real["m_dot"]),
            ("$\\dot{W}_{net}$ (MW)", ideal["W_dot_net"] / 1000.0, real["W_dot_net"] / 1000.0),
        ]
        for label, ideal_value, real_value in comparisons:
            deviation = (real_value - ideal_value) / abs(ideal_value) * 100.0
            comparison_rows.append([
                label,
                f"{ideal_value:.3f}",
                f"{real_value:.3f}",
                f"{deviation:+.2f}",
            ])

        tables[f"state_conditions_{slug}.tex"] = _build_latex_table(
            caption=(
                "Phase 3 real-fluid state conditions for the "
                f"{case.rp_case.label} branch ($r_p = {case.selected_rp:.2f}$)."
            ),
            label=f"tab:phase3_state_conditions_{slug}",
            headers=["State", "$T$ (K)", "$h$ (kJ/kg)", "$s$ (kJ/kg$\\cdot$K)"],
            rows=state_rows,
        )
        tables[f"comparison_{slug}.tex"] = _build_latex_table(
            caption=(
                "Phase 3 comparison between the ideal-gas and real-fluid cycle models "
                f"for the {case.rp_case.label} branch."
            ),
            label=f"tab:phase3_comparison_{slug}",
            headers=["Quantity", "Ideal gas", "Real fluid", "$\\Delta$ (\\%)"],
            rows=comparison_rows,
        )

    return tables


def write_phase3_latex_tables(phase3_cases: list[Phase3Artifacts], output_dir: str | Path) -> None:
    """Write the standard Phase 3 LaTeX tables to disk."""
    write_text_outputs(build_phase3_latex_tables(phase3_cases), output_dir)


def build_phase4_latex_tables(phase4: Phase4Artifacts) -> dict[str, str]:
    """Build the standard Phase 4 LaTeX tables."""
    tables: dict[str, str] = {}
    for case in phase4.cases:
        slug = sanitize_label_slug(case.rp_case.key)
        sweep = case.sweep
        feasible_idx = [idx for idx, feasible in enumerate(sweep["feasible"]) if feasible]

        if feasible_idx:
            best_idx = max(feasible_idx, key=lambda idx: sweep["eta_th"][idx])
            best = sweep["cases"][best_idx]
            first = sweep["cases"][feasible_idx[0]]
            last = sweep["cases"][feasible_idx[-1]]
            rows = [
                ["Source phase", case.rp_case.source_phase],
                ["$r_p$", f"{case.rp_case.rp:.2f}"],
                ["Baseline $\\dot{W}_{net}$ (MW)", f"{case.baseline_phase3['W_dot_net'] / 1000.0:.3f}"],
                ["Baseline $\\eta_{th}$ (\\%)", f"{case.baseline_phase3['eta_th'] * 100.0:.2f}"],
                ["Best $\\Delta T_{approach}$ (K)", f"{best['delta_T_approach']:.1f}"],
                ["$T_2$ (K)", f"{best['T2']:.2f}"],
                ["$T_3$ (K)", f"{best['T3']:.2f}"],
                ["$T_5$ (K)", f"{best['T5']:.2f}"],
                ["$T_6$ (K)", f"{best['T6']:.2f}"],
                ["$P_1=P_5=P_6$ (MPa)", f"{best['P1'] / 1000.0:.3f}"],
                ["$P_2=P_3=P_4$ (MPa)", f"{best['P2'] / 1000.0:.3f}"],
                ["$q_{regen}$ (kJ/kg)", f"{best['q_regen']:.2f}"],
                ["$\\dot{Q}_{regen}$ (MW)", f"{best['Q_dot_regen'] / 1000.0:.3f}"],
                ["$\\dot{m}$ (kg/s)", f"{best['m_dot']:.3f}"],
                ["$\\dot{W}_{net}$ (MW)", f"{best['W_dot_net'] / 1000.0:.3f}"],
                ["$\\eta_{th}$ (\\%)", f"{best['eta_th'] * 100.0:.2f}"],
                [
                    "$\\Delta\\eta_{th}$ vs baseline (percentage points)",
                    f"{(best['eta_th'] - case.baseline_phase3['eta_th']) * 100.0:+.2f}",
                ],
                [
                    "Feasible $\\Delta T$ range (K)",
                    f"{first['delta_T_approach']:.1f} to {last['delta_T_approach']:.1f}",
                ],
                ["Feasible points", f"{len(feasible_idx)}"],
            ]
        else:
            rows = [
                ["Source phase", case.rp_case.source_phase],
                ["$r_p$", f"{case.rp_case.rp:.2f}"],
                ["Baseline $\\dot{W}_{net}$ (MW)", f"{case.baseline_phase3['W_dot_net'] / 1000.0:.3f}"],
                ["Baseline $\\eta_{th}$ (\\%)", f"{case.baseline_phase3['eta_th'] * 100.0:.2f}"],
                ["Sweep status", "No feasible regenerator cases"],
                [
                    "Requested $\\Delta T$ range (K)",
                    f"{case.delta_T_approach_vals[0]:.1f} to {case.delta_T_approach_vals[-1]:.1f}",
                ],
                ["Requested points", f"{len(case.delta_T_approach_vals)}"],
            ]

        tables[f"summary_{slug}.tex"] = _build_latex_table(
            caption=(
                "Phase 4 regenerated-cycle summary for the "
                f"{case.rp_case.label} branch ($r_p = {case.rp_case.rp:.2f}$)."
            ),
            label=f"tab:phase4_summary_{slug}",
            headers=["Quantity", "Value"],
            rows=rows,
        )
    return tables


def write_phase4_latex_tables(phase4: Phase4Artifacts, output_dir: str | Path) -> None:
    """Write the standard Phase 4 LaTeX tables to disk."""
    write_text_outputs(build_phase4_latex_tables(phase4), output_dir)


def build_phase5_selected_case_latex_table(phase5_case_artifact) -> str:
    """Build one LaTeX summary table for a selected Phase 5 case."""
    selected = phase5_case_artifact.selected_phase4_case
    branch_label = phase5_case_artifact.rp_case.label
    branch_slug = sanitize_label_slug(phase5_case_artifact.rp_case.key)
    rows = [
        ["Source phase", phase5_case_artifact.rp_case.source_phase],
        ["$r_p$", f"{phase5_case_artifact.rp_case.rp:.2f}"],
        ["Selection policy", phase5_case_artifact.selection_policy],
        ["$\\Delta T_{approach}$ (K)", f"{selected['delta_T_approach']:.1f}"],
        ["$\\dot{Q}_{regen}$ (MW)", f"{selected['Q_dot_regen'] / 1000.0:.3f}"],
        ["$\\eta_{th}$ (\\%)", f"{selected['eta_th'] * 100.0:.2f}"],
        ["$\\dot{W}_{net}$ (MW)", f"{selected['W_dot_net'] / 1000.0:.3f}"],
        ["$\\dot{m}$ (kg/s)", f"{selected['m_dot']:.3f}"],
        ["$T_0$ (K)", f"{phase5_case_artifact.dead_state['T0']:.2f}"],
        ["$P_0$ (kPa)", f"{phase5_case_artifact.dead_state['P0']:.1f}"],
    ]
    return _build_latex_table(
        caption=(
            "Phase 5 selected regenerated-case summary for the "
            f"{branch_label} branch ($r_p = {phase5_case_artifact.rp_case.rp:.2f}$)."
        ),
        label=f"tab:phase5_selected_case_summary_{branch_slug}",
        headers=["Quantity", "Value"],
        rows=rows,
    )


def build_phase5_latex_table(phase5_case_artifact) -> str:
    """Build one LaTeX stream-exergy table for a selected Phase 5 case."""
    selected = phase5_case_artifact.selected_phase4_case
    branch_label = phase5_case_artifact.rp_case.label
    branch_slug = sanitize_label_slug(phase5_case_artifact.rp_case.key)

    rows = []
    for idx in range(1, 7):
        rows.append(
            "        "
            f"{idx} & "
            f"{selected[f'T{idx}']:.2f} & "
            f"{selected[f'P{idx}']:.1f} & "
            f"{selected[f'h{idx}']:.2f} & "
            f"{selected[f's{idx}']:.3f} & "
            f"{phase5_case_artifact.stream_exergy[f'x{idx}']:.2f} & "
            f"{phase5_case_artifact.stream_exergy_rate[f'X_dot{idx}']:.2f} \\\\"
        )

    table_lines = [
        r"\begin{table}[htbp]",
        r"    \centering",
        (
            r"    \caption{Exergy table for the regenerated Brayton cycle "
            f"selected from the {branch_label} branch "
            f"($r_p = {phase5_case_artifact.rp_case.rp:.2f}$)."
            r"}"
        ),
        f"    \\label{{tab:phase5_exergy_{branch_slug}}}",
        r"    \begin{tabular}{c c c c c c c}",
        r"        \hline",
        (
            r"        State & $T$ (K) & $P$ (kPa) & $h$ (kJ/kg) & "
            r"$s$ (kJ/kg$\cdot$K) & $x$ (kJ/kg) & $\dot{X}$ (kW) \\"
        ),
        r"        \hline",
        *rows,
        r"        \hline",
        r"    \end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(table_lines)


def build_phase5_component_latex_table(phase5_case_artifact) -> str:
    """Build one LaTeX component-exergy-change table for a selected Phase 5 case."""
    branch_label = phase5_case_artifact.rp_case.label
    branch_slug = sanitize_label_slug(phase5_case_artifact.rp_case.key)
    delta_x = phase5_case_artifact.component_summary["delta_x"]
    delta_xdot = phase5_case_artifact.component_summary["delta_X_dot"]
    component_labels = [
        ("compressor", "Compressor"),
        ("heater", "Heater"),
        ("turbine", "Turbine"),
        ("regenerator_cold", "Regenerator cold side"),
        ("regenerator_hot", "Regenerator hot side"),
        ("cooler", "Cooler"),
    ]

    rows = [
        "        "
        f"{label} & {delta_x[key]:.2f} & {delta_xdot[key]:.2f} \\\\"
        for key, label in component_labels
    ]

    table_lines = [
        r"\begin{table}[htbp]",
        r"    \centering",
        (
            r"    \caption{Component exergy changes for the regenerated Brayton cycle "
            f"selected from the {branch_label} branch "
            f"($r_p = {phase5_case_artifact.rp_case.rp:.2f}$)."
            r"}"
        ),
        f"    \\label{{tab:phase5_component_exergy_{branch_slug}}}",
        r"    \begin{tabular}{c c c}",
        r"        \hline",
        r"        Component & $\Delta x$ (kJ/kg) & $\Delta \dot{X}$ (kW) \\",
        r"        \hline",
        *rows,
        r"        \hline",
        r"    \end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(table_lines)


def build_phase5_latex_tables(phase5_artifacts: Phase5Artifacts) -> dict[str, str]:
    """Build one LaTeX stream-exergy table per Phase 5 optimization branch."""
    return {
        case.rp_case.key: build_phase5_latex_table(case)
        for case in phase5_artifacts.case_artifacts
    }


def build_phase5_component_latex_tables(phase5_artifacts: Phase5Artifacts) -> dict[str, str]:
    """Build one LaTeX component-exergy table per Phase 5 optimization branch."""
    return {
        case.rp_case.key: build_phase5_component_latex_table(case)
        for case in phase5_artifacts.case_artifacts
    }


def build_phase5_latex_bundle(
    phase5_artifacts: Phase5Artifacts,
    include_component_tables: bool = False,
) -> str:
    """Build all Phase 5 LaTeX tables as one combined string."""
    chunks: list[str] = []
    for case in phase5_artifacts.case_artifacts:
        chunks.append(build_phase5_latex_table(case))
        if include_component_tables:
            chunks.append(build_phase5_component_latex_table(case))
    return "\n\n".join(chunks)


def build_phase5_output_latex_tables(phase5_artifacts: Phase5Artifacts) -> dict[str, str]:
    """Build the Phase 5 LaTeX files used by run_all output exports."""
    files: dict[str, str] = {}
    for case in phase5_artifacts.case_artifacts:
        branch_slug = sanitize_label_slug(case.rp_case.key)
        files[f"selected_case_summary_{branch_slug}.tex"] = build_phase5_selected_case_latex_table(case)
        files[f"stream_exergy_{branch_slug}.tex"] = build_phase5_latex_table(case)
        files[f"component_exergy_{branch_slug}.tex"] = build_phase5_component_latex_table(case)
    return files


def write_phase5_output_latex_tables(
    phase5_artifacts: Phase5Artifacts,
    output_dir: str | Path,
) -> None:
    """Write the full Phase 5 run_all LaTeX table set to disk."""
    write_text_outputs(build_phase5_output_latex_tables(phase5_artifacts), output_dir)


def write_phase5_latex_tables(
    phase5_artifacts: Phase5Artifacts,
    output_dir: str = "latex_tables",
    include_component_tables: bool = False,
) -> None:
    """Write Phase 5 LaTeX tables to .tex files on disk."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for case in phase5_artifacts.case_artifacts:
        branch_slug = sanitize_label_slug(case.rp_case.key)
        stream_path = output_path / f"phase5_exergy_{branch_slug}.tex"
        stream_path.write_text(build_phase5_latex_table(case), encoding="utf-8")

        if include_component_tables:
            component_path = output_path / f"phase5_component_exergy_{branch_slug}.tex"
            component_path.write_text(
                build_phase5_component_latex_table(case),
                encoding="utf-8",
            )
