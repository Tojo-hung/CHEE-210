"""
export_optimization_table.py
Generate a LaTeX table comparing the ideal-cycle results at the current
Phase 1 and Phase 2 optimization pressure ratios.
"""

from __future__ import annotations

from pathlib import Path

from phase_workflows import build_default_cycle, solve_phase1_workflow, solve_phase2_workflow


OUTPUT_PATH = Path("phase1_phase2_optimization_table.tex")


def _format_table(phase1_point: dict, phase2_point: dict, p1_mpa: float, t1: float, t3: float) -> str:
    """Return a LaTeX table snippet for the two optimization pressure ratios."""
    rp1 = phase1_point["rp"]
    rp2 = phase2_point["rp"]

    return f"""\\begin{{table}}[h!]
\\centering
\\begin{{tabular}}{{lcc}}
\\hline
\\textbf{{Quantity}} & \\textbf{{$r_p = {rp1:.1f}$}} & \\textbf{{$r_p = {rp2:.1f}$}} \\\\
\\hline
$T_1$ (K)              & {t1:.2f}   & {t1:.2f}   \\\\
$T_2$ (K)              & {phase1_point['T2']:.2f}   & {phase2_point['T2']:.2f}   \\\\
$T_3$ (K)              & {t3:.2f}  & {t3:.2f}  \\\\
$T_4$ (K)              & {phase1_point['T4']:.2f}   & {phase2_point['T4']:.2f}   \\\\
$P_1 = P_4$ (MPa)      & {p1_mpa:.3f}    & {p1_mpa:.3f}    \\\\
$P_2 = P_3$ (MPa)      & {p1_mpa * rp1:.3f}   & {p1_mpa * rp2:.3f}    \\\\
$w_c$ (kJ/kg)          & {phase1_point['w_c']:.2f}  & {phase2_point['w_c']:.2f}  \\\\
$w_t$ (kJ/kg)          & {phase1_point['w_t']:.2f}  & {phase2_point['w_t']:.2f}  \\\\
$w_{{net}}$ (kJ/kg)      & {phase1_point['w_net']:.2f}  & {phase2_point['w_net']:.2f}  \\\\
$q_{{in}}$ (kJ/kg)       & {phase1_point['q_in']:.2f}  & {phase2_point['q_in']:.2f}  \\\\
$\\eta_{{th}}$ (\\%)       & {phase1_point['eta_th'] * 100:.2f}   & {phase2_point['eta_th'] * 100:.2f}   \\\\
BWR (\\%)               & {phase1_point['bwr'] * 100:.2f}   & {phase2_point['bwr'] * 100:.2f}   \\\\
$\\dot{{m}}$ (kg/s)       & {phase1_point['m_dot']:.3f}   & {phase2_point['m_dot']:.3f}   \\\\
$\\dot{{W}}_{{net}}$ (MW)   & {phase1_point['W_dot_net'] / 1000:.3f}   & {phase2_point['W_dot_net'] / 1000:.3f}   \\\\
\\hline
\\end{{tabular}}
\\caption{{Cycle state conditions and performance metrics for the two
         optimization-derived pressure ratios under the ideal Brayton cycle model.}}
\\label{{tab:phase1_phase2_optimization_results}}
\\end{{table}}
"""


def main() -> Path:
    """Generate and save the optimization comparison table."""
    cycle = build_default_cycle()
    phase1 = solve_phase1_workflow(cycle)
    phase2 = solve_phase2_workflow(cycle, phase1=phase1)

    point_phase1_opt = cycle.solve_point(phase1.optimization_model["rp_optimal"])
    point_phase2_opt = cycle.solve_point(phase2.optimization_search["rp_optimal"])

    table = _format_table(
        point_phase1_opt,
        point_phase2_opt,
        p1_mpa=cycle.P1 / 1000.0,
        t1=cycle.T1,
        t3=cycle.T3,
    )
    OUTPUT_PATH.write_text(table, encoding="utf-8")
    print(f"Wrote LaTeX table to {OUTPUT_PATH}")
    return OUTPUT_PATH


if __name__ == "__main__":
    main()
