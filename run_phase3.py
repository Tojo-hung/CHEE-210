"""
run_phase3.py
Standalone entry point for Phase 3.
"""

from __future__ import annotations

import matplotlib.pyplot as plt

from phase_workflows import (
    apply_plot_style,
    build_default_cycle,
    create_phase3_optimization_figures,
    solve_phase1_workflow,
    solve_phase2_workflow,
    solve_phase3_optimization_cases,
)
from reporting import print_phase3_report


def main(show_plots: bool = True):
    """Run Phase 3 for the current optimization-derived pressure ratios."""
    apply_plot_style()
    cycle = build_default_cycle()
    phase1 = solve_phase1_workflow(cycle)
    phase2 = solve_phase2_workflow(cycle, phase1=phase1)
    try:
        phase3_cases = solve_phase3_optimization_cases(
            cycle,
            phase1=phase1,
            phase2=phase2,
        )
    except ImportError as exc:
        print(f"\n[Phase 3] Skipped - {exc}")
        plt.close("all")
        return None

    for phase3 in phase3_cases:
        print_phase3_report(phase3)

    create_phase3_optimization_figures(phase3_cases)

    if show_plots:
        plt.show()
    else:
        plt.close("all")

    return phase3_cases


if __name__ == "__main__":
    main()
