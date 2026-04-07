"""
run_phase5.py
Standalone entry point for Phase 5.
"""

from __future__ import annotations

import matplotlib.pyplot as plt

from phase_workflows import (
    apply_plot_style,
    build_default_cycle,
    create_phase5_figures,
    solve_phase1_workflow,
    solve_phase2_workflow,
    solve_phase4_workflow,
    solve_phase5_workflow,
)
from reporting import print_phase5_report, write_phase5_latex_tables


def main(show_plots: bool = True):
    """Run only Phase 5 using selected cases from the Phase 4 workflow."""
    apply_plot_style()
    cycle = build_default_cycle()
    phase1 = solve_phase1_workflow(cycle)
    phase2 = solve_phase2_workflow(cycle, phase1=phase1)

    try:
        phase4 = solve_phase4_workflow(
            cycle,
            phase1=phase1,
            phase2=phase2,
        )
        phase5 = solve_phase5_workflow(
            cycle,
            phase4=phase4,
            selection_policy="manual_index",
            manual_index=300, 
        )
    except ImportError as exc:
        print(f"\n[Phase 5] Skipped - {exc}")
        plt.close("all")
        return None

    print_phase5_report(phase5)
    write_phase5_latex_tables(phase5)
    create_phase5_figures(phase5)
    print("\n[Phase 5] Wrote LaTeX tables to 'latex_tables/'")

    if show_plots:
        plt.show()
    else:
        plt.close("all")

    return phase5


if __name__ == "__main__":
    main()
