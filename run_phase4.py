"""
run_phase4.py
Standalone entry point for Phase 4.
"""

from __future__ import annotations

import matplotlib.pyplot as plt

from phase_workflows import (
    apply_plot_style,
    build_default_cycle,
    create_phase4_figures,
    solve_phase1_workflow,
    solve_phase2_workflow,
    solve_phase4_workflow,
)
from reporting import print_phase4_report


def main(show_plots: bool = True):
    """Run only Phase 4 using the current optimization-derived pressure ratios."""
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
    except ImportError as exc:
        print(f"\n[Phase 4] Skipped - {exc}")
        plt.close("all")
        return None

    print_phase4_report(phase4)
    create_phase4_figures(phase4)

    if show_plots:
        plt.show()
    else:
        plt.close("all")

    return phase4


if __name__ == "__main__":
    main()
