"""
run_phase2.py
Standalone entry point for Phase 2.
"""

from __future__ import annotations

import matplotlib.pyplot as plt

from phase_workflows import (
    apply_plot_style,
    build_default_cycle,
    create_phase2_figures,
    solve_phase1_workflow,
    solve_phase2_workflow,
)
from reporting import print_phase2_report


def main(show_plots: bool = True):
    """Run only Phase 2, using the preferred Phase 1 design pressure ratio."""
    apply_plot_style()
    cycle = build_default_cycle()
    phase1 = solve_phase1_workflow(cycle)
    phase2 = solve_phase2_workflow(cycle, phase1=phase1)
    
    print_phase2_report(phase2)
    create_phase2_figures(phase2)

    if show_plots:
        plt.show()
    else:
        plt.close("all")

    return phase2


if __name__ == "__main__":
    main()
