"""
run_phase1.py
Standalone entry point for Phase 1.
"""

from __future__ import annotations

import matplotlib.pyplot as plt

from phase_workflows import (
    apply_plot_style,
    build_default_cycle,
    create_phase1_figures,
    solve_phase1_workflow,
)
from reporting import print_phase1_report


def main(show_plots: bool = True):
    """Run only Phase 1 using the shared solver and plotting modules."""
    apply_plot_style()
    cycle = build_default_cycle()
    phase1 = solve_phase1_workflow(cycle)

    print_phase1_report(phase1)
    create_phase1_figures(phase1)

    if show_plots:
        plt.show()
    else:
        plt.close("all")

    return phase1


if __name__ == "__main__":
    main()
