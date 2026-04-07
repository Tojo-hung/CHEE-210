
from __future__ import annotations


from visualizer import plot_phase1_optimization, plot_phase1_optimization_old

import matplotlib.pyplot as plt

from phase_workflows import (
    apply_plot_style,
    build_default_cycle,
    solve_phase1_workflow,
)
from reporting import print_phase1_report


def main(show_plots: bool = True):
    """Run only Phase 1 using the shared solver and plotting modules."""
    apply_plot_style()
    cycle = build_default_cycle()
    phase1 = solve_phase1_workflow(cycle)

    plot_phase1_optimization(phase1.sweep)

    print_phase1_report(phase1)


    if show_plots:
        plt.show()
    else:
        plt.close("all")

    return phase1


if __name__ == "__main__":
    main()