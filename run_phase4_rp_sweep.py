"""
run_phase4_rp_sweep.py
Explore Phase 4 over a broad pressure-ratio range with gradient-colored curves.
"""

from __future__ import annotations

import concurrent.futures
import matplotlib.pyplot as plt
import numpy as np
import time

from phase_workflows import (
    DEFAULT_PHASE4_APPROACHES,
    apply_plot_style,
    build_default_cycle,
)
import visualizer as viz


def main(
    show_plots: bool = True,
    num_rp_points: int = 30000,
    delta_T_approach_vals: np.ndarray | tuple[float, ...] = DEFAULT_PHASE4_APPROACHES,
):
    """Run a broad Phase 4 pressure-ratio sweep and plot gradient-colored curves."""
    apply_plot_style()
    cycle = build_default_cycle()

    rp_vals = np.linspace(1.5, cycle.rp_max, num_rp_points)
    dT_vals = np.asarray(delta_T_approach_vals, dtype=float)

    try:
        def _solve(rp: float) -> dict:
            return cycle.solve_phase4_sweep(
                rp=rp,
                delta_T_approach_vals=dT_vals,
                return_cases=False,
            )

        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
            sweep_results = list(pool.map(_solve, rp_vals))
    except ImportError as exc:
        print(f"\n[Phase 4 r_p Sweep] Skipped - {exc}")
        plt.close("all")
        return None

    viz.plot_phase4_rp_sweep_metric(
        rp_vals,
        sweep_results,
        metric_key="eta_th",
        ylabel=r"$\eta_{th}$ (%)",
        title="Phase 4 Regenerator Study Across Pressure Ratio Range: Thermal Efficiency",
    )
    viz.plot_phase4_rp_sweep_metric(
        rp_vals,
        sweep_results,
        metric_key="W_dot_net",
        ylabel=r"$\dot{W}_{net}$ (MW)",
        title="Phase 4 Regenerator Study Across Pressure Ratio Range: Net Power",
    )

    print(
        f"\n[Phase 4 r_p Sweep] Generated curves for {len(rp_vals)} pressure ratios "
        f"from {rp_vals[0]:.2f} to {rp_vals[-1]:.2f}."
    )

    if show_plots:
        plt.show()
    else:
        plt.close("all")

    return {
        "rp_vals": rp_vals,
        "delta_T_approach_vals": dT_vals,
        "sweep_results": sweep_results,
    }


if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"\n[Phase 4 r_p Sweep] Total execution time: {elapsed_time:.2f} seconds")    
