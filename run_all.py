"""
run_all.py
Master entry point that runs the implemented project phases in sequence.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from output_manager import get_phase_output_dirs, save_figures
from phase_workflows import (
    apply_plot_style,
    build_default_cycle,
    create_phase1_figures,
    create_phase2_figures,
    create_phase3_optimization_figures,
    create_phase4_figures,
    create_phase5_figures,
    solve_phase1_workflow,
    solve_phase2_workflow,
    solve_phase3_optimization_cases,
    solve_phase4_workflow,
    solve_phase5_workflow,
)
from reporting import (
    build_phase1_latex_tables,
    build_phase2_latex_tables,
    build_phase3_latex_tables,
    build_phase4_latex_tables,
    build_phase5_output_latex_tables,
    print_phase1_report,
    print_phase2_report,
    print_phase3_report,
    print_phase4_report,
    print_phase5_report,
    sanitize_label_slug,
)
from output_manager import write_text_outputs


def _save_phase_outputs(
    output_root: str | Path,
    phase_name: str,
    figures: list[tuple[str, plt.Figure]],
    latex_tables: dict[str, str],
) -> None:
    """Save one phase's figure and LaTeX outputs into its phase folder."""
    figure_dir, table_dir = get_phase_output_dirs(output_root, phase_name)
    save_figures(figures, figure_dir)
    write_text_outputs(latex_tables, table_dir)


def main(
    show_plots: bool = True,
    save_outputs: bool = True,
    output_root: str | Path = "outputs",
):
    """Run Phase 1 through Phase 5 in project order."""
    apply_plot_style()
    cycle = build_default_cycle()

    phase1 = solve_phase1_workflow(cycle)
    print_phase1_report(phase1)
    phase1_figures = create_phase1_figures(phase1)
    if save_outputs:
        _save_phase_outputs(
            output_root,
            "phase1",
            [
                ("net_power_vs_rp.png", phase1_figures[0]),
                ("thermal_efficiency_vs_rp.png", phase1_figures[1]),
                ("back_work_ratio_vs_rp.png", phase1_figures[2]),
                ("optimization_metric_vs_rp.png", phase1_figures[3]),
            ],
            build_phase1_latex_tables(phase1),
        )

    phase2 = solve_phase2_workflow(cycle, phase1=phase1)
    print_phase2_report(phase2)
    phase2_figures = create_phase2_figures(phase2)
    if save_outputs:
        _save_phase_outputs(
            output_root,
            "phase2",
            [
                ("contours_selected_rp.png", phase2_figures[0]),
                ("mean_power_search.png", phase2_figures[1]),
                ("contours_optimized_rp.png", phase2_figures[2]),
            ],
            build_phase2_latex_tables(phase2),
        )

    phase3_cases = None
    try:
        phase3_cases = solve_phase3_optimization_cases(
            cycle,
            phase1=phase1,
            phase2=phase2,
        )
    except ImportError as exc:
        print(f"\n[Phase 3] Skipped - {exc}")
    else:
        for phase3 in phase3_cases:
            print_phase3_report(phase3)
        phase3_figures = create_phase3_optimization_figures(phase3_cases)
        if save_outputs:
            phase3_named_figures = [
                (
                    f"comparison_{sanitize_label_slug(phase3.rp_case.key)}.png",
                    figure,
                )
                for phase3, figure in zip(phase3_cases, phase3_figures)
            ]
            _save_phase_outputs(
                output_root,
                "phase3",
                phase3_named_figures,
                build_phase3_latex_tables(phase3_cases),
            )

    phase4 = None
    try:
        phase4 = solve_phase4_workflow(
            cycle,
            phase1=phase1,
            phase2=phase2,
        )
    except ImportError as exc:
        print(f"\n[Phase 4] Skipped - {exc}")
    else:
        print_phase4_report(phase4)
        phase4_figures = create_phase4_figures(phase4)
        if save_outputs:
            _save_phase_outputs(
                output_root,
                "phase4",
                [
                    ("thermal_efficiency_vs_qregen.png", phase4_figures[0]),
                    ("net_power_vs_qregen.png", phase4_figures[1]),
                ],
                build_phase4_latex_tables(phase4),
            )

    phase5 = None
    if phase4 is not None:
        try:
            phase5 = solve_phase5_workflow(
                cycle,
                phase4=phase4,
            )
        except ImportError as exc:
            print(f"\n[Phase 5] Skipped - {exc}")
        else:
            print_phase5_report(phase5)
            phase5_figures = create_phase5_figures(phase5)
            if save_outputs:
                phase5_named_figures = [
                    (
                        f"ts_diagram_{sanitize_label_slug(case.rp_case.key)}.png",
                        figure,
                    )
                    for case, figure in zip(phase5.case_artifacts, phase5_figures)
                ]
                _save_phase_outputs(
                    output_root,
                    "phase5",
                    phase5_named_figures,
                    build_phase5_output_latex_tables(phase5),
                )

    if show_plots:
        plt.show()
    else:
        plt.close("all")

    return phase1, phase2, phase3_cases, phase4, phase5


if __name__ == "__main__":
    main()
