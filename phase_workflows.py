"""
phase_workflows.py
Shared phase-level orchestration built on top of the existing solver modules.

This module keeps the thermodynamic logic inside engine.py and the plotting
logic inside visualizer.py. It adds lightweight workflow helpers so each
project phase can be run independently or composed into a full project run.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from constants import MPL_STYLE, P_INLET, P_MAX_CYCLE, Q_DOT_IN, T_INLET, T_MAX
from engine import BraytonCycle
import visualizer as viz

ResultDict = dict[str, Any]

DEFAULT_VERIFICATION_RPS: tuple[float, ...] = (10.0, 5.43, 4.0)
DEFAULT_SAMPLE_EFFICIENCIES: tuple[float, float] = (0.85, 0.85)
DEFAULT_PHASE4_APPROACHES: tuple[float, ...] = tuple(np.linspace(10.0, 250.0, 400))
DEFAULT_PHASE5_SELECTION_POLICY = "max_eta_th"


@dataclass(frozen=True)
class PressureRatioCase:
    """Stable descriptor for a downstream pressure-ratio case."""

    key: str
    label: str
    rp: float
    source_phase: str


@dataclass(frozen=True)
class Phase1Artifacts:
    """Structured outputs for the Phase 1 workflow."""

    T1: float
    T3: float
    P1: float
    Q_dot_in: float
    rp_vals: np.ndarray
    default_rp_case: PressureRatioCase
    optimization_rp_case: PressureRatioCase
    comparison_rp_cases: tuple[PressureRatioCase, ...]
    sweep: ResultDict
    design_max_power: ResultDict
    design_max_specific_work: ResultDict
    optimization_model: ResultDict
    verification_points: dict[float, ResultDict]


@dataclass(frozen=True)
class Phase2Artifacts:
    """Structured outputs for the Phase 2 workflow."""

    T1: float
    T3: float
    P1: float
    Q_dot_in: float
    selected_rp: float
    rp_source: str
    default_rp_case: PressureRatioCase
    optimization_rp_case: PressureRatioCase
    comparison_rp_cases: tuple[PressureRatioCase, ...]
    phase1_design: ResultDict | None
    eta_vals: np.ndarray
    sample_efficiencies: tuple[float, float]
    contour: ResultDict
    sample: ResultDict
    rp_search: np.ndarray
    eta_search: np.ndarray
    optimization_search: ResultDict
    optimized_contour: ResultDict


@dataclass(frozen=True)
class Phase3Artifacts:
    """Structured outputs for the Phase 3 workflow."""

    T1: float
    T3: float
    P1: float
    Q_dot_in: float
    selected_rp: float
    rp_source: str
    rp_case: PressureRatioCase
    rp_candidates: dict[str, float]
    phase1_design: ResultDict | None
    ideal_reference: ResultDict
    real_fluid: ResultDict
    fluid: str


@dataclass(frozen=True)
class Phase4CaseArtifacts:
    """Structured outputs for one Phase 4 regenerator pressure-ratio case."""

    rp_case: PressureRatioCase
    baseline_phase3: ResultDict
    delta_T_approach_vals: np.ndarray
    sweep: ResultDict
    fluid: str


@dataclass(frozen=True)
class Phase4Artifacts:
    """Structured outputs for the Phase 4 workflow."""

    T1: float
    T3: float
    P1: float
    Q_dot_in: float
    fluid: str
    rp_cases: tuple[PressureRatioCase, ...]
    cases: tuple[Phase4CaseArtifacts, ...]


@dataclass(frozen=True)
class Phase5CaseArtifacts:
    """Structured outputs for one selected Phase 5 exergy-analysis case."""

    rp_case: PressureRatioCase
    selection_policy: str
    selected_phase4_case: ResultDict
    dead_state: ResultDict
    stream_exergy: dict[str, float]
    stream_exergy_rate: dict[str, float]
    component_summary: ResultDict
    notes: tuple[str, ...]


@dataclass(frozen=True)
class Phase5Artifacts:
    """Structured outputs for the Phase 5 workflow."""

    comparison_rp_cases: tuple[PressureRatioCase, ...]
    case_artifacts: tuple[Phase5CaseArtifacts, ...]
    default_case: Phase5CaseArtifacts | None
    selection_policy: str
    T_dead: float
    P_dead: float
    fluid: str


def solve_phase1_optimization_model(phase1_results: ResultDict) -> ResultDict:
    """Evaluate the Phase 1 normalized linear-combination optimization model."""
    rp_vals = phase1_results["rp_vals"]
    w_net_spec = phase1_results["w_net_spec"]
    eta_th = phase1_results["eta_th"]
    bwr = phase1_results["bwr"]

    with np.errstate(invalid="ignore", divide="ignore"):
        metric = (
            (w_net_spec / np.nanmax(w_net_spec))
            + (eta_th / np.nanmax(eta_th))
            - (bwr / np.nanmax(bwr))
            - (rp_vals / np.nanmax(rp_vals))
        )

    idx_optimal = int(np.nanargmax(metric))
    return {
        "metric": metric,
        "idx_optimal": idx_optimal,
        "rp_optimal": float(rp_vals[idx_optimal]),
    }


def apply_plot_style() -> None:
    """Apply the shared Matplotlib style used across all phase runners."""
    plt.rcParams.update(MPL_STYLE)


def build_default_cycle() -> BraytonCycle:
    """Build the Brayton cycle object from project default conditions."""
    return BraytonCycle(
        T1=T_INLET,
        T3=T_MAX,
        P1=P_INLET,
        Q_dot_in=Q_DOT_IN,
        P_max=P_MAX_CYCLE,
    )


def _make_rp_case(key: str, label: str, rp: float, source_phase: str) -> PressureRatioCase:
    """Create a stable pressure-ratio case descriptor."""
    return PressureRatioCase(
        key=key,
        label=label,
        rp=float(rp),
        source_phase=source_phase,
    )


def _build_phase1_rp_cases(
    design_max_power: ResultDict,
    design_max_specific_work: ResultDict,
    optimization_model: ResultDict,
) -> tuple[PressureRatioCase, PressureRatioCase, tuple[PressureRatioCase, ...]]:
    """Build the standard Phase 1 pressure-ratio case descriptors."""
    default_case = _make_rp_case(
        "phase1_max_power",
        "Phase 1 max-power design",
        design_max_power["rp"],
        "Phase 1",
    )
    max_specific_work_case = _make_rp_case(
        "phase1_max_specific_work",
        "Phase 1 max-specific-work design",
        design_max_specific_work["rp"],
        "Phase 1",
    )
    optimization_case = _make_rp_case(
        "phase1_optimization",
        "Phase 1 optimization model",
        optimization_model["rp_optimal"],
        "Phase 1",
    )
    comparison_cases = (
        default_case,
        max_specific_work_case,
        optimization_case,
    )
    return default_case, optimization_case, comparison_cases


def _build_phase2_rp_cases(
    selected_rp: float,
    rp_source: str,
    optimization_search: ResultDict,
) -> tuple[PressureRatioCase, PressureRatioCase, tuple[PressureRatioCase, ...]]:
    """Build the standard Phase 2 pressure-ratio case descriptors."""
    default_case = _make_rp_case(
        "phase2_selected",
        f"Phase 2 contour case ({rp_source})",
        selected_rp,
        "Phase 2",
    )
    optimization_case = _make_rp_case(
        "phase2_optimization",
        "Phase 2 mean-power optimization",
        optimization_search["rp_optimal"],
        "Phase 2",
    )
    comparison_cases = (
        default_case,
        optimization_case,
    )
    return default_case, optimization_case, comparison_cases


def _make_user_rp_case(rp: float) -> PressureRatioCase:
    """Build a stable pressure-ratio case for an explicit user-specified rp."""
    return _make_rp_case(
        "user_specified",
        "User-specified pressure ratio",
        rp,
        "User",
    )


def solve_phase1_workflow(
    cycle: BraytonCycle,
    rp_vals: np.ndarray | None = None,
    verification_rps: tuple[float, ...] = DEFAULT_VERIFICATION_RPS,
) -> Phase1Artifacts:
    """Run the full Phase 1 workflow using the existing engine methods."""
    sweep_rps = (
        np.asarray(rp_vals, dtype=float)
        if rp_vals is not None
        else np.linspace(1.5, cycle.rp_max, 500)
    )

    sweep = cycle.solve_phase1(sweep_rps)
    design_max_power = cycle.get_design_point(sweep, criterion="max_power")
    design_max_specific_work = cycle.get_design_point(
        sweep,
        criterion="max_specific_work",
    )
    optimization_model = solve_phase1_optimization_model(sweep)
    default_rp_case, optimization_rp_case, comparison_rp_cases = _build_phase1_rp_cases(
        design_max_power,
        design_max_specific_work,
        optimization_model,
    )
    verification_points = {
        rp: cycle.solve_point(rp)
        for rp in verification_rps
    }

    return Phase1Artifacts(
        T1=cycle.T1,
        T3=cycle.T3,
        P1=cycle.P1,
        Q_dot_in=cycle.Q_dot_in,
        rp_vals=sweep_rps,
        default_rp_case=default_rp_case,
        optimization_rp_case=optimization_rp_case,
        comparison_rp_cases=comparison_rp_cases,
        sweep=sweep,
        design_max_power=design_max_power,
        design_max_specific_work=design_max_specific_work,
        optimization_model=optimization_model,
        verification_points=verification_points,
    )


def solve_phase2_workflow(
    cycle: BraytonCycle,
    phase1: Phase1Artifacts | None = None,
    rp: float | None = None,
    eta_vals: np.ndarray | None = None,
    sample_efficiencies: tuple[float, float] = DEFAULT_SAMPLE_EFFICIENCIES,
    rp_search: np.ndarray | None = None,
    eta_search: np.ndarray | None = None,
) -> Phase2Artifacts:
    """Run the full Phase 2 workflow at a selected pressure ratio."""
    phase1_design = None
    if rp is None:
        phase1 = phase1 or solve_phase1_workflow(cycle)
        phase1_design = phase1.design_max_power
        selected_rp = float(phase1.default_rp_case.rp)
        rp_source = phase1.default_rp_case.label
    else:
        selected_rp = float(rp)
        rp_source = "user-specified pressure ratio"

    eta_grid = (
        np.asarray(eta_vals, dtype=float)
        if eta_vals is not None
        else np.linspace(0.2, 1.0, 500)
    )
    rp_search_vals = (
        np.asarray(rp_search, dtype=float)
        if rp_search is not None
        else np.linspace(1.5, cycle.rp_max, 150)
    )
    eta_search_vals = (
        np.asarray(eta_search, dtype=float)
        if eta_search is not None
        else np.linspace(0.4, 1.0, 80)
    )

    eta_c_sample, eta_t_sample = sample_efficiencies
    contour = cycle.solve_phase2_contour(selected_rp, eta_grid)
    sample = cycle.solve_phase2_sample(selected_rp, eta_c_sample, eta_t_sample)
    optimization_search = cycle.solve_phase2_mean_power_search(
        rp_search_vals,
        eta_search_vals,
    )
    optimized_contour = cycle.solve_phase2_contour(
        optimization_search["rp_optimal"],
        eta_grid,
    )
    default_rp_case, optimization_rp_case, comparison_rp_cases = _build_phase2_rp_cases(
        selected_rp,
        rp_source,
        optimization_search,
    )

    return Phase2Artifacts(
        T1=cycle.T1,
        T3=cycle.T3,
        P1=cycle.P1,
        Q_dot_in=cycle.Q_dot_in,
        selected_rp=selected_rp,
        rp_source=rp_source,
        default_rp_case=default_rp_case,
        optimization_rp_case=optimization_rp_case,
        comparison_rp_cases=comparison_rp_cases,
        phase1_design=phase1_design,
        eta_vals=eta_grid,
        sample_efficiencies=sample_efficiencies,
        contour=contour,
        sample=sample,
        rp_search=rp_search_vals,
        eta_search=eta_search_vals,
        optimization_search=optimization_search,
        optimized_contour=optimized_contour,
    )


def solve_phase3_workflow(
    cycle: BraytonCycle,
    phase1: Phase1Artifacts | None = None,
    phase2: Phase2Artifacts | None = None,
    rp: float | None = None,
    rp_case: PressureRatioCase | None = None,
    fluid: str = "CO2",
    strategy: str = "auto",
) -> Phase3Artifacts:
    """Run the Phase 3 real-fluid comparison at a selected pressure ratio."""
    phase1_design = None
    rp_candidates: dict[str, float] = {}

    if rp_case is not None:
        rp_candidates[rp_case.key] = rp_case.rp
    elif rp is None:
        phase1 = phase1 or solve_phase1_workflow(cycle)
        phase1_design = phase1.design_max_power
        for case in phase1.comparison_rp_cases:
            rp_candidates[case.key] = case.rp

        if phase2 is None and strategy in {"auto", "phase2_optimization"}:
            phase2 = solve_phase2_workflow(cycle, phase1=phase1)

        if phase2 is not None:
            rp_candidates[phase2.optimization_rp_case.key] = phase2.optimization_rp_case.rp

        if strategy == "auto":
            if phase2 is not None:
                rp_case = phase2.optimization_rp_case
            else:
                rp_case = phase1.optimization_rp_case
        elif strategy == "phase1_optimization":
            rp_case = phase1.optimization_rp_case
        elif strategy == "phase2_optimization":
            if phase2 is None:
                raise ValueError("Phase 2 optimization pressure ratio is unavailable.")
            rp_case = phase2.optimization_rp_case
        elif strategy == "phase1_max_power":
            rp_case = phase1.default_rp_case
        elif strategy == "phase1_max_specific_work":
            rp_case = next(
                case for case in phase1.comparison_rp_cases
                if case.key == "phase1_max_specific_work"
            )
        else:
            raise ValueError(
                "Unknown Phase 3 pressure-ratio strategy. "
                "Use 'auto', 'phase1_optimization', 'phase2_optimization', "
                "'phase1_max_power', or 'phase1_max_specific_work'."
            )
    else:
        rp_case = _make_user_rp_case(rp)
        rp_candidates[rp_case.key] = rp_case.rp

    ideal_reference = cycle.solve_point(rp_case.rp)
    real_fluid = cycle.solve_phase3(rp_case.rp, fluid=fluid)

    return Phase3Artifacts(
        T1=cycle.T1,
        T3=cycle.T3,
        P1=cycle.P1,
        Q_dot_in=cycle.Q_dot_in,
        selected_rp=rp_case.rp,
        rp_source=rp_case.label,
        rp_case=rp_case,
        rp_candidates=rp_candidates,
        phase1_design=phase1_design,
        ideal_reference=ideal_reference,
        real_fluid=real_fluid,
        fluid=fluid,
    )


def solve_phase3_optimization_cases(
    cycle: BraytonCycle,
    phase1: Phase1Artifacts | None = None,
    phase2: Phase2Artifacts | None = None,
    fluid: str = "CO2",
) -> list[Phase3Artifacts]:
    """Run Phase 3 for the current optimization-derived pressure ratios."""
    phase1 = phase1 or solve_phase1_workflow(cycle)
    phase2 = phase2 or solve_phase2_workflow(cycle, phase1=phase1)

    rp_cases = (
        phase1.optimization_rp_case,
        phase2.optimization_rp_case,
    )

    return [
        solve_phase3_workflow(
            cycle,
            phase1=phase1,
            phase2=phase2,
            rp_case=rp_case,
            fluid=fluid,
        )
        for rp_case in rp_cases
    ]


def solve_phase4_workflow(
    cycle: BraytonCycle,
    phase1: Phase1Artifacts | None = None,
    phase2: Phase2Artifacts | None = None,
    delta_T_approach_vals: np.ndarray | tuple[float, ...] = DEFAULT_PHASE4_APPROACHES,
    fluid: str = "CO2",
) -> Phase4Artifacts:
    """Run the Phase 4 regenerator sweep for both optimization-derived r_p cases."""
    phase1 = phase1 or solve_phase1_workflow(cycle)
    phase2 = phase2 or solve_phase2_workflow(cycle, phase1=phase1)
    dT_vals = np.asarray(delta_T_approach_vals, dtype=float)

    rp_cases = (
        phase1.optimization_rp_case,
        phase2.optimization_rp_case,
    )
    cases: list[Phase4CaseArtifacts] = []
    for rp_case in rp_cases:
        baseline_phase3 = cycle.solve_phase3(rp_case.rp, fluid=fluid)
        sweep = cycle.solve_phase4_sweep(
            rp=rp_case.rp,
            delta_T_approach_vals=dT_vals,
            fluid=fluid,
        )
        cases.append(
            Phase4CaseArtifacts(
                rp_case=rp_case,
                baseline_phase3=baseline_phase3,
                delta_T_approach_vals=dT_vals.copy(),
                sweep=sweep,
                fluid=fluid,
            )
        )

    return Phase4Artifacts(
        T1=cycle.T1,
        T3=cycle.T3,
        P1=cycle.P1,
        Q_dot_in=cycle.Q_dot_in,
        fluid=fluid,
        rp_cases=rp_cases,
        cases=tuple(cases),
    )


def select_phase4_case_for_exergy(
    phase4_case_artifacts: Phase4CaseArtifacts,
    policy: str = DEFAULT_PHASE5_SELECTION_POLICY,
    manual_index: int | None = None,
) -> ResultDict:
    """Select one feasible Phase 4 sweep case for downstream exergy analysis."""
    sweep_cases = phase4_case_artifacts.sweep["cases"]
    feasible_indices = [
        idx for idx, case in enumerate(sweep_cases)
        if case.get("feasible", False)
    ]
    if not feasible_indices:
        raise ValueError("No feasible Phase 4 cases are available for exergy analysis.")

    if policy == "max_eta_th":
        idx = max(feasible_indices, key=lambda i: sweep_cases[i]["eta_th"])
    elif policy == "max_W_dot_net":
        idx = max(feasible_indices, key=lambda i: sweep_cases[i]["W_dot_net"])
    elif policy == "minimum_delta_T_approach":
        idx = min(feasible_indices, key=lambda i: sweep_cases[i]["delta_T_approach"])
    elif policy == "manual_index":
        if manual_index is None:
            raise ValueError("manual_index must be provided when policy='manual_index'.")
        if manual_index not in feasible_indices:
            raise ValueError("manual_index must point to a feasible Phase 4 case.")
        idx = manual_index
    else:
        raise ValueError(
            "Unknown Phase 5 selection policy. Use 'max_eta_th', "
            "'max_W_dot_net', 'minimum_delta_T_approach', or 'manual_index'."
        )

    return dict(sweep_cases[idx])


def _build_phase5_notes(selected_phase4_case: ResultDict, stream_exergy: dict[str, float]) -> tuple[str, ...]:
    """Create short observations about the selected exergy case."""
    stream_items = [
        (key, value)
        for key, value in stream_exergy.items()
    ]
    max_stream_key, max_stream_value = max(stream_items, key=lambda item: item[1])
    min_stream_key, min_stream_value = min(stream_items, key=lambda item: item[1])
    return (
        f"Highest stream exergy occurs at {max_stream_key} = {max_stream_value:.2f} kJ/kg.",
        f"Lowest stream exergy occurs at {min_stream_key} = {min_stream_value:.2f} kJ/kg.",
        (
            f"Selected regenerated case uses delta_T_approach = "
            f"{selected_phase4_case['delta_T_approach']:.1f} K with "
            f"eta_th = {selected_phase4_case['eta_th'] * 100:.2f}%."
        ),
    )


def solve_phase5_workflow(
    cycle: BraytonCycle,
    phase4: Phase4Artifacts | None = None,
    phase1: Phase1Artifacts | None = None,
    phase2: Phase2Artifacts | None = None,
    selection_policy: str = DEFAULT_PHASE5_SELECTION_POLICY,
    manual_index: int | None = None,
    T_dead: float = 298.15,
    P_dead: float = 100.0,
    fluid: str = "CO2",
) -> Phase5Artifacts:
    """Run Phase 5 exergy analysis for the selected Phase 4 cases."""
    if phase4 is None:
        phase1 = phase1 or solve_phase1_workflow(cycle)
        phase2 = phase2 or solve_phase2_workflow(cycle, phase1=phase1)
        phase4 = solve_phase4_workflow(
            cycle,
            phase1=phase1,
            phase2=phase2,
            fluid=fluid,
        )

    case_artifacts: list[Phase5CaseArtifacts] = []
    for phase4_case in phase4.cases:
        selected_case = select_phase4_case_for_exergy(
            phase4_case,
            policy=selection_policy,
            manual_index=manual_index,
        )
        exergy = cycle.solve_phase5_exergy_case(
            selected_case,
            T_dead=T_dead,
            P_dead=P_dead,
            fluid=fluid,
        )
        notes = _build_phase5_notes(selected_case, exergy["stream_exergy"])
        case_artifacts.append(
            Phase5CaseArtifacts(
                rp_case=phase4_case.rp_case,
                selection_policy=selection_policy,
                selected_phase4_case=selected_case,
                dead_state=exergy["dead_state"],
                stream_exergy=exergy["stream_exergy"],
                stream_exergy_rate=exergy["stream_exergy_rate"],
                component_summary=exergy["component_summary"],
                notes=notes,
            )
        )

    return Phase5Artifacts(
        comparison_rp_cases=phase4.rp_cases,
        case_artifacts=tuple(case_artifacts),
        default_case=case_artifacts[0] if case_artifacts else None,
        selection_policy=selection_policy,
        T_dead=T_dead,
        P_dead=P_dead,
        fluid=fluid,
    )


def create_phase1_figures(phase1: Phase1Artifacts) -> list[plt.Figure]:
    """Create the standard Phase 1 figure set."""
    return [
        viz.plot_phase1_net_power(phase1.sweep, phase1.design_max_power),
        viz.plot_phase1_efficiency(phase1.sweep, phase1.design_max_power),
        viz.plot_phase1_bwr(phase1.sweep, phase1.design_max_power),
        viz.plot_phase1_optimization(phase1.sweep),
    ]


def create_phase2_figures(phase2: Phase2Artifacts) -> list[plt.Figure]:
    """Create the standard Phase 2 figure set."""
    t_max_c = phase2.T3 - 273.15

    return [
        viz.plot_phase2_contours(
            phase2.eta_vals,
            phase2.contour,
            rp=phase2.selected_rp,
            Q_dot_in_MW=phase2.Q_dot_in / 1000.0,
            T_max_C=t_max_c,
        ),
        viz.plot_phase2_mean_power_search(phase2.optimization_search),
        viz.plot_phase2_contours(
            phase2.eta_vals,
            phase2.optimized_contour,
            rp=phase2.optimization_rp_case.rp,
            Q_dot_in_MW=phase2.Q_dot_in / 1000.0,
            T_max_C=t_max_c,
        ),
    ]


def create_phase3_figures(phase3: Phase3Artifacts) -> list[plt.Figure]:
    """Create the standard Phase 3 comparison figure."""
    ideal_reference = dict(phase3.ideal_reference)
    real_fluid = dict(phase3.real_fluid)
    ideal_reference["T1"] = phase3.T1
    real_fluid["T3"] = phase3.T3

    return [
        viz.plot_phase3_comparison(
            ideal_reference,
            real_fluid,
            phase3.selected_rp,
            phase3.Q_dot_in / 1000.0,
        )
    ]


def create_phase3_optimization_figures(
    phase3_cases: list[Phase3Artifacts],
) -> list[plt.Figure]:
    """Create one Phase 3 comparison figure for each optimization case."""
    figures: list[plt.Figure] = []
    for phase3 in phase3_cases:
        figures.extend(create_phase3_figures(phase3))
    return figures


def create_phase4_figures(phase4: Phase4Artifacts) -> list[plt.Figure]:
    """Create the standard Phase 4 regenerator-study figures."""
    return [
        viz.plot_phase4_eta_vs_regen(phase4),
        viz.plot_phase4_power_vs_regen(phase4),
    ]


def create_phase5_figures(phase5: Phase5Artifacts) -> list[plt.Figure]:
    """Create one regenerated-cycle T-s diagram for each selected Phase 5 case."""
    return [
        viz.plot_phase4_ts_diagram(case)
        for case in phase5.case_artifacts
    ]


__all__ = [
    "DEFAULT_PHASE4_APPROACHES",
    "DEFAULT_PHASE5_SELECTION_POLICY",
    "DEFAULT_SAMPLE_EFFICIENCIES",
    "DEFAULT_VERIFICATION_RPS",
    "Phase1Artifacts",
    "Phase2Artifacts",
    "Phase3Artifacts",
    "Phase4Artifacts",
    "Phase4CaseArtifacts",
    "Phase5Artifacts",
    "Phase5CaseArtifacts",
    "PressureRatioCase",
    "apply_plot_style",
    "build_default_cycle",
    "create_phase1_figures",
    "create_phase2_figures",
    "create_phase3_optimization_figures",
    "create_phase3_figures",
    "create_phase4_figures",
    "create_phase5_figures",
    "select_phase4_case_for_exergy",
    "solve_phase1_optimization_model",
    "solve_phase1_workflow",
    "solve_phase2_workflow",
    "solve_phase3_optimization_cases",
    "solve_phase3_workflow",
    "solve_phase4_workflow",
    "solve_phase5_workflow",
]
