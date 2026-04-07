# CHEE 210 - Brayton Cycle Analysis Project

## Project Overview
Thermodynamic analysis of a closed Brayton cycle using CO2 as the working fluid.
All five project phases are now implemented.

The codebase is organized around:
- shared physics/thermodynamic logic
- shared plotting functions
- phase-based workflow helpers
- thin phase runner scripts

This matches the course project structure, where each phase has distinct deliverables and may need to be run independently.

Current implementation notes:
- Phase 1 uses a 500-point default pressure-ratio sweep
- Phase 2 uses a mean-net-power pressure-ratio search
- Phase 3 runs automatically for the current optimization-derived pressure-ratio cases
- Phase 4 runs automatically for those same optimization-derived pressure-ratio cases using a 400-point regenerator approach-temperature sweep from 10 K to 250 K
- Phase 5 performs exergy analysis on selected Phase 4 regenerated cases for those same optimization-derived branches
- Phase 5 (`run_phase5.py`) currently selects cases using `manual_index=300`; the workflow layer also supports `max_eta_th`, `max_W_dot_net`, and `minimum_delta_T_approach` policies
- `solve_phase4_sweep` uses a vectorized inner dT loop with batch CoolProp array calls (replacing the old per-dT Python loop)

**Fixed operating conditions**

| Parameter | Value |
|-----------|-------|
| Compressor inlet temperature `T1` | 298.15 K (25 C) |
| Turbine inlet temperature `T3` | 1073.15 K (800 C) |
| Compressor inlet pressure `P1` | 100 kPa |
| Maximum cycle pressure `P_max` | 6000 kPa (`r_p,max = 60`) |
| Heat input rate `Q_dot_in` | 10000 kW (10 MW) |

---

## File Structure

```text
CHEE 210/
|-- constants.py                # Physical constants and default conditions
|-- fluid_properties.py         # Thermodynamic property functions for CO2
|-- engine.py                   # BraytonCycle class - all solving logic
|-- visualizer.py               # Matplotlib figure functions
|-- phase_workflows.py          # Phase-level orchestration helpers
|-- reporting.py                # Console summary and LaTeX table helpers
|-- run_phase1.py               # Run Phase 1 only
|-- run_phase2.py               # Run Phase 2 only
|-- run_phase3.py               # Run Phase 3 only
|-- run_phase4.py               # Run Phase 4 only
|-- run_phase5.py               # Run Phase 5 only
|-- run_all.py                  # Run all implemented phases
|-- run_phase4_rp_sweep.py      # Broad r_p sweep with gradient-colored Phase 4 curves
|-- export_optimization_table.py # Generate LaTeX comparison table for opt. pressure ratios
|-- main.py                     # Backward-compatible wrapper for run_all.py
|-- test.py                     # Scratch / exploration script (do not rely on)
|-- Optimization Phase 1.py     # Legacy exploration script (do not edit)
`-- brayton_phase1_integral.py  # Original monolithic script (do not edit)
```

---

## Recommended Usage

Run each phase separately when preparing phase-specific outputs:

```bash
python run_phase1.py
python run_phase2.py
python run_phase3.py
python run_phase4.py
python run_phase5.py
```

Run a broad pressure-ratio sweep visualization for Phase 4:

```bash
python run_phase4_rp_sweep.py
```

Generate the LaTeX optimization comparison table:

```bash
python export_optimization_table.py
```

Run everything together:

```bash
python run_all.py
```

Backward-compatible full run:

```bash
python main.py
```

Programmatic use is also supported. Example:

```python
from engine import BraytonCycle
from phase_workflows import (
    solve_phase1_workflow,
    solve_phase2_workflow,
    solve_phase3_optimization_cases,
    solve_phase4_workflow,
    solve_phase5_workflow,
)

cycle = BraytonCycle(T1=298.15, T3=1073.15, P1=100.0, Q_dot_in=10000.0, P_max=6000.0)
phase1 = solve_phase1_workflow(cycle)
phase2 = solve_phase2_workflow(cycle, phase1=phase1)
phase3_cases = solve_phase3_optimization_cases(cycle, phase1=phase1, phase2=phase2)
phase4 = solve_phase4_workflow(cycle, phase1=phase1, phase2=phase2)
phase5 = solve_phase5_workflow(cycle, phase4=phase4)
```

---

## Module Responsibilities

### `constants.py`
Defines module-level constants only. No functions.

| Symbol | Value | Unit | Description |
|--------|-------|------|-------------|
| `CO2_A/B/C/D` | 22.26, 5.981e-2, -3.501e-5, 7.469e-9 | kJ/(kmol*K) | Table A-2c polynomial coefficients |
| `M_CO2` | 44.01 | kg/kmol | Molar mass of CO2 |
| `R_UNIVERSAL` | 8.314 | kJ/(kmol*K) | Universal gas constant |
| `R_CO2` | 0.18893 | kJ/(kg*K) | Specific gas constant (`R_u / M`) |
| `T_INLET` | 298.15 | K | Default `T1` |
| `T_MAX` | 1073.15 | K | Default `T3` |
| `P_INLET` | 100.0 | kPa | Default `P1` |
| `P_MAX_CYCLE` | 6000.0 | kPa | Default `P_max` |
| `Q_DOT_IN` | 10000.0 | kW | Default heat input |
| `MPL_STYLE` | dict | - | Shared Matplotlib rcParams |

### `fluid_properties.py`
Pure thermodynamic-property functions. No global state beyond imported constants.

All temperatures are in Kelvin, enthalpies in `kJ/kg`, and entropies in `kJ/(kg*K)`.

| Function | Inputs | Returns | Notes |
|----------|--------|---------|-------|
| `cp_molar(T)` | `T: float or ndarray` | `kJ/(kmol*K)` | A-2c polynomial |
| `cp(T)` | `T: float or ndarray` | `kJ/(kg*K)` | `cp_molar / M` |
| `delta_h(T_low, T_high)` | two floats | float | `scipy.quad`; scalar only |
| `delta_h_analytic(T_low, T_high)` | float or ndarray | same shape | analytic antiderivative; vectorized |
| `delta_s0(T_low, T_high)` | two floats | float | `int(cp/T dT)` via `scipy.quad` |
| `isentropic_outlet_T(T_in, rp, compress)` | floats + bool | float | isentropic outlet temperature via `brentq` |
| `find_T_from_delta_h(T_ref, dh_target, T_lo, T_hi)` | floats | float | inverse enthalpy lookup |
| `coolprop_state_tp(T, P_kPa, fluid)` | floats + str | dict | real-fluid state from temperature and pressure; LRU-cached |
| `coolprop_state_ps(P_kPa, s, fluid)` | floats + str | dict | real-fluid state from pressure and entropy; LRU-cached |
| `coolprop_state_ph(P_kPa, h, fluid)` | floats + str | dict | real-fluid state from pressure and enthalpy; LRU-cached |
| `coolprop_isobaric_ts_path(P_kPa, T_start, T_end, fluid, n_points)` | floats + str + int | dict of T,s arrays | isobaric path for T-s plotting |
| `coolprop_isentropic_ts_path(P_start_kPa, P_end_kPa, s, fluid, n_points)` | floats + str + int | dict of T,s arrays | isentropic path for T-s plotting |
| `get_coolprop_states(T1, T3, P1_kPa, rp, fluid)` | floats + str | dict | Phase 3 real-fluid state points |

Key distinction:
- `delta_h` is accurate and scalar-oriented
- `delta_h_analytic` is fast and intended for vectorized grids
- The three `coolprop_state_*` wrappers are LRU-cached (maxsize 4096); bypass them with `_get_coolprop_propssi()` when doing batch/vectorized calls

### `engine.py`
Contains the `BraytonCycle` class and all cycle-solving logic.

**Constructor**

```python
BraytonCycle(T1, T3, P1, Q_dot_in, P_max)
```

All temperatures are in K, pressures in kPa, and power in kW.

#### Phase 1 methods

| Method | Returns | Description |
|--------|---------|-------------|
| `solve_phase1(rp_vals)` | dict of ndarrays | pressure-ratio sweep |
| `get_design_point(phase1_results, criterion)` | dict | selects preferred Phase 1 design |
| `solve_point(rp)` | dict | exact state at one pressure ratio |

`solve_phase1` keys:
`rp_vals, W_dot_net, eta_th, bwr, m_dot, w_net_spec, T2, T4`

`get_design_point` / `solve_point` keys:
`idx, rp, T2, T4, w_c, w_t, w_net, q_in, eta_th, bwr, m_dot, W_dot_net`

Phase 1 optimization is now handled in the workflow layer with a normalized linear score built from:
- specific net work
- thermal efficiency
- back work ratio
- pressure ratio penalty

#### Phase 2 methods

| Method | Returns | Description |
|--------|---------|-------------|
| `solve_phase2_contour(rp, eta_vals)` | dict of `N x N` ndarrays | fully vectorized efficiency-grid solve |
| `solve_phase2_sample(rp, eta_C, eta_T)` | dict | exact single-point sample calculation |
| `solve_phase2_mean_power_search(rp_search, eta_search)` | dict | pressure-ratio optimization by mean net power |

`solve_phase2_contour` keys:
`eta_C_grid, eta_T_grid, W_dot_grid, eta_th_grid, m_dot_grid, bwr_grid, w_c_s, w_t_s, T2s, T4s`

`solve_phase2_mean_power_search` keys:
`rp_search, mean_Wdot, rp_optimal, idx_optimal`

#### Phase 3 method

| Method | Returns | Description |
|--------|---------|-------------|
| `solve_phase3(rp, fluid='CO2')` | dict | real-fluid Brayton-cycle solve via CoolProp |

Keys:
`rp, h1, s1, T2, h2, h3, s3, T4, h4, w_c, w_t, w_net, q_in, eta_th, bwr, m_dot, W_dot_net`

#### Phase 4 methods

| Method | Returns | Description |
|--------|---------|-------------|
| `solve_phase4_case(rp, delta_T_approach, fluid='CO2')` | dict | one regenerated real-fluid case |
| `solve_phase4_sweep(rp, delta_T_approach_vals, fluid='CO2', return_cases=True)` | dict | parametric regenerator study over temperature approaches |
| `solve_phase4(rp, delta_T_approach=60.0, fluid='CO2')` | dict | single-case convenience wrapper |

Phase 4 uses a six-state regenerated real-fluid Brayton-cycle model:
- State 1: compressor inlet
- State 2: compressor outlet
- State 3: regenerator cold-side outlet / heater inlet
- State 4: turbine inlet
- State 5: turbine outlet / regenerator hot-side inlet
- State 6: regenerator hot-side outlet / cooler inlet

Phase 4 keys include:
`rp, delta_T_approach, feasible, reason, T1..T6, P1..P6, h1..h6, s1..s6, w_c, w_t, w_net, q_in, q_regen, Q_dot_regen, eta_th, bwr, m_dot, W_dot_net`

Current default Phase 4 study settings:
- regenerator design variable: temperature approach `delta_T_approach`
- default sweep: 400 points from `10 K` to `250 K` (`DEFAULT_PHASE4_APPROACHES` in `phase_workflows.py`)
- comparison cases:
  - the current Phase 1 optimization-model pressure ratio
  - the current Phase 2 mean-net-power pressure ratio
- `solve_phase4_sweep` inner dT loop is vectorized: uses batch `PropsSI` array calls for state 3 (h3) and state 6 (T6), replacing N_dT individual Python-level CoolProp calls per rp. Use `return_cases=False` to skip case-dict construction and get only the aggregated NumPy arrays.

#### Phase 5 methods

| Method | Returns | Description |
|--------|---------|-------------|
| `solve_phase5_exergy_case(phase4_case, T_dead=298.15, P_dead=100.0, fluid='CO2')` | dict | exergy analysis for one selected Phase 4 case |
| `solve_phase5_exergy(phase4_case, T_dead=298.15, P_dead=100.0, fluid='CO2')` | dict | compatibility wrapper |

Phase 5 uses the selected Phase 4 state data directly rather than recomputing the cycle.
Dead-state properties are evaluated at:
- `T0 = 298.15 K`
- `P0 = 100 kPa`

Current default Phase 5 settings:
- available case-selection policies: `max_eta_th`, `max_W_dot_net`, `minimum_delta_T_approach`, `manual_index`
- `run_phase5.py` currently uses `manual_index=300`
- comparison cases:
  - the selected Phase 4 case from the Phase 1 optimization branch
  - the selected Phase 4 case from the Phase 2 optimization branch

### `visualizer.py`
Contains plotting functions only. Each function accepts precomputed results and returns a `matplotlib.figure.Figure`.

| Function | Inputs | Figure content |
|----------|--------|----------------|
| `plot_phase1_net_power(p1_results, design_point)` | sweep dict + design dict | net power vs `r_p` |
| `plot_phase1_efficiency(p1_results, design_point)` | sweep dict + design dict | thermal efficiency vs `r_p` |
| `plot_phase1_bwr(p1_results, design_point)` | sweep dict + design dict | BWR vs `r_p` |
| `plot_phase1_summary_table(dp_best, dp_mmp, P1, Q_dot_in)` | two design dicts | Phase 1 comparison table |
| `plot_phase1_optimization(p1_results)` | sweep dict | normalized linear-combination optimization metric plot |
| `plot_phase2_contours(eta_vals, contour_results, rp, ...)` | contour dict | 2x2 contour figure |
| `plot_phase2_mean_power_search(optimization_results)` | optimization dict | single-panel mean-net-power search plot |
| `plot_phase3_comparison(phase1_point, phase3_results, rp, ...)` | two point dicts | T-s comparison + bar chart |
| `plot_phase4_eta_vs_regen(phase4)` | Phase 4 artifacts | thermal efficiency vs regenerator heat-transfer rate |
| `plot_phase4_power_vs_regen(phase4)` | Phase 4 artifacts | net power vs regenerator heat-transfer rate |
| `plot_phase4_rp_sweep_metric(rp_vals, sweep_results, metric_key, ylabel, title)` | rp array + list of sweep dicts | gradient-colored metric curves across the full r_p range |
| `plot_phase4_ts_diagram(phase5_case_artifact)` | Phase 5 case artifact | full T-s diagram of the selected six-state regenerated cycle |

Note: `plot_phase1_optimization_old` also exists in the file as a legacy multiplicative metric formulation — it is not called by any runner.

### `phase_workflows.py`
Shared orchestration layer for running project phases.

This module sits above `engine.py` and `visualizer.py` and keeps:
- solving
- plotting
- reporting

separate from each other.

Module-level constants:
- `DEFAULT_VERIFICATION_RPS = (10.0, 5.43, 4.0)`
- `DEFAULT_SAMPLE_EFFICIENCIES = (0.85, 0.85)`
- `DEFAULT_PHASE4_APPROACHES = tuple(np.linspace(10.0, 250.0, 400))` — 400 points from 10 K to 250 K
- `DEFAULT_PHASE5_SELECTION_POLICY = "max_eta_th"`

It provides:
- `build_default_cycle()`
- `apply_plot_style()`
- `solve_phase1_optimization_model()`
- `solve_phase1_workflow()`
- `solve_phase2_workflow()`
- `solve_phase3_workflow()` — accepts `strategy` param: `'auto'`, `'phase1_optimization'`, `'phase2_optimization'`, `'phase1_max_power'`, `'phase1_max_specific_work'`
- `solve_phase3_optimization_cases()`
- `solve_phase4_workflow()`
- `select_phase4_case_for_exergy()` — policies: `'max_eta_th'`, `'max_W_dot_net'`, `'minimum_delta_T_approach'`, `'manual_index'`
- `solve_phase5_workflow()`
- `create_phase1_figures()`
- `create_phase2_figures()`
- `create_phase3_figures()`
- `create_phase3_optimization_figures()`
- `create_phase4_figures()`
- `create_phase5_figures()` — one T-s diagram per selected Phase 5 case

It also defines lightweight dataclasses:
- `PressureRatioCase`
- `Phase1Artifacts`
- `Phase2Artifacts`
- `Phase3Artifacts`
- `Phase4CaseArtifacts`
- `Phase4Artifacts`
- `Phase5CaseArtifacts`
- `Phase5Artifacts`

These dataclasses wrap the existing result dicts without changing the underlying physics layer.
The stable `PressureRatioCase` objects make it easier for downstream phases to
consume whichever pressure ratios are currently selected by the active
optimization models.

Current artifact intent:
- `default_rp_case`: the phase's default downstream pressure ratio
- `optimization_rp_case`: the active optimization-derived pressure ratio for downstream comparison work
- `comparison_rp_cases`: the named pressure-ratio cases tracked by that phase
- `selected_phase4_case`: the specific feasible Phase 4 case chosen for Phase 5 exergy analysis

### `reporting.py`
Console output and LaTeX table helpers for each runner.

Console print functions:
- `print_phase1_report()`
- `print_phase2_report()`
- `print_phase3_report()`
- `print_phase4_report()`
- `print_phase5_report()`

LaTeX table builders (Phase 5):
- `build_phase5_latex_table(phase5_case_artifact)` — stream-exergy table for one case
- `build_phase5_component_latex_table(phase5_case_artifact)` — component exergy-change table for one case
- `build_phase5_latex_tables(phase5_artifacts)` — dict of stream-exergy tables keyed by branch
- `build_phase5_component_latex_tables(phase5_artifacts)` — dict of component tables keyed by branch
- `build_phase5_latex_bundle(phase5_artifacts, include_component_tables=False)` — all tables as one combined string
- `write_phase5_latex_tables(phase5_artifacts, output_dir='latex_tables', include_component_tables=False)` — write `.tex` files to disk

### Runner scripts

#### `run_phase1.py`
Runs only Phase 1:
- solve pressure-ratio sweep
- identify design points
- evaluate the Phase 1 optimization model
- print summary and verification points
- create Phase 1 figures

#### `run_phase2.py`
Runs only Phase 2:
- computes the preferred Phase 1 design point first
- uses that selected `r_p` for the Phase 2 contour and sample calculations
- computes a separate Phase 2 pressure-ratio optimization based on mean net power over the efficiency grid
- prints the Phase 2 summary
- creates the Phase 2 figures

#### `run_phase3.py`
Runs only Phase 3:
- computes Phase 1 and Phase 2 first
- automatically pulls the current optimization-derived pressure ratios
- runs Phase 3 for both optimization cases
- prints two Phase 3 summaries
- creates two Phase 3 comparison figures automatically
- skips gracefully if CoolProp is not installed

#### `run_phase4.py`
Runs only Phase 4:
- computes Phase 1 and Phase 2 first
- automatically pulls the current optimization-derived pressure ratios
- runs the regenerator sweep for both optimization cases
- uses the default 400-point `delta_T_approach` sweep from `10 K` to `250 K` unless overridden programmatically
- prints a Phase 4 report for each case
- creates Phase 4 comparison figures automatically
- skips gracefully if CoolProp is not installed

#### `run_phase5.py`
Runs only Phase 5:
- computes Phase 1, Phase 2, and Phase 4 first
- selects one feasible regenerated case per optimization branch using `manual_index=300`
- performs exergy analysis using the selected Phase 4 state data
- prints Phase 5 exergy tables for both optimization branches
- writes LaTeX exergy tables to `latex_tables/`
- creates T-s diagrams for each selected case via `create_phase5_figures()`
- skips gracefully if CoolProp is not installed

#### `run_phase4_rp_sweep.py`
Standalone broad pressure-ratio sweep for Phase 4 visualization:
- sweeps `r_p` from 1.5 to `r_p,max` over 30 000 points
- uses `DEFAULT_PHASE4_APPROACHES` (400 dT values from 10 K to 250 K) per rp
- parallelizes over rp values using `ThreadPoolExecutor` (16 workers)
- `solve_phase4_sweep` inner dT loop is vectorized with batch CoolProp calls
- plots gradient-colored thermal efficiency and net power curves vs Q_dot_regen
- skips gracefully if CoolProp is not installed

#### `export_optimization_table.py`
Utility script for generating a LaTeX comparison table:
- runs Phase 1 and Phase 2 workflows
- evaluates ideal-cycle state at both optimization pressure ratios
- writes a `phase1_phase2_optimization_table.tex` file to the working directory

#### `run_all.py`
Runs Phase 1 -> Phase 2 -> Phase 3 -> Phase 4 -> Phase 5 in project order.
Also creates Phase 5 T-s figures via `create_phase5_figures()`.

#### `main.py`
Thin backward-compatible wrapper around `run_all.py`.

---

## Phase Data Flow

The phase runners are designed around the course workflow:

- Phase 1 identifies the preferred pressure ratio
- Phase 2 uses the selected Phase 1 pressure ratio by default
- Phase 3 can evaluate multiple optimization-derived pressure ratios automatically
- Phase 4 uses those optimization-derived pressure ratios dynamically for the regenerator study
- Phase 5 selects cases from Phase 4 and performs exergy analysis on them

Default behavior:
- `run_phase2.py` recomputes Phase 1 internally, then carries the Phase 1 max-power design into the Phase 2 contour/sample solve
- `run_phase2.py` also computes a separate mean-net-power pressure-ratio optimization for Phase 2
- `run_phase3.py` recomputes Phase 1 and Phase 2 internally, then runs Phase 3 for:
  - the Phase 1 optimization-model pressure ratio
  - the Phase 2 mean-net-power pressure ratio
- `run_phase4.py` recomputes Phase 1 and Phase 2 internally, then runs the Phase 4 regenerator sweep for:
  - the Phase 1 optimization-model pressure ratio
  - the Phase 2 mean-net-power pressure ratio
- `run_phase5.py` recomputes Phase 1, Phase 2, and Phase 4 internally, then selects one feasible regenerated case from each optimization branch for exergy analysis
- `run_all.py` computes the earlier phases once and passes those results forward into Phase 3, Phase 4, and Phase 5

This keeps each phase runner self-contained while still matching the engineering design sequence.

### Current automatic optimization-linked pressure ratios

With the current models, the code is set up to track these pressure-ratio choices automatically:

- `phase1_max_power`: highest total net power from the Phase 1 sweep
- `phase1_max_specific_work`: highest specific net work from the Phase 1 sweep
- `phase1_optimization`: optimum from the normalized linear Phase 1 optimization model
- `phase2_mean_power`: optimum from the Phase 2 mean-net-power search

Phase 3 now uses the optimization outputs dynamically rather than relying on a fixed hard-coded comparison pressure ratio.
Phase 4 follows the same pattern, so changes to the Phase 1 or Phase 2 optimization logic automatically propagate into the regenerator study.
Phase 5 consumes Phase 4 cases rather than raw pressure ratios, so it follows those same upstream changes automatically.

### Current default downstream pressure-ratio behavior

- `Phase1Artifacts.default_rp_case` is the Phase 1 max-power design
- `Phase1Artifacts.optimization_rp_case` is the Phase 1 normalized linear-model optimum
- `Phase2Artifacts.default_rp_case` is the Phase 2 contour/sample pressure ratio
- `Phase2Artifacts.optimization_rp_case` is the Phase 2 mean-net-power optimum
- `run_phase3.py` compares the Phase 1 and Phase 2 optimization cases
- `run_phase4.py` studies those same two optimization cases
- `run_phase5.py` analyzes selected regenerated cases from those same two optimization cases

---

## Physics Reference

**Isentropic condition (ideal gas, variable `cp`)**

```text
int[T_in to T_out] cp(T)/T dT =  R_CO2 * ln(r_p)   [compression]
                                = -R_CO2 * ln(r_p)   [expansion]
```

**Cycle energy balance (per unit mass)**

```text
w_c   = delta_h(T1 -> T2)      compressor work input
w_t   = delta_h(T4 -> T3)      turbine work output
w_net = w_t - w_c
q_in  = delta_h(T2 -> T3)
eta_th = w_net / q_in
BWR    = w_c / w_t
m_dot  = Q_dot_in / q_in
```

**Phase 2 actual works**

```text
w_c_act = w_c_s / eta_C    (eta_C < 1 -> more compressor work)
w_t_act = w_t_s * eta_T    (eta_T < 1 -> less turbine work)
```

**Phase 2 pressure-ratio optimization**

```text
Choose the pressure ratio that maximizes mean_Wdot over the eta_C, eta_T grid
```

**Phase 3 automatic comparison cases**

```text
Run the real-fluid comparison for both:
1. the Phase 1 optimization-model pressure ratio
2. the Phase 2 mean-net-power pressure ratio
```

**Phase 4 regenerated real-fluid cycle**

```text
w_c        = h2 - h1
w_t        = h4 - h5
w_net      = w_t - w_c
q_in       = h4 - h3
eta_th     = w_net / q_in
m_dot      = Q_dot_in / q_in
q_regen    = h3 - h2 = h5 - h6
Q_dot_regen = m_dot * q_regen
```

**Phase 4 dynamic comparison cases**

```text
Run the regenerator study for both:
1. the Phase 1 optimization-model pressure ratio
2. the Phase 2 mean-net-power pressure ratio
```

**Phase 4 default parametric sweep**

```text
delta_T_approach = 400 evenly spaced points from 10 K to 250 K
plot eta_th and W_dot_net against Q_dot_regen
```

**Phase 5 specific exergy**

```text
x_i = (h_i - h0) - T0 * (s_i - s0)
X_dot_i = m_dot * x_i
```

**Phase 5 component exergy changes**

```text
compressor       = x2 - x1
heater           = x4 - x3
turbine          = x4 - x5
regenerator_cold = x3 - x2
regenerator_hot  = x5 - x6
cooler           = x6 - x1
```

**Phase 5 case-selection policies**

```text
max_eta_th               — select the feasible case with highest thermal efficiency
max_W_dot_net            — select the feasible case with highest net power
minimum_delta_T_approach — select the feasible case with the smallest approach temperature
manual_index             — select a specific index from the sweep case list
```

---

## Dependencies

| Package | Purpose | Required for |
|---------|---------|--------------|
| `numpy` | array math | all phases |
| `scipy` | `quad`, `brentq` | all phases |
| `matplotlib` | plotting | all phases |
| `CoolProp` | real-fluid properties | Phase 3, Phase 4, and Phase 5 |

Install optional real-fluid dependency:

```bash
pip install CoolProp
```

---

## Adding a New Phase

Recommended pattern for future extensions:

1. Add the new solving method to `BraytonCycle` in `engine.py`
2. Add any new plotting function to `visualizer.py`
3. Add a workflow helper to `phase_workflows.py`
4. Add console reporting to `reporting.py` if needed
5. Add a new runner script such as `run_phase4.py`
6. Call the new phase from `run_all.py`
7. Update this document

This preserves the current architecture while making future extension straightforward.
