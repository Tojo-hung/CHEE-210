"""
engine.py
Class-based Brayton cycle solver for CO2 working fluid.

Phases implemented
------------------
Phase 1 – Ideal gas, variable cp (Table A-2c polynomial integral method).
Phase 2 – Non-ideal compressor and turbine with isentropic efficiencies.
Phase 3 – Real fluid properties via CoolProp (Span-Wagner EOS).

Extension hooks
---------------
Phase 4 (regenerator) and Phase 5 (exergy) stubs are provided; implement by
overriding solve_phase4 / solve_phase5_exergy in a subclass or filling in the
NotImplementedError bodies when those phases are required.
"""

from __future__ import annotations

import numpy as np

import fluid_properties as fp


class BraytonCycle:
    """Brayton cycle solver for CO2 working fluid.

    Args:
        T1:       Compressor inlet temperature in Kelvin.
        T3:       Turbine inlet temperature in Kelvin.
        P1:       Compressor inlet pressure in kPa.
        Q_dot_in: Total heat input rate in kW.
        P_max:    Maximum allowable cycle pressure in kPa.
    """

    def __init__(
        self,
        T1: float,
        T3: float,
        P1: float,
        Q_dot_in: float,
        P_max: float,
    ) -> None:
        self.T1 = T1
        self.T3 = T3
        self.P1 = P1
        self.Q_dot_in = Q_dot_in
        self.P_max = P_max
        self.rp_max: float = P_max / P1

    # ── Phase 1 ───────────────────────────────────────────────────────────────

    def solve_phase1(self, rp_vals: np.ndarray) -> dict:
        """Pressure-ratio sweep for the ideal-gas variable-cp Brayton cycle.

        For each rp in rp_vals the method finds isentropic compressor and turbine
        outlet temperatures via brentq, then computes specific works and cycle
        performance.  Non-viable points (w_net ≤ 0 or q_in ≤ 0) are stored as
        np.nan so downstream code (plotting, argmax) handles them gracefully.

        Args:
            rp_vals: 1-D array of pressure ratios to evaluate.

        Returns:
            Dictionary with keys:
                rp_vals    (np.ndarray): Pressure ratios (echoed from input).
                W_dot_net  (np.ndarray): Net power output in kW.
                eta_th     (np.ndarray): Thermal efficiency, dimensionless (0–1).
                bwr        (np.ndarray): Back work ratio, dimensionless (0–1).
                m_dot      (np.ndarray): Mass flow rate in kg/s.
                w_net_spec (np.ndarray): Specific net work in kJ/kg.
                T2         (np.ndarray): Compressor outlet temperature in K.
                T4         (np.ndarray): Turbine outlet temperature in K.
        """
        T1, T3, Q_dot_in = self.T1, self.T3, self.Q_dot_in
        n = len(rp_vals)
        W_dot_net  = np.full(n, np.nan)
        eta_th     = np.full(n, np.nan)
        bwr        = np.full(n, np.nan)
        m_dot      = np.full(n, np.nan)
        w_net_spec = np.full(n, np.nan)
        T2_arr     = np.full(n, np.nan)
        T4_arr     = np.full(n, np.nan)

        for i, rp in enumerate(rp_vals):
            T2 = fp.isentropic_outlet_T(T1, rp, compress=True)
            T4 = fp.isentropic_outlet_T(T3, rp, compress=False)

            w_c   = fp.delta_h(T1, T2)
            w_t   = fp.delta_h(T4, T3)
            w_net = w_t - w_c
            q_in  = fp.delta_h(T2, T3)

            if q_in <= 0 or w_net <= 0:
                continue

            m_dot_i = Q_dot_in / q_in
            W_dot_net[i]  = m_dot_i * w_net
            eta_th[i]     = w_net / q_in
            bwr[i]        = w_c / w_t
            m_dot[i]      = m_dot_i
            w_net_spec[i] = w_net
            T2_arr[i]     = T2
            T4_arr[i]     = T4

        return {
            "rp_vals":    rp_vals,
            "W_dot_net":  W_dot_net,
            "eta_th":     eta_th,
            "bwr":        bwr,
            "m_dot":      m_dot,
            "w_net_spec": w_net_spec,
            "T2":         T2_arr,
            "T4":         T4_arr,
        }

    def get_design_point(
        self,
        phase1_results: dict,
        criterion: str = "max_power",
    ) -> dict:
        """Extract the optimal design point from a Phase 1 sweep.

        Args:
            phase1_results: Dictionary returned by solve_phase1.
            criterion: Optimality criterion –
                       ``'max_power'``        maximise W_dot_net,
                       ``'max_specific_work'`` maximise w_net per kg.

        Returns:
            Dictionary with keys: idx, rp, T2, T4, w_c, w_t, w_net, q_in,
            eta_th, bwr, m_dot, W_dot_net (kW).

        Raises:
            ValueError: If criterion is not recognised.
        """
        res = phase1_results
        if criterion == "max_power":
            idx = int(np.nanargmax(res["W_dot_net"]))
        elif criterion == "max_specific_work":
            idx = int(np.nanargmax(res["w_net_spec"]))
        else:
            raise ValueError(
                f"Unknown criterion '{criterion}'. "
                "Use 'max_power' or 'max_specific_work'."
            )
        return self.solve_point(float(res["rp_vals"][idx]), _idx=idx)

    def solve_point(self, rp: float, _idx: int | None = None) -> dict:
        """Compute the exact Phase 1 cycle state at a single pressure ratio.

        Args:
            rp: Pressure ratio.

        Returns:
            Dictionary with keys: idx (None unless supplied internally), rp,
            T2, T4, w_c, w_t, w_net, q_in, eta_th, bwr, m_dot, W_dot_net (kW).
        """
        T2 = fp.isentropic_outlet_T(self.T1, rp, compress=True)
        T4 = fp.isentropic_outlet_T(self.T3, rp, compress=False)
        w_c   = fp.delta_h(self.T1, T2)
        w_t   = fp.delta_h(T4, self.T3)
        w_net = w_t - w_c
        q_in  = fp.delta_h(T2, self.T3)
        m_dot = self.Q_dot_in / q_in
        return {
            "idx":       _idx,
            "rp":        rp,
            "T2":        T2,
            "T4":        T4,
            "w_c":       w_c,
            "w_t":       w_t,
            "w_net":     w_net,
            "q_in":      q_in,
            "eta_th":    w_net / q_in,
            "bwr":       w_c / w_t,
            "m_dot":     m_dot,
            "W_dot_net": m_dot * w_net,
        }

    # ── Phase 2 ───────────────────────────────────────────────────────────────

    def solve_phase2_contour(
        self,
        rp: float,
        eta_vals: np.ndarray,
    ) -> dict:
        """Vectorized Phase 2 contour grid over (η_C, η_T) efficiency pairs.

        For a fixed pressure ratio, sweeps a full 2-D efficiency grid without
        any nested Python loops.  The key vectorization:

            w_c_act = w_c_s / η_C_grid    (N × N)
            w_t_act = w_t_s × η_T_grid    (N × N)

        T2_act is approximated using the local average cp for full array support.
        q_in is then evaluated with the analytic polynomial antiderivative.
        Non-viable points (w_net ≤ 0 or q_in ≤ 0) are stored as np.nan.

        Args:
            rp:       Pressure ratio (sets the isentropic reference works w_c_s, w_t_s).
            eta_vals: 1-D array of isentropic efficiency values for both axes (0–1).

        Returns:
            Dictionary with keys:
                eta_C_grid  (N×N ndarray): Compressor efficiency values.
                eta_T_grid  (N×N ndarray): Turbine efficiency values.
                W_dot_grid  (N×N ndarray): Net power in MW (nan = not viable).
                eta_th_grid (N×N ndarray): Thermal efficiency in % (nan = not viable).
                m_dot_grid  (N×N ndarray): Mass flow rate in kg/s (nan = not viable).
                bwr_grid    (N×N ndarray): Back work ratio in % (nan = not viable).
                w_c_s  (float): Isentropic compressor work in kJ/kg.
                w_t_s  (float): Isentropic turbine work in kJ/kg.
                T2s    (float): Isentropic compressor outlet temperature in K.
                T4s    (float): Isentropic turbine outlet temperature in K.
        """
        T1, T3, Q_dot_in = self.T1, self.T3, self.Q_dot_in

        T2s = fp.isentropic_outlet_T(T1, rp, compress=True)
        T4s = fp.isentropic_outlet_T(T3, rp, compress=False)
        w_c_s = fp.delta_h(T1, T2s)
        w_t_s = fp.delta_h(T4s, T3)

        eta_C_grid, eta_T_grid = np.meshgrid(eta_vals, eta_vals)

        # Vectorized actual works (the two key lines)
        w_c_act = w_c_s / eta_C_grid   # (N, N)
        w_t_act = w_t_s * eta_T_grid   # (N, N)
        w_net   = w_t_act - w_c_act    # (N, N)

        # Approximate actual compressor outlet temperature (vectorized via avg cp)
        cp_avg = fp.cp((T1 + T2s) / 2.0)
        T2_act = np.clip(T1 + w_c_act / cp_avg, T1 + 1.0, T3 - 1.0)

        # q_in from analytic antiderivative — supports array T2_act
        q_in = fp.delta_h_analytic(T2_act, T3)

        # Feasibility mask: blank out non-viable regions
        viable = (w_net > 0) & (q_in > 0)
        w_net = np.where(viable, w_net, np.nan)
        q_in  = np.where(viable, q_in,  np.nan)

        m_dot    = Q_dot_in / q_in
        W_dot_MW = m_dot * w_net / 1000.0
        eta_th   = w_net / q_in * 100.0
        bwr      = np.where(viable, w_c_act / w_t_act * 100.0, np.nan)

        return {
            "eta_C_grid":  eta_C_grid,
            "eta_T_grid":  eta_T_grid,
            "W_dot_grid":  W_dot_MW,
            "eta_th_grid": eta_th,
            "m_dot_grid":  m_dot,
            "bwr_grid":    bwr,
            "w_c_s":  w_c_s,
            "w_t_s":  w_t_s,
            "T2s":    T2s,
            "T4s":    T4s,
        }

    def solve_phase2_sample(
        self,
        rp: float,
        eta_C: float,
        eta_T: float,
    ) -> dict:
        """Exact Phase 2 performance at a single (η_C, η_T) operating point.

        Uses brentq to find T2_act exactly (no approximation), suitable for
        reporting verification values.

        Args:
            rp:    Pressure ratio.
            eta_C: Isentropic compressor efficiency (0–1).
            eta_T: Isentropic turbine efficiency (0–1).

        Returns:
            Dictionary with keys: rp, T2s, T4s, w_c_s, w_t_s, w_c_act,
            w_t_act, T2_act, q_in, w_net, eta_th, bwr, m_dot, W_dot_net (kW).
        """
        T1, T3, Q_dot_in = self.T1, self.T3, self.Q_dot_in

        T2s = fp.isentropic_outlet_T(T1, rp, compress=True)
        T4s = fp.isentropic_outlet_T(T3, rp, compress=False)
        w_c_s = fp.delta_h(T1, T2s)
        w_t_s = fp.delta_h(T4s, T3)

        w_c_act = w_c_s / eta_C
        w_t_act = w_t_s * eta_T
        w_net   = w_t_act - w_c_act

        T2_act = fp.find_T_from_delta_h(T1, w_c_act, T1 + 0.1, 2000.0)
        q_in   = fp.delta_h(T2_act, T3)
        m_dot  = Q_dot_in / q_in

        return {
            "rp":        rp,
            "T2s":       T2s,
            "T4s":       T4s,
            "w_c_s":     w_c_s,
            "w_t_s":     w_t_s,
            "w_c_act":   w_c_act,
            "w_t_act":   w_t_act,
            "T2_act":    T2_act,
            "q_in":      q_in,
            "w_net":     w_net,
            "eta_th":    w_net / q_in,
            "bwr":       w_c_act / w_t_act,
            "m_dot":     m_dot,
            "W_dot_net": m_dot * w_net,
        }

    def solve_phase2_mean_power_search(
        self,
        rp_search: np.ndarray,
        eta_search: np.ndarray,
    ) -> dict:
        """Find the Phase 2 pressure ratio that maximises mean net power.

        For each rp, sweeps a coarse (η_C, η_T) grid and computes the mean net
        power over the full efficiency grid. Non-viable operating points are
        assigned zero net power before averaging, so the search favours rp
        values that perform well across the efficiency range without using a
        separate balance score.

        Args:
            rp_search:  1-D array of pressure ratios to search over.
            eta_search: 1-D array of efficiency values for the coarse grid.

        Returns:
            Dictionary with keys:
                rp_search   (np.ndarray): Pressure ratios (echoed from input).
                mean_Wdot   (np.ndarray): Mean net power for each rp in kW.
                rp_optimal  (float): rp that maximises mean_Wdot.
                idx_optimal (int): Index into rp_search for rp_optimal.
        """
        T1, T3, Q_dot_in = self.T1, self.T3, self.Q_dot_in

        eC, eT = np.meshgrid(eta_search, eta_search)
        mean_Wdot   = np.empty(len(rp_search))

        for k, rp in enumerate(rp_search):
            T2s = fp.isentropic_outlet_T(T1, rp, compress=True)
            T4s = fp.isentropic_outlet_T(T3, rp, compress=False)
            wcs = fp.delta_h(T1, T2s)
            wts = fp.delta_h(T4s, T3)

            wc_act = wcs / eC          # vectorized over grid
            wt_act = wts * eT
            wnet   = wt_act - wc_act

            # Scalar q_in approximation (isentropic T2 reference; fast)
            q_in_approx  = fp.delta_h(T2s, T3)
            Wdot         = np.where(wnet > 0, (Q_dot_in / q_in_approx) * wnet, 0.0)
            mean_Wdot[k] = np.mean(Wdot)

        idx_optimal = int(np.argmax(mean_Wdot))

        return {
            "rp_search":   rp_search,
            "mean_Wdot":   mean_Wdot,
            "rp_optimal":  float(rp_search[idx_optimal]),
            "idx_optimal": idx_optimal,
        }

    # ── Phase 3 ───────────────────────────────────────────────────────────────

    def solve_phase3(self, rp: float, fluid: str = "CO2") -> dict:
        """Ideal Brayton cycle with real-fluid properties from CoolProp.

        Compressor and turbine remain isentropic (η = 1).  Non-ideality comes
        entirely from the Span-Wagner equation of state for CO2.

        Args:
            rp:     Pressure ratio.
            fluid:  CoolProp fluid identifier (default "CO2").

        Returns:
            Dictionary with keys: rp, h1, s1, T2, h2, h3, s3, T4, h4,
            w_c, w_t, w_net, q_in, eta_th, bwr, m_dot, W_dot_net (kW).

        Raises:
            ImportError: If CoolProp is not installed.
        """
        states = fp.get_coolprop_states(self.T1, self.T3, self.P1, rp, fluid)

        w_c   = states["h2"] - states["h1"]
        w_t   = states["h3"] - states["h4"]
        w_net = w_t - w_c
        q_in  = states["h3"] - states["h2"]
        m_dot = self.Q_dot_in / q_in

        return {
            "rp": rp,
            **states,
            "w_c":       w_c,
            "w_t":       w_t,
            "w_net":     w_net,
            "q_in":      q_in,
            "eta_th":    w_net / q_in,
            "bwr":       w_c / w_t,
            "m_dot":     m_dot,
            "W_dot_net": m_dot * w_net,
        }

    # ── Phase 4 hook (regenerator) ────────────────────────────────────────────

    def solve_phase4_case(
        self,
        rp: float,
        delta_T_approach: float,
        fluid: str = "CO2",
    ) -> dict:
        """Solve one regenerated real-fluid Brayton-cycle case."""
        P1 = self.P1
        P2 = self.P1 * rp

        st1 = fp.coolprop_state_tp(self.T1, P1, fluid)
        st2 = fp.coolprop_state_ps(P2, st1["s"], fluid)
        st4 = fp.coolprop_state_tp(self.T3, P2, fluid)
        st5 = fp.coolprop_state_ps(P1, st4["s"], fluid)

        result = {
            "rp": rp,
            "delta_T_approach": delta_T_approach,
            "P1": P1,
            "P2": P2,
            "P3": P2,
            "P4": P2,
            "P5": P1,
            "P6": P1,
            "T1": st1["T"],
            "h1": st1["h"],
            "s1": st1["s"],
            "T2": st2["T"],
            "h2": st2["h"],
            "s2": st2["s"],
            "T4": st4["T"],
            "h4": st4["h"],
            "s4": st4["s"],
            "T5": st5["T"],
            "h5": st5["h"],
            "s5": st5["s"],
        }

        if st5["T"] <= st2["T"]:
            return {
                **result,
                "feasible": False,
                "reason": "turbine exhaust is not hotter than compressor outlet",
            }

        T3_regen = st5["T"] - delta_T_approach
        if T3_regen <= st2["T"]:
            return {
                **result,
                "feasible": False,
                "reason": "requested temperature approach eliminates useful regeneration",
            }

        st3 = fp.coolprop_state_tp(T3_regen, P2, fluid)
        q_regen = st3["h"] - st2["h"]
        if q_regen <= 0:
            return {
                **result,
                "feasible": False,
                "reason": "regenerator heat transfer is non-positive",
                "T3": st3["T"],
                "h3": st3["h"],
                "s3": st3["s"],
            }

        h6 = st5["h"] - q_regen
        st6 = fp.coolprop_state_ph(P1, h6, fluid)

        q_in = st4["h"] - st3["h"]
        w_c = st2["h"] - st1["h"]
        w_t = st4["h"] - st5["h"]
        w_net = w_t - w_c

        if q_in <= 0 or w_net <= 0:
            return {
                **result,
                "feasible": False,
                "reason": "cycle is not viable at this regenerator setting",
                "T3": st3["T"],
                "h3": st3["h"],
                "s3": st3["s"],
                "T6": st6["T"],
                "h6": st6["h"],
                "s6": st6["s"],
                "q_regen": q_regen,
            }

        m_dot = self.Q_dot_in / q_in
        Q_dot_regen = m_dot * q_regen

        return {
            **result,
            "feasible": True,
            "reason": "",
            "T3": st3["T"],
            "h3": st3["h"],
            "s3": st3["s"],
            "T6": st6["T"],
            "h6": st6["h"],
            "s6": st6["s"],
            "w_c": w_c,
            "w_t": w_t,
            "w_net": w_net,
            "q_in": q_in,
            "q_regen": q_regen,
            "Q_dot_regen": Q_dot_regen,
            "eta_th": w_net / q_in,
            "bwr": w_c / w_t,
            "m_dot": m_dot,
            "W_dot_net": m_dot * w_net,
        }

    def solve_phase4_sweep(
        self,
        rp: float,
        delta_T_approach_vals: np.ndarray,
        fluid: str = "CO2",
        return_cases: bool = True,
    ) -> dict:
        """Run a Phase 4 regenerator sweep over temperature approaches."""
        dT_vals = np.asarray(delta_T_approach_vals, dtype=float)

        P1 = self.P1
        P2 = self.P1 * rp
        st1 = fp.coolprop_state_tp(self.T1, P1, fluid)
        st2 = fp.coolprop_state_ps(P2, st1["s"], fluid)
        st4 = fp.coolprop_state_tp(self.T3, P2, fluid)
        st5 = fp.coolprop_state_ps(P1, st4["s"], fluid)

        w_c = st2["h"] - st1["h"]
        w_t = st4["h"] - st5["h"]
        w_net = w_t - w_c
        bwr = w_c / w_t

        cases: list[dict] = []
        feasible_list: list[bool] = []
        q_regen_list: list[float] = []
        q_in_list: list[float] = []
        q_dot_regen_list: list[float] = []
        eta_th_list: list[float] = []
        m_dot_list: list[float] = []
        w_dot_list: list[float] = []
        t2_list: list[float] = []
        t3_list: list[float] = []
        t5_list: list[float] = []
        t6_list: list[float] = []

        def append_values(
            *,
            feasible: bool,
            q_regen: float = np.nan,
            q_in: float = np.nan,
            q_dot_regen: float = np.nan,
            eta_th: float = np.nan,
            m_dot: float = np.nan,
            w_dot_net: float = np.nan,
            T2: float = np.nan,
            T3: float = np.nan,
            T5: float = np.nan,
            T6: float = np.nan,
        ) -> None:
            feasible_list.append(feasible)
            q_regen_list.append(q_regen)
            q_in_list.append(q_in)
            q_dot_regen_list.append(q_dot_regen)
            eta_th_list.append(eta_th)
            m_dot_list.append(m_dot)
            w_dot_list.append(w_dot_net)
            t2_list.append(T2)
            t3_list.append(T3)
            t5_list.append(T5)
            t6_list.append(T6)

        if st5["T"] <= st2["T"] or w_net <= 0:
            reason = (
                "turbine exhaust is not hotter than compressor outlet"
                if st5["T"] <= st2["T"]
                else "cycle is not viable at this regenerator setting"
            )
            base_case = {
                "rp": rp,
                "P1": P1,
                "P2": P2,
                "P3": P2,
                "P4": P2,
                "P5": P1,
                "P6": P1,
                "T1": st1["T"],
                "h1": st1["h"],
                "s1": st1["s"],
                "T2": st2["T"],
                "h2": st2["h"],
                "s2": st2["s"],
                "T4": st4["T"],
                "h4": st4["h"],
                "s4": st4["s"],
                "T5": st5["T"],
                "h5": st5["h"],
                "s5": st5["s"],
                "feasible": False,
                "reason": reason,
            }
            for dT in dT_vals:
                append_values(
                    feasible=False,
                    T2=st2["T"],
                    T5=st5["T"],
                )
                if return_cases:
                    cases.append({**base_case, "delta_T_approach": float(dT)})
        else:
            base_result = {
                "rp": rp,
                "P1": P1,
                "P2": P2,
                "P3": P2,
                "P4": P2,
                "P5": P1,
                "P6": P1,
                "T1": st1["T"],
                "h1": st1["h"],
                "s1": st1["s"],
                "T2": st2["T"],
                "h2": st2["h"],
                "s2": st2["s"],
                "T4": st4["T"],
                "h4": st4["h"],
                "s4": st4["s"],
                "T5": st5["T"],
                "h5": st5["h"],
                "s5": st5["s"],
                "w_c": w_c,
                "w_t": w_t,
                "w_net": w_net,
                "bwr": bwr,
            }

            # ── vectorized dT sweep ──────────────────────────────────────
            # Replaces the per-dT loop with 2–4 batch CoolProp calls.
            _PropsSI = fp._get_coolprop_propssi()
            P1_Pa = P1 * 1e3
            P2_Pa = P2 * 1e3
            n = len(dT_vals)

            # T3 is purely arithmetic — no CoolProp call needed
            T3_vals = st5["T"] - dT_vals                        # (n,)
            ok1 = T3_vals > st2["T"]                            # T3 above compressor outlet

            # Batch call: h3 for all feasible T3 values
            h3_arr = np.full(n, np.nan)
            if ok1.any():
                h3_arr[ok1] = (
                    _PropsSI("H", "T", T3_vals[ok1], "P", P2_Pa, fluid) / 1e3
                )

            q_regen_arr = h3_arr - st2["h"]        # NaN where ok1 is False
            ok2 = ok1 & (q_regen_arr > 0)

            h6_arr = np.full(n, np.nan)
            h6_arr[ok2] = st5["h"] - q_regen_arr[ok2]

            q_in_arr = st4["h"] - h3_arr           # NaN where h3 is NaN
            ok3 = ok2 & (q_in_arr > 0)

            # Batch call: T6 for all cases where q_regen is positive
            T6_arr = np.full(n, np.nan)
            if ok2.any():
                T6_arr[ok2] = _PropsSI(
                    "T", "P", P1_Pa, "H", h6_arr[ok2] * 1e3, fluid
                )

            # Metrics — computed only for fully feasible indices
            m_dot_arr = np.full(n, np.nan)
            m_dot_arr[ok3] = self.Q_dot_in / q_in_arr[ok3]

            q_dot_regen_arr = np.full(n, np.nan)
            q_dot_regen_arr[ok3] = m_dot_arr[ok3] * q_regen_arr[ok3]

            eta_th_arr = np.full(n, np.nan)
            eta_th_arr[ok3] = w_net / q_in_arr[ok3]

            w_dot_arr = np.full(n, np.nan)
            w_dot_arr[ok3] = m_dot_arr[ok3] * w_net

            # Bulk-extend the shared result lists
            # q_regen is reported for ok2 (includes q_in<=0 infeasible cases)
            feasible_list.extend(ok3.tolist())
            q_regen_list.extend(np.where(ok2, q_regen_arr, np.nan).tolist())
            q_in_list.extend(np.where(ok3, q_in_arr, np.nan).tolist())
            q_dot_regen_list.extend(q_dot_regen_arr.tolist())
            eta_th_list.extend(eta_th_arr.tolist())
            m_dot_list.extend(m_dot_arr.tolist())
            w_dot_list.extend(w_dot_arr.tolist())
            t2_list.extend([st2["T"]] * n)
            t3_list.extend(np.where(ok1, T3_vals, np.nan).tolist())
            t5_list.extend([st5["T"]] * n)
            t6_list.extend(T6_arr.tolist())

            if return_cases:
                # Two extra batch calls for entropy (needed in full state dicts)
                s3_arr = np.full(n, np.nan)
                if ok1.any():
                    s3_arr[ok1] = (
                        _PropsSI("S", "T", T3_vals[ok1], "P", P2_Pa, fluid) / 1e3
                    )
                s6_arr = np.full(n, np.nan)
                if ok2.any():
                    s6_arr[ok2] = (
                        _PropsSI("S", "P", P1_Pa, "H", h6_arr[ok2] * 1e3, fluid)
                        / 1e3
                    )
                for i, dT in enumerate(dT_vals):
                    entry = {**base_result, "delta_T_approach": float(dT)}
                    if not ok1[i]:
                        cases.append({
                            **entry,
                            "feasible": False,
                            "reason": "requested temperature approach eliminates useful regeneration",
                        })
                    elif not ok2[i]:
                        cases.append({
                            **entry,
                            "feasible": False,
                            "reason": "regenerator heat transfer is non-positive",
                            "T3": float(T3_vals[i]),
                            "h3": float(h3_arr[i]),
                            "s3": float(s3_arr[i]),
                        })
                    elif not ok3[i]:
                        cases.append({
                            **entry,
                            "feasible": False,
                            "reason": "cycle is not viable at this regenerator setting",
                            "T3": float(T3_vals[i]),
                            "h3": float(h3_arr[i]),
                            "s3": float(s3_arr[i]),
                            "T6": float(T6_arr[i]),
                            "h6": float(h6_arr[i]),
                            "s6": float(s6_arr[i]),
                            "q_regen": float(q_regen_arr[i]),
                        })
                    else:
                        m_i = float(m_dot_arr[i])
                        qr_i = float(q_regen_arr[i])
                        qi_i = float(q_in_arr[i])
                        cases.append({
                            **entry,
                            "feasible": True,
                            "reason": "",
                            "T3": float(T3_vals[i]),
                            "h3": float(h3_arr[i]),
                            "s3": float(s3_arr[i]),
                            "T6": float(T6_arr[i]),
                            "h6": float(h6_arr[i]),
                            "s6": float(s6_arr[i]),
                            "q_in": qi_i,
                            "q_regen": qr_i,
                            "Q_dot_regen": m_i * qr_i,
                            "eta_th": w_net / qi_i,
                            "m_dot": m_i,
                            "W_dot_net": m_i * w_net,
                        })

        result = {
            "rp": rp,
            "delta_T_approach_vals": dT_vals,
            "feasible": np.array(feasible_list, dtype=bool),
            "Q_dot_regen": np.array(q_dot_regen_list, dtype=float),
            "eta_th": np.array(eta_th_list, dtype=float),
            "m_dot": np.array(m_dot_list, dtype=float),
            "W_dot_net": np.array(w_dot_list, dtype=float),
            "q_in": np.array(q_in_list, dtype=float),
            "q_regen": np.array(q_regen_list, dtype=float),
            "T2": np.array(t2_list, dtype=float),
            "T3": np.array(t3_list, dtype=float),
            "T5": np.array(t5_list, dtype=float),
            "T6": np.array(t6_list, dtype=float),
        }
        if return_cases:
            result["cases"] = cases
        return result

    def solve_phase4_placeholder(self, rp: float, delta_T_approach: float = 60.0, fluid: str = "CO2") -> dict:
        """[Placeholder] Brayton cycle with an ideal recuperator/regenerator.

        Args:
            rp:      Pressure ratio.
            epsilon: Regenerator effectiveness (0–1).

        Raises:
            NotImplementedError: Until Phase 4 is implemented.
        """
        raise NotImplementedError("Phase 4 (regenerator) is not yet implemented.")

    def solve_phase4(
        self,
        rp: float,
        delta_T_approach: float = 60.0,
        fluid: str = "CO2",
    ) -> dict:
        """Compatibility wrapper for a single Phase 4 regenerator case."""
        return self.solve_phase4_case(rp, delta_T_approach, fluid=fluid)

    # ── Phase 5 hook (exergy) ─────────────────────────────────────────────────

    def solve_phase5_exergy_case(
        self,
        phase4_case: dict,
        T_dead: float = 298.15,
        P_dead: float = 100.0,
        fluid: str = "CO2",
    ) -> dict:
        """Perform exergy analysis for one selected regenerated Phase 4 case."""
        if not phase4_case.get("feasible", False):
            raise ValueError("Phase 5 requires a feasible Phase 4 case.")

        dead_state = fp.coolprop_state_tp(T_dead, P_dead, fluid)
        h0 = dead_state["h"]
        s0 = dead_state["s"]
        m_dot = phase4_case["m_dot"]

        stream_exergy: dict[str, float] = {}
        stream_exergy_rate: dict[str, float] = {}
        for idx in range(1, 7):
            h_i = phase4_case[f"h{idx}"]
            s_i = phase4_case[f"s{idx}"]
            x_i = (h_i - h0) - T_dead * (s_i - s0)
            stream_exergy[f"x{idx}"] = x_i
            stream_exergy_rate[f"X_dot{idx}"] = m_dot * x_i

        component_delta_x = {
            "compressor": stream_exergy["x2"] - stream_exergy["x1"],
            "heater": stream_exergy["x4"] - stream_exergy["x3"],
            "turbine": stream_exergy["x4"] - stream_exergy["x5"],
            "regenerator_cold": stream_exergy["x3"] - stream_exergy["x2"],
            "regenerator_hot": stream_exergy["x5"] - stream_exergy["x6"],
            "cooler": stream_exergy["x6"] - stream_exergy["x1"],
        }
        component_delta_xdot = {
            name: m_dot * value
            for name, value in component_delta_x.items()
        }

        return {
            "dead_state": {
                "T0": T_dead,
                "P0": P_dead,
                "h0": h0,
                "s0": s0,
            },
            "stream_exergy": stream_exergy,
            "stream_exergy_rate": stream_exergy_rate,
            "component_summary": {
                "delta_x": component_delta_x,
                "delta_X_dot": component_delta_xdot,
            },
        }

    def solve_phase5_exergy_placeholder(
        self,
        phase4_case: dict,
        T_dead: float = 298.15,
        P_dead: float = 100.0,
        fluid: str = "CO2",
    ) -> dict:
        """[Placeholder] Exergy analysis of the Brayton cycle."""
        raise NotImplementedError("Phase 5 (exergy analysis) is not yet implemented.")

    def solve_phase5_exergy(
        self,
        phase4_case: dict,
        T_dead: float = 298.15,
        P_dead: float = 100.0,
        fluid: str = "CO2",
    ) -> dict:
        """Compatibility wrapper for a selected Phase 4 exergy case."""
        return self.solve_phase5_exergy_case(
            phase4_case,
            T_dead=T_dead,
            P_dead=P_dead,
            fluid=fluid,
        )
