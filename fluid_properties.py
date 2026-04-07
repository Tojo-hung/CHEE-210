"""
fluid_properties.py
CO2 thermodynamic property functions.

Two integration strategies are provided:
    * Numerical (scipy.integrate.quad) – high accuracy for scalar inputs.
    * Analytic polynomial antiderivative  – supports NumPy array inputs for
      fully vectorized Phase 2 grid sweeps.

CoolProp wrappers for real-fluid (Phase 3) state-point computation are also
included, guarded by a try/import so the module loads without CoolProp.
"""

from __future__ import annotations

from functools import lru_cache
import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
from typing import Union

from constants import CO2_A, CO2_B, CO2_C, CO2_D, M_CO2, R_CO2

# Type alias for functions that accept either a scalar or an ndarray
_Numeric = Union[float, np.ndarray]


# ── Polynomial cp ─────────────────────────────────────────────────────────────

def cp_molar(T: _Numeric) -> _Numeric:
    """Molar isobaric heat capacity of CO2 from the Table A-2c polynomial.

    Args:
        T: Temperature in Kelvin (scalar or ndarray). Valid range 273–1800 K.

    Returns:
        Molar cp in kJ/(kmol·K).
    """
    return CO2_A + CO2_B * T + CO2_C * T**2 + CO2_D * T**3


def cp(T: _Numeric) -> _Numeric:
    """Specific isobaric heat capacity of CO2.

    Args:
        T: Temperature in Kelvin (scalar or ndarray). Valid range 273–1800 K.

    Returns:
        Specific cp in kJ/(kg·K).
    """
    return cp_molar(T) / M_CO2


# ── Enthalpy ──────────────────────────────────────────────────────────────────

def delta_h(T_low: float, T_high: float) -> float:
    """Specific enthalpy difference h(T_high) - h(T_low) via numerical quadrature.

    Uses scipy.integrate.quad for high accuracy. Suitable for scalar inputs
    only (e.g., single-point verification and root-finding).

    Args:
        T_low:  Lower temperature in Kelvin.
        T_high: Upper temperature in Kelvin.

    Returns:
        Enthalpy difference in kJ/kg.
    """
    val, _ = quad(cp, T_low, T_high)
    return val


def delta_h_analytic(T_low: _Numeric, T_high: _Numeric) -> _Numeric:
    """Specific enthalpy difference via the analytic polynomial antiderivative.

    Because cp(T) is a cubic polynomial, its antiderivative H(T) is a quartic
    that can be evaluated exactly.  This function accepts NumPy arrays for
    fully vectorized Phase 2 grid computations (no loops required).

    Args:
        T_low:  Lower temperature in Kelvin (scalar or ndarray).
        T_high: Upper temperature in Kelvin (scalar or ndarray).

    Returns:
        Enthalpy difference in kJ/kg (same shape as inputs).
    """
    def H(T: _Numeric) -> _Numeric:
        return (
            CO2_A * T
            + CO2_B / 2.0 * T**2
            + CO2_C / 3.0 * T**3
            + CO2_D / 4.0 * T**4
        ) / M_CO2

    return H(T_high) - H(T_low)


# ── Entropy function ──────────────────────────────────────────────────────────

def delta_s0(T_low: float, T_high: float) -> float:
    """Entropy function difference s°(T_high) - s°(T_low) via numerical quadrature.

    Defined as ∫_{T_low}^{T_high} cp(T)/T dT.  Used in the isentropic condition.

    Args:
        T_low:  Lower temperature in Kelvin.
        T_high: Upper temperature in Kelvin.

    Returns:
        Entropy function difference in kJ/(kg·K).
    """
    val, _ = quad(lambda T: cp(T) / T, T_low, T_high)
    return val


# ── Isentropic state-point solvers ────────────────────────────────────────────

def isentropic_outlet_T(T_in: float, rp: float, compress: bool = True) -> float:
    """Find the isentropic outlet temperature for a compressor or turbine.

    Solves: ∫_{T_in}^{T_out} cp(T)/T dT = ±R_CO2 · ln(rp)
    using scipy.optimize.brentq.

    Args:
        T_in:     Inlet temperature in Kelvin.
        rp:       Pressure ratio (P_high / P_low, always > 1).
        compress: True → compression (T_out > T_in, target = +R·ln rp).
                  False → expansion  (T_out < T_in, target = -R·ln rp).

    Returns:
        Isentropic outlet temperature in Kelvin.
    """
    if compress:
        target = R_CO2 * np.log(rp)
        bracket = (T_in + 0.1, 2000.0)
    else:
        target = -R_CO2 * np.log(rp)
        bracket = (200.0, T_in - 0.1)

    f = lambda T_out: delta_s0(T_in, T_out) - target
    return brentq(f, bracket[0], bracket[1], xtol=1e-4)


def find_T_from_delta_h(
    T_ref: float,
    dh_target: float,
    T_lo: float,
    T_hi: float,
) -> float:
    """Find temperature T such that delta_h(T_ref, T) = dh_target.

    Uses scipy.optimize.brentq.  Needed to recover the actual compressor outlet
    temperature T2' given the actual (non-isentropic) compressor work.

    Args:
        T_ref:     Reference temperature (lower bound of the enthalpy integral) in K.
        dh_target: Target enthalpy difference in kJ/kg.
        T_lo:      Lower bracket temperature in Kelvin.
        T_hi:      Upper bracket temperature in Kelvin.

    Returns:
        Temperature in Kelvin.
    """
    f = lambda T: delta_h(T_ref, T) - dh_target
    return brentq(f, T_lo, T_hi, xtol=1e-4)


# ── CoolProp wrappers (Phase 3) ───────────────────────────────────────────────

def _get_coolprop_propssi():
    """Import and return CoolProp.PropsSI with a helpful real-fluid error."""
    try:
        from CoolProp.CoolProp import PropsSI
    except ImportError as exc:
        raise ImportError(
            "CoolProp is required for real-fluid phases. Install with: pip install CoolProp"
        ) from exc
    return PropsSI


@lru_cache(maxsize=4096)
def _coolprop_state_tp_cached(T: float, P_kPa: float, fluid: str) -> tuple[float, float, float, float]:
    """Cached real-fluid state from temperature and pressure."""
    PropsSI = _get_coolprop_propssi()
    P_Pa = P_kPa * 1e3
    h = PropsSI("H", "T", T, "P", P_Pa, fluid) / 1e3
    s = PropsSI("S", "T", T, "P", P_Pa, fluid) / 1e3
    return T, P_kPa, h, s


@lru_cache(maxsize=4096)
def _coolprop_state_ps_cached(P_kPa: float, s: float, fluid: str) -> tuple[float, float, float, float]:
    """Cached real-fluid state from pressure and entropy."""
    PropsSI = _get_coolprop_propssi()
    P_Pa = P_kPa * 1e3
    T = PropsSI("T", "P", P_Pa, "S", s * 1e3, fluid)
    h = PropsSI("H", "T", T, "P", P_Pa, fluid) / 1e3
    return T, P_kPa, h, s


@lru_cache(maxsize=4096)
def _coolprop_state_ph_cached(P_kPa: float, h: float, fluid: str) -> tuple[float, float, float, float]:
    """Cached real-fluid state from pressure and enthalpy."""
    PropsSI = _get_coolprop_propssi()
    P_Pa = P_kPa * 1e3
    T = PropsSI("T", "P", P_Pa, "H", h * 1e3, fluid)
    s = PropsSI("S", "T", T, "P", P_Pa, fluid) / 1e3
    return T, P_kPa, h, s


def coolprop_state_tp(
    T: float,
    P_kPa: float,
    fluid: str = "CO2",
) -> dict:
    """Return a real-fluid state from temperature and pressure."""
    T, P_kPa, h, s = _coolprop_state_tp_cached(T, P_kPa, fluid)
    return {
        "T": T,
        "P": P_kPa,
        "h": h,
        "s": s,
    }


def coolprop_state_ps(
    P_kPa: float,
    s: float,
    fluid: str = "CO2",
) -> dict:
    """Return a real-fluid state from pressure and entropy."""
    T, P_kPa, h, s = _coolprop_state_ps_cached(P_kPa, s, fluid)
    return {
        "T": T,
        "P": P_kPa,
        "h": h,
        "s": s,
    }


def coolprop_state_ph(
    P_kPa: float,
    h: float,
    fluid: str = "CO2",
) -> dict:
    """Return a real-fluid state from pressure and enthalpy."""
    T, P_kPa, h, s = _coolprop_state_ph_cached(P_kPa, h, fluid)
    return {
        "T": T,
        "P": P_kPa,
        "h": h,
        "s": s,
    }


def coolprop_isobaric_ts_path(
    P_kPa: float,
    T_start: float,
    T_end: float,
    fluid: str = "CO2",
    n_points: int = 80,
) -> dict:
    """Sample a real-fluid isobaric path for T-s plotting."""
    T_vals = np.linspace(T_start, T_end, n_points)
    states = [coolprop_state_tp(T, P_kPa, fluid) for T in T_vals]
    return {
        "T": np.array([state["T"] for state in states], dtype=float),
        "s": np.array([state["s"] for state in states], dtype=float),
    }


def coolprop_isentropic_ts_path(
    P_start_kPa: float,
    P_end_kPa: float,
    s: float,
    fluid: str = "CO2",
    n_points: int = 80,
) -> dict:
    """Sample a real-fluid isentropic path for T-s plotting."""
    P_vals = np.linspace(P_start_kPa, P_end_kPa, n_points)
    states = [coolprop_state_ps(P, s, fluid) for P in P_vals]
    return {
        "T": np.array([state["T"] for state in states], dtype=float),
        "s": np.array([state["s"] for state in states], dtype=float),
    }


def get_coolprop_states(
    T1: float,
    T3: float,
    P1_kPa: float,
    rp: float,
    fluid: str = "CO2",
) -> dict:
    """Compute isentropic cycle state points using CoolProp (real-fluid EOS).

    Machines are assumed isentropic (η = 1).  Non-ideality comes entirely from
    the fluid's equation of state (Span-Wagner for CO2).

    Args:
        T1:      Compressor inlet temperature in Kelvin.
        T3:      Turbine inlet temperature in Kelvin.
        P1_kPa:  Compressor inlet pressure in kPa.
        rp:      Pressure ratio (P2 / P1).
        fluid:   CoolProp fluid identifier (default "CO2").

    Returns:
        Dictionary with keys (all in kJ/kg or kJ/(kg·K)):
            h1, s1  – State 1 enthalpy and entropy.
            T2, h2  – Isentropic compressor outlet temperature and enthalpy.
            h3, s3  – State 3 enthalpy and entropy.
            T4, h4  – Isentropic turbine outlet temperature and enthalpy.

    Raises:
        ImportError: If CoolProp is not installed.
    """
    P2_kPa = P1_kPa * rp

    st1 = coolprop_state_tp(T1, P1_kPa, fluid)
    st2 = coolprop_state_ps(P2_kPa, st1["s"], fluid)
    st3 = coolprop_state_tp(T3, P2_kPa, fluid)
    st4 = coolprop_state_ps(P1_kPa, st3["s"], fluid)

    return {
        "h1": st1["h"], "s1": st1["s"],
        "T2": st2["T"], "h2": st2["h"],
        "h3": st3["h"], "s3": st3["s"],
        "T4": st4["T"], "h4": st4["h"],
    }
