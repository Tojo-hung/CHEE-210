"""
constants.py
Physical constants and default operating conditions for the Brayton cycle project.
"""

# ── CO2 Table A-2c polynomial coefficients (Cengel & Boles) ──────────────────
# cp_bar(T) = CO2_A + CO2_B*T + CO2_C*T^2 + CO2_D*T^3  [kJ/(kmol·K)], T in K
# Valid range: 273–1800 K
CO2_A: float = 22.26
CO2_B: float = 5.981e-2
CO2_C: float = -3.501e-5
CO2_D: float = 7.469e-9

# ── Molar mass and gas constants ──────────────────────────────────────────────
M_CO2: float = 44.01            # kg/kmol
R_UNIVERSAL: float = 8.314      # kJ/(kmol·K)
R_CO2: float = R_UNIVERSAL / M_CO2  # kJ/(kg·K) ≈ 0.18893

# ── Default cycle operating conditions ───────────────────────────────────────
T_INLET: float = 298.15         # K   compressor inlet temperature  (T1)
T_MAX: float = 1073.15          # K   turbine inlet temperature      (T3)
P_INLET: float = 100.0          # kPa compressor inlet pressure      (P1)
P_MAX_CYCLE: float = 6_000.0    # kPa maximum allowable cycle pressure
Q_DOT_IN: float = 10_000.0      # kW  total heat input rate

# ── Matplotlib style (Computer Modern / LaTeX default serif) ─────────────────
MPL_STYLE: dict = {
    "font.family":      "serif",
    "mathtext.fontset": "cm",
    "font.serif":       ["cmr10", "Computer Modern Roman", "DejaVu Serif"],
    "xtick.labelsize":  10,
    "ytick.labelsize":  10,
}
