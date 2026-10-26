"""
config.py
---------
Canonical configuration for the Springer major-revision pipeline:
Channel-Aware Adaptive Hybrid RF/FSO/THz Maritime Wireless Networking

This file is the single source of truth for:
- scenario ranges,
- physical channel parameters,
- QoS constraints,
- reproducibility seeds,
- dataset sweep settings.

Scientific parameters are only populated when supported by the current
implementation/report. The two FSO geometry parameters that were previously
hidden inside an effective optical gain are deliberately left UNSET and must be
defined before the revised dataset is generated.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


# ============================================================
# 1) Reproducibility
# ============================================================

MASTER_SEED = 2026

# Independent experiment seeds for repeated ML/RL runs.
EXPERIMENT_SEEDS = [11, 22, 33, 44, 55, 66, 77, 88, 99, 110]


# ============================================================
# 2) Dataset sweep
# ============================================================

DISTANCE_MIN_M = 10.0
DISTANCE_MAX_M = 4000.0
DISTANCE_STEP_M = 10.0


# ============================================================
# 3) Scenario ranges
# ============================================================

SCENARIO_RANGES = {
    "clear": {
        "rain_rate_mmph": (0.0, 0.0),
        "humidity_pct": (35.0, 45.0),
        "visibility_km": (15.0, 25.0),
        "Cn2": (1e-15, 1e-15),
    },
    "rain": {
        "rain_rate_mmph": (25.0, 50.0),
        "humidity_pct": (60.0, 80.0),
        "visibility_km": (3.0, 7.0),
        "Cn2": (5e-15, 5e-15),
    },
    "fog": {
        "rain_rate_mmph": (5.0, 20.0),
        "humidity_pct": (85.0, 95.0),
        "visibility_km": (0.3, 1.0),
        "Cn2": (1e-14, 1e-14),
    },
    "worst": {
        "rain_rate_mmph": (50.0, 80.0),
        "humidity_pct": (90.0, 98.0),
        "visibility_km": (0.1, 0.4),
        "Cn2": (5e-14, 5e-14),
    },
}


# ============================================================
# 4) RF parameters
# ============================================================

RF_FREQUENCY_HZ = 5e9
RF_BANDWIDTH_HZ = 20e6
RF_TX_POWER_DBM = 20.0
RF_NOISE_FIGURE_DB = 5.0

RF_RAIN_K = 2.162e-4
RF_RAIN_ALPHA = 1.6969
RF_SHADOW_SIGMA_DB = 2.0


# ============================================================
# 5) FSO parameters
# ============================================================

FSO_WAVELENGTH_M = 1550e-9
FSO_BANDWIDTH_HZ = 1e9
FSO_TX_POWER_DBM = 10.0
FSO_NOISE_FIGURE_DB = 3.0

# Visibility convention used by the revised P.1814-1 implementation.
FSO_VISIBILITY_DEFINITION = "MOR_5pct"

# Rain DSD shape parameter used by the P.1814-1 optical rain model.
FSO_RAIN_DSD_MU = 0

# No independent pointing/misalignment loss is assumed unless explicitly set.
FSO_SYSTEM_LOSS_DB = 0.0

# IMPORTANT:
# These were previously hidden inside an empirical effective optical gain.
# They must be supplied from the intended FSO terminal specification/literature
# before dataset regeneration.
FSO_BEAM_DIVERGENCE_MRAD: Optional[float] = 2.0
FSO_RECEIVER_DIAMETER_M: Optional[float] = 0.20


# ============================================================
# 6) THz parameters
# ============================================================

THZ_FREQUENCY_HZ = 300e9
THZ_BANDWIDTH_HZ = 5e9
THZ_TX_POWER_DBM = 15.0
THZ_NOISE_FIGURE_DB = 7.0

THZ_ABSORPTION_A0_DB_PER_KM = 0.1
THZ_ABSORPTION_A1_DB_PER_KM_PER_PCT = 0.03
THZ_DIRECTIONAL_GAIN_DB = 65.0


# ============================================================
# 7) QoS / reference-policy constraints
# ============================================================

MIN_CAPACITY_BPS = 1e5   # 100 kbps
MIN_SNR_DB = 0.0

CHANNEL_NAMES = {0: "RF", 1: "FSO", 2: "THz"}


# ============================================================
# 8) Validation
# ============================================================

def validate_config(require_fso_geometry: bool = True) -> None:
    """
    Validate configuration before dataset generation.

    Set require_fso_geometry=False for code paths that do not yet invoke
    the revised FSO geometrical model.
    """
    if DISTANCE_MIN_M <= 0:
        raise ValueError("DISTANCE_MIN_M must be > 0.")
    if DISTANCE_MAX_M < DISTANCE_MIN_M:
        raise ValueError("DISTANCE_MAX_M must be >= DISTANCE_MIN_M.")
    if DISTANCE_STEP_M <= 0:
        raise ValueError("DISTANCE_STEP_M must be > 0.")

    if require_fso_geometry:
        if FSO_BEAM_DIVERGENCE_MRAD is None:
            raise ValueError(
                "FSO_BEAM_DIVERGENCE_MRAD is not defined. "
                "Set it from the adopted FSO terminal specification before "
                "regenerating the revised dataset."
            )
        if FSO_RECEIVER_DIAMETER_M is None:
            raise ValueError(
                "FSO_RECEIVER_DIAMETER_M is not defined. "
                "Set it from the adopted FSO terminal specification before "
                "regenerating the revised dataset."
            )
        if FSO_BEAM_DIVERGENCE_MRAD <= 0:
            raise ValueError("FSO_BEAM_DIVERGENCE_MRAD must be > 0.")
        if FSO_RECEIVER_DIAMETER_M <= 0:
            raise ValueError("FSO_RECEIVER_DIAMETER_M must be > 0.")
