"""
fso_channel_vectorized_v2.py
----------------------------
Revised FSO propagation model for the Springer major revision.

Main changes relative to the previous implementation:
    1) Removes the empirical visibility "severity factor".
    2) Removes the arbitrary 10 dB clipping of scintillation loss.
    3) Replaces the optical-FSPL-minus-effective-gain approximation with
       explicit geometrical attenuation based on transmitter beam divergence
       and receiver aperture.
    4) Uses the visibility-dependent suspended-particle attenuation model
       recommended in ITU-R P.1814-1 for 0.4–1.55 µm.
    5) Optionally includes rain attenuation using the ITU-R P.1814-1
       power-law model.
    6) Uses the ITU-R P.1814-1 weak-turbulence scintillation expression.
    7) Keeps system-dependent losses (misalignment, optical losses, beam
       wander, etc.) explicit through system_loss_db instead of hiding them
       inside an effective gain.

IMPORTANT
---------
- beam_divergence_mrad and receiver_diameter_m are explicit physical system
  parameters and MUST be supplied by the experiment configuration.
- system_loss_db is set to 0 dB by default. Therefore, pointing/misalignment
  is NOT independently modelled unless a non-zero value is explicitly used.
- The deterministic sum A_sp + A_rain is used here to create scenario-level
  synthetic samples. ITU-R P.1814-1 also provides a statistical CCDF-based
  framework for long-term link-availability planning.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------
# 1) Geometrical attenuation
# ---------------------------------------------------------------------

def geometrical_attenuation_db(
    d_km,
    beam_divergence_mrad: float,
    receiver_diameter_m: float,
):
    """
    Geometrical attenuation following the area-ratio formulation:

        A_geo = 10 log10(S_d / S_capture)

    where

        S_d       = (pi/4) (d * theta)^2
        S_capture = pi (D_r/2)^2

    With d in km and theta in mrad, d*theta is directly in metres because
    1 km * 1 mrad = 1 m.

    If the receiver aperture is larger than the beam footprint, A_geo = 0 dB.
    """
    d_km = np.asarray(d_km, dtype=float)

    if beam_divergence_mrad <= 0:
        raise ValueError("beam_divergence_mrad must be > 0.")
    if receiver_diameter_m <= 0:
        raise ValueError("receiver_diameter_m must be > 0.")

    beam_diameter_m = d_km * float(beam_divergence_mrad)
    S_d = (np.pi / 4.0) * beam_diameter_m**2
    S_capture = np.pi * (float(receiver_diameter_m) / 2.0) ** 2

    ratio = np.maximum(S_d / S_capture, 1.0)
    return 10.0 * np.log10(ratio)


# ---------------------------------------------------------------------
# 2) Suspended-particle attenuation from visibility
# ---------------------------------------------------------------------

def _q_itu_p1814(V_2pct_km):
    """
    Wavelength exponent q from ITU-R P.1814-1.

    V is the visibility associated with the 2% threshold.
    """
    V = np.asarray(V_2pct_km, dtype=float)

    q = np.zeros_like(V, dtype=float)

    q = np.where(V > 50.0, 1.6, q)
    q = np.where((V > 6.0) & (V <= 50.0), 1.3, q)
    q = np.where((V >= 1.0) & (V <= 6.0), 0.16 * V + 0.34, q)
    q = np.where((V >= 0.5) & (V < 1.0), V - 0.5, q)
    q = np.where(V < 0.5, 0.0, q)

    return q


def suspended_particle_specific_attenuation_db_per_km(
    visibility_km,
    wavelength_m: float = 1550e-9,
    visibility_definition: str = "MOR_5pct",
):
    """
    Specific attenuation due to suspended particles.

    For 0.4 µm <= lambda <= 1.55 µm:

        gamma_sp(lambda) =
            17 / V_2% * (0.55 µm / lambda)^q       [dB/km]

    Visibility handling
    -------------------
    ITU-R P.1814-1 expresses the wavelength-dependent relation using
    visibility defined at the 2% threshold.

    If the supplied visibility is a standard MOR value based on the WMO 5%
    threshold, it is converted as:

        V_2% = 1.31 * V_5%

    Set visibility_definition="2pct" if the supplied values already correspond
    to the 2% threshold.
    """
    V = np.asarray(visibility_km, dtype=float)
    if np.any(V <= 0):
        raise ValueError("visibility_km must be > 0.")

    lambda_um = float(wavelength_m) * 1e6
    if not (0.4 <= lambda_um <= 1.55):
        raise ValueError(
            "This ITU-R P.1814-1 visibility model is used here only for "
            "0.4 µm <= wavelength <= 1.55 µm."
        )

    definition = visibility_definition.strip().lower()
    if definition in {"mor_5pct", "5pct", "mor"}:
        V_2pct = 1.31 * V
    elif definition in {"2pct", "v2"}:
        V_2pct = V
    else:
        raise ValueError(
            "visibility_definition must be 'MOR_5pct' or '2pct'."
        )

    q = _q_itu_p1814(V_2pct)

    gamma = (17.0 / V_2pct) * (0.55 / lambda_um) ** q
    return gamma


def suspended_particle_path_attenuation_db(
    d_km,
    visibility_km,
    wavelength_m: float = 1550e-9,
    visibility_definition: str = "MOR_5pct",
):
    """A_sp = gamma_sp * L."""
    d_km = np.asarray(d_km, dtype=float)
    gamma = suspended_particle_specific_attenuation_db_per_km(
        visibility_km=visibility_km,
        wavelength_m=wavelength_m,
        visibility_definition=visibility_definition,
    )
    return gamma * d_km


# ---------------------------------------------------------------------
# 3) Rain attenuation
# ---------------------------------------------------------------------

# ITU-R P.1814-1 Table values for gamma_rain = k R^alpha.
_RAIN_COEFFICIENTS = {
    -2: (2.2838, 0.4050),
    -1: (1.5921, 0.5506),
     0: (1.2924, 0.6436),
     1: (1.1394, 0.7057),
     2: (1.0505, 0.7497),
}


def rain_specific_attenuation_db_per_km(
    rain_rate_mmph,
    dsd_shape_mu: int = 0,
):
    """
    Specific optical rain attenuation:

        gamma_rain = k R^alpha     [dB/km]

    Coefficients are selected from ITU-R P.1814-1 for the requested Gamma-DSD
    shape parameter mu in {-2, -1, 0, 1, 2}.

    mu=0 is used as the default explicit modelling assumption.
    """
    if dsd_shape_mu not in _RAIN_COEFFICIENTS:
        raise ValueError("dsd_shape_mu must be one of {-2,-1,0,1,2}.")

    R = np.asarray(rain_rate_mmph, dtype=float)
    R = np.maximum(R, 0.0)

    k_rain, alpha_rain = _RAIN_COEFFICIENTS[dsd_shape_mu]
    gamma = k_rain * np.power(R, alpha_rain)
    return np.where(R > 0.0, gamma, 0.0)


def rain_path_attenuation_db(
    d_km,
    rain_rate_mmph,
    dsd_shape_mu: int = 0,
):
    """
    Scenario-level deterministic approximation:

        A_rain = gamma_rain * L

    This is suitable for the controlled synthetic scenario snapshots used in
    the present framework. It is not the full long-term CCDF procedure.
    """
    d_km = np.asarray(d_km, dtype=float)
    gamma = rain_specific_attenuation_db_per_km(
        rain_rate_mmph=rain_rate_mmph,
        dsd_shape_mu=dsd_shape_mu,
    )
    return gamma * d_km


# ---------------------------------------------------------------------
# 4) Scintillation
# ---------------------------------------------------------------------

def scintillation_attenuation_db(
    d_km,
    Cn2,
    wavelength_m: float = 1550e-9,
):
    """
    Weak-turbulence scintillation attenuation from ITU-R P.1814-1.

    Log-amplitude variance:

        sigma_chi^2 =
            23.17 * k^(7/6) * Cn2 * L^(11/6)       [dB^2]

        k = 2*pi/lambda

    The Recommendation states that attenuation due to scintillation can be
    represented as:

        A_scint = 2 * sigma_chi                  [dB]

    No arbitrary clipping is applied.

    Note:
    For strong turbulence, the weak-turbulence relation may overestimate the
    validity range because variance saturation can occur. This limitation must
    be reported rather than hidden by numerical clipping.
    """
    d_km = np.asarray(d_km, dtype=float)
    L_m = d_km * 1000.0

    Cn2 = np.asarray(Cn2, dtype=float)
    if np.any(Cn2 < 0):
        raise ValueError("Cn2 must be >= 0.")

    k_wave = 2.0 * np.pi / float(wavelength_m)

    sigma_chi_sq = (
        23.17
        * (k_wave ** (7.0 / 6.0))
        * Cn2
        * (L_m ** (11.0 / 6.0))
    )

    sigma_chi_sq = np.maximum(sigma_chi_sq, 0.0)
    sigma_chi = np.sqrt(sigma_chi_sq)

    return 2.0 * sigma_chi


# ---------------------------------------------------------------------
# 5) Total deterministic FSO attenuation
# ---------------------------------------------------------------------

def fso_atten_db(
    d_km,
    visibility_km,
    wavelength_m: float,
    Cn2,
    beam_divergence_mrad: float,
    receiver_diameter_m: float,
    rain_rate_mmph=0.0,
    system_loss_db: float = 0.0,
    visibility_definition: str = "MOR_5pct",
    dsd_shape_mu: int = 0,
):
    """
    Deterministic scenario-level FSO attenuation:

        L_FSO =
            A_geo
          + A_sp
          + A_rain
          + A_scint
          + A_system

    where:
        A_geo    : geometrical attenuation
        A_sp     : suspended-particle attenuation from visibility
        A_rain   : rain attenuation
        A_scint  : turbulence/scintillation attenuation
        A_system : explicit fixed system-dependent loss

    Pointing/misalignment
    ---------------------
    No separate stochastic pointing-error model is introduced here.
    If a fixed system loss is used to represent implementation losses, it must
    be declared explicitly through system_loss_db in the experiment config.
    """
    d_km = np.asarray(d_km, dtype=float)

    A_geo = geometrical_attenuation_db(
        d_km=d_km,
        beam_divergence_mrad=beam_divergence_mrad,
        receiver_diameter_m=receiver_diameter_m,
    )

    A_sp = suspended_particle_path_attenuation_db(
        d_km=d_km,
        visibility_km=visibility_km,
        wavelength_m=wavelength_m,
        visibility_definition=visibility_definition,
    )

    A_rain = rain_path_attenuation_db(
        d_km=d_km,
        rain_rate_mmph=rain_rate_mmph,
        dsd_shape_mu=dsd_shape_mu,
    )

    A_scint = scintillation_attenuation_db(
        d_km=d_km,
        Cn2=Cn2,
        wavelength_m=wavelength_m,
    )

    A_system = float(system_loss_db)

    return A_geo + A_sp + A_rain + A_scint + A_system
