"""
fso_channel.py
--------------
FSO channel attenuation model (~1550 nm): Beer–Lambert (Kruse visibility) + turbulence (scintillation) + pointing loss.

Formulas (dB):
  Atmospheric (Beer–Lambert + Kruse):
    sigma(V, lambda) = 3.91/V * (lambda/550nm)^(-q)
    L_atm = 4.343 * sigma * d_km
  Turbulence (scintillation index -> margin):
    sigma_R^2 = 1.23 * Cn2 * k^(7/6) * d_m^(11/6),  k=2*pi/lambda
    sigma_I^2 ≈ exp( Phi(sigma_R^2) ) - 1
    L_turb ≈ 10*log10(1 + sigma_I^2)
  Pointing (Gaussian beam; geometric approx):
    h_p ≈ A0 * exp( -2 r^2 / w_e^2 )
    L_point = -10*log10(h_p) ≈ 10*log10(1/A0) + 8.686 * (r^2 / w_e^2)

Total:
  L_FSO = L_atm + L_turb + L_point

References:
  - M. A. Khalighi & M. Uysal (2014), IEEE Comm. Surveys & Tutorials.
  - L. C. Andrews & R. L. Phillips (2005), "Laser Beam Propagation through Random Media".
  - M. Uysal et al. (2007), IEEE TWC, pointing errors.

Usage:
  >>> import numpy as np
  >>> from fso_channel import fso_atten_db
  >>> d_km = np.array([0.2, 1.0, 2.0])   # km
  >>> L = fso_atten_db(d_km, visibility_km=5.0, wavelength_m=1550e-9, Cn2=1e-14, r_m=0.005, w_e_m=0.05, A0=0.95)
  >>> print(np.round(L,2))
"""
import numpy as np

def kruse_sigma_per_km(visibility_km: float, wavelength_m: float) -> float:
    """Kruse model for extinction coefficient sigma [1/km]."""
    V = float(visibility_km)
    lam_nm = float(wavelength_m) * 1e9
    if V > 50:
        q = 1.6
    elif V > 6:
        q = 1.3
    else:
        q = 0.585 * (V**(1/3))
    sigma = 3.91 / max(V, 1e-6) * (lam_nm/550.0) ** (-q)
    return sigma

def beer_lambert_atm_loss_db(d_km: np.ndarray, visibility_km: float, wavelength_m: float) -> np.ndarray:
    """Atmospheric loss L_atm = 4.343 * sigma * d_km [dB]."""
    d_km = np.asarray(d_km, dtype=float)
    sigma = kruse_sigma_per_km(visibility_km, wavelength_m)  # 1/km
    return 4.343 * sigma * d_km

def rytov_variance(Cn2: float, wavelength_m: float, d_m: np.ndarray) -> np.ndarray:
    """Rytov variance sigma_R^2 = 1.23 * Cn2 * k^(7/6) * d^(11/6)."""
    k = 2.0 * np.pi / float(wavelength_m)
    d_m = np.asarray(d_m, dtype=float)
    return 1.23 * float(Cn2) * (k ** (7.0/6.0)) * (d_m ** (11.0/6.0))

def turbulence_loss_db(Cn2: float, wavelength_m: float, d_km: np.ndarray) -> np.ndarray:
    """Convert scintillation index to an average margin in dB: L_turb ≈ 10*log10(1+sigma_I^2).

    We use a compact surrogate: sigma_I^2 ≈ exp(a*sigma_R^2) - 1 with a=1
    for didactic purposes (captures growth trend with distance and Cn2).
    """
    d_km = np.asarray(d_km, dtype=float)
    d_m = d_km * 1000.0
    sigmaR2 = rytov_variance(Cn2, wavelength_m, d_m)
    sigmaI2 = np.exp(np.minimum(sigmaR2, 10.0)) - 1.0  # cap exponent for numerical safety
    return 10.0 * np.log10(1.0 + sigmaI2)

def pointing_loss_db(r_m: np.ndarray, w_e_m: float, A0: float = 0.95) -> np.ndarray:
    """Pointing loss: L_point = -10*log10(h_p) with h_p ≈ A0 * exp(-2 r^2 / w_e^2)."""
    r_m = np.asarray(r_m, dtype=float)
    A0 = float(A0)
    w_e_m = float(w_e_m)
    hp = max(A0, 1e-6) * np.exp(-2.0 * (r_m**2) / (w_e_m**2 + 1e-12))
    return -10.0 * np.log10(np.clip(hp, 1e-12, 1.0))

def fso_atten_db(d_km, visibility_km=5.0, wavelength_m=1550e-9, Cn2=1e-14, r_m=0.0, w_e_m=0.05, A0=0.95):
    """Total FSO attenuation (dB): Beer–Lambert (Kruse) + turbulence + pointing.

    Parameters
    ----------
    d_km : array-like
        Propagation distance(s) in kilometers.
    visibility_km : float
        Meteorological visibility in km.
    wavelength_m : float
        Optical wavelength in meters (default 1550 nm).
    Cn2 : float
        Refractive index structure parameter (m^(-2/3)), e.g., 1e-15 to 1e-13.
    r_m : array-like or float
        Pointing radial offset at receiver in meters (can be scalar or array-like aligned with d_km).
    w_e_m : float
        Effective beam radius at receiver plane (m).
    A0 : float
        Geometric coupling efficiency (0< A0 <=1).

    Returns
    -------
    L_total_db : np.ndarray
        Total attenuation in dB.
    """
    d_km = np.asarray(d_km, dtype=float)
    if np.isscalar(r_m):
        r_m = np.full_like(d_km, float(r_m))
    L_atm = beer_lambert_atm_loss_db(d_km, visibility_km, wavelength_m)
    L_turb = turbulence_loss_db(Cn2, wavelength_m, d_km)
    L_point = pointing_loss_db(r_m, w_e_m, A0=A0)
    return L_atm + L_turb + L_point

if __name__ == "__main__":
    import numpy as np
    d_test_km = np.array([0.2, 1.0, 2.0, 3.0, 4.0], dtype=float)
    L = fso_atten_db(d_test_km, visibility_km=5.0, wavelength_m=1550e-9, Cn2=1e-14, r_m=0.005, w_e_m=0.05, A0=0.95)
    print("FSO Attenuation (dB):", np.round(L, 2))
