"""
rf_channel.py
-------------
RF channel attenuation model (~5 GHz): FSPL + rain attenuation (ITU-R P.838-3) + log-normal shadowing.

Formulas (dB):
  FSPL: L_FS = 20*log10(d_m) + 20*log10(f_Hz) - 147.55
  Rain: L_rain = (k * R**alpha) * d_km
  Shadowing (large-scale): X_sigma ~ N(0, sigma^2) [dB]

Total:
  L_RF = L_FS + L_rain + X_sigma

References:
  - T. S. Rappaport, "Wireless Communications: Principles and Practice", 2nd ed., 2002.
  - ITU-R P.838-3, "Specific attenuation model for rain for use in prediction methods", 2005.
  - A. Goldsmith, "Wireless Communications", 2005.

Usage:
  >>> import numpy as np
  >>> from rf_channel import rf_atten_db
  >>> d = np.array([100, 1000, 3000])           # meters
  >>> f = 5e9                                    # Hz
  >>> R = np.array([0, 5, 20])                   # rain rate mm/h
  >>> L = rf_atten_db(d_m=d, f_Hz=f, rain_rate_mmph=R, k=1e-4, alpha=0.9, shadow_sigma_db=2.0, rng_seed=42)
  >>> print(np.round(L,2))
"""
import numpy as np

def fspl_db(d_m: np.ndarray, f_Hz: float) -> np.ndarray:
    """Free-space path loss in dB with d in meters and f in Hz."""
    d_m = np.asarray(d_m, dtype=float)
    return 20.0*np.log10(d_m) + 20.0*np.log10(float(f_Hz)) - 147.55

def rain_specific_atten_db_per_km(rain_rate_mmph: np.ndarray, k: float, alpha: float) -> np.ndarray:
    """ITU-R P.838-3: gamma_R = k * R**alpha [dB/km]. Values at 5 GHz are typically small."""
    R = np.asarray(rain_rate_mmph, dtype=float)
    gamma = k * np.power(np.maximum(R, 0.0), alpha)
    gamma[R <= 0] = 0.0
    return gamma

def rf_atten_db(d_m, f_Hz=5e9, rain_rate_mmph=0.0, k=1e-4, alpha=0.9, shadow_sigma_db=2.0, rng_seed=None):
    """Compute total RF attenuation (dB): FSPL + rain + log-normal shadowing in dB.

    Parameters
    ----------
    d_m : array-like
        Distance(s) in meters.
    f_Hz : float
        Carrier frequency in Hz (default 5e9).
    rain_rate_mmph : array-like or float
        Rain rate R in mm/h (can be scalar or array-like aligned with d_m).
    k, alpha : float
        ITU-R P.838-3 coefficients (k ~ 1e-4, alpha ~ 0.9 for ~5 GHz).
    shadow_sigma_db : float
        Standard deviation of log-normal shadowing in dB (typ. 2-6 dB).
    rng_seed : int or None
        Seed for reproducible shadowing.

    Returns
    -------
    L_total_db : np.ndarray
        Total attenuation in dB.
    """
    d_m = np.asarray(d_m, dtype=float)
    R = np.asarray(rain_rate_mmph, dtype=float)
    if R.size == 1 and d_m.size > 1:
        R = np.full_like(d_m, float(R))

    # Components
    L_fs = fspl_db(d_m, f_Hz)
    gamma_R = rain_specific_atten_db_per_km(R, k, alpha)  # dB/km
    L_rain = gamma_R * (d_m/1000.0)

    # Shadowing
    rng = np.random.default_rng(rng_seed)
    X_sigma = rng.normal(0.0, shadow_sigma_db, size=d_m.shape)

    return L_fs + L_rain + X_sigma

if __name__ == "__main__":
    import numpy as np
    d_test = np.array([100, 1000, 3000], dtype=float)
    L = rf_atten_db(d_test, f_Hz=5e9, rain_rate_mmph=np.array([0, 5, 20]), k=1e-4, alpha=0.9, shadow_sigma_db=2.0, rng_seed=42)
    print("RF Attenuation (dB):", np.round(L, 2))
