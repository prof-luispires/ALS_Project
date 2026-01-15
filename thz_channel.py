"""
thz_channel.py
--------------
THz channel attenuation model (~300 GHz): FSPL + molecular absorption (humidity-dependent).

Formulas (dB):
  FSPL: L_FS = 20*log10(d_m) + 20*log10(f_Hz) - 147.55
  Gas absorption (simplified practical surrogate parameterization):
    gamma_gas(f, H) ≈ a0(f) + a1(f) * H   [dB/km], H in %RH
    L_abs = gamma_gas * d_km

Total:
  L_THz = L_FS + L_abs

Notes:
  - For high-fidelity, compute gamma_gas from ITU-R P.676-12 (function of f, pressure, temperature, and water vapor partial pressure).
  - The linear surrogate is useful for quick simulations around a given band (e.g., ~300 GHz).

References:
  - ITU-R P.676-12, "Attenuation by atmospheric gases", 2019.
  - J. M. Jornet & I. F. Akyildiz, IEEE TWC, 2011.

Usage:
  >>> import numpy as np
  >>> from thz_channel import thz_atten_db
  >>> d = np.array([100, 1000, 3000])  # meters
  >>> L = thz_atten_db(d_m=d, f_Hz=3e11, humidity_pct=70.0, a0_dbkm=5.0, a1_dbkm_per_pct=0.02)
  >>> print(np.round(L,2))
"""
import numpy as np

def fspl_db(d_m: np.ndarray, f_Hz: float) -> np.ndarray:
    """Free-space path loss in dB with d in meters and f in Hz."""
    d_m = np.asarray(d_m, dtype=float)
    return 20.0*np.log10(d_m) + 20.0*np.log10(float(f_Hz)) - 147.55

def gamma_gas_db_per_km_linear(humidity_pct: float, a0_dbkm: float = 5.0, a1_dbkm_per_pct: float = 0.02) -> float:
    """Linear surrogate for gas absorption around a target band (e.g., ~300 GHz).

    gamma_gas(H) = a0 + a1 * H  [dB/km], with H in %RH.
    Defaults loosely reflect increased attenuation with humidity.
    """
    H = float(humidity_pct)
    return float(a0_dbkm) + float(a1_dbkm_per_pct) * H

def thz_atten_db(d_m, f_Hz=3e11, humidity_pct=60.0, a0_dbkm=5.0, a1_dbkm_per_pct=0.02):
    """Total THz attenuation (dB): FSPL + molecular absorption (linear surrogate).

    Parameters
    ----------
    d_m : array-like
        Distance(s) in meters.
    f_Hz : float
        Carrier frequency in Hz (default 3e11, i.e., 300 GHz).
    humidity_pct : float
        Relative humidity in percent (0-100).
    a0_dbkm, a1_dbkm_per_pct : float
        Surrogate coefficients for gamma_gas(H) = a0 + a1*H [dB/km].

    Returns
    -------
    L_total_db : np.ndarray
        Total attenuation in dB.
    """
    d_m = np.asarray(d_m, dtype=float)
    L_fs = fspl_db(d_m, f_Hz)
    gamma = gamma_gas_db_per_km_linear(humidity_pct, a0_dbkm=a0_dbkm, a1_dbkm_per_pct=a1_dbkm_per_pct)
    L_abs = gamma * (d_m/1000.0)
    return L_fs + L_abs

if __name__ == "__main__":
    import numpy as np
    d_test = np.array([100, 1000, 3000, 4000], dtype=float)
    L = thz_atten_db(d_test, f_Hz=3e11, humidity_pct=70.0, a0_dbkm=5.0, a1_dbkm_per_pct=0.02)
    print("THz Attenuation (dB):", np.round(L, 2))
