"""
rf_channel_vectorized.py
------------------------

Modelo RF vetorizado para o sistema híbrido RF/FSO/THz.

Inclui:
    - FSPL (Free-Space Path Loss)
    - Atenuação por chuva (modelo power-law tipo ITU)
    - Termo de shadowing lognormal em dB

Função principal:
    rf_atten_db(...)
"""

import numpy as np

C0 = 3e8  # velocidade da luz (m/s)


# -------------------------------------------------------------------
# 1) FSPL vetorizado
# -------------------------------------------------------------------
def fspl_db(d_m, f_Hz, c0=C0):
    """
    Free-Space Path Loss (FSPL) em dB:

        L_FSPL = 20·log10(4 π d f / c)

    d_m : distância em metros (escalar ou array)
    f_Hz: frequência em Hz
    """
    d_m = np.asarray(d_m, dtype=float)
    return 20.0 * np.log10(4.0 * np.pi * d_m * f_Hz / c0 + 1e-30)


# -------------------------------------------------------------------
# 2) Atenuação específica da chuva (dB/km)
# -------------------------------------------------------------------
def rain_specific_atten_db_per_km(R_mmph, k=5e-4, alpha=1.1):
    """
    Atenuação específica da chuva (modelo ITU simplificado):

        gamma_R [dB/km] = k · R^alpha

    R_mmph : taxa de chuva em mm/h (escalar ou array)
    k, alpha: parâmetros empíricos (ajustáveis)
    """
    R = np.asarray(R_mmph, dtype=float)
    gamma = k * np.power(np.maximum(R, 0.0), alpha)
    # Sem chuva → sem atenuação
    gamma = np.where(R <= 0.0, 0.0, gamma)
    return gamma


# -------------------------------------------------------------------
# 3) Atenuação total RF (FSPL + chuva + shadowing)
# -------------------------------------------------------------------
def rf_atten_db(
    d_m,
    f_Hz,
    rain_rate_mmph,
    k=5e-4,
    alpha=1.1,
    shadow_sigma_db=2.0,
    rng_seed=None,
):
    """
    Calcula a atenuação total RF em dB:

        L_total(dB) = L_FSPL + gamma_R · d_km + Shadowing

    Parâmetros
    ----------
    d_m : distância [m] (escalar ou array)
    f_Hz : frequência [Hz]
    rain_rate_mmph : taxa de chuva [mm/h]
    k, alpha : parâmetros do modelo de chuva
    shadow_sigma_db : desvio padrão do shadowing em dB
    rng_seed : semente opcional para reprodutibilidade

    Retorna
    -------
    L_total : array de atenuação em dB (mesma forma de d_m)
    """
    d_m = np.asarray(d_m, dtype=float)
    d_km = d_m / 1000.0

    # FSPL
    L_fs = fspl_db(d_m, f_Hz)

    # Atenuação por chuva
    gamma_R = rain_specific_atten_db_per_km(rain_rate_mmph, k=k, alpha=alpha)
    L_rain = gamma_R * d_km

    # Shadowing lognormal (em dB ~ N(0, sigma²))
    if shadow_sigma_db > 0:
        rng = np.random.default_rng(rng_seed)
        shadow = rng.normal(0.0, shadow_sigma_db, size=d_m.shape)
    else:
        shadow = 0.0

    return L_fs + L_rain + shadow
