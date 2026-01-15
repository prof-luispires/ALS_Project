"""
fso_channel_vectorized.py — versão recalibrada
----------------------------------------------
Modelo FSO vetorizado para o sistema híbrido RF/FSO/THz.

Inclui:
    • FSPL óptico com ganho efetivo das aperturas (G_opt_db)
    • Beer–Lambert (Kruse) para atenuação atmosférica, com fator de severidade
    • Scintillation (Rytov simplificado, suavizado)
"""

import numpy as np

C0 = 3e8  # velocidade da luz (m/s)


# ------------------------------------------------------------
# 1) FSPL óptico com ganho das aperturas
# ------------------------------------------------------------
def fspl_optical_db(d_km, wavelength_m, G_opt_db: float = 140.0):
    """
    FSPL óptico:
        L_fspl = 20·log10(4π d / λ)
    compensado com o ganho efetivo das aperturas (G_opt_db).

    d_km        : distância em km
    wavelength_m: comprimento de onda (m), ex: 1550e-9
    G_opt_db    : ganho óptico aproximado (Tx/Rx), afinado para que o
                  FSO seja competitivo em condições 'clear', mas degradado
                  em 'fog' e 'worst'.
    """
    d_km = np.asarray(d_km, dtype=float)
    d_m = d_km * 1000.0

    L_fspl = 20.0 * np.log10(4.0 * np.pi * d_m / wavelength_m + 1e-30)
    return L_fspl - G_opt_db


# ------------------------------------------------------------
# 2) Beer–Lambert (Kruse) com fator de severidade
# ------------------------------------------------------------
def beer_lambert_atm_loss_db(d_km, visibility_km, wavelength_m):
    """
    Atenuação atmosférica baseada no modelo de Kruse, com fator de severidade
    em função da visibilidade:

        V >= 15 km   -> cenário muito favorável   (factor ≈ 0.3)
        5 <= V < 15  -> cenário moderado         (factor ≈ 1.0)
        1 <= V < 5   -> cenário desfavorável     (factor ≈ 2.0)
        V < 1        -> cenário extremo (fog)    (factor ≈ 3.0)

    Isto permite:
        - pouca penalização em 'clear' (V=20 km)
        - penalização forte em 'fog' (V=1 km) e 'worst' (V=0.5 km)
    """
    d_km = np.asarray(d_km, dtype=float)
    V = float(visibility_km)

    # Expoente de Kruse (simplificado)
    if V > 50:
        q = 1.6
    elif V > 6:
        q = 1.3
    else:
        q = 0.585 * V ** (1.0 / 3.0)

    # Coeficiente de extinção "base"
    sigma_base = 3.912 / max(V, 0.1) * (wavelength_m / 550e-9) ** (-q)

    # Fator de severidade
    if V >= 15.0:
        severity = 0.3
    elif V >= 5.0:
        severity = 1.0
    elif V >= 1.0:
        severity = 2.0
    else:
        severity = 3.0

    sigma = sigma_base * severity

    # Atenuação linear em nepers
    L_atm_lin = sigma * d_km

    # Converter nepers -> dB (1 neper ≈ 4.343 dB)
    return 4.343 * L_atm_lin


# ------------------------------------------------------------
# 3) Scintillation (Rytov simplificado e suavizado)
# ------------------------------------------------------------
def scintillation_loss_db(d_km, Cn2, wavelength_m):
    """
    Cn2: parâmetro de turbulência (m^(−2/3)).

    A fórmula clássica da variância de Rytov pode originar perdas
    muito grandes para distâncias elevadas. Aqui usamos:

        sigma_R^2 = 1.23 · Cn2 · k^(7/6) · d^(11/6)
        penalty_dB ≈ 2 · sigma_R   (regime de pequenas flutuações)

    e limitamos a penalização máxima a 10 dB para evitar valores
    irrealistas que destruiriam completamente o FSO no dataset.
    """
    d_km = np.asarray(d_km, dtype=float)
    d_m = d_km * 1000.0

    k = 2.0 * np.pi / wavelength_m  # número de onda
    Cn2 = float(Cn2)

    sigma_R2 = 1.23 * Cn2 * (k ** (7.0 / 6.0)) * (d_m ** (11.0 / 6.0))
    sigma_R2 = np.maximum(sigma_R2, 0.0)

    sigma_R = np.sqrt(sigma_R2)

    # Penalização em dB (suavizada)
    L_scin = 2.0 * sigma_R

    # Limitar entre 0 e 10 dB
    L_scin = np.clip(L_scin, 0.0, 10.0)
    return L_scin


# ------------------------------------------------------------
# 4) Atenuação total FSO
# ------------------------------------------------------------
def fso_atten_db(
    d_km,
    visibility_km,
    wavelength_m=1550e-9,
    Cn2=1e-14,
    G_opt_db=140.0,
):
    """
    Atenuação total em dB:

        L_total = L_fspl_optical + L_Beer-Lambert + L_scintillation
    """
    d_km = np.asarray(d_km, dtype=float)

    L_fs = fspl_optical_db(d_km, wavelength_m, G_opt_db=G_opt_db)
    L_beer = beer_lambert_atm_loss_db(d_km, visibility_km, wavelength_m)
    L_scin = scintillation_loss_db(d_km, Cn2, wavelength_m)

    return L_fs + L_beer + L_scin
