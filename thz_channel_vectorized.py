"""
thz_channel_vectorized.py — versão recalibrada
----------------------------------------------
Modelo THz vetorizado para o sistema híbrido RF/FSO/THz.

Inclui:
    • FSPL em THz
    • Absorção gasosa dependente da humidade (modelo linear ajustável)
    • Ganho direcional de antenas (G_thz_db)

Objetivo da calibração:
    - THz competitivo/ótimo a distâncias curtas em 'clear'
    - Fortemente penalizado à medida que a humidade aumenta (rain/fog/worst)
"""

import numpy as np

C0 = 3e8  # velocidade da luz (m/s)


# ------------------------------------------------------------
# 1) FSPL vetorizado
# ------------------------------------------------------------
def fspl_db(d_m, f_Hz, c0=C0):
    """
    FSPL:
        L = 20·log10(4π d f / c)
    """
    d_m = np.asarray(d_m, dtype=float)
    return 20 * np.log10(4 * np.pi * d_m * f_Hz / c0 + 1e-30)


# ------------------------------------------------------------
# 2) Absorção gasosa (modelo linear em função da humidade)
# ------------------------------------------------------------
def gamma_gas_db_per_km_linear(
    humidity_pct,
    a0_dbkm: float = 0.1,
    a1_dbkm_per_pct: float = 0.03,
):
    """
    Atenuação gasosa aproximada:
        gamma = a0 + a1 * humidade(%)

    Valores típicos com estes parâmetros:
        humid = 40% (clear)  -> gamma ≈ 0.1 + 0.03*40 = 1.3 dB/km
        humid = 70% (rain)   -> gamma ≈ 2.2 dB/km
        humid = 90% (fog)    -> gamma ≈ 2.8 dB/km
        humid = 95% (worst)  -> gamma ≈ 2.95 dB/km

    Assim, em 'clear' o THz é penalizado de forma moderada,
    mas em 'fog'/'worst' a penalização total aumenta rapidamente
    com a distância.
    """
    H = np.asarray(humidity_pct, dtype=float)
    return a0_dbkm + a1_dbkm_per_pct * H


# ------------------------------------------------------------
# 3) Atenuação total THz
# ------------------------------------------------------------
def thz_atten_db(
    d_m,
    f_Hz,
    humidity_pct,
    a0_dbkm: float = 0.1,
    a1_dbkm_per_pct: float = 0.03,
    G_thz_db: float = 65.0,
):
    """
    L_total = FSPL + absorção_gasosa - ganho_antenas

    G_thz_db ≈ 65 dB reflete antenas altamente direcionais.
    Combinado com a absorção gasosa dependente da humidade, isto faz com que:
        - Em 'clear' e distâncias curtas, o THz tenha SNR muito elevado.
        - Em 'rain/fog/worst' e/ou distâncias longas, a absorção gasosa
          faça o THz colapsar e deixe de ser competitivo.
    """
    d_m = np.asarray(d_m, dtype=float)
    d_km = d_m / 1000.0

    # FSPL
    L_fs = fspl_db(d_m, f_Hz)

    # Atenuação gasosa
    gamma = gamma_gas_db_per_km_linear(humidity_pct, a0_dbkm, a1_dbkm_per_pct)
    L_gas = gamma * d_km

    # Ganho das antenas (Tx/Rx altamente direcionais)
    return L_fs + L_gas - G_thz_db
