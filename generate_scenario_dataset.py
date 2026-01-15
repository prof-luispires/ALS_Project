#!/usr/bin/env python3
"""
generate_scenario_dataset.py
----------------------------
Gera datasets para os 4 cenários (clear, rain, fog, worst)
com parâmetros ambientais aleatórios dentro de intervalos
físicos realistas.

Para cada cenário gera:
    dataset_<scenario>.csv

e no fim:
    dataset_all_scenarios.csv
    dataset_all_scenarios.xlsx
"""

import numpy as np
import pandas as pd

# ------------------------------------------------------------
# PARTE 1 — Amostragem aleatória dos parâmetros por cenário
# ------------------------------------------------------------

def sample_clear_params():
    """
    Cenário CLEAR:
      - Sem chuva
      - Humidade moderada
      - Visibilidade alta
    """
    return {
        "rain_rate_mmph": 0.0,
        "humidity_pct": float(np.random.uniform(35.0, 45.0)),
        "visibility_km": float(np.random.uniform(15.0, 25.0)),
        "Cn2": 1e-15,
    }


def sample_rain_params():
    """
    Cenário RAIN:
      - chuva moderada/forte: 25–50 mm/h
      - humidade elevada
      - visibilidade média
    """
    return {
        "rain_rate_mmph": float(np.random.uniform(25.0, 50.0)),  # > 25 mm/h
        "humidity_pct": float(np.random.uniform(60.0, 80.0)),
        "visibility_km": float(np.random.uniform(3.0, 7.0)),
        "Cn2": 5e-15,
    }


def sample_fog_params():
    """
    Cenário FOG:
      - chuva fraca: 5–20 mm/h  (< 25 mm/h)
      - humidade muito elevada
      - visibilidade muito reduzida
    """
    return {
        "rain_rate_mmph": float(np.random.uniform(5.0, 20.0)),   # < 25 mm/h
        "humidity_pct": float(np.random.uniform(85.0, 95.0)),
        "visibility_km": float(np.random.uniform(0.3, 1.0)),
        "Cn2": 1e-14,
    }


def sample_worst_params():
    """
    Cenário WORST:
      - chuva muito forte: 50–80 mm/h
      - humidade quase saturada
      - visibilidade extremamente reduzida
    """
    return {
        "rain_rate_mmph": float(np.random.uniform(50.0, 80.0)),  # máximo ~80 mm/h
        "humidity_pct": float(np.random.uniform(90.0, 98.0)),
        "visibility_km": float(np.random.uniform(0.1, 0.4)),
        "Cn2": 5e-14,
    }


SCENARIOS = {
    "clear": sample_clear_params,
    "rain": sample_rain_params,
    "fog": sample_fog_params,
    "worst": sample_worst_params,
}

# ------------------------------------------------------------
# PARTE 2 — Importação dos modelos de canal vetorizados
# ------------------------------------------------------------

from rf_channel_vectorized import rf_atten_db
from fso_channel_vectorized import fso_atten_db
from thz_channel_vectorized import thz_atten_db

PT_DBM = 0      # Potência Tx em dBm
GT_DB = 0
GR_DB = 0
NF_DB = 3
BW_HZ = 1e9


def snr_capacity_energy(L_dB):
    """
    Calcula SNR, capacidade e energia por bit para um dado L_dB (array).
    """
    PT_lin = 1e-3 * 10**(PT_DBM / 10)
    N0 = 1e-3 * 10**((-174 + NF_DB) / 10)  # ruído térmico + NF

    L_lin = 10**(-L_dB / 10)
    Prx_lin = PT_lin * L_lin

    SNR_lin = Prx_lin / (N0 * BW_HZ)
    SNR_dB = 10 * np.log10(SNR_lin + 1e-30)

    C = BW_HZ * np.log2(1 + SNR_lin)
    Eb = Prx_lin / (C + 1e-30)

    return SNR_dB, C, Eb


# ------------------------------------------------------------
# PARTE 3 — Constantes físicas e auxiliares
# ------------------------------------------------------------

# Bandas (aproximadas) por tecnologia
B_RF_HZ  = 20e6     # 20 MHz
B_FSO_HZ = 1e9      # 1 GHz
B_THz_HZ = 5e9      # 5 GHz

# Noise figure (dB)
NF_RF_dB  = 5.0
NF_FSO_dB = 3.0
NF_THz_dB = 7.0

# Potências Tx (dBm)
P_TX_RF_dBm  = 20.0   # 100 mW
P_TX_FSO_dBm = 10.0   # 10 mW
P_TX_THz_dBm = 15.0   # 32 mW

# Requisitos de QoS
C_MIN_REQ_BPS  = 1e5   # 100 kbps mínimos
SNR_MIN_dB     = 0.0   # SNR mínima aceitável

# Ruído térmico a 290K
N0_dBm_per_Hz = -174.0


def snr_db(P_tx_dBm, path_loss_dB, B_Hz, NF_dB):
    """
    Calcula SNR em dB:
        P_rx_dBm = P_tx_dBm - L_dB
        N_dBm    = N0 + 10 log10(B) + NF
        SNR_dB   = P_rx_dBm - N_dBm
    """
    P_tx_dBm = np.asarray(P_tx_dBm, dtype=float)
    path_loss_dB = np.asarray(path_loss_dB, dtype=float)

    N_dBm = N0_dBm_per_Hz + 10.0 * np.log10(B_Hz) + NF_dB
    P_rx_dBm = P_tx_dBm - path_loss_dB
    return P_rx_dBm - N_dBm


def capacity_bps(B_Hz, snr_dB, C_min=1e-3, C_max=1e9):
    """
    Capacidade de Shannon:
        C = B log2(1 + SNR)
    com limites para evitar 0 e valores absurdos.
    """
    snr_dB = np.asarray(snr_dB, dtype=float)
    snr_lin = 10.0 ** (snr_dB / 10.0)
    C = B_Hz * np.log2(1.0 + snr_lin)
    return np.clip(C, C_min, C_max)


def ptx_watt(P_tx_dBm):
    """Converte dBm em Watt."""
    P_tx_dBm = np.asarray(P_tx_dBm, dtype=float)
    return 10.0 ** ((P_tx_dBm - 30.0) / 10.0)


def energy_per_bit_joule(P_tx_dBm, C_bps):
    """
    Energia por bit:
        Eb = P_tx / C
    com saturação em limites razoáveis.
    """
    C_bps = np.asarray(C_bps, dtype=float)
    P_W = ptx_watt(P_tx_dBm)
    Eb = P_W / C_bps
    return np.clip(Eb, 1e-12, 1e-1)


# ------------------------------------------------------------
# PARTE 4 — Geração de sweep de distâncias por cenário
# ------------------------------------------------------------

def generate_scenario_sweep(scenario_name, params, max_dist=4000, step=10):
    """
    Gera um sweep de distâncias para um conjunto fixo de parâmetros
    de cenário e calcula:

      - SNR_RF/FSO/THz
      - C_RF/FSO/THz
      - Eb_RF/FSO/THz
      - best_channel (0=RF,1=FSO,2=THz) por critério de Eb + outage.
    """

    distances = np.arange(step, max_dist + step, step).astype(float)  # 10 m → 4000 m
    d_m = distances
    d_km = distances / 1000.0

    rain_rate_mmph = params["rain_rate_mmph"]
    humidity_pct   = params["humidity_pct"]
    visibility_km  = params["visibility_km"]
    Cn2            = params["Cn2"]

    # --------------------------------------------------------
    # RF (ar)
    # --------------------------------------------------------
    L_RF = rf_atten_db(
        d_m,
        f_Hz=5e9,
        rain_rate_mmph=rain_rate_mmph,
        k=5e-4,
        alpha=1.1,
        shadow_sigma_db=2.0,
        rng_seed=None,
    )
    SNR_RF_dB = snr_db(P_TX_RF_dBm, L_RF, B_RF_HZ, NF_RF_dB)
    C_RF_bps  = capacity_bps(B_RF_HZ, SNR_RF_dB)
    Eb_RF_J   = energy_per_bit_joule(P_TX_RF_dBm, C_RF_bps)

    # --------------------------------------------------------
    # FSO (ar)
    # --------------------------------------------------------
    L_FSO = fso_atten_db(
        d_km,
        visibility_km=visibility_km,
        wavelength_m=1550e-9,
        Cn2=Cn2,
        # G_opt_db usa o default do fso_channel_vectorized (ajustado)
    )
    SNR_FSO_dB = snr_db(P_TX_FSO_dBm, L_FSO, B_FSO_HZ, NF_FSO_dB)
    C_FSO_bps  = capacity_bps(B_FSO_HZ, SNR_FSO_dB)
    Eb_FSO_J   = energy_per_bit_joule(P_TX_FSO_dBm, C_FSO_bps)

    # --------------------------------------------------------
    # THz (ar)
    # --------------------------------------------------------
    L_THz = thz_atten_db(
        d_m,
        f_Hz=300e9,
        humidity_pct=humidity_pct,
        # base_clear_dbkm, slope_dbkm_per_pct e G_thz_db usam defaults
    )
    SNR_THz_dB = snr_db(P_TX_THz_dBm, L_THz, B_THz_HZ, NF_THz_dB)
    C_THz_bps  = capacity_bps(B_THz_HZ, SNR_THz_dB)
    Eb_THz_J   = energy_per_bit_joule(P_TX_THz_dBm, C_THz_bps)

    # --------------------------------------------------------
    # Critério de outage e best_channel
    # --------------------------------------------------------
    C_all   = np.vstack([C_RF_bps,  C_FSO_bps,  C_THz_bps])      # (3, N)
    Eb_all  = np.vstack([Eb_RF_J,   Eb_FSO_J,   Eb_THz_J])
    SNR_all = np.vstack([SNR_RF_dB, SNR_FSO_dB, SNR_THz_dB])

    valid = (C_all >= C_MIN_REQ_BPS) & (SNR_all >= SNR_MIN_dB)

    # Penalizar canais inválidos com Eb enorme
    Eb_penalized = np.where(valid, Eb_all, 1e9)
    best_channel = np.argmin(Eb_penalized, axis=0).astype(int)

    # fallback: se todos inválidos, usar RF (0)
    all_invalid = ~valid.any(axis=0)
    best_channel[all_invalid] = 0

    # --------------------------------------------------------
    # Construção do DataFrame
    # --------------------------------------------------------
    df = pd.DataFrame(
        {
            "distance_m": d_m,
            "rain_rate_mmph": rain_rate_mmph,
            "humidity_pct": humidity_pct,
            "visibility_km": visibility_km,
            "Cn2": Cn2,

            "SNR_RF_dB": SNR_RF_dB,
            "SNR_FSO_dB": SNR_FSO_dB,
            "SNR_THz_dB": SNR_THz_dB,

            "C_RF_bps": C_RF_bps,
            "C_FSO_bps": C_FSO_bps,
            "C_THz_bps": C_THz_bps,

            "Eb_RF_J_per_bit": Eb_RF_J,
            "Eb_FSO_J_per_bit": Eb_FSO_J,
            "Eb_THz_J_per_bit": Eb_THz_J,

            "best_channel": best_channel,
            "scenario": scenario_name,
        }
    )

    return df


# ------------------------------------------------------------
# PARTE 5 — Main: gera datasets individuais e unificado
# ------------------------------------------------------------

def main():
    all_dfs = []

    for scen, sampler in SCENARIOS.items():
        print(f"[INFO] Gerando dataset para o cenário '{scen}'...")
        params = sampler()
        print(f"       Parâmetros usados: {params}")

        df = generate_scenario_sweep(scen, params)
        fname = f"dataset_{scen}.csv"
        df.to_csv(fname, index=False)

        print(f"[OK] Guardado: {fname} ({len(df)} linhas)")
        all_dfs.append(df)

    # --------------------------------------------------------
    # Dataset unificado
    # --------------------------------------------------------
    df_all = pd.concat(all_dfs, ignore_index=True)

    df_all.to_csv("dataset_all_scenarios.csv", index=False)
    try:
        df_all.to_excel("dataset_all_scenarios.xlsx", index=False)
    except Exception as e:
        print(f"[AVISO] Não foi possível guardar XLSX: {e}")

    print("\n[OK] dataset_all_scenarios.csv (e .xlsx se possível) guardados.")
    print("\nResumo por cenário:")
    print(df_all["scenario"].value_counts())

    print("\nResumo de best_channel (0=RF,1=FSO,2=THz):")
    print(df_all["best_channel"].value_counts())


if __name__ == "__main__":
    main()
