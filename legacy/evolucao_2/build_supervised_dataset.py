#!/usr/bin/env python3
"""
build_supervised_dataset.py
---------------------------
Gera um dataset supervisionado para treino de modelos Random Forest e SVM
para seleção adaptativa de canal (RF, FSO, THz).

Cada amostra corresponde a um conjunto de condições de propagação e potência,
e inclui:

Features (X):
    - distance_m
    - visibility_km
    - humidity_pct
    - rain_rate_mmph
    - Cn2
    - Pt_RF_dBm
    - Pt_FSO_dBm
    - Pt_THz_dBm
    - SNR_RF_dB
    - SNR_FSO_dB
    - SNR_THz_dB
    - Eb_RF_J
    - Eb_FSO_J
    - Eb_THz_J

Label (y):
    - best_channel:
        0 = RF
        1 = FSO
        2 = THz

Definição do canal ideal:
    - Primeiro selecciona-se o canal com maior SNR (argmax dos 3).
    - Em caso de empate em SNR, escolhe-se o canal com menor energia por bit.

Saída:
    - dataset_supervised_channels.csv
    - dataset_supervised_channels.xlsx

Requisitos:
    - rf_channel.py
    - thz_channel.py
    - fso_channel_vectorized.py
    - channel_metrics.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import rf_channel
import rf_channel_vectorized as rf_channel
import thz_channel_vectorized as thz_channel
import fso_channel_vectorized as fso_channel
import channel_metrics as cm


# ---------------------------------------------------------------------------
# Funções auxiliares
# ---------------------------------------------------------------------------

def noise_power_dbm(bandwidth_hz: float, nf_db: float) -> float:
    """Calcula potência de ruído em dBm para largura de banda B e NF dados.

    Fórmula:
        N0 = -174 dBm/Hz (ruído térmico)
        N = N0 + 10*log10(B) + NF
    """
    return -174.0 + 10.0 * np.log10(bandwidth_hz) + nf_db


def sample_log_uniform(low: float, high: float, size=None) -> np.ndarray:
    """Amostragem log-uniforme entre low e high (valores >0)."""
    low = float(low)
    high = float(high)
    if low <= 0 or high <= 0:
        raise ValueError("log-uniform requer limites > 0.")
    u = np.random.uniform(np.log10(low), np.log10(high), size=size)
    return 10.0 ** u


# ---------------------------------------------------------------------------
# Geração do dataset
# ---------------------------------------------------------------------------

def build_dataset_supervised(
    n_samples: int = 50000,
    seed: int | None = 42,
) -> pd.DataFrame:
    """Gera o dataset supervisionado com n_samples amostras."""

    if seed is not None:
        np.random.seed(seed)

    # 1) Amostragem das variáveis de ambiente e distância
    distance_m = np.random.uniform(10.0, 4000.0, size=n_samples)        # [10 m, 4 km]
    visibility_km = np.random.uniform(0.5, 25.0, size=n_samples)        # FSO
    humidity_pct = np.random.uniform(30.0, 100.0, size=n_samples)       # THz
    rain_rate_mmph = np.random.uniform(0.0, 100.0, size=n_samples)      # RF
    # Cn2: log-uniform para abranger 1e-15 a 5e-14
    Cn2 = sample_log_uniform(1e-15, 5e-14, size=n_samples)

    # 2) Potências e débitos por canal (em dBm e bit/s)
    Pt_RF_dBm = np.random.uniform(20.0, 40.0, size=n_samples)
    Pt_FSO_dBm = np.random.uniform(0.0, 20.0, size=n_samples)
    Pt_THz_dBm = np.random.uniform(0.0, 20.0, size=n_samples)

    BW_RF_Hz = 20e6
    BW_FSO_Hz = 1e9
    BW_THz_Hz = 5e9

    NF_RF_dB = 5.0
    NF_FSO_dB = 5.0
    NF_THz_dB = 8.0

    Rb_RF = 10e6      # 10 Mbit/s
    Rb_FSO = 1e9      # 1 Gbit/s
    Rb_THz = 1e9      # 1 Gbit/s

    # 3) Ruído por canal
    N_RF_dBm = noise_power_dbm(BW_RF_Hz, NF_RF_dB)
    N_FSO_dBm = noise_power_dbm(BW_FSO_Hz, NF_FSO_dB)
    N_THz_dBm = noise_power_dbm(BW_THz_Hz, NF_THz_dB)

    # 4) Cálculo SNR por canal, amostra a amostra

    # RF: FSPL + chuva (sem shadowing aleatório, para não poluir o dataset)
    f_RF = 5e9
    Lfs_RF = rf_channel.fspl_db(distance_m, f_RF)
    gamma_R = rf_channel.rain_specific_atten_db_per_km(
        rain_rate_mmph,
        k=1e-4,
        alpha=0.9,
    )
    Lrain_RF = gamma_R * (distance_m / 1000.0)
    Ltot_RF = Lfs_RF + Lrain_RF
    SNR_RF_dB = Pt_RF_dBm - Ltot_RF - N_RF_dBm

    # FSO: usa função vetorizada para atenuação total
    wavelength_m = 1550e-9
    Ltot_FSO = fso_channel.fso_attenuation_total_db_vec(
        d_m=distance_m,
        visibility_km=visibility_km,
        Cn2=Cn2,
        wavelength_m=wavelength_m,
        r_m=0.005,
        w_e_m=0.05,
        A0=0.95,
    )
    SNR_FSO_dB = Pt_FSO_dBm - Ltot_FSO - N_FSO_dBm

    # THz: FSPL + absorção molecular dependente da humidade
    f_THz = 3e11
    Lfs_THz = thz_channel.fspl_db(distance_m, f_THz)
    gamma_gas = thz_channel.gamma_gas_db_per_km_linear(
        humidity_pct,
        a0_dbkm=5.0,
        a1_dbkm_per_pct=0.02,
    )
    Labs_THz = gamma_gas * (distance_m / 1000.0)
    Ltot_THz = Lfs_THz + Labs_THz
    SNR_THz_dB = Pt_THz_dBm - Ltot_THz - N_THz_dBm

    # 5) Energia por bit de cada canal
    Eb_RF = np.array(
        [cm.energy_per_bit_joule(P_dBm=p, bitrate_bps=Rb_RF) for p in Pt_RF_dBm]
    )
    Eb_FSO = np.array(
        [cm.energy_per_bit_joule(P_dBm=p, bitrate_bps=Rb_FSO) for p in Pt_FSO_dBm]
    )
    Eb_THz = np.array(
        [cm.energy_per_bit_joule(P_dBm=p, bitrate_bps=Rb_THz) for p in Pt_THz_dBm]
    )

    # 6) Label: best_channel (0=RF, 1=FSO, 2=THz)
    #    Regra:
    #      - escolhe-se o índice do canal com maior SNR;
    #      - se houver empate em SNR (muito raro em float), escolhe o de menor Eb.
    SNR_stack = np.stack([SNR_RF_dB, SNR_FSO_dB, SNR_THz_dB], axis=1)
    Eb_stack = np.stack([Eb_RF, Eb_FSO, Eb_THz], axis=1)

    best_idx = np.argmax(SNR_stack, axis=1)

    # Tratamento de empates de SNR (casos raros)
    for i in range(n_samples):
        snr_row = SNR_stack[i, :]
        max_snr = snr_row[best_idx[i]]
        ties = np.where(np.isclose(snr_row, max_snr, atol=1e-6))[0]
        if len(ties) > 1:
            eb_row = Eb_stack[i, ties]
            best_tie_idx = ties[np.argmin(eb_row)]
            best_idx[i] = best_tie_idx

    best_channel = best_idx  # 0=RF,1=FSO,2=THz

    # 7) Construção do DataFrame final
    df = pd.DataFrame(
        {
            "distance_m": distance_m,
            "visibility_km": visibility_km,
            "humidity_pct": humidity_pct,
            "rain_rate_mmph": rain_rate_mmph,
            "Cn2": Cn2,
            "Pt_RF_dBm": Pt_RF_dBm,
            "Pt_FSO_dBm": Pt_FSO_dBm,
            "Pt_THz_dBm": Pt_THz_dBm,
            "SNR_RF_dB": SNR_RF_dB,
            "SNR_FSO_dB": SNR_FSO_dB,
            "SNR_THz_dB": SNR_THz_dB,
            "Eb_RF_J": Eb_RF,
            "Eb_FSO_J": Eb_FSO,
            "Eb_THz_J": Eb_THz,
            "best_channel": best_channel,
        }
    )

    return df


def main():
    df = build_dataset_supervised(n_samples=50000, seed=42)
    df.to_csv("dataset_supervised_channels.csv", index=False)
    df.to_excel("dataset_supervised_channels.xlsx", index=False)
    print("Dataset supervisionado gerado:")
    print(" - dataset_supervised_channels.csv")
    print(" - dataset_supervised_channels.xlsx")
    print("Nº de amostras:", len(df))


if __name__ == "__main__":
    main()
