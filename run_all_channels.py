#!/usr/bin/env python3
"""
run_all_channels.py
-------------------
Gera resultados para os seis canais (RF, RF subaquático, THz, THz subaquático,
FSO atmosférico e FSO subaquático) até 4 km, com passo de 10 m,
exportando para CSV e Excel.

Requer:
    rf_channel.py
    rf_underwater.py
    thz_channel.py
    thz_underwater.py
    fso_channel.py
    fso_underwater.py

Todos estes ficheiros devem estar na mesma pasta.
"""

import numpy as np
import pandas as pd

import rf_channel
import rf_underwater
import thz_channel
import thz_underwater
import fso_channel
import fso_underwater


def generate_rf_air(d_min=10.0, d_max=4000.0, step=10.0):
    d_m = np.arange(d_min, d_max + step, step, dtype=float)

    # Parâmetros RF
    f_Hz = 5e9
    rain_rate = 20.0  # mm/h
    k = 1e-4
    alpha = 0.9
    shadow_sigma_db = 4.0
    rng_seed = 42

    # Componentes
    L_fs = rf_channel.fspl_db(d_m, f_Hz)
    gamma_R = rf_channel.rain_specific_atten_db_per_km(
        np.full_like(d_m, rain_rate), k, alpha
    )
    L_rain = gamma_R * (d_m / 1000.0)

    rng = np.random.default_rng(rng_seed)
    L_shadow = rng.normal(0.0, shadow_sigma_db, size=d_m.shape)

    L_total = L_fs + L_rain + L_shadow

    df = pd.DataFrame(
        {
            "distance_m": d_m,
            "FSPL_dB": L_fs,
            "rain_loss_dB": L_rain,
            "shadowing_dB": L_shadow,
            "total_loss_dB": L_total,
        }
    )
    df.to_csv("rf_channel_4km.csv", index=False)
    df.to_excel("rf_channel_4km.xlsx", index=False)
    return df


def generate_rf_underwater(d_min=10.0, d_max=4000.0, step=10.0):
    d = np.arange(d_min, d_max + step, step, dtype=float)

    # Usamos a mesma formulação interna de rf_underwater.rf_underwater_snr_capacity,
    # mas com passo fixo de 10 m e gama de distâncias até 4 km.
    f_hz = 100e3
    sigma = 4.0
    mu = 4 * np.pi * 1e-7
    Pt_dBm = 30.0
    Gt_dB = 0.0
    Gr_dB = 0.0
    B_hz = 1000.0
    NF_dB = 5.0

    alpha = np.sqrt(np.pi * f_hz * mu * sigma)
    L_dB = 8.686 * alpha * d
    Pr_dBm = Pt_dBm - L_dB + Gt_dB + Gr_dB
    N_dBm = -174.0 + 10.0 * np.log10(B_hz) + NF_dB
    SNR_dB = Pr_dBm - N_dBm
    SNR_lin = 10.0 ** (SNR_dB / 10.0)
    C_bps = B_hz * np.log2(1.0 + SNR_lin)

    df = pd.DataFrame(
        {
            "distance_m": d,
            "alpha_Np_per_m": np.full_like(d, alpha),
            "path_loss_dB": L_dB,
            "Pr_dBm": Pr_dBm,
            "SNR_dB": SNR_dB,
            "Capacity_bps": C_bps,
        }
    )
    df.to_csv("rf_underwater_4km.csv", index=False)
    df.to_excel("rf_underwater_4km.xlsx", index=False)
    return df


def generate_thz_air(d_min=10.0, d_max=4000.0, step=10.0):
    d_m = np.arange(d_min, d_max + step, step, dtype=float)

    f_Hz = 3e11
    humidity_pct = 70.0
    a0_dbkm = 5.0
    a1_dbkm_per_pct = 0.02

    L_fs = thz_channel.fspl_db(d_m, f_Hz)
    gamma = thz_channel.gamma_gas_db_per_km_linear(
        humidity_pct, a0_dbkm=a0_dbkm, a1_dbkm_per_pct=a1_dbkm_per_pct
    )
    L_abs = gamma * (d_m / 1000.0)
    L_total = L_fs + L_abs

    df = pd.DataFrame(
        {
            "distance_m": d_m,
            "FSPL_dB": L_fs,
            "gas_absorption_dB": L_abs,
            "total_loss_dB": L_total,
        }
    )
    df.to_csv("thz_channel_4km.csv", index=False)
    df.to_excel("thz_channel_4km.xlsx", index=False)
    return df


def generate_thz_underwater(d_min=10.0, d_max=4000.0, step=10.0):
    d = np.arange(d_min, d_max + step, step, dtype=float)

    # Parâmetros base do módulo thz_underwater
    f_hz = 300e9
    alpha_abs_dB_per_m = 100.0
    Pt_dBm = 0.0
    Gt_dB = 0.0
    Gr_dB = 0.0
    B_hz = 100e6
    NF_dB = 6.0

    C0 = thz_underwater.C0

    FSPL_dB = 20.0 * np.log10(4.0 * np.pi * f_hz * d / C0)
    L_dB = FSPL_dB + alpha_abs_dB_per_m * d
    Pr_dBm = Pt_dBm - L_dB + Gt_dB + Gr_dB
    N_dBm = -174.0 + 10.0 * np.log10(B_hz) + NF_dB
    SNR_dB = Pr_dBm - N_dBm
    SNR_lin = 10.0 ** (SNR_dB / 10.0)
    C_bps = B_hz * np.log2(1.0 + SNR_lin)

    df = pd.DataFrame(
        {
            "distance_m": d,
            "FSPL_dB": FSPL_dB,
            "path_loss_dB": L_dB,
            "Pr_dBm": Pr_dBm,
            "SNR_dB": SNR_dB,
            "Capacity_bps": C_bps,
        }
    )
    df.to_csv("thz_underwater_4km.csv", index=False)
    df.to_excel("thz_underwater_4km.xlsx", index=False)
    return df


def generate_fso_air(d_min=10.0, d_max=4000.0, step=10.0):
    d_m = np.arange(d_min, d_max + step, step, dtype=float)
    d_km = d_m / 1000.0

    visibility_km = 5.0
    wavelength_m = 1550e-9
    Cn2 = 1e-14
    r_m = 0.005
    w_e_m = 0.05
    A0 = 0.95

    L_atm = fso_channel.beer_lambert_atm_loss_db(d_km, visibility_km, wavelength_m)
    L_turb = fso_channel.turbulence_loss_db(Cn2, wavelength_m, d_km)
    L_point = fso_channel.pointing_loss_db(np.full_like(d_km, r_m), w_e_m, A0=A0)
    L_total = L_atm + L_turb + L_point

    df = pd.DataFrame(
        {
            "distance_m": d_m,
            "distance_km": d_km,
            "L_atm_dB": L_atm,
            "L_turb_dB": L_turb,
            "L_point_dB": L_point,
            "L_total_dB": L_total,
        }
    )
    df.to_csv("fso_channel_4km.csv", index=False)
    df.to_excel("fso_channel_4km.xlsx", index=False)
    return df


def generate_fso_underwater(d_min=10.0, d_max=4000.0, step=10.0):
    d = np.arange(d_min, d_max + step, step, dtype=float)

    c_per_m = 0.2
    G_pointing_dB = -1.0
    Pt_dBm = 10.0
    Gt_dB = 0.0
    Gr_dB = 0.0
    B_hz = 10e6
    NF_dB = 5.0

    L_dB = 4.343 * c_per_m * d - G_pointing_dB
    Pr_dBm = Pt_dBm - L_dB + Gt_dB + Gr_dB
    N_dBm = -174.0 + 10.0 * np.log10(B_hz) + NF_dB
    SNR_dB = Pr_dBm - N_dBm
    SNR_lin = 10.0 ** (SNR_dB / 10.0)
    C_bps = B_hz * np.log2(1.0 + SNR_lin)

    df = pd.DataFrame(
        {
            "distance_m": d,
            "extinction_per_m": np.full_like(d, c_per_m),
            "path_loss_dB": L_dB,
            "Pr_dBm": Pr_dBm,
            "SNR_dB": SNR_dB,
            "Capacity_bps": C_bps,
        }
    )
    df.to_csv("fso_underwater_4km.csv", index=False)
    df.to_excel("fso_underwater_4km.xlsx", index=False)
    return df


def main():
    print("Generating RF (air) results...")
    generate_rf_air()
    print("Generating RF (underwater) results...")
    generate_rf_underwater()
    print("Generating THz (air) results...")
    generate_thz_air()
    print("Generating THz (underwater) results...")
    generate_thz_underwater()
    print("Generating FSO (air) results...")
    generate_fso_air()
    print("Generating FSO (underwater) results...")
    generate_fso_underwater()
    print("All CSV and Excel files generated (4 km, step 10 m).")


if __name__ == "__main__":
    main()
