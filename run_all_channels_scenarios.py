#!/usr/bin/env python3
"""
run_all_channels_scenarios.py
-----------------------------
Gera resultados para os seis canais (RF, RF subaquático, THz, THz subaquático,
FSO atmosférico e FSO subaquático) até 4 km, com passo de 10 m, para vários
cenários físico-ambientais.

Requer que os seguintes módulos/ficheiros estejam na MESMA PASTA:
    rf_channel.py
    rf_underwater.py
    thz_channel.py
    thz_underwater.py
    fso_channel.py
    fso_underwater.py

Cada cenário gera um par CSV/XLSX por tecnologia, com nomes do tipo:
    rf_channel_clear_4km.csv
    fso_underwater_fog_4km.xlsx
"""

import numpy as np
import pandas as pd

import rf_channel
import rf_underwater
import thz_channel
import thz_underwater
import fso_channel
import fso_underwater


# ---------------------------------------------------------------------------
# Definição dos cenários
# ---------------------------------------------------------------------------

SCENARIOS = {
    "clear": {
        "rf_air":       {"rain_rate": 0.0},
        "thz_air":      {"humidity_pct": 40.0},
        "fso_air":      {"visibility_km": 20.0, "Cn2": 1e-15},
        "rf_under":     {"sigma": 3.0},
        "thz_under":    {"alpha_abs_dB_per_m": 20.0},
        "fso_under":    {"c_per_m": 0.05},
    },
    "rain": {
        "rf_air":       {"rain_rate": 25.0},
        "thz_air":      {"humidity_pct": 70.0},
        "fso_air":      {"visibility_km": 5.0, "Cn2": 5e-15},
        "rf_under":     {"sigma": 4.0},
        "thz_under":    {"alpha_abs_dB_per_m": 50.0},
        "fso_under":    {"c_per_m": 0.2},
    },
    "fog": {
        "rf_air":       {"rain_rate": 50.0},
        "thz_air":      {"humidity_pct": 90.0},
        "fso_air":      {"visibility_km": 1.0, "Cn2": 1e-14},
        "rf_under":     {"sigma": 5.0},
        "thz_under":    {"alpha_abs_dB_per_m": 100.0},
        "fso_under":    {"c_per_m": 0.5},
    },
    "worst": {
        "rf_air":       {"rain_rate": 80.0},
        "thz_air":      {"humidity_pct": 95.0},
        "fso_air":      {"visibility_km": 0.5, "Cn2": 5e-14},
        "rf_under":     {"sigma": 6.0},
        "thz_under":    {"alpha_abs_dB_per_m": 150.0},
        "fso_under":    {"c_per_m": 1.0},
    },
}


# ---------------------------------------------------------------------------
# Funções geradoras por tecnologia
# ---------------------------------------------------------------------------

def generate_rf_air_scenario(scen_name, rain_rate,
                             d_min=10.0, d_max=4000.0, step=10.0):
    """RF em ar: FSPL + chuva + shadowing."""
    d_m = np.arange(d_min, d_max + step, step, dtype=float)

    f_Hz = 5e9
    k = 1e-4
    alpha = 0.9
    shadow_sigma_db = 4.0
    rng_seed = 42

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
    df.to_csv(f"rf_channel_{scen_name}_4km.csv", index=False)
    df.to_excel(f"rf_channel_{scen_name}_4km.xlsx", index=False)
    return df


def generate_rf_underwater_scenario(scen_name, sigma,
                                    d_min=10.0, d_max=4000.0, step=10.0):
    """RF subaquático: meio condutor homogéneo."""
    d = np.arange(d_min, d_max + step, step, dtype=float)

    # parâmetros base (consistentes com rf_underwater.rf_underwater_snr_capacity)
    f_hz = 100e3
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
    df.to_csv(f"rf_underwater_{scen_name}_4km.csv", index=False)
    df.to_excel(f"rf_underwater_{scen_name}_4km.xlsx", index=False)
    return df


def generate_thz_air_scenario(scen_name, humidity_pct,
                              d_min=10.0, d_max=4000.0, step=10.0):
    """THz em ar: FSPL + absorção gasosa dependente da humidade."""
    d_m = np.arange(d_min, d_max + step, step, dtype=float)

    f_Hz = 3e11
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
    df.to_csv(f"thz_channel_{scen_name}_4km.csv", index=False)
    df.to_excel(f"thz_channel_{scen_name}_4km.xlsx", index=False)
    return df


def generate_thz_underwater_scenario(scen_name, alpha_abs_dB_per_m,
                                     d_min=10.0, d_max=4000.0, step=10.0):
    """THz subaquático: FSPL + absorção muito forte no meio."""
    d = np.arange(d_min, d_max + step, step, dtype=float)

    f_hz = 300e9
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
    df.to_csv(f"thz_underwater_{scen_name}_4km.csv", index=False)
    df.to_excel(f"thz_underwater_{scen_name}_4km.xlsx", index=False)
    return df


def generate_fso_air_scenario(scen_name, visibility_km, Cn2,
                              d_min=10.0, d_max=4000.0, step=10.0):
    """FSO atmosférico: Beer–Lambert (Kruse) + turbulência + pointing."""
    d_m = np.arange(d_min, d_max + step, step, dtype=float)
    d_km = d_m / 1000.0

    wavelength_m = 1550e-9
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
    df.to_csv(f"fso_channel_{scen_name}_4km.csv", index=False)
    df.to_excel(f"fso_channel_{scen_name}_4km.xlsx", index=False)
    return df


def generate_fso_underwater_scenario(scen_name, c_per_m,
                                     d_min=10.0, d_max=4000.0, step=10.0):
    """FSO/VLC subaquático: Beer–Lambert em água + penalização de pointing."""
    d = np.arange(d_min, d_max + step, step, dtype=float)

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
    df.to_csv(f"fso_underwater_{scen_name}_4km.csv", index=False)
    df.to_excel(f"fso_underwater_{scen_name}_4km.xlsx", index=False)
    return df


# ---------------------------------------------------------------------------
# Função principal
# ---------------------------------------------------------------------------

def main():
    for scen_name, params in SCENARIOS.items():
        print(f"=== Cenário {scen_name} ===")
        generate_rf_air_scenario(scen_name, **params["rf_air"])
        generate_rf_underwater_scenario(scen_name, **params["rf_under"])
        generate_thz_air_scenario(scen_name, **params["thz_air"])
        generate_thz_underwater_scenario(scen_name, **params["thz_under"])
        generate_fso_air_scenario(scen_name, **params["fso_air"])
        generate_fso_underwater_scenario(scen_name, **params["fso_under"])
    print("Todos os cenários gerados (CSV + Excel, 4 km, passo 10 m).")


if __name__ == "__main__":
    main()
