#!/usr/bin/env python3
"""
run_all_channels_scenarios_metrics.py
-------------------------------------
Gera resultados para os seis canais (RF, RF subaquático, THz, THz subaquático,
FSO atmosférico e FSO subaquático) até 4 km, com passo de 10 m, para vários
cenários físico-ambientais.

Adiciona, para cada canal e cenário:
    - Energia por bit (Eb) em joule.
    - Capacidade de Shannon (se aplicável).
    - Probabilidade de outage baseada em capacidade (C < R_target).

Requer que os seguintes módulos/ficheiros estejam na MESMA PASTA:
    rf_channel.py
    rf_underwater.py
    thz_channel.py
    thz_underwater.py
    fso_channel.py
    fso_underwater.py
    channel_metrics.py

Cada cenário gera um par CSV/XLSX por tecnologia, com nomes do tipo:
    rf_channel_clear_4km.csv
    fso_underwater_fog_4km.xlsx
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import rf_channel
import rf_underwater
import thz_channel
import thz_underwater
import fso_channel
import fso_underwater
import channel_metrics as cm


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
# Parâmetros globais para SNR / capacidade / Eb
# (pode ajustar facilmente estes valores)
# ---------------------------------------------------------------------------

# RF ar
RF_AIR_PT_DBM = 30.0
RF_AIR_BW_HZ = 20e6          # 20 MHz
RF_AIR_RATE_TARGET = 10e6    # 10 Mbit/s (débito alvo)
RF_AIR_RB_EB = 10e6          # 10 Mbit/s para Eb
RF_AIR_NF_DB = 5.0

# THz ar
THZ_AIR_PT_DBM = 10.0
THZ_AIR_BW_HZ = 5e9          # 5 GHz
THZ_AIR_RATE_TARGET = 1e9    # 1 Gbit/s
THZ_AIR_RB_EB = 1e9          # 1 Gbit/s
THZ_AIR_NF_DB = 8.0

# FSO ar
FSO_AIR_PT_DBM = 10.0
FSO_AIR_BW_HZ = 1e9          # 1 GHz
FSO_AIR_RATE_TARGET = 1e9    # 1 Gbit/s
FSO_AIR_RB_EB = 1e9
FSO_AIR_NF_DB = 5.0

# RF subaquático
RF_UNDER_PT_DBM = 30.0
RF_UNDER_BW_HZ = 1e3         # 1 kHz
RF_UNDER_RATE_TARGET = 500.0 # 500 bit/s
RF_UNDER_RB_EB = 1e3
RF_UNDER_NF_DB = 5.0

# THz subaquático
THZ_UNDER_PT_DBM = 0.0
THZ_UNDER_BW_HZ = 100e6      # 100 MHz
THZ_UNDER_RATE_TARGET = 10e6 # 10 Mbit/s
THZ_UNDER_RB_EB = 10e6
THZ_UNDER_NF_DB = 6.0

# FSO subaquático
FSO_UNDER_PT_DBM = 10.0
FSO_UNDER_BW_HZ = 10e6       # 10 MHz
FSO_UNDER_RATE_TARGET = 10e6 # 10 Mbit/s
FSO_UNDER_RB_EB = 10e6
FSO_UNDER_NF_DB = 5.0


def noise_power_dbm(bandwidth_hz: float, nf_db: float) -> float:
    """Calcula potência de ruído em dBm para largura de banda B e NF dados.

    Fórmula:
        N0 = -174 dBm/Hz (ruído térmico)
        N = N0 + 10*log10(B) + NF
    """
    return -174.0 + 10.0 * np.log10(bandwidth_hz) + nf_db


# ---------------------------------------------------------------------------
# Funções geradoras por tecnologia, com métricas
# ---------------------------------------------------------------------------

def generate_rf_air_scenario(scen_name: str, rain_rate: float,
                             d_min: float = 10.0, d_max: float = 4000.0, step: float = 10.0):
    """RF em ar: FSPL + chuva + shadowing + métricas (Eb, capacidade, outage)."""
    d_m = np.arange(d_min, d_max + step, step, dtype=float)

    f_Hz = 5e9
    k = 1e-4
    alpha = 0.9
    shadow_sigma_db = 4.0
    rng_seed = 42

    # Perdas
    L_fs = rf_channel.fspl_db(d_m, f_Hz)
    gamma_R = rf_channel.rain_specific_atten_db_per_km(
        np.full_like(d_m, rain_rate), k, alpha
    )
    L_rain = gamma_R * (d_m / 1000.0)
    rng = np.random.default_rng(rng_seed)
    L_shadow = rng.normal(0.0, shadow_sigma_db, size=d_m.shape)
    L_total = L_fs + L_rain + L_shadow

    # SNR e capacidade
    Pt_dBm = RF_AIR_PT_DBM
    B_hz = RF_AIR_BW_HZ
    N_dBm = noise_power_dbm(B_hz, RF_AIR_NF_DB)
    SNR_dB = Pt_dBm - L_total - N_dBm

    df = pd.DataFrame(
        {
            "distance_m": d_m,
            "FSPL_dB": L_fs,
            "rain_loss_dB": L_rain,
            "shadowing_dB": L_shadow,
            "total_loss_dB": L_total,
            "SNR_dB": SNR_dB,
        }
    )

    # Adiciona Eb, capacidade e outage (capacidade)
    df, summary = cm.add_metrics_to_df_capacity(
        df,
        snr_col="SNR_dB",
        pt_dBm=Pt_dBm,
        bitrate_bps=RF_AIR_RB_EB,
        bandwidth_hz=B_hz,
        rate_target_bps=RF_AIR_RATE_TARGET,
        eb_col="Eb_J",
        outage_col="outage_flag_cap",
        cap_col="capacity_bps",
    )

    df.to_csv(f"rf_channel_{scen_name}_4km.csv", index=False)
    df.to_excel(f"rf_channel_{scen_name}_4km.xlsx", index=False)
    return df, summary


def generate_rf_underwater_scenario(scen_name: str, sigma: float,
                                    d_min: float = 10.0, d_max: float = 4000.0, step: float = 10.0):
    """RF subaquático: meio condutor homogéneo + métricas."""
    d = np.arange(d_min, d_max + step, step, dtype=float)

    # parâmetros base (consistentes com rf_underwater.rf_underwater_snr_capacity)
    f_hz = 100e3
    mu = 4 * np.pi * 1e-7
    Pt_dBm = RF_UNDER_PT_DBM
    Gt_dB = 0.0
    Gr_dB = 0.0
    B_hz = RF_UNDER_BW_HZ

    alpha = np.sqrt(np.pi * f_hz * mu * sigma)
    L_dB = 8.686 * alpha * d
    Pr_dBm = Pt_dBm - L_dB + Gt_dB + Gr_dB
    N_dBm = noise_power_dbm(B_hz, RF_UNDER_NF_DB)
    SNR_dB = Pr_dBm - N_dBm

    df = pd.DataFrame(
        {
            "distance_m": d,
            "alpha_Np_per_m": np.full_like(d, alpha),
            "path_loss_dB": L_dB,
            "Pr_dBm": Pr_dBm,
            "SNR_dB": SNR_dB,
        }
    )

    df, summary = cm.add_metrics_to_df_capacity(
        df,
        snr_col="SNR_dB",
        pt_dBm=Pt_dBm,
        bitrate_bps=RF_UNDER_RB_EB,
        bandwidth_hz=B_hz,
        rate_target_bps=RF_UNDER_RATE_TARGET,
        eb_col="Eb_J",
        outage_col="outage_flag_cap",
        cap_col="capacity_bps",
    )

    df.to_csv(f"rf_underwater_{scen_name}_4km.csv", index=False)
    df.to_excel(f"rf_underwater_{scen_name}_4km.xlsx", index=False)
    return df, summary


def generate_thz_air_scenario(scen_name: str, humidity_pct: float,
                              d_min: float = 10.0, d_max: float = 4000.0, step: float = 10.0):
    """THz em ar: FSPL + absorção gasosa + métricas."""
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

    Pt_dBm = THZ_AIR_PT_DBM
    B_hz = THZ_AIR_BW_HZ
    N_dBm = noise_power_dbm(B_hz, THZ_AIR_NF_DB)
    SNR_dB = Pt_dBm - L_total - N_dBm

    df = pd.DataFrame(
        {
            "distance_m": d_m,
            "FSPL_dB": L_fs,
            "gas_absorption_dB": L_abs,
            "total_loss_dB": L_total,
            "SNR_dB": SNR_dB,
        }
    )

    df, summary = cm.add_metrics_to_df_capacity(
        df,
        snr_col="SNR_dB",
        pt_dBm=Pt_dBm,
        bitrate_bps=THZ_AIR_RB_EB,
        bandwidth_hz=B_hz,
        rate_target_bps=THZ_AIR_RATE_TARGET,
        eb_col="Eb_J",
        outage_col="outage_flag_cap",
        cap_col="capacity_bps",
    )

    df.to_csv(f"thz_channel_{scen_name}_4km.csv", index=False)
    df.to_excel(f"thz_channel_{scen_name}_4km.xlsx", index=False)
    return df, summary


def generate_thz_underwater_scenario(scen_name: str, alpha_abs_dB_per_m: float,
                                     d_min: float = 10.0, d_max: float = 4000.0, step: float = 10.0):
    """THz subaquático: FSPL + absorção muito forte + métricas."""
    d = np.arange(d_min, d_max + step, step, dtype=float)

    f_hz = 300e9
    Pt_dBm = THZ_UNDER_PT_DBM
    Gt_dB = 0.0
    Gr_dB = 0.0
    B_hz = THZ_UNDER_BW_HZ
    C0 = thz_underwater.C0

    FSPL_dB = 20.0 * np.log10(4.0 * np.pi * f_hz * d / C0)
    L_dB = FSPL_dB + alpha_abs_dB_per_m * d
    Pr_dBm = Pt_dBm - L_dB + Gt_dB + Gr_dB
    N_dBm = noise_power_dbm(B_hz, THZ_UNDER_NF_DB)
    SNR_dB = Pr_dBm - N_dBm

    df = pd.DataFrame(
        {
            "distance_m": d,
            "FSPL_dB": FSPL_dB,
            "path_loss_dB": L_dB,
            "Pr_dBm": Pr_dBm,
            "SNR_dB": SNR_dB,
        }
    )

    df, summary = cm.add_metrics_to_df_capacity(
        df,
        snr_col="SNR_dB",
        pt_dBm=Pt_dBm,
        bitrate_bps=THZ_UNDER_RB_EB,
        bandwidth_hz=B_hz,
        rate_target_bps=THZ_UNDER_RATE_TARGET,
        eb_col="Eb_J",
        outage_col="outage_flag_cap",
        cap_col="capacity_bps",
    )

    df.to_csv(f"thz_underwater_{scen_name}_4km.csv", index=False)
    df.to_excel(f"thz_underwater_{scen_name}_4km.xlsx", index=False)
    return df, summary


def generate_fso_air_scenario(scen_name: str, visibility_km: float, Cn2: float,
                              d_min: float = 10.0, d_max: float = 4000.0, step: float = 10.0):
    """FSO atmosférico: Beer–Lambert (Kruse) + turbulência + pointing + métricas."""
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

    Pt_dBm = FSO_AIR_PT_DBM
    B_hz = FSO_AIR_BW_HZ
    N_dBm = noise_power_dbm(B_hz, FSO_AIR_NF_DB)
    SNR_dB = Pt_dBm - L_total - N_dBm

    df = pd.DataFrame(
        {
            "distance_m": d_m,
            "distance_km": d_km,
            "L_atm_dB": L_atm,
            "L_turb_dB": L_turb,
            "L_point_dB": L_point,
            "L_total_dB": L_total,
            "SNR_dB": SNR_dB,
        }
    )

    df, summary = cm.add_metrics_to_df_capacity(
        df,
        snr_col="SNR_dB",
        pt_dBm=Pt_dBm,
        bitrate_bps=FSO_AIR_RB_EB,
        bandwidth_hz=B_hz,
        rate_target_bps=FSO_AIR_RATE_TARGET,
        eb_col="Eb_J",
        outage_col="outage_flag_cap",
        cap_col="capacity_bps",
    )

    df.to_csv(f"fso_channel_{scen_name}_4km.csv", index=False)
    df.to_excel(f"fso_channel_{scen_name}_4km.xlsx", index=False)
    return df, summary


def generate_fso_underwater_scenario(scen_name: str, c_per_m: float,
                                     d_min: float = 10.0, d_max: float = 4000.0, step: float = 10.0):
    """FSO/VLC subaquático: Beer–Lambert em água + penalização de pointing + métricas."""
    d = np.arange(d_min, d_max + step, step, dtype=float)

    G_pointing_dB = -1.0
    Pt_dBm = FSO_UNDER_PT_DBM
    Gt_dB = 0.0
    Gr_dB = 0.0
    B_hz = FSO_UNDER_BW_HZ

    L_dB = 4.343 * c_per_m * d - G_pointing_dB
    Pr_dBm = Pt_dBm - L_dB + Gt_dB + Gr_dB
    N_dBm = noise_power_dbm(B_hz, FSO_UNDER_NF_DB)
    SNR_dB = Pr_dBm - N_dBm

    df = pd.DataFrame(
        {
            "distance_m": d,
            "extinction_per_m": np.full_like(d, c_per_m),
            "path_loss_dB": L_dB,
            "Pr_dBm": Pr_dBm,
            "SNR_dB": SNR_dB,
        }
    )

    df, summary = cm.add_metrics_to_df_capacity(
        df,
        snr_col="SNR_dB",
        pt_dBm=Pt_dBm,
        bitrate_bps=FSO_UNDER_RB_EB,
        bandwidth_hz=B_hz,
        rate_target_bps=FSO_UNDER_RATE_TARGET,
        eb_col="Eb_J",
        outage_col="outage_flag_cap",
        cap_col="capacity_bps",
    )

    df.to_csv(f"fso_underwater_{scen_name}_4km.csv", index=False)
    df.to_excel(f"fso_underwater_{scen_name}_4km.xlsx", index=False)
    return df, summary


# ---------------------------------------------------------------------------
# Função principal
# ---------------------------------------------------------------------------

def main():
    summaries = {}
    for scen_name, params in SCENARIOS.items():
        print(f"=== Cenário {scen_name} ===")

        summaries[scen_name] = {}

        df_rf_air, s_rf_air = generate_rf_air_scenario(scen_name, **params["rf_air"])
        summaries[scen_name]["rf_air"] = s_rf_air

        df_rf_under, s_rf_under = generate_rf_underwater_scenario(scen_name, **params["rf_under"])
        summaries[scen_name]["rf_under"] = s_rf_under

        df_thz_air, s_thz_air = generate_thz_air_scenario(scen_name, **params["thz_air"])
        summaries[scen_name]["thz_air"] = s_thz_air

        df_thz_under, s_thz_under = generate_thz_underwater_scenario(scen_name, **params["thz_under"])
        summaries[scen_name]["thz_under"] = s_thz_under

        df_fso_air, s_fso_air = generate_fso_air_scenario(scen_name, **params["fso_air"])
        summaries[scen_name]["fso_air"] = s_fso_air

        df_fso_under, s_fso_under = generate_fso_underwater_scenario(scen_name, **params["fso_under"])
        summaries[scen_name]["fso_under"] = s_fso_under

    # Opcional: criar um resumo em Excel com Eb e P_out por cenário/canal
    rows = []
    for scen, techs in summaries.items():
        for tech, s in techs.items():
            row = {"scenario": scen, "tech": tech}
            row.update(s)
            rows.append(row)
    df_summary = pd.DataFrame(rows)
    df_summary.to_excel("summary_metrics_scenarios.xlsx", index=False)

    print("Todos os cenários gerados (CSV + Excel, 4 km, passo 10 m).")
    print("Resumo de métricas guardado em 'summary_metrics_scenarios.xlsx'.")


if __name__ == "__main__":
    main()
