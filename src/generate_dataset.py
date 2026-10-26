"""
generate_dataset.py
-------------------
Canonical reproducible dataset generator for the Springer major revision:
Channel-Aware Adaptive Hybrid RF/FSO/THz Maritime Wireless Networking.

Key properties
--------------
- Single source of parameters through config.py.
- Canonical reference policy through reference_policy.py.
- Multiple atmospheric realizations per scenario.
- Explicit random seeds.
- Deterministic distance sweep for each realization.
- Reproducible RF shadowing per realization.
- Revised FSO model with explicit geometry.
- CSV, XLSX, per-scenario files, and JSON metadata.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import config
from reference_policy import select_reference_channel

from rf_channel_vectorized import rf_atten_db
from fso_channel_vectorized_v2 import fso_atten_db
from thz_channel_vectorized import thz_atten_db


# ============================================================
# Helpers: link metrics
# ============================================================

N0_DBM_PER_HZ = -174.0


def snr_db(P_tx_dBm, path_loss_dB, bandwidth_hz, noise_figure_db):
    path_loss_dB = np.asarray(path_loss_dB, dtype=float)
    noise_dBm = (
        N0_DBM_PER_HZ
        + 10.0 * np.log10(float(bandwidth_hz))
        + float(noise_figure_db)
    )
    p_rx_dBm = float(P_tx_dBm) - path_loss_dB
    return p_rx_dBm - noise_dBm


def capacity_bps(bandwidth_hz, snr_dB):
    snr_dB = np.asarray(snr_dB, dtype=float)
    snr_lin = 10.0 ** (snr_dB / 10.0)
    return float(bandwidth_hz) * np.log2(1.0 + snr_lin)


def tx_power_watt(P_tx_dBm):
    return 10.0 ** ((float(P_tx_dBm) - 30.0) / 10.0)


def energy_per_bit_joule(P_tx_dBm, capacity):
    capacity = np.asarray(capacity, dtype=float)
    return tx_power_watt(P_tx_dBm) / np.maximum(capacity, 1e-30)


# ============================================================
# Scenario sampling
# ============================================================

def sample_range(rng, bounds):
    lo, hi = map(float, bounds)
    if lo == hi:
        return lo
    return float(rng.uniform(lo, hi))


def sample_scenario_params(rng, scenario_name):
    ranges = config.SCENARIO_RANGES[scenario_name]
    return {
        "rain_rate_mmph": sample_range(rng, ranges["rain_rate_mmph"]),
        "humidity_pct": sample_range(rng, ranges["humidity_pct"]),
        "visibility_km": sample_range(rng, ranges["visibility_km"]),
        "Cn2": sample_range(rng, ranges["Cn2"]),
    }


# ============================================================
# One realization
# ============================================================

def generate_realization(
    scenario_name: str,
    realization_id: int,
    scenario_params: dict,
    rf_shadow_seed: int,
):
    distances_m = np.arange(
        config.DISTANCE_MIN_M,
        config.DISTANCE_MAX_M + config.DISTANCE_STEP_M,
        config.DISTANCE_STEP_M,
        dtype=float,
    )
    distances_km = distances_m / 1000.0

    rain = scenario_params["rain_rate_mmph"]
    humidity = scenario_params["humidity_pct"]
    visibility = scenario_params["visibility_km"]
    Cn2 = scenario_params["Cn2"]

    # ---------------- RF ----------------
    L_RF_dB = rf_atten_db(
        distances_m,
        f_Hz=config.RF_FREQUENCY_HZ,
        rain_rate_mmph=rain,
        k=config.RF_RAIN_K,
        alpha=config.RF_RAIN_ALPHA,
        shadow_sigma_db=config.RF_SHADOW_SIGMA_DB,
        rng_seed=int(rf_shadow_seed),
    )
    SNR_RF_dB = snr_db(
        config.RF_TX_POWER_DBM,
        L_RF_dB,
        config.RF_BANDWIDTH_HZ,
        config.RF_NOISE_FIGURE_DB,
    )
    C_RF_bps = capacity_bps(config.RF_BANDWIDTH_HZ, SNR_RF_dB)
    Eb_RF_J = energy_per_bit_joule(config.RF_TX_POWER_DBM, C_RF_bps)

    # ---------------- FSO ----------------
    L_FSO_dB = fso_atten_db(
        distances_km,
        visibility_km=visibility,
        wavelength_m=config.FSO_WAVELENGTH_M,
        Cn2=Cn2,
        beam_divergence_mrad=config.FSO_BEAM_DIVERGENCE_MRAD,
        receiver_diameter_m=config.FSO_RECEIVER_DIAMETER_M,
        rain_rate_mmph=rain,
        system_loss_db=config.FSO_SYSTEM_LOSS_DB,
        visibility_definition=config.FSO_VISIBILITY_DEFINITION,
        dsd_shape_mu=config.FSO_RAIN_DSD_MU,
    )
    SNR_FSO_dB = snr_db(
        config.FSO_TX_POWER_DBM,
        L_FSO_dB,
        config.FSO_BANDWIDTH_HZ,
        config.FSO_NOISE_FIGURE_DB,
    )
    C_FSO_bps = capacity_bps(config.FSO_BANDWIDTH_HZ, SNR_FSO_dB)
    Eb_FSO_J = energy_per_bit_joule(config.FSO_TX_POWER_DBM, C_FSO_bps)

    # ---------------- THz ----------------
    L_THz_dB = thz_atten_db(
        distances_m,
        f_Hz=config.THZ_FREQUENCY_HZ,
        humidity_pct=humidity,
        a0_dbkm=config.THZ_ABSORPTION_A0_DB_PER_KM,
        a1_dbkm_per_pct=config.THZ_ABSORPTION_A1_DB_PER_KM_PER_PCT,
        G_thz_db=config.THZ_DIRECTIONAL_GAIN_DB,
    )
    SNR_THz_dB = snr_db(
        config.THZ_TX_POWER_DBM,
        L_THz_dB,
        config.THZ_BANDWIDTH_HZ,
        config.THZ_NOISE_FIGURE_DB,
    )
    C_THz_bps = capacity_bps(config.THZ_BANDWIDTH_HZ, SNR_THz_dB)
    Eb_THz_J = energy_per_bit_joule(config.THZ_TX_POWER_DBM, C_THz_bps)

    # ---------------- Canonical reference policy ----------------
    SNR_all = np.vstack([SNR_RF_dB, SNR_FSO_dB, SNR_THz_dB])
    C_all = np.vstack([C_RF_bps, C_FSO_bps, C_THz_bps])
    Eb_all = np.vstack([Eb_RF_J, Eb_FSO_J, Eb_THz_J])

    best_channel = select_reference_channel(
        SNR_all,
        C_all,
        Eb_all,
    )

    valid = (
        (SNR_all >= config.MIN_SNR_DB)
        & (C_all >= config.MIN_CAPACITY_BPS)
    )
    all_infeasible = ~valid.any(axis=0)

    return pd.DataFrame(
        {
            "scenario": scenario_name,
            "realization_id": int(realization_id),
            "distance_m": distances_m,
            "rain_rate_mmph": rain,
            "humidity_pct": humidity,
            "visibility_km": visibility,
            "Cn2": Cn2,
            "RF_shadow_seed": int(rf_shadow_seed),

            "L_RF_dB": L_RF_dB,
            "L_FSO_dB": L_FSO_dB,
            "L_THz_dB": L_THz_dB,

            "SNR_RF_dB": SNR_RF_dB,
            "SNR_FSO_dB": SNR_FSO_dB,
            "SNR_THz_dB": SNR_THz_dB,

            "C_RF_bps": C_RF_bps,
            "C_FSO_bps": C_FSO_bps,
            "C_THz_bps": C_THz_bps,

            "Eb_RF_J_per_bit": Eb_RF_J,
            "Eb_FSO_J_per_bit": Eb_FSO_J,
            "Eb_THz_J_per_bit": Eb_THz_J,

            "feasible_RF": valid[0].astype(int),
            "feasible_FSO": valid[1].astype(int),
            "feasible_THz": valid[2].astype(int),
            "all_channels_infeasible": all_infeasible.astype(int),

            "best_channel": best_channel,
        }
    )


# ============================================================
# Full dataset
# ============================================================

def generate_dataset(
    n_realizations_per_scenario: int = 10,
    master_seed: int = config.MASTER_SEED,
):
    config.validate_config(require_fso_geometry=True)

    if n_realizations_per_scenario < 1:
        raise ValueError("n_realizations_per_scenario must be >= 1.")

    master_rng = np.random.default_rng(int(master_seed))
    all_frames = []
    realization_records = []

    for scenario_name in config.SCENARIO_RANGES:
        scenario_rng_seed = int(master_rng.integers(0, 2**32 - 1))
        scenario_rng = np.random.default_rng(scenario_rng_seed)

        for realization_id in range(n_realizations_per_scenario):
            params = sample_scenario_params(scenario_rng, scenario_name)
            rf_shadow_seed = int(scenario_rng.integers(0, 2**32 - 1))

            df_r = generate_realization(
                scenario_name=scenario_name,
                realization_id=realization_id,
                scenario_params=params,
                rf_shadow_seed=rf_shadow_seed,
            )
            all_frames.append(df_r)

            realization_records.append(
                {
                    "scenario": scenario_name,
                    "realization_id": realization_id,
                    "scenario_rng_seed": scenario_rng_seed,
                    "rf_shadow_seed": rf_shadow_seed,
                    **params,
                }
            )

    df = pd.concat(all_frames, ignore_index=True)
    realizations_df = pd.DataFrame(realization_records)
    return df, realizations_df


# ============================================================
# Save outputs
# ============================================================

def save_outputs(
    df: pd.DataFrame,
    realizations_df: pd.DataFrame,
    output_dir: str | Path = ".",
    n_realizations_per_scenario: int = 10,
    master_seed: int = config.MASTER_SEED,
):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    csv_path = out / "dataset_all_scenarios.csv"
    xlsx_path = out / "dataset_all_scenarios.xlsx"
    realization_path = out / "scenario_realizations.csv"
    metadata_path = out / "dataset_metadata.json"

    df.to_csv(csv_path, index=False)
    df.to_excel(xlsx_path, index=False)
    realizations_df.to_csv(realization_path, index=False)

    for scenario_name, df_s in df.groupby("scenario", sort=False):
        df_s.to_csv(out / f"dataset_{scenario_name}.csv", index=False)

    counts = (
        df.groupby(["scenario", "best_channel"])
        .size()
        .unstack(fill_value=0)
        .rename(columns=config.CHANNEL_NAMES)
    )

    metadata = {
        "master_seed": int(master_seed),
        "n_realizations_per_scenario": int(n_realizations_per_scenario),
        "distance_min_m": config.DISTANCE_MIN_M,
        "distance_max_m": config.DISTANCE_MAX_M,
        "distance_step_m": config.DISTANCE_STEP_M,
        "samples_per_realization": int(
            ((config.DISTANCE_MAX_M - config.DISTANCE_MIN_M)
             / config.DISTANCE_STEP_M) + 1
        ),
        "total_samples": int(len(df)),
        "scenario_ranges": config.SCENARIO_RANGES,
        "fso_geometry": {
            "wavelength_m": config.FSO_WAVELENGTH_M,
            "beam_divergence_mrad": config.FSO_BEAM_DIVERGENCE_MRAD,
            "receiver_diameter_m": config.FSO_RECEIVER_DIAMETER_M,
            "system_loss_db": config.FSO_SYSTEM_LOSS_DB,
            "visibility_definition": config.FSO_VISIBILITY_DEFINITION,
            "rain_dsd_mu": config.FSO_RAIN_DSD_MU,
        },
        "qos_reference_policy": {
            "min_snr_db": config.MIN_SNR_DB,
            "min_capacity_bps": config.MIN_CAPACITY_BPS,
            "selection": "minimum energy per bit among feasible channels",
            "fallback_if_all_infeasible": "RF",
        },
        "best_channel_counts_by_scenario": counts.to_dict(orient="index"),
    }

    metadata_path.write_text(
        json.dumps(metadata, indent=2, default=float),
        encoding="utf-8",
    )

    return {
        "csv": csv_path,
        "xlsx": xlsx_path,
        "realizations": realization_path,
        "metadata": metadata_path,
    }


def main():
    n_realizations = 10
    df, realizations_df = generate_dataset(
        n_realizations_per_scenario=n_realizations,
        master_seed=config.MASTER_SEED,
    )
    paths = save_outputs(
        df,
        realizations_df,
        output_dir=".",
        n_realizations_per_scenario=n_realizations,
        master_seed=config.MASTER_SEED,
    )

    print("Generated dataset:")
    for key, value in paths.items():
        print(f" - {key}: {value}")

    print("\nSamples by scenario:")
    print(df["scenario"].value_counts())

    print("\nBest-channel distribution:")
    print(
        df.groupby(["scenario", "best_channel"])
        .size()
        .unstack(fill_value=0)
        .rename(columns=config.CHANNEL_NAMES)
    )

    print("\nAll-channel-infeasible samples:")
    print(df.groupby("scenario")["all_channels_infeasible"].sum())


if __name__ == "__main__":
    main()
