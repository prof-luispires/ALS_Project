#!/usr/bin/env python3
"""
plot_metrics_7_3.py
-------------------
Gera as figuras:
 - Figura 7.3.1 — SNR vs distance
 - Figura 7.3.2 — Outage probability vs distance
 - Figura 7.3.3 — Energy-per-bit vs distance
A partir do dataset_all_scenarios.csv.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

DATASET = "dataset_all_scenarios.csv"
MIN_CAPACITY_Mbps = 10  # threshold para outage

def snr_to_capacity(snr_db, bandwidth_hz=1e9):
    """Shannon capacity."""
    snr_lin = 10 ** (snr_db / 10)
    return bandwidth_hz * np.log2(1 + snr_lin) / 1e6  # Mbps

def main():
    df = pd.read_csv(DATASET)
    scenarios = df["scenario"].unique()

    # ============================
    # FIGURA 7.3.1 — SNR vs distance
    # ============================
    for scen in scenarios:
        d = df[df["scenario"] == scen]

        plt.figure(figsize=(8, 4))
        plt.plot(d["distance_m"], d["SNR_RF_dB"], label="RF")
        plt.plot(d["distance_m"], d["SNR_FSO_dB"], label="FSO")
        plt.plot(d["distance_m"], d["SNR_THz_dB"], label="THz")
        plt.xlabel("Distance (m)")
        plt.ylabel("SNR (dB)")
      #  plt.title(f"Figura 7.3.1 — SNR vs distance ({scen})")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"fig_7_3_1_snr_{scen}.png", dpi=200)
        plt.close()

    # ============================
    # FIGURA 7.3.2 — OUTAGE vs distance
    # ============================
    for scen in scenarios:
        d = df[df["scenario"] == scen]

        cap_RF = snr_to_capacity(d["SNR_RF_dB"])
        cap_FSO = snr_to_capacity(d["SNR_FSO_dB"])
        cap_THz = snr_to_capacity(d["SNR_THz_dB"])

        outage_RF = (cap_RF < MIN_CAPACITY_Mbps).astype(int)
        outage_FSO = (cap_FSO < MIN_CAPACITY_Mbps).astype(int)
        outage_THz = (cap_THz < MIN_CAPACITY_Mbps).astype(int)

        plt.figure(figsize=(8, 4))
        plt.plot(d["distance_m"], outage_RF, label="RF")
        plt.plot(d["distance_m"], outage_FSO, label="FSO")
        plt.plot(d["distance_m"], outage_THz, label="THz")
        plt.xlabel("Distance (m)")
        plt.ylabel("Outage Probability")
      #  plt.title(f"Figura 7.3.2 — Outage vs distance ({scen})")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"fig_7_3_2_outage_{scen}.png", dpi=200)
        plt.close()

    # ============================
    # FIGURA 7.3.3 — ENERGY PER BIT
    # ============================
    for scen in scenarios:
        d = df[df["scenario"] == scen]

        plt.figure(figsize=(8, 4))
        plt.plot(d["distance_m"], d["Eb_RF_J_per_bit"], label="RF")
        plt.plot(d["distance_m"], d["Eb_FSO_J_per_bit"], label="FSO")
        plt.plot(d["distance_m"], d["Eb_THz_J_per_bit"], label="THz")
        plt.xlabel("Distance (m)")
        plt.ylabel("Energy per bit (J/bit)")
       # plt.title(f"Figura 7.3.3 — Energy-per-bit vs distance ({scen})")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"fig_7_3_3_energy_{scen}.png", dpi=200)
        plt.close()

    print("Figuras 7.3.1, 7.3.2 e 7.3.3 geradas com sucesso.")

if __name__ == "__main__":
    main()
