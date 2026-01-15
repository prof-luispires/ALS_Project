#!/usr/bin/env python3
"""
plot_feature_correlation_5_2.py
Gera a Figura 5.2: Heatmap de correlação das principais features.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

DATASET = "dataset_all_scenarios.csv"

def main():
    df = pd.read_csv(DATASET)

    # Selecionar colunas relevantes
    cols = [
        "distance_m",
        "rain_rate_mmph",
        "humidity_pct",
        "visibility_km",
        "Cn2",
        "SNR_RF_dB",
        "SNR_FSO_dB",
        "SNR_THz_dB",
        "Eb_RF_J_per_bit",
        "Eb_FSO_J_per_bit",
        "Eb_THz_J_per_bit",
    ]

    for col in cols:
        if col not in df.columns:
            raise ValueError(f"Coluna em falta no CSV: {col}")

    corr = df[cols].corr()

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(corr.values, interpolation="nearest")

    ax.set_xticks(np.arange(len(cols)))
    ax.set_yticks(np.arange(len(cols)))
    ax.set_xticklabels(cols, rotation=45, ha="right")
    ax.set_yticklabels(cols)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
   # ax.set_title("Figure 5.2 – Correlation heatmap of environmental and channel features")

    plt.tight_layout()
    fname = "fig_5_2_feature_correlation.png"
    plt.savefig(fname, dpi=200)
    plt.close()
    print(f"Guardado: {fname}")

if __name__ == "__main__":
    main()
