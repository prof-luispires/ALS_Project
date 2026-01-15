#!/usr/bin/env python3
"""
plot_env_distributions_5_1.py
Gera as figuras da Figura 5.1:
Distribuições das variáveis ambientais por cenário.
"""

import pandas as pd
import matplotlib.pyplot as plt

DATASET = "dataset_all_scenarios.csv"

def main():
    df = pd.read_csv(DATASET)

    # Verificar colunas esperadas
    expected_cols = [
        "scenario",
        "distance_m",
        "rain_rate_mmph",
        "humidity_pct",
        "visibility_km",
        "Cn2",
    ]
    for col in expected_cols:
        if col not in df.columns:
            raise ValueError(f"Coluna em falta no CSV: {col}")

    scenarios = df["scenario"].unique()

    for scen in scenarios:
        d = df[df["scenario"] == scen]

        fig, axes = plt.subplots(2, 2, figsize=(8, 6))
        fig.suptitle(f"Figure 5.1 – Environmental distributions ({scen} scenario)")

        # Rain
        axes[0, 0].hist(d["rain_rate_mmph"], bins=20)
        axes[0, 0].set_title("Rain rate (mm/h)")
        axes[0, 0].set_xlabel("Rain rate (mm/h)")
        axes[0, 0].set_ylabel("Count")
        axes[0, 0].grid(True, linestyle="--", alpha=0.5)

        # Humidity
        axes[0, 1].hist(d["humidity_pct"], bins=20)
        axes[0, 1].set_title("Humidity (%)")
        axes[0, 1].set_xlabel("Humidity (%)")
        axes[0, 1].set_ylabel("Count")
        axes[0, 1].grid(True, linestyle="--", alpha=0.5)

        # Visibility
        axes[1, 0].hist(d["visibility_km"], bins=20)
        axes[1, 0].set_title("Visibility (km)")
        axes[1, 0].set_xlabel("Visibility (km)")
        axes[1, 0].set_ylabel("Count")
        axes[1, 0].grid(True, linestyle="--", alpha=0.5)

        # Cn2
        axes[1, 1].hist(d["Cn2"], bins=20)
        axes[1, 1].set_title(r"$C_n^2$ (m$^{-2/3}$)")
        axes[1, 1].set_xlabel(r"$C_n^2$ (m$^{-2/3}$)")
        axes[1, 1].set_ylabel("Count")
        axes[1, 1].grid(True, linestyle="--", alpha=0.5)

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        fname = f"fig_5_1_env_distributions_{scen}.png"
        plt.savefig(fname, dpi=200)
        plt.close()
        print(f"Guardado: {fname}")

if __name__ == "__main__":
    main()
