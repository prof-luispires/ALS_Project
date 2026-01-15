#!/usr/bin/env python3
"""
plot_best_channel_vs_distance.py
--------------------------------
Mostra, para cada cenário, o canal ótimo (best_channel)
em função da distância, usando o dataset_all_scenarios.csv.
"""

import pandas as pd
import matplotlib.pyplot as plt

# Ajuste o caminho se necessário
DATASET_PATH = "dataset_all_scenarios.csv"

def main():
    df = pd.read_csv(DATASET_PATH)

    # Garantir colunas esperadas
    assert "scenario" in df.columns, "Falta coluna 'scenario'"
    assert "distance_m" in df.columns, "Falta coluna 'distance_m'"
    assert "best_channel" in df.columns, "Falta coluna 'best_channel'"

    scenarios = df["scenario"].unique()

    for scen in scenarios:
        df_s = df[df["scenario"] == scen].copy()

        plt.figure(figsize=(8, 4))
        # scatter com pequena transparência para ver densidade
        plt.scatter(
            df_s["distance_m"],
            df_s["best_channel"],
            alpha=0.6,
            s=15,
        )
        plt.yticks([0, 1, 2], ["RF (0)", "FSO (1)", "THz (2)"])
        plt.xlabel("Distance (m)")
        plt.ylabel("Best channel")
      #  plt.title(f"Optimal channel vs distance — scenario: {scen}")
        plt.grid(True, axis="y", linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(f"best_channel_vs_distance_{scen}.png", dpi=200)
        plt.close()

    print("Gráficos guardados como best_channel_vs_distance_*.png")

if __name__ == "__main__":
    main()
