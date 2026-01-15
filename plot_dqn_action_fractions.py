#!/usr/bin/env python3
"""
plot_dqn_action_fractions.py
----------------------------
A partir de dqn_evaluation_results.csv, calcula e plota,
para cada cenário, a fracção de ações escolhidas pelo DQN
(RF=0, FSO=1, THz=2) por intervalo de distância.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

CSV_PATH = "dqn_evaluation_results.csv"

def main():
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"{CSV_PATH} não encontrado.")

    df = pd.read_csv(CSV_PATH)

    # garantir colunas essenciais
    for col in ["scenario", "step", "action", "best_channel"]:
        assert col in df.columns, f"Falta coluna '{col}' em {CSV_PATH}"

    # distância em metros a partir do step (como no visualize_dqn_policy)
    if "distance_m" not in df.columns:
        df["distance_m"] = (df["step"] + 1) * 10.0

    scenarios = df["scenario"].unique()
    n_bins = 30  # pode ajustar

    for scen in scenarios:
        df_s = df[df["scenario"] == scen].copy()

        d_min = df_s["distance_m"].min()
        d_max = df_s["distance_m"].max()
        bins = np.linspace(d_min, d_max, n_bins + 1)

        # para cada bin, calcular fração de ações 0, 1, 2
        frac_RF = []
        frac_FSO = []
        frac_THz = []
        centers = []

        for i in range(n_bins):
            d_lo = bins[i]
            d_hi = bins[i+1]
            mask = (df_s["distance_m"] >= d_lo) & (df_s["distance_m"] < d_hi)
            df_bin = df_s[mask]

            if len(df_bin) == 0:
                # sem dados neste bin
                frac_RF.append(0.0)
                frac_FSO.append(0.0)
                frac_THz.append(0.0)
            else:
                total = len(df_bin)
                frac_RF.append((df_bin["action"] == 0).sum() / total)
                frac_FSO.append((df_bin["action"] == 1).sum() / total)
                frac_THz.append((df_bin["action"] == 2).sum() / total)

            centers.append(0.5 * (d_lo + d_hi))

        centers = np.array(centers)

        plt.figure(figsize=(8, 4))
        plt.plot(centers, frac_RF, label="RF (ação 0)")
        plt.plot(centers, frac_FSO, label="FSO (ação 1)")
        plt.plot(centers, frac_THz, label="THz (ação 2)")
        plt.ylim(0.0, 1.05)
        plt.xlabel("Distance (m)")
        plt.ylabel("Fraction of actions")
        plt.title(f"DQN action fractions vs distance — scenario: {scen}")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"dqn_action_fractions_{scen}.png", dpi=200)
        plt.close()

    print("Gráficos guardados como dqn_action_fractions_*.png")

if __name__ == "__main__":
    main()
