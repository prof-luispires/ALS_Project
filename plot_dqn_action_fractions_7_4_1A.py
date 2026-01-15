#!/usr/bin/env python3
"""
Figura 7.4.1A — Action Fractions vs Distance.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

CSV = "dqn_evaluation_results.csv"

def main():
    df = pd.read_csv(CSV)
    if "distance_m" not in df.columns:
        df["distance_m"] = (df["step"] + 1) * 10.0

    scenarios = df["scenario"].unique()
    bins = 30

    for scen in scenarios:
        d = df[df["scenario"] == scen]

        centers = []
        frac_RF = []
        frac_FSO = []
        frac_THz = []

        edges = np.linspace(d["distance_m"].min(), d["distance_m"].max(), bins+1)

        for i in range(bins):
            lo, hi = edges[i], edges[i+1]
            mask = (d["distance_m"] >= lo) & (d["distance_m"] < hi)
            sub = d[mask]

            centers.append((lo + hi)/2)
            if len(sub) == 0:
                frac_RF.append(0)
                frac_FSO.append(0)
                frac_THz.append(0)
            else:
                total = len(sub)
                frac_RF.append((sub["action"] == 0).sum() / total)
                frac_FSO.append((sub["action"] == 1).sum() / total)
                frac_THz.append((sub["action"] == 2).sum() / total)

        plt.figure(figsize=(8, 4))
        plt.plot(centers, frac_RF, label="RF (0)")
        plt.plot(centers, frac_FSO, label="FSO (1)")
        plt.plot(centers, frac_THz, label="THz (2)")
        plt.xlabel("Distance (m)")
        plt.ylabel("Fraction of Actions")
       # plt.title(f"Figura 7.4.1A — Action Fractions vs Distance ({scen})")
        plt.grid(True, linestyle="--")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"fig_7_4_1A_action_fractions_{scen}.png", dpi=200)
        plt.close()

    print("Figura 7.4.1A gerada para todos os cenários.")

if __name__ == "__main__":
    main()
