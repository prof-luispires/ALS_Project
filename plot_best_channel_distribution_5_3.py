#!/usr/bin/env python3
"""
plot_best_channel_distribution_5_3.py
Gera a Figura 5.3: Distribuição do best_channel por cenário.
best_channel: 0=RF, 1=FSO, 2=THz
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

DATASET = "dataset_all_scenarios.csv"

CHANNEL_MAP = {
    0: "RF",
    1: "FSO",
    2: "THz",
}

def main():
    df = pd.read_csv(DATASET)

    if "scenario" not in df.columns or "best_channel" not in df.columns:
        raise ValueError("É necessário ter colunas 'scenario' e 'best_channel' no CSV.")

    scenarios = df["scenario"].unique()
    channels = [0, 1, 2]

    # Preparar matriz: linhas cenários, colunas canais
    counts = []
    for scen in scenarios:
        d = df[df["scenario"] == scen]
        row = []
        for ch in channels:
            row.append((d["best_channel"] == ch).sum())
        counts.append(row)

    counts = np.array(counts, dtype=float)
    totals = counts.sum(axis=1, keepdims=True)
    perc = counts / totals  # percentagens

    x = np.arange(len(scenarios))  # posição dos cenários
    width = 0.25

    fig, ax = plt.subplots(figsize=(8, 5))

    for i, ch in enumerate(channels):
        ax.bar(
            x + (i - 1) * width,
            perc[:, i],
            width,
            label=CHANNEL_MAP[ch],
        )

    ax.set_xticks(x)
    ax.set_xticklabels(scenarios)
    ax.set_ylabel("Fraction of samples (best_channel)")
   # ax.set_title("Figure 5.3 – Best channel distribution per scenario")
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)
    ax.legend()

    plt.tight_layout()
    fname = "fig_5_3_best_channel_distribution.png"
    plt.savefig(fname, dpi=200)
    plt.close()
    print(f"Guardado: {fname}")

if __name__ == "__main__":
    main()
