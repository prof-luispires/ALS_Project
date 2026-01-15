#!/usr/bin/env python3
"""
analyze_best_channel.py
-----------------------
Analisa o dataset_all_scenarios.* e produz:
  • Tabela de contagens de best_channel por cenário
  • Tabela de percentagens de best_channel por cenário
  • SNR médio por tecnologia e cenário
  • Ficheiro Excel "best_channel_analysis.xlsx" com 3 folhas:
        - counts
        - percentages
        - snr_means
  • Gráficos simples de percentagem de best_channel por cenário
    guardados na pasta "best_channel_plots"
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_dataset(path_xlsx="dataset_all_scenarios.xlsx",
                 path_csv="dataset_all_scenarios.csv") -> pd.DataFrame:
    if os.path.exists(path_xlsx):
        print(f"[INFO] A carregar {path_xlsx}")
        return pd.read_excel(path_xlsx)
    elif os.path.exists(path_csv):
        print(f"[INFO] A carregar {path_csv}")
        return pd.read_csv(path_csv)
    else:
        raise FileNotFoundError(
            "Não encontrei 'dataset_all_scenarios.xlsx' nem 'dataset_all_scenarios.csv'."
        )


def analyze_best_channel(df: pd.DataFrame):
    # Garantir que existe coluna 'best_channel' e 'scenario'
    if "best_channel" not in df.columns or "scenario" not in df.columns:
        raise ValueError("O dataset não contém as colunas 'best_channel' e/ou 'scenario'.")

    # Contagens globais
    global_counts = df["best_channel"].value_counts().sort_index()

    # Contagens por cenário
    counts_by_scenario = (
        df.groupby("scenario")["best_channel"]
        .value_counts()
        .unstack(fill_value=0)
        .sort_index(axis=1)
    )

    # Percentagens por cenário
    percentages_by_scenario = counts_by_scenario.div(
        counts_by_scenario.sum(axis=1), axis=0
    ) * 100.0

    # SNR médio por tecnologia e cenário
    snr_cols = ["SNR_RF_dB", "SNR_FSO_dB", "SNR_THz_dB"]
    missing = [c for c in snr_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Faltam colunas de SNR no dataset: {missing}")

    snr_means = df.groupby("scenario")[snr_cols].mean()

    return global_counts, counts_by_scenario, percentages_by_scenario, snr_means


def save_to_excel(counts_by_scenario, percentages_by_scenario, snr_means,
                  out_path="best_channel_analysis.xlsx"):
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        counts_by_scenario.to_excel(writer, sheet_name="counts")
        percentages_by_scenario.to_excel(writer, sheet_name="percentages")
        snr_means.to_excel(writer, sheet_name="snr_means")
    print(f"[OK] Resultados detalhados guardados em: {out_path}")


def plot_best_channel_percentages(percentages_by_scenario,
                                  out_dir="best_channel_plots"):
    os.makedirs(out_dir, exist_ok=True)

    scenarios = percentages_by_scenario.index.tolist()
    channels = percentages_by_scenario.columns.tolist()

    for scen in scenarios:
        row = percentages_by_scenario.loc[scen]

        plt.figure(figsize=(6, 4))
        plt.bar([str(ch) for ch in channels], row.values)
        plt.xlabel("best_channel (0=RF, 1=FSO, 2=THz)")
        plt.ylabel("Percentagem (%)")
        plt.title(f"Distribuição de best_channel — cenário: {scen}")
        plt.grid(axis="y")
        plt.tight_layout()

        fname = os.path.join(out_dir, f"best_channel_percent_{scen}.png")
        plt.savefig(fname)
        plt.close()
        print(f"[OK] Gráfico guardado: {fname}")


def main():
    df = load_dataset()
    global_counts, counts_by_scenario, percentages_by_scenario, snr_means = analyze_best_channel(df)

    print("\n[GLOBAL] Contagens de best_channel:")
    print(global_counts)

    print("\n[POR CENÁRIO] Contagens de best_channel:")
    print(counts_by_scenario)

    print("\n[POR CENÁRIO] Percentagens de best_channel (%):")
    print(percentages_by_scenario.round(2))

    print("\n[POR CENÁRIO] SNR médio por tecnologia (dB):")
    print(snr_means.round(2))

    save_to_excel(counts_by_scenario, percentages_by_scenario, snr_means)
    plot_best_channel_percentages(percentages_by_scenario)


if __name__ == "__main__":
    main()
