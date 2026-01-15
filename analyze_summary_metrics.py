#!/usr/bin/env python3
"""
analyze_summary_metrics.py
--------------------------
Script para ler o ficheiro 'summary_metrics_scenarios.xlsx' gerado pelo
run_all_channels_scenarios_metrics.py e produzir:

1) Tabelas de resumo (pivots) Eb_J e P_out por cenário e tecnologia;
2) Gráficos em barras (PNG) de:
   - Probabilidade de outage por cenário/tecnologia;
   - Energia por bit por cenário/tecnologia.

Requisitos:
    - summary_metrics_scenarios.xlsx na mesma pasta.
    - Bibliotecas: pandas, numpy, matplotlib.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os


def load_summary(filename: str = "summary_metrics_scenarios.xlsx") -> pd.DataFrame:
    """Carrega o ficheiro de resumo gerado anteriormente."""
    if not os.path.exists(filename):
        raise FileNotFoundError(
            f"Ficheiro '{filename}' não encontrado. "
            "Certifique-se de que já executou run_all_channels_scenarios_metrics.py."
        )
    df = pd.read_excel(filename)
    return df


def create_pivot_tables(df: pd.DataFrame, out_filename: str = "summary_metrics_tables.xlsx") -> None:
    """Cria tabelas resumo (pivot) para Eb e P_out e guarda em Excel."""
    # Pivot: Eb por cenário/tecnologia
    if "Eb_J" in df.columns:
        pivot_eb = df.pivot_table(
            index="scenario",
            columns="tech",
            values="Eb_J",
            aggfunc="mean"
        )
    else:
        pivot_eb = None

    # Pivot: P_out_capacity por cenário/tecnologia
    if "P_out_capacity" in df.columns:
        pivot_pout = df.pivot_table(
            index="scenario",
            columns="tech",
            values="P_out_capacity",
            aggfunc="mean"
        )
    else:
        pivot_pout = None

    # Guardar em Excel (várias sheets)
    with pd.ExcelWriter(out_filename) as writer:
        df.to_excel(writer, sheet_name="raw_summary", index=False)
        if pivot_eb is not None:
            pivot_eb.to_excel(writer, sheet_name="Eb_J_mean")
        if pivot_pout is not None:
            pivot_pout.to_excel(writer, sheet_name="P_out_capacity_mean")


def plot_bar_per_scenario(df: pd.DataFrame,
                          value_col: str,
                          ylabel: str,
                          prefix: str) -> None:
    """Cria um gráfico de barras por cenário (um ficheiro PNG por cenário).

    value_col : coluna com o valor (Eb_J, P_out_capacity, ...)
    ylabel    : texto a colocar no eixo Y
    prefix    : prefixo do nome do ficheiro, ex: 'Pout' -> 'Pout_clear.png'
    """
    scenarios = sorted(df["scenario"].unique())
    for scen in scenarios:
        df_s = df[df["scenario"] == scen].copy()

        # Ordenar tecnologias por nome para consistência visual
        df_s = df_s.sort_values("tech")

        x = np.arange(len(df_s))
        y = df_s[value_col].values
        labels = df_s["tech"].values

        plt.figure()
        plt.bar(x, y)
        plt.xticks(x, labels, rotation=45, ha="right")
        plt.ylabel(ylabel)
        plt.xlabel("Technology")
        plt.title(f"{ylabel} - scenario: {scen}")
        plt.tight_layout()
        filename = f"{prefix}_{scen}.png"
        plt.savefig(filename, dpi=150)
        plt.close()


def main():
    df = load_summary()

    # Cria tabelas em Excel
    create_pivot_tables(df)

    # Gráficos de P_out_capacity por cenário/tecnologia
    if "P_out_capacity" in df.columns:
        plot_bar_per_scenario(
            df,
            value_col="P_out_capacity",
            ylabel="Outage probability (capacity-based)",
            prefix="Pout_capacity"
        )

    # Gráficos de Eb_J por cenário/tecnologia
    if "Eb_J" in df.columns:
        plot_bar_per_scenario(
            df,
            value_col="Eb_J",
            ylabel="Energy per bit Eb [J/bit]",
            prefix="Eb"
        )

    print("Análise concluída:")
    print(" - summary_metrics_tables.xlsx criado com pivots.")
    print(" - Gráficos PNG por cenário para P_out_capacity e Eb_J criados.")


if __name__ == "__main__":
    main()
