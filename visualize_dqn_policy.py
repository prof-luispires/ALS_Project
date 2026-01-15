#!/usr/bin/env python3
"""
visualize_dqn_policy.py
-----------------------
Gera visualizações da política DQN a partir do ficheiro
dqn_evaluation_results.csv gerado por evaluate_dqn_multiscenario.py.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def load_logs(csv_path: str = "dqn_evaluation_results.csv") -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["distance_m"] = (df["step"] + 1) * 10.0
    return df

def plot_actions_per_scenario(df: pd.DataFrame, out_dir: str = "plots_dqn"):
    os.makedirs(out_dir, exist_ok=True)
    scenarios = df["scenario"].unique()
    summary_rows = []

    for scen in scenarios:
        df_s = df[df["scenario"] == scen].copy()
        correct = (df_s["action"] == df_s["best_channel"]).sum()
        total = len(df_s)
        acc = correct / total if total > 0 else 0.0
        summary_rows.append({"scenario": scen, "accuracy": acc, "total_steps": total})

        plt.figure(figsize=(10, 5))
        plt.scatter(df_s["distance_m"], df_s["best_channel"], marker="o", alpha=0.5, label="Best channel")
        plt.scatter(df_s["distance_m"], df_s["action"], marker="x", alpha=0.7, label="DQN action")
        plt.xlabel("Distance (m)")
        plt.ylabel("Channel index (0=RF, 1=FSO, 2=THz)")
        plt.title(f"Agent actions vs optimal channel ({scen})")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"actions_vs_optimal_{scen}.png"))
        plt.close()

        d_bins = np.linspace(df_s["distance_m"].min(), df_s["distance_m"].max(), 50)
        a_bins = [-0.5, 0.5, 1.5, 2.5]
        H, xedges, yedges = np.histogram2d(df_s["distance_m"], df_s["action"], bins=[d_bins, a_bins])

        plt.figure(figsize=(10, 4))
        plt.imshow(
            H.T,
            aspect="auto",
            origin="lower",
            extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
        )
        plt.colorbar(label="Counts")
        plt.yticks([0, 1, 2])
        plt.xlabel("Distance (m)")
        plt.ylabel("Action (channel index)")
        plt.title(f"Policy heatmap (distance vs action) — {scen}")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"heatmap_actions_{scen}.png"))
        plt.close()

    df_summary = pd.DataFrame(summary_rows)
    excel_path = os.path.join(out_dir, "policy_accuracy_summary.xlsx")
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        df_summary.to_excel(writer, sheet_name="accuracy_per_scenario", index=False)

    print("Resumo de accuracy por cenário:")
    print(df_summary)
    print(f"\nResumo guardado em: {excel_path}")
    print(f"Gráficos guardados na pasta: {out_dir}")

def main():
    df = load_logs("dqn_evaluation_results.csv")
    plot_actions_per_scenario(df)

if __name__ == "__main__":
    main()
