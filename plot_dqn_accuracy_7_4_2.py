#!/usr/bin/env python3
"""
Figura 7.4.2 — DQN accuracy per scenario.
"""

import pandas as pd
import matplotlib.pyplot as plt

EXCEL = "policy_accuracy_summary.xlsx"

def main():
    df = pd.read_excel(EXCEL)

    plt.figure(figsize=(6,4))
    plt.bar(df["scenario"], df["accuracy"], color="gray")
    plt.xlabel("Scenario")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1.05)
    plt.title("Figura 7.4.2 — DQN Policy Accuracy per Scenario")
    plt.grid(True, axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig("fig_7_4_2_accuracy.png", dpi=200)
    plt.close()

    print("Figura 7.4.2 criada com sucesso.")

if __name__ == "__main__":
    main()
