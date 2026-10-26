#!/usr/bin/env python3
"""
train_test_unseen_scenarios.py
------------------------------
Treina modelos supervisionados apenas nos cenários:
    - clear
    - rain

Testa os modelos apenas nos cenários:
    - fog
    - worst

Objetivo:
avaliar a capacidade de generalização para condições não vistas.

Outputs:
    - unseen_generalization_summary.csv
    - unseen_generalization_summary.xlsx
    - unseen_confusion_matrices.xlsx
"""

from __future__ import annotations

import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)


# ---------------------------------------------------------
# 1) Configuração
# ---------------------------------------------------------
DATASET_PATH = "dataset_all_scenarios.csv"

TRAIN_SCENARIOS = ["clear", "fog"]
TEST_SCENARIOS = ["rain", "worst"]

#TRAIN_SCENARIOS = ["clear", "rain", "fog"]
#TEST_SCENARIOS = ["worst"]

FEATURE_COLS = [
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

TARGET_COL = "best_channel"


# ---------------------------------------------------------
# 2) Carregamento do dataset
# ---------------------------------------------------------
def load_dataset(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    required_cols = FEATURE_COLS + [TARGET_COL, "scenario"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Faltam colunas no dataset: {missing}")

    return df


# ---------------------------------------------------------
# 3) Split por cenário
# ---------------------------------------------------------
def split_by_scenario(df: pd.DataFrame):
    df_train = df[df["scenario"].isin(TRAIN_SCENARIOS)].copy()
    df_test = df[df["scenario"].isin(TEST_SCENARIOS)].copy()

    X_train = df_train[FEATURE_COLS].values
    y_train = df_train[TARGET_COL].values

    X_test = df_test[FEATURE_COLS].values
    y_test = df_test[TARGET_COL].values

    return df_train, df_test, X_train, y_train, X_test, y_test


# ---------------------------------------------------------
# 4) Modelos
# ---------------------------------------------------------
def build_models():
    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            random_state=42,
            n_jobs=-1,
        ),
        "SVM Linear": Pipeline([
            ("scaler", StandardScaler()),
            ("svc", SVC(kernel="linear"))
        ]),
        "SVM RBF": Pipeline([
            ("scaler", StandardScaler()),
            ("svc", SVC(kernel="rbf", gamma="scale"))
        ]),
    }
    return models


# ---------------------------------------------------------
# 5) Avaliação
# ---------------------------------------------------------
def evaluate_model(model, X_train, y_train, X_test, y_test, test_scenarios):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    overall_acc = accuracy_score(y_test, y_pred)
    overall_f1 = f1_score(y_test, y_pred, average="macro")
    overall_cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])

    report_dict = classification_report(
        y_test,
        y_pred,
        labels=[0, 1, 2],
        output_dict=True,
        zero_division=0
    )

    per_scenario_rows = []
    for scen in test_scenarios:
        mask = test_scenarios == scen
        y_true_s = y_test[mask]
        y_pred_s = y_pred[mask]

        acc_s = accuracy_score(y_true_s, y_pred_s)
        f1_s = f1_score(y_true_s, y_pred_s, average="macro")
        cm_s = confusion_matrix(y_true_s, y_pred_s, labels=[0, 1, 2])

        per_scenario_rows.append({
            "scenario": scen,
            "accuracy": acc_s,
            "f1_macro": f1_s,
            "n_samples": len(y_true_s),
            "cm_00_trueRF_predRF": cm_s[0, 0],
            "cm_01_trueRF_predFSO": cm_s[0, 1],
            "cm_02_trueRF_predTHz": cm_s[0, 2],
            "cm_10_trueFSO_predRF": cm_s[1, 0],
            "cm_11_trueFSO_predFSO": cm_s[1, 1],
            "cm_12_trueFSO_predTHz": cm_s[1, 2],
            "cm_20_trueTHz_predRF": cm_s[2, 0],
            "cm_21_trueTHz_predFSO": cm_s[2, 1],
            "cm_22_trueTHz_predTHz": cm_s[2, 2],
        })

    return {
        "y_pred": y_pred,
        "overall_acc": overall_acc,
        "overall_f1": overall_f1,
        "overall_cm": overall_cm,
        "report_dict": report_dict,
        "per_scenario": pd.DataFrame(per_scenario_rows),
    }


# ---------------------------------------------------------
# 6) Execução principal
# ---------------------------------------------------------
def main():
    df = load_dataset(DATASET_PATH)
    df_train, df_test, X_train, y_train, X_test, y_test = split_by_scenario(df)

    print("=== Split por cenário ===")
    print("Treino:", df_train["scenario"].value_counts().to_dict())
    print("Teste :", df_test["scenario"].value_counts().to_dict())
    print()

    print("=== Distribuição de classes ===")
    print("Treino - classes:", sorted(df_train["best_channel"].unique()))
    print(df_train["best_channel"].value_counts().sort_index())
    print()
    print("Teste - classes:", sorted(df_test["best_channel"].unique()))
    print(df_test["best_channel"].value_counts().sort_index())
    print()

    n_classes_train = df_train["best_channel"].nunique()
    if n_classes_train < 2:
        raise ValueError(
            f"O conjunto de treino tem apenas {n_classes_train} classe. "
            f"Não é possível treinar SVM. "
            f"Altere os cenários de treino ou regenere o dataset para incluir "
            f"pelo menos duas classes em best_channel."
        )

    test_scenarios_array = df_test["scenario"].values
    models = build_models()

    summary_rows = []
    confusion_sheets = {}

    for model_name, model in models.items():
        print(f"=== {model_name} ===")
        result = evaluate_model(
            model,
            X_train,
            y_train,
            X_test,
            y_test,
            test_scenarios_array
        )

        print(f"Overall test accuracy   : {result['overall_acc']:.4f}")
        print(f"Overall test F1-macro   : {result['overall_f1']:.4f}")
        print("Confusion matrix [0=RF,1=FSO,2=THz]:")
        print(result["overall_cm"])
        print()

        summary_rows.append({
            "model": model_name,
            "train_scenarios": ", ".join(TRAIN_SCENARIOS),
            "test_scenarios": ", ".join(TEST_SCENARIOS),
            "overall_test_accuracy": result["overall_acc"],
            "overall_test_f1_macro": result["overall_f1"],
            "train_samples": len(df_train),
            "test_samples": len(df_test),
        })

        per_scen = result["per_scenario"].copy()
        per_scen.insert(0, "model", model_name)
        confusion_sheets[model_name[:31]] = per_scen

    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv("unseen_generalization_summary.csv", index=False)

    with pd.ExcelWriter("unseen_generalization_summary.xlsx", engine="openpyxl") as writer:
        df_summary.to_excel(writer, sheet_name="summary", index=False)

    with pd.ExcelWriter("unseen_confusion_matrices.xlsx", engine="openpyxl") as writer:
        for sheet_name, df_sheet in confusion_sheets.items():
            df_sheet.to_excel(writer, sheet_name=sheet_name, index=False)

    print("[OK] Guardado: unseen_generalization_summary.csv")
    print("[OK] Guardado: unseen_generalization_summary.xlsx")
    print("[OK] Guardado: unseen_confusion_matrices.xlsx")

if __name__ == "__main__":
    main()