#!/usr/bin/env python3
"""
train_test_unseen_protocols.py
------------------------------
Avalia generalização supervisionada em múltiplos protocolos de cenários
não vistos, com e sem ruído gaussiano nas SNRs.

Protocolos:
    A: treino = clear + fog        | teste = rain + worst
    B: treino = clear + rain       | teste = fog + worst
    C: treino = clear + rain + fog | teste = worst

Condições:
    - BASELINE: sem ruído
    - NOISE   : com ruído gaussiano nas SNRs

Outputs:
    - unseen_protocols_all_results.csv
    - unseen_protocols_all_results.xlsx
    - unseen_protocols_confusion_matrices.xlsx
"""

from __future__ import annotations

import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix


# ---------------------------------------------------------
# 1) Configuração global
# ---------------------------------------------------------
DATASET_PATH = "dataset_all_scenarios.csv"

FEATURE_COLS = [
    "distance_m",
    "rain_rate_mmph",
    "humidity_pct",
    "visibility_km",
    "Cn2",
]

TARGET_COL = "best_channel"

PROTOCOLS = [
    {
        "protocol": "A",
        "train_scenarios": ["clear", "fog"],
        "test_scenarios": ["rain", "worst"],
    },
    {
        "protocol": "B",
        "train_scenarios": ["clear", "rain"],
        "test_scenarios": ["fog", "worst"],
    },
    {
        "protocol": "C",
        "train_scenarios": ["clear", "rain", "fog"],
        "test_scenarios": ["worst"],
    },
]

NOISE_SETTINGS = [
    {
        "noise_label": "BASELINE",
        "use_noise": False,
        "noise_std_db": 0.0,
        "noise_seed": 42,
    },
    {
        "noise_label": "NOISE",
        "use_noise": True,
        "noise_std_db": 2.0,
        "noise_seed": 42,
    },
]


# ---------------------------------------------------------
# 2) Utilitários
# ---------------------------------------------------------
def load_dataset(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    required_cols = FEATURE_COLS + [TARGET_COL, "scenario"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Faltam colunas no dataset: {missing}")

    return df


def add_snr_noise(df: pd.DataFrame, std_db: float = 2.0, seed: int = 42) -> pd.DataFrame:
    """
    Adiciona ruído gaussiano independente às SNRs.
    Não altera a coluna best_channel.
    """
    rng = np.random.default_rng(seed)
    df_noisy = df.copy()

    df_noisy["SNR_RF_dB"] += rng.normal(0.0, std_db, len(df_noisy))
    df_noisy["SNR_FSO_dB"] += rng.normal(0.0, std_db, len(df_noisy))
    df_noisy["SNR_THz_dB"] += rng.normal(0.0, std_db, len(df_noisy))

    return df_noisy


def split_by_protocol(df: pd.DataFrame, train_scenarios: list[str], test_scenarios: list[str]):
    df_train = df[df["scenario"].isin(train_scenarios)].copy()
    df_test = df[df["scenario"].isin(test_scenarios)].copy()

    X_train = df_train[FEATURE_COLS].values
    y_train = df_train[TARGET_COL].values

    X_test = df_test[FEATURE_COLS].values
    y_test = df_test[TARGET_COL].values

    return df_train, df_test, X_train, y_train, X_test, y_test


def build_models():
    return {
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


def evaluate_model(model, X_train, y_train, X_test, y_test, test_scenarios_array):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    overall_acc = accuracy_score(y_test, y_pred)
    overall_f1 = f1_score(y_test, y_pred, average="macro")
    overall_cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])

    per_scenario_rows = []
    unique_test_scenarios = pd.unique(test_scenarios_array)

    for scen in unique_test_scenarios:
        mask = test_scenarios_array == scen
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
        "overall_acc": overall_acc,
        "overall_f1": overall_f1,
        "overall_cm": overall_cm,
        "per_scenario": pd.DataFrame(per_scenario_rows),
    }


# ---------------------------------------------------------
# 3) Execução
# ---------------------------------------------------------
def main():
    df_original = load_dataset(DATASET_PATH)
    models = build_models()

    summary_rows = []
    detailed_rows = []
    confusion_sheets = {}

    for noise_cfg in NOISE_SETTINGS:
        noise_label = noise_cfg["noise_label"]
        use_noise = noise_cfg["use_noise"]
        noise_std_db = noise_cfg["noise_std_db"]
        noise_seed = noise_cfg["noise_seed"]

        if use_noise:
            print(f"\n========== {noise_label} ==========")
            print(f"[INFO] A aplicar ruído gaussiano às SNRs: σ = {noise_std_db:.2f} dB")
            df_eval = add_snr_noise(df_original, std_db=noise_std_db, seed=noise_seed)
        else:
            print(f"\n========== {noise_label} ==========")
            print("[INFO] Execução baseline (sem ruído)")
            df_eval = df_original.copy()

        for proto in PROTOCOLS:
            proto_name = proto["protocol"]
            train_scenarios = proto["train_scenarios"]
            test_scenarios = proto["test_scenarios"]

            print(f"\n===== Protocolo {proto_name} =====")
            print("Treino:", train_scenarios)
            print("Teste :", test_scenarios)

            df_train, df_test, X_train, y_train, X_test, y_test = split_by_protocol(
                df_eval,
                train_scenarios=train_scenarios,
                test_scenarios=test_scenarios,
            )

            print("Distribuição treino:", df_train["scenario"].value_counts().to_dict())
            print("Distribuição teste :", df_test["scenario"].value_counts().to_dict())
            print("Classes treino     :", sorted(df_train[TARGET_COL].unique()))
            print(df_train[TARGET_COL].value_counts().sort_index())
            print("Classes teste      :", sorted(df_test[TARGET_COL].unique()))
            print(df_test[TARGET_COL].value_counts().sort_index())

            n_classes_train = df_train[TARGET_COL].nunique()
            if n_classes_train < 2:
                print(f"[AVISO] Protocolo {proto_name} ignorado: treino mono-classe.\n")
                continue

            test_scenarios_array = df_test["scenario"].values

            for model_name, model in models.items():
                print(f"\n--- {model_name} ---")
                result = evaluate_model(
                    model,
                    X_train,
                    y_train,
                    X_test,
                    y_test,
                    test_scenarios_array
                )

                print(f"Overall test accuracy : {result['overall_acc']:.4f}")
                print(f"Overall test F1-macro : {result['overall_f1']:.4f}")
                print("Confusion matrix [0=RF,1=FSO,2=THz]:")
                print(result["overall_cm"])

                summary_rows.append({
                    "protocol": proto_name,
                    "noise_label": noise_label,
                    "noise_enabled": use_noise,
                    "noise_std_db": noise_std_db,
                    "model": model_name,
                    "train_scenarios": ", ".join(train_scenarios),
                    "test_scenarios": ", ".join(test_scenarios),
                    "train_classes": str(sorted(df_train[TARGET_COL].unique().tolist())),
                    "test_classes": str(sorted(df_test[TARGET_COL].unique().tolist())),
                    "train_samples": len(df_train),
                    "test_samples": len(df_test),
                    "overall_test_accuracy": result["overall_acc"],
                    "overall_test_f1_macro": result["overall_f1"],
                    "cm_00": result["overall_cm"][0, 0],
                    "cm_01": result["overall_cm"][0, 1],
                    "cm_02": result["overall_cm"][0, 2],
                    "cm_10": result["overall_cm"][1, 0],
                    "cm_11": result["overall_cm"][1, 1],
                    "cm_12": result["overall_cm"][1, 2],
                    "cm_20": result["overall_cm"][2, 0],
                    "cm_21": result["overall_cm"][2, 1],
                    "cm_22": result["overall_cm"][2, 2],
                })

                per_scen = result["per_scenario"].copy()
                per_scen.insert(0, "protocol", proto_name)
                per_scen.insert(1, "noise_label", noise_label)
                per_scen.insert(2, "noise_enabled", use_noise)
                per_scen.insert(3, "noise_std_db", noise_std_db)
                per_scen.insert(4, "model", model_name)
                per_scen.insert(5, "train_scenarios", ", ".join(train_scenarios))
                per_scen.insert(6, "test_scenarios", ", ".join(test_scenarios))

                detailed_rows.append(per_scen)

                sheet_name = f"{proto_name}_{noise_label}_{model_name}".replace(" ", "_")
                sheet_name = sheet_name[:31]
                confusion_sheets[sheet_name] = per_scen

    if not summary_rows:
        raise RuntimeError(
            "Nenhum protocolo/modelo foi executado com sucesso. "
            "Verifique a distribuição de classes no dataset."
        )

    df_summary = pd.DataFrame(summary_rows)
    df_detailed = pd.concat(detailed_rows, ignore_index=True) if detailed_rows else pd.DataFrame()

    csv_name = "unseen_protocols_all_results.csv"
    xlsx_name = "unseen_protocols_all_results.xlsx"
    xlsx_cm_name = "unseen_protocols_confusion_matrices.xlsx"

    df_summary.to_csv(csv_name, index=False)

    with pd.ExcelWriter(xlsx_name, engine="openpyxl") as writer:
        df_summary.to_excel(writer, sheet_name="summary", index=False)
        if not df_detailed.empty:
            df_detailed.to_excel(writer, sheet_name="per_scenario", index=False)

    with pd.ExcelWriter(xlsx_cm_name, engine="openpyxl") as writer:
        for sheet_name, df_sheet in confusion_sheets.items():
            df_sheet.to_excel(writer, sheet_name=sheet_name, index=False)

    print(f"\n[OK] Guardado: {csv_name}")
    print(f"[OK] Guardado: {xlsx_name}")
    print(f"[OK] Guardado: {xlsx_cm_name}")


if __name__ == "__main__":
    main()