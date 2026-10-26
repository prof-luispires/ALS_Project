#!/usr/bin/env python3
"""
train_supervised_models.py
--------------------------
Treino dos modelos supervisionados para seleção de canal:
    - Random Forest
    - SVM (kernel linear e RBF)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix


def load_dataset(path: str = "dataset_supervised_channels.csv"):
    df = pd.read_csv(path)
    if "best_channel" not in df.columns:
        raise ValueError("Coluna 'best_channel' não encontrada no dataset.")
    X = df.drop(columns=["best_channel"])
    y = df["best_channel"]
    return X, y


def train_random_forest(X_train, X_test, y_train, y_test):
    print("\n=== Random Forest ===")
    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        n_jobs=-1,
        random_state=42,
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")
    print(f"Accuracy : {acc:.4f}")
    print(f"F1-score : {f1:.4f}")
    print("\nClassification report:\n", classification_report(y_test, y_pred))
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
    return clf


def make_svm_pipeline(kernel: str = "linear", C: float = 1.0, gamma: str | float = "scale"):
    svm = SVC(kernel=kernel, C=C, gamma=gamma)
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("svm", svm),
        ]
    )
    return pipe


def train_svm(X_train, X_test, y_train, y_test, kernel: str):
    print(f"\n=== SVM (kernel={kernel}) ===")
    if kernel == "linear":
        model = make_svm_pipeline(kernel="linear", C=1.0)
    else:  # RBF
        model = make_svm_pipeline(kernel="rbf", C=5.0, gamma="scale")

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")
    print(f"Accuracy : {acc:.4f}")
    print(f"F1-score : {f1:.4f}")
    print("\nClassification report:\n", classification_report(y_test, y_pred))
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
    return model


def main():
    X, y = load_dataset()
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # Random Forest
    rf_model = train_random_forest(X_train, X_test, y_train, y_test)

    # SVM linear
    svm_linear = train_svm(X_train, X_test, y_train, y_test, kernel="linear")

    # SVM RBF
    svm_rbf = train_svm(X_train, X_test, y_train, y_test, kernel="rbf")

    print("\nTreino concluído.")
    print("Sugestão: guardar modelos com joblib se quiser reutilizá-los em produção.")


if __name__ == "__main__":
    main()
