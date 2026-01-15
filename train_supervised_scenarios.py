"""
train_supervised_scenarios.py
-----------------------------
Treina modelos Random Forest e SVM (linear e RBF) para prever o canal ótimo
(0=RF, 1=FSO, 2=THz) usando o dataset unificado de cenários.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

def load_dataset(path: str = "dataset_all_scenarios.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    return df

def prepare_features(df: pd.DataFrame):
    # Features simples; pode ajustar conforme necessário
    feature_cols = [
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
    X = df[feature_cols].values
    y = df["best_channel"].values
    return X, y, feature_cols

def train_and_eval_model(model, X, y, name: str):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print(f"\n=== {name} ===")
    print(classification_report(y_test, y_pred))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))

def main():
    df = load_dataset()
    X, y, feature_cols = prepare_features(df)
    print(f"Dataset carregado com {len(df)} amostras.")
    print("Features:", feature_cols)

    # Random Forest
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        random_state=42,
        n_jobs=-1,
    )
    train_and_eval_model(rf, X, y, "Random Forest")

    # SVM linear
    svm_lin = Pipeline([
        ("scaler", StandardScaler()),
        ("svc", SVC(kernel="linear"))
    ])
    train_and_eval_model(svm_lin, X, y, "SVM (linear)")

    # SVM RBF
    svm_rbf = Pipeline([
        ("scaler", StandardScaler()),
        ("svc", SVC(kernel="rbf", gamma="scale"))
    ])
    train_and_eval_model(svm_rbf, X, y, "SVM (rbf)")

if __name__ == "__main__":
    main()
