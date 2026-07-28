# Detecção de anomalias genérica com Isolation Forest.
# Funciona com qualquer CSV: usa todas as colunas numéricas como features.
# Se existir uma coluna de rótulo (0 = normal, 1 = anomalia), ela entra só na avaliação, nunca no treino.
#
# Uso: python deteccao_anomalias.py caminho/para/dados.csv

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report

DATA_PATH = "creditcard.csv"
LABEL_COL = None          # nome da coluna de rótulo, se houver (ex: "Class", "is_fraud")
CONTAMINATION = "auto"    # proporção esperada de anomalias; "auto" deixa o sklearn estimar


def load_data(path):
    try:
        return pd.read_csv(path), LABEL_COL
    except FileNotFoundError:
        return generate_synthetic_data()


def generate_synthetic_data(n=3000, n_features=8, anomaly_rate=0.03, seed=42):
    rng = np.random.default_rng(seed)
    n_anom = int(n * anomaly_rate)
    n_normal = n - n_anom

    normal = rng.normal(0, 1, size=(n_normal, n_features))
    anomalies = rng.normal(0, 3, size=(n_anom, n_features)) + rng.choice([-6, 6], size=(n_anom, n_features))

    X = np.vstack([normal, anomalies])
    y = np.array([0] * n_normal + [1] * n_anom)

    df = pd.DataFrame(X, columns=[f"feature_{i+1}" for i in range(n_features)])
    df["label"] = y
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    return df, "label"


def select_features(df, label_col):
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    if label_col in numeric_cols:
        numeric_cols.remove(label_col)
    return numeric_cols


def detect_anomalies(df, label_col, contamination):
    feature_cols = select_features(df, label_col)
    X = StandardScaler().fit_transform(df[feature_cols])

    model = IsolationForest(contamination=contamination, random_state=42, n_jobs=-1)
    model.fit(X)

    df["anomaly_score"] = model.decision_function(X)
    df["is_anomaly"] = model.predict(X) == -1
    return df, X


def plot_anomalies(X, is_anomaly, path="../resultados/anomalias.png"):
    coords = PCA(n_components=2, random_state=42).fit_transform(X) if X.shape[1] > 2 else X

    plt.figure(figsize=(8, 6))
    plt.scatter(coords[~is_anomaly, 0], coords[~is_anomaly, 1], s=8, alpha=0.4, label="normal")
    plt.scatter(coords[is_anomaly, 0], coords[is_anomaly, 1], s=20, color="red", label="anomalia")
    plt.xlabel("Componente 1")
    plt.ylabel("Componente 2")
    plt.title("Detecção de Anomalias - Isolation Forest")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else DATA_PATH
    df, label_col = load_data(path)
    df, X = detect_anomalies(df, label_col, CONTAMINATION)

    print(f"Anômalas: {df['is_anomaly'].sum()} de {len(df)} registros")

    if label_col and label_col in df.columns:
        print(classification_report(df[label_col], df["is_anomaly"], target_names=["normal", "anomalia"]))

    plot_anomalies(X, df["is_anomaly"].values)
    df.to_csv("../resultados/resultado_anomalias.csv", index=False)


if __name__ == "__main__":
    main()
