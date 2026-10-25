"""
ML Network Intrusion Detector
Binary classification of CIC-IDS2017 network-flow records using Random Forest.

Dataset:
https://www.unb.ca/cic/datasets/ids-2017.html

Place the extracted CSV files in a folder named "data" beside this script.
Install dependencies: pip install -r requirements.txt
Run: python YourName_ML_Network_Intrusion_Detector.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, classification_report, confusion_matrix)

RANDOM_STATE = 42
DATA_DIR = Path("data")
MAX_ROWS = 150_000


def main():
    csv_files = sorted(DATA_DIR.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(
            "No CSV files found. Put CIC-IDS2017 CSV files in a 'data' folder beside this script."
        )

    frames = []
    for file in csv_files:
        frame = pd.read_csv(file, low_memory=False, encoding_errors="replace")
        frame.columns = [str(c).strip() for c in frame.columns]
        frames.append(frame)
        print(f"Loaded {file.name}: {len(frame):,} rows")

    data = pd.concat(frames, ignore_index=True, sort=False)
    data.columns = [str(c).strip() for c in data.columns]
    label_candidates = [c for c in data.columns if c.lower() == "label"]
    if not label_candidates:
        raise ValueError("Could not find a Label column in the input CSV files.")

    label_col = label_candidates[0]
    data = data.dropna(subset=[label_col]).copy()
    labels = data[label_col].astype(str).str.strip()
    y = (labels.str.upper() != "BENIGN").astype("int8")
    X = data.drop(columns=[label_col])

    identifier_columns = [
        c for c in X.columns
        if c.strip().lower() in {
            "flow id", "source ip", "destination ip", "timestamp",
            "src ip", "dst ip"
        }
    ]
    X = X.drop(columns=identifier_columns, errors="ignore")
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.dropna(axis=1, how="all")
    X = X.fillna(X.median(numeric_only=True)).fillna(0)
    X = X.loc[:, X.nunique(dropna=False) > 1]
    X = X.loc[:, ~X.columns.duplicated()]

    if len(X) > MAX_ROWS:
        sampled_idx = data.sample(n=MAX_ROWS, random_state=RANDOM_STATE).index
        X, y = X.loc[sampled_idx], y.loc[sampled_idx]

    if X.empty or y.nunique() < 2:
        raise ValueError("Need numeric features and both benign and attack labels.")

    print("\nRows used:", len(X))
    print("Class counts (0=Benign, 1=Attack):")
    print(y.value_counts().rename(index={0: "Benign", 1: "Attack"}))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )
    model = RandomForestClassifier(
        n_estimators=150, class_weight="balanced_subsample",
        random_state=RANDOM_STATE, n_jobs=-1, min_samples_leaf=2
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("\nAccuracy:", round(accuracy_score(y_test, y_pred), 4))
    print("Attack precision:", round(precision_score(y_test, y_pred, zero_division=0), 4))
    print("Attack recall:", round(recall_score(y_test, y_pred, zero_division=0), 4))
    print("Attack F1-score:", round(f1_score(y_test, y_pred, zero_division=0), 4))
    print("\nClassification report:")
    print(classification_report(
        y_test, y_pred, target_names=["Benign", "Attack"], zero_division=0
    ))
    print("Confusion matrix (rows=actual, columns=predicted):")
    print(confusion_matrix(y_test, y_pred, labels=[0, 1]))


if __name__ == "__main__":
    main()
