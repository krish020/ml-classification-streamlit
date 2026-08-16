"""
train_models.py
===============
Trains six classification models on the UCI Breast Cancer Wisconsin (Diagnostic)
dataset and persists them, together with their evaluation metrics, so that the
Streamlit application can load everything without re-training.

Dataset  : Breast Cancer Wisconsin (Diagnostic) - UCI ML Repository (ID 17)
Instances: 569          (requirement: >= 500)
Features : 30 numeric   (requirement: >= 12)
Target   : diagnosis -> 1 = Malignant (positive class), 0 = Benign

Run with:  python model/train_models.py
"""

import json
import os
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
TEST_SIZE = 0.30

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA_DIR = os.path.join(ROOT, "data")
TARGET_COL = "diagnosis"

os.makedirs(DATA_DIR, exist_ok=True)


# --------------------------------------------------------------------------- #
# 1. Load and prepare the dataset
# --------------------------------------------------------------------------- #
def load_dataset() -> pd.DataFrame:
    """Load the UCI Breast Cancer Wisconsin dataset into a tidy DataFrame."""
    raw = load_breast_cancer(as_frame=True)
    df = raw.data.copy()

    # Clean column names: "mean radius" -> "mean_radius"
    df.columns = [c.strip().replace(" ", "_").lower() for c in df.columns]

    # sklearn encodes 0 = malignant, 1 = benign.
    # Flip it so that the clinically interesting class (Malignant) is the
    # positive class (1) -- this makes Recall/Precision/AUC meaningful.
    df[TARGET_COL] = (raw.target == 0).astype(int)
    return df


# --------------------------------------------------------------------------- #
# 2. Model zoo
# --------------------------------------------------------------------------- #
def build_models() -> dict:
    """Return the six classifiers, each wrapped in a scaling Pipeline.

    Wrapping every model in an identical Pipeline means the Streamlit app can
    feed raw (unscaled) CSV values to any model and get correct predictions.
    """
    return {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=5000, C=1.0,
                                       random_state=RANDOM_STATE)),
        ]),
        "Decision Tree": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", DecisionTreeClassifier(max_depth=5, min_samples_leaf=5,
                                           criterion="gini",
                                           random_state=RANDOM_STATE)),
        ]),
        "kNN": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", KNeighborsClassifier(n_neighbors=7, weights="distance",
                                         metric="minkowski")),
        ]),
        "Naive Bayes": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", GaussianNB()),
        ]),
        "Random Forest (Ensemble)": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(n_estimators=300, max_depth=None,
                                           min_samples_leaf=1, n_jobs=-1,
                                           random_state=RANDOM_STATE)),
        ]),
        "Gradient Boosting (Ensemble)": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", GradientBoostingClassifier(n_estimators=200,
                                               learning_rate=0.1, max_depth=3,
                                               random_state=RANDOM_STATE)),
        ]),
    }


# --------------------------------------------------------------------------- #
# 3. Metric computation (shared with app.py)
# --------------------------------------------------------------------------- #
def evaluate(model, X, y) -> dict:
    """Compute the six required evaluation metrics for a fitted model."""
    y_pred = model.predict(X)

    # AUC needs a continuous score, not hard labels.
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X)[:, 1]
    else:
        y_score = model.decision_function(X)

    return {
        "Accuracy": round(float(accuracy_score(y, y_pred)), 4),
        "AUC": round(float(roc_auc_score(y, y_score)), 4),
        "Precision": round(float(precision_score(y, y_pred, zero_division=0)), 4),
        "Recall": round(float(recall_score(y, y_pred, zero_division=0)), 4),
        "F1": round(float(f1_score(y, y_pred, zero_division=0)), 4),
        "MCC": round(float(matthews_corrcoef(y, y_pred)), 4),
    }


# --------------------------------------------------------------------------- #
# 4. Main training routine
# --------------------------------------------------------------------------- #
def main() -> None:
    print("=" * 72)
    print("Breast Cancer Wisconsin (Diagnostic) - Multi-Model Classification")
    print("=" * 72)

    df = load_dataset()
    print(f"\nDataset shape          : {df.shape[0]} instances x "
          f"{df.shape[1] - 1} features")
    print(f"Class balance          : Malignant={int(df[TARGET_COL].sum())}, "
          f"Benign={int((df[TARGET_COL] == 0).sum())}")
    print(f"Missing values         : {int(df.isnull().sum().sum())}")

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"Train / Test split     : {len(X_train)} / {len(X_test)} "
          f"(stratified, {int(TEST_SIZE * 100)}% test)")

    # -- Persist the data splits ------------------------------------------- #
    df.to_csv(os.path.join(DATA_DIR, "full_dataset.csv"), index=False)

    train_df = X_train.copy()
    train_df[TARGET_COL] = y_train
    train_df.to_csv(os.path.join(DATA_DIR, "train_data.csv"), index=False)

    test_df = X_test.copy()
    test_df[TARGET_COL] = y_test
    # test_data.csv lives at the repo root -- this is what users upload
    # to the Streamlit app.
    test_df.to_csv(os.path.join(ROOT, "test_data.csv"), index=False)
    test_df.to_csv(os.path.join(DATA_DIR, "test_data.csv"), index=False)
    print(f"Saved test_data.csv    : {test_df.shape[0]} rows "
          f"x {test_df.shape[1]} cols (incl. target)")

    # -- Train, evaluate, persist ------------------------------------------ #
    models = build_models()
    results, confusions = {}, {}

    print("\n" + "-" * 72)
    print(f"{'Model':<30}{'Accuracy':>9}{'AUC':>8}{'Prec':>8}"
          f"{'Recall':>8}{'F1':>8}{'MCC':>8}")
    print("-" * 72)

    for name, model in models.items():
        model.fit(X_train, y_train)

        test_metrics = evaluate(model, X_test, y_test)
        train_metrics = evaluate(model, X_train, y_train)

        cv = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy")

        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        confusions[name] = cm.tolist()

        results[name] = {
            "test": test_metrics,
            "train": train_metrics,
            "cv_accuracy_mean": round(float(cv.mean()), 4),
            "cv_accuracy_std": round(float(cv.std()), 4),
            "confusion_matrix": cm.tolist(),
        }

        slug = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        joblib.dump(model, os.path.join(HERE, f"{slug}.pkl"))

        print(f"{name:<30}"
              f"{test_metrics['Accuracy']:>9.4f}"
              f"{test_metrics['AUC']:>8.4f}"
              f"{test_metrics['Precision']:>8.4f}"
              f"{test_metrics['Recall']:>8.4f}"
              f"{test_metrics['F1']:>8.4f}"
              f"{test_metrics['MCC']:>8.4f}")

    print("-" * 72)

    # -- Winner ------------------------------------------------------------ #
    winner = max(results, key=lambda k: (results[k]["test"]["MCC"],
                                         results[k]["test"]["AUC"]))
    print(f"\nOverall winner (by MCC, tie-break AUC): {winner}")

    # -- Persist metadata --------------------------------------------------- #
    metadata = {
        "dataset": {
            "name": "Breast Cancer Wisconsin (Diagnostic)",
            "source": "UCI Machine Learning Repository (ID 17)",
            "url": "https://archive.ics.uci.edu/dataset/17/"
                   "breast+cancer+wisconsin+diagnostic",
            "n_instances": int(df.shape[0]),
            "n_features": int(df.shape[1] - 1),
            "target": TARGET_COL,
            "classes": {"0": "Benign", "1": "Malignant"},
            "task": "Binary classification",
        },
        "split": {
            "test_size": TEST_SIZE,
            "random_state": RANDOM_STATE,
            "n_train": int(len(X_train)),
            "n_test": int(len(X_test)),
        },
        "feature_names": list(X.columns),
        "results": results,
        "winner": winner,
    }

    with open(os.path.join(HERE, "metrics.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Artifacts written to   : {HERE}")
    print("Done.\n")


if __name__ == "__main__":
    main()
