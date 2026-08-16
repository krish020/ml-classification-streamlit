"""
app.py
======
Interactive Streamlit application for comparing six classification models
trained on the UCI Breast Cancer Wisconsin (Diagnostic) dataset.

Required features implemented:
  a. Dataset upload option (CSV, test data only)
  b. Model selection dropdown
  c. Display of evaluation metrics (Accuracy, AUC, Precision, Recall, F1, MCC)
  d. Confusion matrix AND classification report

Run locally with:  streamlit run app.py
"""

import io
import json
import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
st.set_page_config(
    page_title="Multi-Model Classification Explorer",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(ROOT, "model")
DATA_DIR = os.path.join(ROOT, "data")
TARGET_COL = "diagnosis"
CLASS_NAMES = ["Benign (0)", "Malignant (1)"]

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "kNN": "knn.pkl",
    "Naive Bayes": "naive_bayes.pkl",
    "Random Forest (Ensemble)": "random_forest_ensemble.pkl",
    "Gradient Boosting (Ensemble)": "gradient_boosting_ensemble.pkl",
}

sns.set_theme(style="whitegrid")


# --------------------------------------------------------------------------- #
# Loading helpers
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Loading trained models…")
def load_models():
    """Load the pickled pipelines.

    Falls back to re-training from data/train_data.csv if the pickles are
    missing or were written by an incompatible scikit-learn version. This
    keeps the deployed app working regardless of the resolved dependency
    versions on Streamlit Community Cloud.
    """
    models, note = {}, None
    try:
        for name, fname in MODEL_FILES.items():
            path = os.path.join(MODEL_DIR, fname)
            if not os.path.exists(path):
                raise FileNotFoundError(path)
            models[name] = joblib.load(path)
        return models, note
    except Exception as exc:  # noqa: BLE001
        note = (
            f"Saved model files could not be loaded ({type(exc).__name__}). "
            "Models were re-trained from `data/train_data.csv` on startup — "
            "results are identical."
        )

    # ---- Fallback: rebuild from the training split ------------------------ #
    import sys

    sys.path.insert(0, MODEL_DIR)
    from train_models import build_models  # noqa: WPS433

    train_df = pd.read_csv(os.path.join(DATA_DIR, "train_data.csv"))
    X_train = train_df.drop(columns=[TARGET_COL])
    y_train = train_df[TARGET_COL]

    models = build_models()
    for model in models.values():
        model.fit(X_train, y_train)
    return models, note


@st.cache_data(show_spinner=False)
def load_metadata():
    """Load metrics.json produced during training."""
    path = os.path.join(MODEL_DIR, "metrics.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


@st.cache_data(show_spinner=False)
def load_bundled_test_data():
    """Load the test_data.csv shipped with the repository."""
    path = os.path.join(ROOT, "test_data.csv")
    return pd.read_csv(path) if os.path.exists(path) else None


# --------------------------------------------------------------------------- #
# Metric helpers
# --------------------------------------------------------------------------- #
def get_scores(model, X):
    """Return probability/decision scores for the positive class."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)


def compute_metrics(y_true, y_pred, y_score):
    """Compute the six required evaluation metrics."""
    try:
        auc = roc_auc_score(y_true, y_score)
    except ValueError:
        auc = float("nan")  # only one class present in the uploaded file
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": auc,
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }


def validate(df, expected_features):
    """Check an uploaded DataFrame against the schema the models expect."""
    if TARGET_COL not in df.columns:
        return False, (
            f"The uploaded CSV has no `{TARGET_COL}` column. "
            "The target column is required so that metrics can be computed."
        )
    missing = [c for c in expected_features if c not in df.columns]
    if missing:
        preview = ", ".join(missing[:6])
        more = f" (+{len(missing) - 6} more)" if len(missing) > 6 else ""
        return False, f"Missing {len(missing)} required feature column(s): {preview}{more}"
    return True, ""


def plot_confusion_matrix(cm, title):
    fig, ax = plt.subplots(figsize=(4.2, 3.4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", cbar=False,
        xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
        annot_kws={"size": 13, "weight": "bold"}, ax=ax,
    )
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(title, fontsize=11, weight="bold")
    fig.tight_layout()
    return fig


def plot_roc(curves):
    fig, ax = plt.subplots(figsize=(5.4, 4.2))
    for label, (fpr, tpr, auc) in curves.items():
        ax.plot(fpr, tpr, lw=1.8, label=f"{label} (AUC = {auc:.4f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Chance (AUC = 0.5000)")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve", fontsize=11, weight="bold")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
models, load_note = load_models()
meta = load_metadata()
feature_names = meta["feature_names"] if meta else None

st.sidebar.title("⚙️ Controls")

# ---- (a) Dataset upload option ------------------------------------------- #
st.sidebar.subheader("1 · Upload test data")
uploaded = st.sidebar.file_uploader(
    "Upload a test CSV",
    type=["csv"],
    help="Must contain the 30 feature columns plus the `diagnosis` target column.",
)

use_bundled = st.sidebar.checkbox(
    "Use the bundled `test_data.csv` instead", value=True,
    help="Uncheck to evaluate only your own uploaded file.",
)

# ---- (b) Model selection dropdown ---------------------------------------- #
st.sidebar.subheader("2 · Select model")
model_choice = st.sidebar.selectbox(
    "Classification model", list(MODEL_FILES.keys()), index=4,
)

show_all = st.sidebar.checkbox("Compare all models side-by-side", value=True)

st.sidebar.divider()
if meta:
    st.sidebar.caption(
        f"**Dataset:** {meta['dataset']['name']}  \n"
        f"**Source:** {meta['dataset']['source']}  \n"
        f"**Size:** {meta['dataset']['n_instances']} instances × "
        f"{meta['dataset']['n_features']} features  \n"
        f"**Trained on:** {meta['split']['n_train']} rows "
        f"({int((1 - meta['split']['test_size']) * 100)}%)"
    )

# --------------------------------------------------------------------------- #
# Header
# --------------------------------------------------------------------------- #
st.title("🔬 Multi-Model Classification Explorer")
st.markdown(
    "Six classifiers trained on the **UCI Breast Cancer Wisconsin (Diagnostic)** "
    "dataset. Upload a test CSV, pick a model, and inspect its evaluation "
    "metrics, confusion matrix and classification report."
)

if load_note:
    st.info(load_note, icon="ℹ️")

# --------------------------------------------------------------------------- #
# Resolve which dataset to evaluate
# --------------------------------------------------------------------------- #
test_df, source_label = None, ""

if uploaded is not None:
    try:
        test_df = pd.read_csv(uploaded)
        source_label = f"uploaded file — `{uploaded.name}`"
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not read the uploaded CSV: {exc}")
elif use_bundled:
    test_df = load_bundled_test_data()
    source_label = "bundled `test_data.csv`"

if test_df is None:
    st.warning(
        "Upload a test CSV in the sidebar, or tick **Use the bundled "
        "`test_data.csv`** to explore the app with the held-out test split.",
        icon="📤",
    )
    bundled = load_bundled_test_data()
    if bundled is not None:
        st.download_button(
            "⬇️ Download a sample test_data.csv",
            bundled.to_csv(index=False).encode(),
            file_name="test_data.csv",
            mime="text/csv",
        )
    st.stop()

# ---- Validate schema ------------------------------------------------------ #
if feature_names is None:
    feature_names = [c for c in test_df.columns if c != TARGET_COL]

ok, message = validate(test_df, feature_names)
if not ok:
    st.error(message, icon="🚫")
    st.stop()

X_test = test_df[feature_names]
y_test = test_df[TARGET_COL].astype(int)

# --------------------------------------------------------------------------- #
# Data preview
# --------------------------------------------------------------------------- #
c1, c2, c3, c4 = st.columns(4)
c1.metric("Rows", f"{len(test_df):,}")
c2.metric("Features", f"{len(feature_names)}")
c3.metric("Malignant (1)", f"{int(y_test.sum()):,}")
c4.metric("Benign (0)", f"{int((y_test == 0).sum()):,}")

st.caption(f"Evaluating on the {source_label}.")

with st.expander("Preview the test data"):
    st.dataframe(test_df.head(20), width="stretch")

st.divider()

# --------------------------------------------------------------------------- #
# Single-model results
# --------------------------------------------------------------------------- #
model = models[model_choice]
y_pred = model.predict(X_test)
y_score = get_scores(model, X_test)
metrics = compute_metrics(y_test, y_pred, y_score)

st.header(f"📈 Results — {model_choice}")

# ---- (c) Display of evaluation metrics ------------------------------------ #
m = st.columns(6)
for col, (label, value) in zip(m, metrics.items()):
    col.metric(label, "—" if np.isnan(value) else f"{value:.4f}")

st.divider()

# ---- (d) Confusion matrix + classification report ------------------------- #
left, right = st.columns([1, 1.15])

with left:
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    st.pyplot(plot_confusion_matrix(cm, model_choice))

    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        st.caption(
            f"True Negatives **{tn}** · False Positives **{fp}** · "
            f"False Negatives **{fn}** · True Positives **{tp}**"
        )

with right:
    st.subheader("Classification Report")
    report = classification_report(
        y_test, y_pred, target_names=CLASS_NAMES,
        output_dict=True, zero_division=0,
    )
    report_df = pd.DataFrame(report).transpose().round(4)
    st.dataframe(report_df, width="stretch")

    st.subheader("ROC Curve")
    if len(np.unique(y_test)) > 1:
        fpr, tpr, _ = roc_curve(y_test, y_score)
        st.pyplot(plot_roc({model_choice: (fpr, tpr, metrics["AUC"])}))
    else:
        st.info("ROC curve needs both classes present in the test data.")

# ---- Downloadable predictions --------------------------------------------- #
pred_df = test_df.copy()
pred_df["predicted"] = y_pred
pred_df["prob_malignant"] = np.round(y_score, 4)
pred_df["correct"] = (pred_df["predicted"] == y_test).map({True: "✓", False: "✗"})

with st.expander("Row-level predictions"):
    st.dataframe(
        pred_df[[TARGET_COL, "predicted", "prob_malignant", "correct"]],
        width="stretch", height=320,
    )
    st.download_button(
        "⬇️ Download predictions as CSV",
        pred_df.to_csv(index=False).encode(),
        file_name=f"predictions_{model_choice.replace(' ', '_').lower()}.csv",
        mime="text/csv",
    )

# --------------------------------------------------------------------------- #
# All-model comparison
# --------------------------------------------------------------------------- #
if show_all:
    st.divider()
    st.header("🏁 Comparison across all models")

    rows, roc_curves = [], {}
    for name, mdl in models.items():
        preds = mdl.predict(X_test)
        scores = get_scores(mdl, X_test)
        row = {"ML Model Name": name}
        row.update(compute_metrics(y_test, preds, scores))
        rows.append(row)
        if len(np.unique(y_test)) > 1:
            fpr, tpr, _ = roc_curve(y_test, scores)
            roc_curves[name] = (fpr, tpr, row["AUC"])

    comp_df = pd.DataFrame(rows).set_index("ML Model Name")

    st.dataframe(
        comp_df.style
        .format("{:.4f}")
        .highlight_max(axis=0, color="#c8e6c9")
        .highlight_min(axis=0, color="#ffcdd2"),
        width="stretch",
    )
    st.caption("Green = best score in the column, red = worst.")

    winner = comp_df["MCC"].idxmax()
    st.success(
        f"**Overall winner on this test set: {winner}** "
        f"(MCC {comp_df.loc[winner, 'MCC']:.4f}, "
        f"Accuracy {comp_df.loc[winner, 'Accuracy']:.4f}, "
        f"AUC {comp_df.loc[winner, 'AUC']:.4f})",
        icon="🏆",
    )

    gl, gr = st.columns(2)

    with gl:
        st.subheader("Metric comparison")
        plot_df = comp_df.reset_index().melt(
            id_vars="ML Model Name", var_name="Metric", value_name="Score"
        )
        fig, ax = plt.subplots(figsize=(7.5, 4.4))
        sns.barplot(data=plot_df, x="Metric", y="Score",
                    hue="ML Model Name", ax=ax)
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("")
        ax.legend(fontsize=7, loc="lower right", ncol=2)
        fig.tight_layout()
        st.pyplot(fig)

    with gr:
        st.subheader("ROC curves")
        if roc_curves:
            st.pyplot(plot_roc(roc_curves))
        else:
            st.info("ROC curves need both classes present in the test data.")

    st.download_button(
        "⬇️ Download the comparison table",
        comp_df.to_csv().encode(),
        file_name="model_comparison.csv",
        mime="text/csv",
    )

# --------------------------------------------------------------------------- #
# Footer
# --------------------------------------------------------------------------- #
st.divider()
st.caption(
    "Built with Streamlit · scikit-learn · Dataset: "
    "Breast Cancer Wisconsin (Diagnostic), UCI ML Repository (ID 17). "
    "Positive class = Malignant."
)
