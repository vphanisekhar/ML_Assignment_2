# app.py
import os
import traceback

import streamlit as st
import pandas as pd
import joblib

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
)
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# Streamlit Page Config
# -----------------------------
st.set_page_config(
    page_title="Bank Marketing Classification App",
    layout="wide"
)

st.title("📊 Bank Marketing Classification App")

# -----------------------------
# Helpers
# -----------------------------
MODEL_DIR = os.path.join("model", "saved")

MODEL_OPTIONS = [
    "Logistic Regression",
    "Decision Tree",
    "KNN",
    "Naive Bayes",
    "Random Forest",
    "XGBoost"
]


def load_csv(uploaded_file) -> pd.DataFrame:
    """
    Load CSV robustly.
    Bank Marketing CSV from UCI is usually ';' separated.
    If it is comma separated, fallback automatically.
    """
    uploaded_file.seek(0)
    df = pd.read_csv(uploaded_file, sep=";")

    # If separator is wrong, everything might land in 1 column -> fallback to comma.
    if df.shape[1] == 1:
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file)

    return df


def safe_predict_proba(model, X):
    """
    Return probability of positive class if available; else None.
    """
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        # Binary classification: take probability of class 1
        if proba is not None and proba.shape[1] >= 2:
            return proba[:, 1]
    return None


# -----------------------------
# UI Controls
# -----------------------------
model_name = st.selectbox("Select ML Model", MODEL_OPTIONS)

uploaded_file = st.file_uploader(
    "Upload CSV Test Data (should include target column 'y')",
    type=["csv"]
)

# -----------------------------
# Main Flow
# -----------------------------
if not uploaded_file:
    st.info("👆 Please upload a CSV file to see predictions and evaluation output.")
    st.stop()

try:
    # 1) Read CSV
    df = load_csv(uploaded_file)

    # Show a quick preview (helps evaluators)
    with st.expander("🔎 Preview uploaded data", expanded=False):
        st.write("Rows, Cols:", df.shape)
        st.write("Columns:", df.columns.tolist())
        st.dataframe(df.head())

    # 2) Validate target column
    if "y" not in df.columns:
        st.error("❌ Column 'y' not found in uploaded CSV. Please upload a CSV that includes the target column 'y'.")
        st.stop()

    # Normalize target values safely
    df["y"] = df["y"].astype(str).str.strip().str.lower()
    y_true = df["y"].map({"yes": 1, "no": 0})

    if y_true.isna().any():
        bad_vals = df.loc[y_true.isna(), "y"].unique()
        st.error(f"❌ Found unexpected values in target column 'y': {bad_vals}. Expected only 'yes'/'no'.")
        st.stop()

    X = df.drop("y", axis=1)

    # 3) Load model
    model_path = os.path.join(MODEL_DIR, f"{model_name}.pkl")
    if not os.path.exists(model_path):
        st.error(
            f"❌ Model file not found: {model_path}\n\n"
            f"Please ensure you committed model files to GitHub under '{MODEL_DIR}/'."
        )
        st.stop()

    model = joblib.load(model_path)
    st.success("✅ Model loaded successfully")

    # 4) Predict
    y_pred = model.predict(X)
    y_prob = safe_predict_proba(model, X)  # used for AUC (if available)
    st.success("✅ Prediction completed")

    # 5) Metrics (Rubric-safe: Accuracy, AUC, Precision, Recall, F1, MCC)
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob) if y_prob is not None else None

    st.subheader("📌 Evaluation Metrics")
    c1, c2, c3 = st.columns(3)
    c1.metric("Accuracy", f"{acc:.4f}")
    c2.metric("AUC", f"{auc:.4f}" if auc is not None else "N/A")
    c3.metric("MCC", f"{mcc:.4f}")

    c4, c5, c6 = st.columns(3)
    c4.metric("Precision", f"{prec:.4f}")
    c5.metric("Recall", f"{rec:.4f}")
    c6.metric("F1 Score", f"{f1:.4f}")

    # 6) Report + Confusion Matrix (side-by-side)
    cm = confusion_matrix(y_true, y_pred)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📈 Classification Report")
        st.code(classification_report(y_true, y_pred))

    with col2:
        st.subheader("📊 Confusion Matrix")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=True, ax=ax)
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
        # ✅ Optional tiny improvement (clarity of 0/1)
        ax.set_xticklabels(["No (0)", "Yes (1)"])
        ax.set_yticklabels(["No (0)", "Yes (1)"])
        st.pyplot(fig, use_container_width=False)

except Exception:
    st.error("❌ App crashed. Full error below (useful for debugging):")
    st.code(traceback.format_exc())
