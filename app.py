# app.py
import os
import traceback

import streamlit as st
import pandas as pd
import joblib

from sklearn.metrics import confusion_matrix, classification_report
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

def load_csv(uploaded_file: st.runtime.uploaded_file_manager.UploadedFile) -> pd.DataFrame:
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
        st.error(f"❌ Model file not found: {model_path}\n\n"
                 f"Please ensure you committed model files to GitHub under '{MODEL_DIR}/'.")
        st.stop()

    model = joblib.load(model_path)
    st.success("✅ Model loaded successfully")

    # 4) Predict
    y_pred = model.predict(X)
    st.success("✅ Prediction completed")

    # 5) Evaluation outputs
    cm = confusion_matrix(y_true, y_pred)

    # Side-by-side layout for clean alignment
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
        st.pyplot(fig, use_container_width=False)

except Exception:
    st.error("❌ App crashed. Full error below (useful for debugging):")
    st.code(traceback.format_exc())
