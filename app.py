import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import traceback

st.set_page_config(page_title="ML Assignment 2", layout="wide")

st.title("📊 Bank Marketing Classification App")

model_name = st.selectbox(
    "Select ML Model",
    ["Logistic Regression", "Decision Tree", "KNN",
     "Naive Bayes", "Random Forest", "XGBoost"]
)

uploaded_file = st.file_uploader("Upload CSV Test Data", type="csv")

if uploaded_file:
   try:
    st.write("File uploaded ✅")
    df = pd.read_csv(uploaded_file, sep=';')
    st.write("Shape:", df.shape)
    st.write("Columns:", df.columns)
    st.write(df.head())

    st.write("✅ Columns detected:", df.columns.tolist())
    st.write("✅ Last 5 columns:", df.columns.tolist()[-5:])
    st.write("✅ Does y exist?:", 'y' in df.columns)


    if df.shape[1] == 1:
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file)  # fallback comma
   except Exception:
    uploaded_file.seek(0)
    df = pd.read_csv(uploaded_file)  # fallback

    if 'y' in df.columns:
        y_true = df['y'].map({'yes': 1, 'no': 0})
        X = df.drop('y', axis=1)
    else:
        st.error("Target column 'y' not found!")
        st.stop()

try:
    model = joblib.load(f"model/saved/{model_name}.pkl")
    st.success("✅ Model loaded successfully")

    # Prepare X, y
    df['y'] = df['y'].astype(str).str.strip().str.lower()
    y_true = df['y'].map({'yes': 1, 'no': 0})

    if y_true.isna().any():
        bad_vals = df.loc[y_true.isna(), 'y'].unique()
        st.error(f"❌ Found unexpected target values in 'y': {bad_vals}")
        st.stop()

    X = df.drop('y', axis=1)

    y_pred = model.predict(X)
    st.success("✅ Prediction completed")

    st.subheader("📈 Classification Report")
    st.text(classification_report(y_true, y_pred))

    st.subheader("📊 Confusion Matrix")
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    st.pyplot(fig)

except Exception as e:
    st.error("❌ App crashed while predicting. Full error below:")
    st.code(traceback.format_exc())
