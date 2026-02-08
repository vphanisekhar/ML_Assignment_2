import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="ML Assignment 2", layout="wide")

st.title("📊 Bank Marketing Classification App")

model_name = st.selectbox(
    "Select ML Model",
    ["Logistic Regression", "Decision Tree", "KNN",
     "Naive Bayes", "Random Forest", "XGBoost"]
)

uploaded_file = st.file_uploader("Upload CSV Test Data", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file, sep=';')

    if 'y' in df.columns:
        y_true = df['y'].map({'yes': 1, 'no': 0})
        X = df.drop('y', axis=1)
    else:
        st.error("Target column 'y' not found!")
        st.stop()

    model = joblib.load(f"model/saved/{model_name}.pkl")

    y_pred = model.predict(X)

    st.subheader("📈 Classification Report")
    st.text(classification_report(y_true, y_pred))

    st.subheader("📊 Confusion Matrix")
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    st.pyplot(fig)
