import streamlit as st
import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, matthews_corrcoef, confusion_matrix
)

st.title("Classification Models Demo")

# Upload test dataset
uploaded_file = st.file_uploader("Upload your test dataset (CSV)", type="csv")

# Model selection dropdown
model_choice = st.selectbox(
    "Select a model",
    ["Logistic Regression", "Decision Tree", "KNN", "Naive Bayes", "Random Forest"]
)

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.write("Dataset Preview:", data.head())

    # Assume last column is target
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    # Load chosen model
    model_filename = f"model/{model_choice.lower().replace(' ', '_')}.pkl"
    model = joblib.load(model_filename)

    # Predictions
    y_pred = model.predict(X)

    # Metrics
    acc = accuracy_score(y, y_pred)
    prec = precision_score(y, y_pred, zero_division=0)
    rec = recall_score(y, y_pred, zero_division=0)
    f1 = f1_score(y, y_pred, zero_division=0)
    mcc = matthews_corrcoef(y, y_pred)

    try:
        y_prob = model.predict_proba(X)[:,1]
        auc = roc_auc_score(y, y_prob)
    except:
        auc = None

    st.subheader("Evaluation Metrics")
    st.write(f"Accuracy: {acc:.3f}")
    st.write(f"Precision: {prec:.3f}")
    st.write(f"Recall: {rec:.3f}")
    st.write(f"F1 Score: {f1:.3f}")
    st.write(f"MCC: {mcc:.3f}")
    if auc is not None:
        st.write(f"AUC: {auc:.3f}")

    # Confusion Matrix
    cm = confusion_matrix(y, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    st.subheader("Confusion Matrix")
    st.pyplot(fig)
