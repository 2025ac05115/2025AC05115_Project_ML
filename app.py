import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from models.logistic_regression import LogisticRegressionScratch
from models.knn import KNN
from models.naive_bayes import GaussianNB
from models.decision_tree import DecisionTree
from models.random_forest import RandomForest
from models.metrics import accuracy, precision, recall, f1_score, mcc, auc_score, confusion_matrix

st.title("Classification Models Without sklearn")

# Dataset upload option
uploaded_file = st.file_uploader("Upload your dataset (CSV)", type="csv")
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file, sep=";")
    st.write("Dataset Preview:", data.head())

    X = data.iloc[:, :-1].values
    y = (data.iloc[:, -1].values >= 6).astype(int)

    # Train/Test Split
    np.random.seed(42)
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    split = int(0.7 * len(X))
    train_idx, test_idx = indices[:split], indices[split:]
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Models
    models = {
        "Logistic Regression": LogisticRegressionScratch(),
        "Decision Tree": DecisionTree(max_depth=5),
        "KNN": KNN(k=5),
        "Naive Bayes": GaussianNB(),
        "Random Forest": RandomForest(n_estimators=10, max_depth=5)
    }

    # Model selection dropdown
    choice = st.selectbox("Select a model", list(models.keys()))
    model = models[choice]
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Metrics
    acc = accuracy(y_test, y_pred)
    prec = precision(y_test, y_pred)
    rec = recall(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    mcc_val = mcc(y_test, y_pred)

    try:
        y_prob = model.predict_proba(X_test)
        auc = auc_score(y_test, y_prob)
    except:
        auc = None

    # Display metrics
    st.subheader("Evaluation Metrics")
    st.write(f"Accuracy: {acc:.3f}")
    st.write(f"Precision: {prec:.3f}")
    st.write(f"Recall: {rec:.3f}")
    st.write(f"F1 Score: {f1:.3f}")
    st.write(f"MCC: {mcc_val:.3f}")
    if auc is not None:
        st.write(f"AUC: {auc:.3f}")

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Pred 0","Pred 1"],
                yticklabels=["Actual 0","Actual 1"], ax=ax)
    ax.set_title(f"Confusion Matrix - {choice}")
    st.pyplot(fig)
