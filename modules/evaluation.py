import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, classification_report
)

def run_evaluation(model, results):
    st.header("📊 Évaluation du Modèle")

    if model is None or results is None:
        st.warning("⚠️ Aucun modèle trouvé. Veuillez entraîner un modèle d'abord.")
        return

    # Récupération des données
    X_test = results.get("X_test")
    y_test = results.get("y_test")
    feature_cols = results.get("feature_cols", [])
    model_type = results.get("model_type", "Model")

    # Prédictions Test
    y_pred = model.predict(X_test)

    # Prédictions Train (pour comparaison)
    X_train = st.session_state.X_train[feature_cols]
    y_train = st.session_state.y_train
    y_pred_train = model.predict(X_train)

    # Métriques
    metrics = {
        "Accuracy": (
            accuracy_score(y_train, y_pred_train),
            accuracy_score(y_test, y_pred)
        ),
        "Precision": (
            precision_score(y_train, y_pred_train, average="weighted", zero_division=0),
            precision_score(y_test, y_pred, average="weighted", zero_division=0)
        ),
        "Recall": (
            recall_score(y_train, y_pred_train, average="weighted", zero_division=0),
            recall_score(y_test, y_pred, average="weighted", zero_division=0)
        ),
        "F1-Score": (
            f1_score(y_train, y_pred_train, average="weighted", zero_division=0),
            f1_score(y_test, y_pred, average="weighted", zero_division=0)
        )
    }

    # 📌 Affichage metrics
    st.subheader("📦 Performance du Modèle (Train vs Test)")
    cols = st.columns(4)
    for idx, (metric, (train_val, test_val)) in enumerate(metrics.items()):
        cols[idx].metric(
            metric,
            f"{test_val:.3f}",
            f"{test_val - train_val:+.3f}",
            help=f"Train: {train_val:.3f} vs Test: {test_val:.3f}"
        )

    st.write("---")

    # 🔥 Matrice de confusion
    st.subheader("📌 Matrice de Confusion")

    fig, ax = plt.subplots(figsize=(4, 3))

    labels = list(st.session_state.label_mapping.keys())
    ConfusionMatrixDisplay.from_predictions(
        y_test,
        model.predict(X_test),
        cmap="Blues",
        ax=ax
    )
    ax.set_title("Matrice de Confusion")
    st.pyplot(fig)

    st.write("---")

    # 📑 Classification Report
    st.subheader("📋 Classification Report")
    report_df = pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).transpose()
    st.dataframe(report_df.style.format("{:.3f}"), width="stretch")

    st.success("✅ Évaluation terminée")
