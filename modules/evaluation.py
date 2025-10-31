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
    st.header("Évaluation du modèle")
 
    if model is None or results is None:
        st.warning("⚠️ Aucun modèle trouvé. Veuillez entraîner un modèle d'abord.")
        return
 
    # --- Récupération des données ---
    X_test = results.get("X_test")
    y_test = results.get("y_test")
    feature_cols = results.get("feature_cols", [])
    model_type = results.get("model_type", "Model")
 
    # --- Prédictions Test & Train ---
    y_pred_test = model.predict(X_test)
    X_train = st.session_state.X_train[feature_cols]
    y_train = st.session_state.y_train
    y_pred_train = model.predict(X_train)
 
    # --- Calcul des métriques ---
    metrics = {
        "Accuracy": (
            accuracy_score(y_train, y_pred_train),
            accuracy_score(y_test, y_pred_test)
        ),
        "Precision": (
            precision_score(y_train, y_pred_train, average="weighted", zero_division=0),
            precision_score(y_test, y_pred_test, average="weighted", zero_division=0)
        ),
        "Recall": (
            recall_score(y_train, y_pred_train, average="weighted", zero_division=0),
            recall_score(y_test, y_pred_test, average="weighted", zero_division=0)
        ),
        "F1-score": (
            f1_score(y_train, y_pred_train, average="weighted", zero_division=0),
            f1_score(y_test, y_pred_test, average="weighted", zero_division=0)
        )
    }
 
    # --- Display metrics cleanly ---
    st.subheader("Performance du modèle (Train vs Test)")
 
 # Réduire la taille du texte des metrics
    st.markdown("""
    <style>
    [data-testid="stMetricValue"] {
        font-size: 16px !important;     /* valeur par défaut ~28px */
    }
    [data-testid="stMetricLabel"] {
        font-size: 12px !important;     /* label plus petit */
    }
    </style>
    """, unsafe_allow_html=True)
 
 
    cols = st.columns(4)
    for idx, (metric, (train_val, test_val)) in enumerate(metrics.items()):
        cols[idx].metric(
            metric,
            f"Train: {train_val:.3f} | Test: {test_val:.3f}"
        )
 
    st.write("---")
 
    # --- Matrices de confusion ---
    st.subheader("Matrices de confusion")
 
    # Test
    col1, col2 = st.columns(2)
 
    with col1:
        st.write("### Jeu de Test")
        fig, ax = plt.subplots(figsize=(4, 3))
        ConfusionMatrixDisplay.from_predictions(y_test, y_pred_test, cmap="Blues", ax=ax)
        st.pyplot(fig)
 
    # Train
    with col2:
        st.write("### Jeu d'entrainement")
        fig, ax = plt.subplots(figsize=(4, 3))
        ConfusionMatrixDisplay.from_predictions(y_train, y_pred_train, cmap="Greens", ax=ax)
        st.pyplot(fig)
 
    st.write("---")
 
    # --- Rapport classification ---
    st.subheader("Rapport de classification (Test)")
    report_df = pd.DataFrame(classification_report(y_test, y_pred_test, output_dict=True)).transpose()
    st.dataframe(report_df.style.format("{:.3f}"), width="stretch")
 
    st.success("✅ Évaluation terminée")