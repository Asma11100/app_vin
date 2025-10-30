# modules/evaluation.py
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def run_evaluation(model, results):
    """
    Fonction pour exécuter l'évaluation du modèle
    """
    try:
        st.header("📊 Évaluation du Modèle")
        
        if model is None:
            st.warning("Aucun modèle n'a été entraîné.")
            return
        
        if results is None:
            st.warning("Aucun résultat d'évaluation disponible.")
            return
        
        # Métriques principales
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Accuracy", f"{results.get('accuracy', 0):.3f}")
        
        with col2:
            precision = results.get('classification_report', {}).get('weighted avg', {}).get('precision', 0)
            st.metric("Precision", f"{precision:.3f}")
        
        with col3:
            recall = results.get('classification_report', {}).get('weighted avg', {}).get('recall', 0)
            st.metric("Recall", f"{recall:.3f}")
        
        with col4:
            f1 = results.get('classification_report', {}).get('weighted avg', {}).get('f1-score', 0)
            st.metric("F1-Score", f"{f1:.3f}")
        
        # Matrice de confusion
        if 'confusion_matrix' in results:
            st.subheader("📈 Matrice de Confusion")
            fig, ax = plt.subplots()
            sns.heatmap(results['confusion_matrix'], annot=True, fmt='d', ax=ax, cmap='Blues')
            ax.set_xlabel('Prédiction')
            ax.set_ylabel('Vérité terrain')
            st.pyplot(fig)
        
        # Rapport de classification détaillé
        if 'classification_report' in results:
            st.subheader("📋 Rapport de Classification Détaillé")
            report_df = pd.DataFrame(results['classification_report']).transpose()
            st.dataframe(report_df.style.format("{:.3f}"))
            
    except Exception as e:
        st.error(f"Erreur lors de l'évaluation : {str(e)}")