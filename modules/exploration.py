import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

def run_exploration(df):
    st.title("🔍 Exploration des Données")
    
    if df.empty:
        st.warning("Aucune donnée à explorer")
        return
    
    # A. Histogramme avec sélection de colonnes
    st.header("📊 Histogrammes")
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    
    if numeric_cols:
        selected_hist_cols = st.multiselect(
            "Choisir les colonnes pour l'histogramme :",
            options=numeric_cols,
            default=numeric_cols[:3] if len(numeric_cols) >= 3 else numeric_cols
        )
        
        if selected_hist_cols:
            n_cols = min(2, len(selected_hist_cols))
            n_rows = (len(selected_hist_cols) + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
            axes = axes.flatten() if n_rows > 1 else [axes]
            
            for i, col in enumerate(selected_hist_cols):
                if i < len(axes):
                    axes[i].hist(df[col].dropna(), bins=20, alpha=0.7, color='skyblue', edgecolor='black')
                    axes[i].set_title(f'Histogramme de {col}')
                    axes[i].set_xlabel(col)
                    axes[i].set_ylabel('Fréquence')
            
            # Masquer les axes vides
            for j in range(len(selected_hist_cols), len(axes)):
                axes[j].set_visible(False)
            
            plt.tight_layout()
            st.pyplot(fig)
    
    # B. Pairplot avec interactions
    st.header("📈 Pairplot avec Interactions")
    
    if len(numeric_cols) >= 2:
        selected_pair_cols = st.multiselect(
            "Choisir les colonnes pour le pairplot :",
            options=numeric_cols,
            default=numeric_cols[:4] if len(numeric_cols) >= 4 else numeric_cols
        )
        
        # Sélection de la colonne pour la couleur (catégorielle)
        color_col = st.selectbox(
            "Colonne pour la couleur (optionnel) :",
            options=['Aucune'] + df.columns.tolist()
        )
        
        if len(selected_pair_cols) >= 2:
            if st.button("Générer Pairplot"):
                try:
                    if color_col != 'Aucune':
                        fig = px.scatter_matrix(df[selected_pair_cols + [color_col]], 
                                              dimensions=selected_pair_cols,
                                              color=color_col,
                                              title="Pairplot avec Interactions")
                    else:
                        fig = px.scatter_matrix(df[selected_pair_cols], 
                                              dimensions=selected_pair_cols,
                                              title="Pairplot")
                    
                    st.plotly_chart(fig)
                except Exception as e:
                    st.error(f"Erreur lors de la génération du pairplot : {e}")