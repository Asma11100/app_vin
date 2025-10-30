import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
import numpy as np

def run_preprocessing(df):
    st.title("⚙️ Prétraitement des Données")
    
    if df.empty:
        st.warning("Aucune donnée à prétraiter")
        return df
    
    df_processed = df.copy()
    
    # A. Sélection des colonnes à conserver
    st.header("🗂️ Sélection des Colonnes")
    
    all_columns = df_processed.columns.tolist()
    selected_columns = st.multiselect(
        "Choisir les colonnes à conserver :",
        options=all_columns,
        default=all_columns,
        help="Désélectionnez les colonnes que vous ne souhaitez pas utiliser"
    )
    
    if selected_columns:
        df_processed = df_processed[selected_columns]
        st.success(f"✅ {len(selected_columns)} colonnes sélectionnées")
    else:
        st.warning("⚠️ Veuillez sélectionner au moins une colonne")
        return df
    
    # B. Matrice de corrélation avec filtrage
    st.header("📊 Matrice de Corrélation")
    
    numeric_cols = df_processed.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) > 1:
        # Sélection des colonnes pour la corrélation
        corr_columns = st.multiselect(
            "Choisir les colonnes pour la matrice de corrélation :",
            options=numeric_cols,
            default=numeric_cols[:min(6, len(numeric_cols))],
            help="Sélectionnez les colonnes numériques à inclure dans la matrice de corrélation"
        )
        
        if len(corr_columns) > 1:
            # Calcul et affichage de la matrice de corrélation
            corr_matrix = df_processed[corr_columns].corr()
            
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(corr_matrix, 
                       annot=True, 
                       cmap='coolwarm', 
                       center=0, 
                       ax=ax, 
                       fmt='.2f',
                       square=True)
            ax.set_title('Matrice de Corrélation')
            st.pyplot(fig)
            
            # Détection des colonnes fortement corrélées
            st.subheader("🔍 Détection des colonnes fortement corrélées")
            
            # Trouver les paires avec corrélation > 0.8
            high_corr_pairs = []
            for i in range(len(corr_columns)):
                for j in range(i+1, len(corr_columns)):
                    if abs(corr_matrix.iloc[i, j]) > 0.8:
                        high_corr_pairs.append({
                            'Colonne 1': corr_columns[i],
                            'Colonne 2': corr_columns[j], 
                            'Corrélation': corr_matrix.iloc[i, j]
                        })
            
            if high_corr_pairs:
                high_corr_df = pd.DataFrame(high_corr_pairs)
                st.write("Colonnes fortement corrélées (|r| > 0.8):")
                st.dataframe(high_corr_df, use_container_width=True)
                
                # Suggestion de suppression
                cols_to_consider = set()
                for pair in high_corr_pairs:
                    cols_to_consider.add(pair['Colonne 1'])
                    cols_to_consider.add(pair['Colonne 2'])
                
                st.info(f"💡 Considérer la suppression d'une colonne parmi : {', '.join(cols_to_consider)}")
            else:
                st.success("✅ Aucune forte corrélation détectée entre les colonnes")
    
    # C. Gestion des valeurs manquantes
    st.header("🎯 Gestion des Valeurs Manquantes")
    
    missing_cols = df_processed.columns[df_processed.isnull().any()].tolist()
    
    if missing_cols:
        st.warning(f"Colonnes avec valeurs manquantes : {', '.join(missing_cols)}")
        
        for col in missing_cols:
            st.subheader(f"Traitement de la colonne : {col}")
            missing_count = df_processed[col].isnull().sum()
            missing_percent = (missing_count / len(df_processed)) * 100
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.write(f"Valeurs manquantes : {missing_count}")
            with col2:
                st.write(f"Pourcentage : {missing_percent:.1f}%")
            with col3:
                if df_processed[col].dtype in ['int64', 'float64']:
                    st.write("Type : Numérique")
                else:
                    st.write("Type : Catégoriel")
            
            # Stratégie de traitement
            strategy = st.radio(
                f"Stratégie pour {col} :",
                ["Supprimer les lignes", "Remplacer par la moyenne/médiane", "Remplacer par le mode"],
                key=f"strategy_{col}"
            )
            
            if strategy == "Supprimer les lignes":
                if st.button(f"Supprimer les lignes avec {col} manquante", key=f"drop_{col}"):
                    initial_count = len(df_processed)
                    df_processed = df_processed.dropna(subset=[col])
                    final_count = len(df_processed)
                    st.success(f"Lignes supprimées : {initial_count - final_count}")
            
            elif strategy == "Remplacer par la moyenne/médiane" and df_processed[col].dtype in ['int64', 'float64']:
                replace_with = st.radio("Remplacer par :", ["Moyenne", "Médiane"], key=f"replace_{col}")
                if st.button(f"Appliquer pour {col}", key=f"apply_num_{col}"):
                    if replace_with == "Moyenne":
                        df_processed[col] = df_processed[col].fillna(df_processed[col].mean())
                    else:
                        df_processed[col] = df_processed[col].fillna(df_processed[col].median())
                    st.success(f"Valeurs manquantes remplacées par la {replace_with.lower()}")
            
            elif strategy == "Remplacer par le mode":
                if st.button(f"Appliquer pour {col}", key=f"apply_cat_{col}"):
                    mode_value = df_processed[col].mode()[0] if not df_processed[col].mode().empty else "Unknown"
                    df_processed[col] = df_processed[col].fillna(mode_value)
                    st.success(f"Valeurs manquantes remplacées par le mode : {mode_value}")
    else:
        st.success("✅ Aucune valeur manquante détectée")
    
    # D. Normalisation des données
    st.header("📏 Normalisation des Données Numériques")
    
    numeric_cols = df_processed.select_dtypes(include=[np.number]).columns.tolist()
    
    if numeric_cols:
        normalize_cols = st.multiselect(
            "Choisir les colonnes à normaliser :",
            options=numeric_cols,
            help="Sélectionnez les colonnes numériques à normaliser"
        )
        
        if normalize_cols:
            scaler = StandardScaler()
            df_processed[normalize_cols] = scaler.fit_transform(df_processed[normalize_cols])
            st.success(f"✅ {len(normalize_cols)} colonnes normalisées avec StandardScaler")
    
    # E. Encodage des variables catégorielles
    st.header("🔤 Encodage des Variables Catégorielles")
    
    categorical_cols = df_processed.select_dtypes(include=['object']).columns.tolist()
    
    if categorical_cols:
        for col in categorical_cols:
            with st.expander(f"Colonne : {col}"):
                st.write(f"Valeurs uniques : {df_processed[col].nunique()}")
                st.write(f"Exemples : {', '.join(map(str, df_processed[col].unique()[:5]))}")
                
                if st.button(f"Encoder {col}", key=f"encode_{col}"):
                    le = LabelEncoder()
                    df_processed[col] = le.fit_transform(df_processed[col].astype(str))
                    st.success(f"✅ Colonne {col} encodée ({df_processed[col].nunique()} catégories)")
    
    # F. Résultat final
    st.header("📋 Résultat du Prétraitement")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Avant prétraitement")
        st.write(f"Dimensions : {df.shape}")
        st.write(f"Colonnes : {len(df.columns)}")
        st.write(f"Valeurs manquantes : {df.isnull().sum().sum()}")
    
    with col2:
        st.subheader("Après prétraitement")
        st.write(f"Dimensions : {df_processed.shape}")
        st.write(f"Colonnes : {len(df_processed.columns)}")
        st.write(f"Valeurs manquantes : {df_processed.isnull().sum().sum()}")
    
    st.subheader("Aperçu des données transformées")
    st.dataframe(df_processed.head(), use_container_width=True)
    
    # Option de téléchargement
    if st.button("💾 Sauvegarder les données prétraitées"):
        csv = df_processed.to_csv(index=False)
        st.download_button(
            label="📥 Télécharger les données prétraitées",
            data=csv,
            file_name="donnees_pretraitees.csv",
            mime="text/csv"
        )
    
    return df_processed