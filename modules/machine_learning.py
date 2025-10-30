import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler

def run_machine_learning(df):
    """
    Module d'entraînement de modèles de classification
    """
    st.title("🤖 Entraînement du Modèle de Classification")
    
    if df is None or df.empty:
        st.error("❌ Aucune donnée disponible pour l'entraînement")
        return None, None
    
    # A. INITIALISATION ET VÉRIFICATIONS
    st.header("📋 Vérification des Données")
    
    # Afficher les dimensions
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Lignes", df.shape[0])
    with col2:
        st.metric("Colonnes", df.shape[1])
    with col3:
        st.metric("Données manquantes", df.isnull().sum().sum())
    
    # B. SÉLECTION DE LA TARGET - FIXE
    st.header("🎯 Variable Cible")
    
    all_columns = df.columns.tolist()
    
    # DÉTERMINATION AUTOMATIQUE DE LA TARGET
    if 'target_encoded' in all_columns:
        target_col = 'target_encoded'
        st.success("✅ **Target utilisée :** `target_encoded`")
        
        # Afficher le mapping si disponible
        if 'label_mapping' in st.session_state:
            st.info("**Correspondance des classes :**")
            for text_label, numeric_value in st.session_state.label_mapping.items():
                st.write(f"  - `{text_label}` → Classe **{numeric_value}**")
    
    elif 'target' in all_columns:
        # Si target existe mais n'est pas encodée, on l'encode automatiquement
        target_col = 'target'
        st.warning("⚠️ **Target détectée :** `target` (encodage automatique nécessaire)")
        
        # Encodage automatique
        if df[target_col].dtype == 'object':
            try:
                le = LabelEncoder()
                df['target_encoded'] = le.fit_transform(df[target_col])
                target_col = 'target_encoded'
                
                # Sauvegarder le mapping
                st.session_state.label_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
                st.success("✅ **Target encodée automatiquement en :** `target_encoded`")
                
                st.info("**Correspondance créée :**")
                for text_label, numeric_value in st.session_state.label_mapping.items():
                    st.write(f"  - `{text_label}` → Classe **{numeric_value}**")
                    
            except Exception as e:
                st.error(f"❌ Erreur lors de l'encodage automatique : {e}")
                return None, None
        else:
            # Si target est déjà numérique
            st.success("✅ **Target utilisée :** `target` (déjà numérique)")
    
    else:
        st.error("❌ **Aucune variable target trouvée**")
        st.info("""
        **Solutions :**
        - Vérifiez que votre dataset contient une colonne 'target' ou 'target_encoded'
        - Utilisez la section 'Nettoyage' pour encoder votre variable cible
        """)
        return None, None
    
    # Afficher la target sélectionnée (lecture seule)
    st.text_input(
        "Variable cible sélectionnée :",
        value=target_col,
        disabled=True,
        help="Cette variable est déterminée automatiquement et ne peut pas être modifiée"
    )
    
    # C. SÉLECTION DES FEATURES
    st.header("🔧 Sélection des Variables Explicatives")
    
    # Exclure TOUTES les colonnes de target
    available_features = [col for col in all_columns 
                         if col != 'target' 
                         and col != 'target_encoded'
                         and not col.startswith('target_')]
    
    if not available_features:
        st.error("❌ Aucune variable explicative disponible")
        return None, None
    
    # Sélection des features avec valeurs par défaut intelligentes
    default_features = available_features[:min(5, len(available_features))]
    
    feature_cols = st.multiselect(
        "Sélectionnez les variables features :",
        options=available_features,
        default=default_features,
        help="Choisissez les variables qui serviront à prédire la target"
    )
    
    if not feature_cols:
        st.warning("⚠️ Veuillez sélectionner au moins une variable feature")
        return None, None
    
    # Vérifier que les features sont numériques
    non_numeric_features = []
    for col in feature_cols:
        if not pd.api.types.is_numeric_dtype(df[col]):
            non_numeric_features.append(col)
    
    if non_numeric_features:
        st.error(f"❌ Variables non numériques : {', '.join(non_numeric_features)}")
        st.info("💡 Veuillez encoder ces variables dans la section 'Nettoyage'")
        return None, None
    
    # D. ANALYSE DE LA TARGET
    st.header("📊 Analyse de la Variable Cible")
    
    y = df[target_col]
    
    # Vérifier que la target est prête pour l'entraînement
    if y.dtype == 'object':
        st.error("❌ La target est toujours textuelle après encodage automatique")
        return None, None
    
    # Analyse des classes
    n_classes = len(np.unique(y))
    class_distribution = pd.Series(y).value_counts().sort_index()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Nombre de classes", n_classes)
        st.metric("Total échantillons", len(y))
    
    with col2:
        st.write("**Distribution des classes :**")
        for cls, count in class_distribution.items():
            percentage = (count / len(y)) * 100
            
            # Afficher le label texte si le mapping existe
            if 'label_mapping' in st.session_state:
                text_label = [k for k, v in st.session_state.label_mapping.items() if v == cls]
                label_display = f" ({text_label[0]})" if text_label else ""
            else:
                label_display = ""
                
            st.write(f"- Classe {cls}{label_display} : {count} échantillons ({percentage:.1f}%)")
    
    if n_classes < 2:
        st.error("❌ La variable cible doit avoir au moins 2 classes différentes")
        return None, None
    
    # E. CHOIX DU MODÈLE
    st.header("🧠 Configuration du Modèle")
    
    model_choice = st.radio(
        "Sélectionnez l'algorithme :",
        options=["Random Forest", "Régression Logistique"],
        horizontal=True
    )
    
    # Paramètres du modèle
    if model_choice == "Random Forest":
        st.subheader("🌳 Paramètres Random Forest")
        col1, col2 = st.columns(2)
        with col1:
            n_estimators = st.slider("Nombre d'arbres", 10, 200, 100)
        with col2:
            max_depth = st.slider("Profondeur maximale", 2, 30, 10)
        
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42
        )
        
    else:  # Régression Logistique
        st.subheader("📈 Paramètres Régression Logistique")
        col1, col2 = st.columns(2)
        with col1:
            C = st.slider("Régularisation (C)", 0.01, 10.0, 1.0, 0.1)
        with col2:
            max_iter = st.slider("Itérations max", 100, 2000, 500)
        
        model = LogisticRegression(
            C=C,
            max_iter=max_iter,
            random_state=42
        )
    
    # F. PARAMÈTRES D'ENTRAÎNEMENT
    st.header("⚙️ Paramètres d'Entraînement")
    
    col1, col2 = st.columns(2)
    with col1:
        test_size = st.slider("Taille du jeu de test", 0.1, 0.4, 0.2, 0.05)
    with col2:
        random_state = st.number_input("Seed aléatoire", value=42)
    
    # Option de normalisation
    normalize_data = st.checkbox("Normaliser les données", value=True)
    
    # G. ENTRAÎNEMENT
    st.header("🚀 Entraînement du Modèle")
    
    if st.button("🎯 Lancer l'Entraînement", type="primary", use_container_width=True):
        try:
            with st.spinner("Entraînement en cours..."):
                # Préparation des données
                X = df[feature_cols]
                
                # Normalisation si demandée
                if normalize_data:
                    scaler = StandardScaler()
                    X = pd.DataFrame(scaler.fit_transform(X), columns=feature_cols)
                    st.info("✅ Données normalisées avec StandardScaler")
                
                # Split des données
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, 
                    test_size=test_size, 
                    random_state=random_state, 
                    stratify=y
                )
                
                # Entraînement
                model.fit(X_train, y_train)
                
                # Prédictions
                y_pred_train = model.predict(X_train)
                y_pred_test = model.predict(X_test)
                
                # Évaluation
                accuracy_train = accuracy_score(y_train, y_pred_train)
                accuracy_test = accuracy_score(y_test, y_pred_test)
                cm = confusion_matrix(y_test, y_pred_test)
                report = classification_report(y_test, y_pred_test, output_dict=True)
                
            # H. RÉSULTATS
            st.header("📊 Résultats de l'Entraînement")
            
            # Métriques principales
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Accuracy Train", f"{accuracy_train:.3f}")
            with col2:
                st.metric("Accuracy Test", f"{accuracy_test:.3f}")
            with col3:
                diff = accuracy_train - accuracy_test
                st.metric("Différence", f"{diff:.3f}", 
                         delta="Sur-ajustement" if diff > 0.1 else "Bon équilibre")
            with col4:
                st.metric("Modèle", model_choice)
            
            # Matrice de confusion
            st.subheader("📈 Matrice de Confusion")
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, 
                       xticklabels=[f'Classe {i}' for i in range(n_classes)],
                       yticklabels=[f'Classe {i}' for i in range(n_classes)])
            ax.set_xlabel('Prédiction')
            ax.set_ylabel('Vérité Terrain')
            ax.set_title('Matrice de Confusion - Jeu de Test')
            st.pyplot(fig)
            
            # Rapport de classification
            st.subheader("📋 Rapport de Classification Détaillé")
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df.style.format("{:.3f}"), use_container_width=True)
            
            # Importance des features (Random Forest seulement)
            if model_choice == "Random Forest":
                st.subheader("🎯 Importance des Variables")
                feature_importance = pd.DataFrame({
                    'Variable': feature_cols,
                    'Importance': model.feature_importances_
                }).sort_values('Importance', ascending=False)
                
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.barplot(data=feature_importance, x='Importance', y='Variable', ax=ax)
                ax.set_title('Importance des Variables - Random Forest')
                ax.set_xlabel('Score d\'importance')
                plt.tight_layout()
                st.pyplot(fig)
                
                # Tableau d'importance
                st.dataframe(feature_importance, use_container_width=True)
            
            # Coefficients (Régression Logistique seulement)
            elif model_choice == "Régression Logistique":
                st.subheader("📊 Coefficients du Modèle")
                coefficients = pd.DataFrame({
                    'Variable': feature_cols,
                    'Coefficient': model.coef_[0]
                }).sort_values('Coefficient', key=abs, ascending=False)
                
                fig, ax = plt.subplots(figsize=(10, 6))
                colors = ['red' if x < 0 else 'blue' for x in coefficients['Coefficient']]
                sns.barplot(data=coefficients, x='Coefficient', y='Variable', palette=colors, ax=ax)
                ax.axvline(x=0, color='black', linestyle='--')
                ax.set_title('Coefficients - Régression Logistique')
                plt.tight_layout()
                st.pyplot(fig)
                
                # Tableau des coefficients
                st.dataframe(coefficients, use_container_width=True)
            
            # I. SAUVEGARDE DES RÉSULTATS
            results = {
                'accuracy_train': accuracy_train,
                'accuracy_test': accuracy_test,
                'confusion_matrix': cm,
                'classification_report': report,
                'feature_names': feature_cols,
                'n_classes': n_classes,
                'model_type': model_choice,
                'model': model
            }
            
            st.success("✅ **Modèle entraîné avec succès !**")
            st.balloons()
            
            return model, results
            
        except Exception as e:
            st.error(f"❌ **Erreur lors de l'entraînement :** {str(e)}")
            st.info("""
            **Conseils de dépannage :**
            - Vérifiez que toutes les variables sont numériques
            - Assurez-vous qu'il n'y a pas de valeurs manquantes  
            - Vérifiez que la target a au moins 2 classes
            - Réduisez le nombre de features si nécessaire
            """)
            return None, None
    
    return None, None