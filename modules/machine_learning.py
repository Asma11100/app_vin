import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

def run_machine_learning(df):
    st.title("🤖 Entraînement du Modèle de Classification")

    if df is None or df.empty:
        st.error("❌ Aucune donnée disponible pour l'entraînement")
        return None
    
    st.header("📋 Vérification des Données")
    col1, col2, col3 = st.columns(3)
    col1.metric("Lignes", df.shape[0])
    col2.metric("Colonnes", df.shape[1])
    col3.metric("Données manquantes", df.isnull().sum().sum())

    # -------------------------------
    # ✅ Target
    # -------------------------------
    st.header("🎯 Target")

    all_columns = df.columns.tolist()

    if 'target_encoded' in all_columns:
        target_col = 'target_encoded'
        st.success("✅ Target utilisée : `target_encoded`")

        if 'label_mapping' in st.session_state:
            st.info("**Correspondance des classes :**")
            for text_label, numeric_value in st.session_state.label_mapping.items():
                st.write(f"• **{text_label}** → Classe {numeric_value}")

    elif 'target' in all_columns:
        target_col = 'target'
        st.warning("⚠️ `target` détectée — encodage en cours…")

        if df[target_col].dtype == 'object':
            le = LabelEncoder()
            df['target_encoded'] = le.fit_transform(df[target_col])
            target_col = 'target_encoded'
            st.session_state.label_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
            st.success("✅ Target encodée → `target_encoded`")

            st.info("**Correspondance :**")
            for text_label, numeric_value in st.session_state.label_mapping.items():
                st.write(f"• **{text_label}** → Classe {numeric_value}")
    else:
        st.error("❌ Aucune target trouvée (`target` ou `target_encoded`)")
        return None

    st.text_input("Target sélectionnée :", target_col, disabled=True)

    # -------------------------------
    # Analyse de la target
    # -------------------------------
    st.header("📊 Analyse de la Target")
    y = df[target_col]
    n_classes = len(np.unique(y))

    col1, col2 = st.columns(2)
    col1.metric("Nombre de classes", n_classes)
    col1.metric("Total échantillons", len(y))

    col2.write("**Distribution des classes :**")
    for cls, count in pd.Series(y).value_counts().sort_index().items():
        percentage = count / len(y) * 100
        label_text = ""
        if 'label_mapping' in st.session_state:
            label_text = [k for k, v in st.session_state.label_mapping.items() if v == cls][0]
        col2.write(f"• Classe {cls} ({label_text}) : {count} échantillons ({percentage:.1f}%)")

    # -------------------------------
    # Définition des features
    # -------------------------------
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in numeric_cols if c not in ['target', 'target_encoded']]

    #st.info(f"✅ **Variables utilisées automatiquement :** {feature_cols}")

    # -------------------------------
    # Récupération du split depuis la préparation
    # -------------------------------
    if not all(k in st.session_state for k in ["X_train", "X_test", "y_train", "y_test"]):
        st.error("❌ Veuillez d'abord préparer les données (split manquant)")
        return None

    X_train = st.session_state.X_train[feature_cols]
    X_test = st.session_state.X_test[feature_cols]
    y_train = st.session_state.y_train
    y_test = st.session_state.y_test

    # -------------------------------
    # ✅ Choix du modèle
    # -------------------------------
    st.header("🧠 Configuration du Modèle")

    model_choice = st.radio("Sélectionnez l'algorithme :", ["Random Forest", "Régression Logistique"], horizontal=True)

    if model_choice == "Random Forest":
        n_estimators = st.slider("Nombre d'arbres", 10, 200, 100)
        max_depth = st.slider("Profondeur max", 2, 30, 10)
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    else:
        C = st.slider("Régularisation (C)", 0.01, 10.0, 1.0)
        max_iter = st.slider("Itérations max", 100, 2000, 500)
        model = LogisticRegression(C=C, max_iter=max_iter, random_state=42)

    # -------------------------------
    # ✅ Entraînement
    # -------------------------------
    st.header("🚀 Entraînement du Modèle")

    if st.button("🚀 Lancer l'Entraînement", type="primary"):

        model.fit(X_train, y_train)
        st.success("✅ Modèle entraîné !")

        # ✅ Importance des variables
        st.subheader("📌 Importance des Variables")

        importances = model.feature_importances_ if model_choice == "Random Forest" else np.abs(model.coef_[0])

        feat_imp = pd.DataFrame({"Variable": feature_cols, "Importance": importances}).sort_values(by="Importance", ascending=False)
        st.dataframe(feat_imp, width="stretch")

        # Graphique importance
        st.subheader("📊 Importance des Variables")
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(data=feat_imp, x="Importance", y="Variable", hue="Variable", palette="viridis", legend=False, ax=ax)
        ax.set_title(f"Importance des Variables - {model_choice}")
        st.pyplot(fig)

        # ✅ Tree example for RF
        if model_choice == "Random Forest":
            from sklearn import tree
            estimator = model.estimators_[0]
            fig, ax = plt.subplots(figsize=(25, 12))
            tree.plot_tree(estimator, feature_names=feature_cols, filled=True, rounded=True, fontsize=7)
            st.subheader("🌳 Présentation du Random Forest")
            st.pyplot(fig)

        # Stocker résultats pour l'évaluation
        results = {
            "X_test": X_test,
            "y_test": y_test,
            "feature_cols": feature_cols,
            "model_type": model_choice
        }

        return model, results
