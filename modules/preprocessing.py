import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

def run_preprocessing(df):
    st.title("Préparation des données")

    if df.empty:
        st.warning("❗ Aucune donnée à prétraiter")
        return df

    df_processed = df.copy()

    # =====================================================
    # 1️⃣ Sélection des colonnes
    # =====================================================
    st.header("Sélection des colonnes")

    selected_columns = st.multiselect(
        "Choisissez les colonnes à conserver :",
        options=df_processed.columns.tolist(),
        default=df_processed.columns.tolist()
    )

    if not selected_columns:
        st.error("❌ Vous devez sélectionner au moins une colonne")
        return df

    df_processed = df_processed[selected_columns]
    st.success(f"✅ {len(selected_columns)} colonnes sélectionnées")

    # =====================================================
    # 2️⃣ Gestion des valeurs manquantes
    # =====================================================
    st.header("Gestion des valeurs manquantes")

    missing_cols = df_processed.columns[df_processed.isnull().any()]

    if missing_cols.any():
        st.warning(f"Colonnes avec valeurs manquantes : {', '.join(missing_cols)}")

        for col in missing_cols:
            st.subheader(f"📌 Colonne : {col}")
            missing_count = df_processed[col].isnull().sum()
            missing_percent = missing_count / len(df_processed) * 100

            st.write(f"- Manquantes : {missing_count} ({missing_percent:.1f}%)")
            st.write(f"- Type : {'Numérique' if pd.api.types.is_numeric_dtype(df_processed[col]) else 'Catégoriel'}")

            strategy = st.radio(
                f"Stratégie pour {col}",
                ["Supprimer les lignes", "Remplacer par la moyenne/médiane", "Remplacer par le mode"],
                key=f"strategy_{col}"
            )

            if strategy == "Supprimer les lignes":
                if st.button(f"Supprimer lignes {col}", key=f"drop_{col}"):
                    df_processed.dropna(subset=[col], inplace=True)
                    st.success(f"✅ Lignes supprimées pour {col}")

            elif strategy == "Remplacer par la moyenne/médiane" and pd.api.types.is_numeric_dtype(df_processed[col]):
                method = st.radio("Méthode :", ["Moyenne", "Médiane"], key=f"method_{col}")
                if st.button(f"Remplacer {col}", key=f"replace_{col}"):
                    value = df_processed[col].mean() if method == "Moyenne" else df_processed[col].median()
                    df_processed[col].fillna(value, inplace=True)
                    st.success(f"✅ Valeurs remplacées ({method.lower()})")

            else:
                if st.button(f"Remplacer {col} par mode", key=f"mode_{col}"):
                    df_processed[col].fillna(df_processed[col].mode()[0], inplace=True)
                    st.success(f"✅ Mode appliqué pour {col}")

    else:
        st.success("✅ Aucune valeur manquante")

 
    # =====================================================
    # 4️⃣ Encodage des variables catégorielles
    # =====================================================
    st.header("Encodage des variables catégorielles")

    cat_cols = df_processed.select_dtypes(include='object').columns

    for col in cat_cols:
        with st.expander(f"Encodage : {col}"):
            st.write(f"Valeurs uniques : {df_processed[col].nunique()}")
            st.write(df_processed[col].unique()[:5])

            if st.button(f"Encoder {col}", key=f"encode_{col}"):
                le = LabelEncoder()
                df_processed[col] = le.fit_transform(df_processed[col].astype(str))
                st.success(f"✅ '{col}' encodée")

    # =====================================================
    # 5️⃣ Division du Train/Test
    # =====================================================
    st.header("Division du Train / Test")

    target_cols = [c for c in df_processed.columns if "target" in c.lower()]
    if not target_cols:
        st.error("❌ Target non trouvée — assurez-vous d'avoir une colonne contenant 'target'")
        return df_processed

    target = target_cols[0]

    test_size = st.slider("Taille Test (%)", 0.1, 0.4, 0.2, step=0.05)

    X = df_processed.drop(columns=[target])
    y = df_processed[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=42
    )

    # Stock session
    st.session_state.X_train = X_train
    st.session_state.X_test = X_test
    st.session_state.y_train = y_train
    st.session_state.y_test = y_test

    st.success(f"✅ Split réalisé : Train={len(X_train)} | Test={len(X_test)}")

    # =====================================================
    # Résumé final
    # =====================================================
    st.header("Résultat de la préparation des données")

    col1, col2 = st.columns(2)

    with col1:
        st.write("**Avant** :", df.shape, "colonnes :", len(df.columns))
    with col2:
        st.write("**Après** :", df_processed.shape, "colonnes :", len(df_processed.columns))

    st.subheader("Aperçu des données transformées")
    st.dataframe(df_processed.head())

    # Option de téléchargement
    if st.button("Sauvegardez les données préparées"):
        csv = df_processed.to_csv(index=False)
        st.download_button(
            label="📥 Téléchargez les données préparées",
            data=csv,
            file_name="donnees_preparees.csv",
            mime="text/csv"
        )

    return df_processed
