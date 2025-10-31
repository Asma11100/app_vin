import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import altair as alt

def run_exploration(df):
    st.title("Exploration des données")
    
    if df.empty:
        st.warning("Aucune donnée à explorer")
        return
    
    # --- Sélection des colonnes  ---
    # --- Sélection des colonnes (inclut target) ---
    all_cols = df.columns.tolist()
    selected_cols = st.multiselect(
        "Choisissez les colonnes à afficher",
        options=all_cols,
        default=all_cols
    )

    if not selected_cols:
        st.warning("Merci de sélectionner au moins une colonne.")
        return

    # --- DataFrame filtré ---
    df_selected = df[selected_cols]

    # Séparer target des features
    if "target" in selected_cols:
        feature_cols = [c for c in selected_cols if c != "target"]
    else:
        feature_cols = selected_cols

    # --- Choix des visuels ---
    visual_choice = st.selectbox(
        "Choisissez le type de visualisation",
        ["Histogrammes", "Histogrammes segmentés par catégorie", "Pairplot", "Matrice de corrélation"]
    )

    st.markdown("<br><br>", unsafe_allow_html=True)

    # if visuel_choice == "Histogramme par variable":
    #     st.subheader("Histogramme par variable")
    #     st.bar_chart(df["target"].value_counts(), color="#E3A587")

    
    # ===========================
    # Histogramme par variable (Target)
    # ===========================
    if visual_choice == "Histogrammes segmentés par catégorie":

        # ---- Cas 1 : target non sélectionné ----
        # Vérifier que target est sélectionné
        if "target" not in selected_cols:
            st.warning("⚠️ Veuillez sélectionner la colonne 'target' pour afficher cet histogramme.")
            return

        # ---- Cas 2 : target + autres colonnes = non ----
        if len(selected_cols) > 1:
            st.warning("⚠️ Cet histogramme n'est affiché que lorsque seule la colonne 'target' est sélectionnée.")
            return

        # ---- Cas 3 : uniquement target sélectionné ➜ afficher ----
        st.subheader("Histogrammes segmentés par catégorie")

        # Compter les classes
        counts = df["target"].value_counts().reset_index()
        counts.columns = ["target", "count"]

        # Identifier la classe la plus fréquente
        max_class = counts.loc[counts["count"].idxmax(), "target"]

        # Palette
        colors = alt.Scale(
            domain=counts["target"].tolist(),  # Classes dans l'ordre
            range=[
                "#E3A587" if cls != max_class else "#722E1E"  # 👈 foncé si classe max
                for cls in counts["target"]
            ]
        )

        chart = (
            alt.Chart(counts)
            .mark_bar()
            .encode(
                x=alt.X("target:N", title="Classe"),
                y=alt.Y("count:Q", title="Nombre d'observations"),
                color=alt.Color("target:N", scale=colors, title="target"),
                tooltip=["target", "count"]
            )
            .properties(width=400, height=300)
        )
        st.altair_chart(chart, use_container_width=True)
        st.success(f"Classe la plus représentée : **{max_class}**")

    # ===========================
    # Histogrammes par feature
    # ===========================   

    elif visual_choice == "Histogrammes":
        st.subheader("Histogrammes")

        if "target" not in selected_cols:
            st.warning("⚠️ Sélectionnez la colonne 'target' pour afficher les histogrammes.")
            return
        
        for col in feature_cols:
            fig, ax = plt.subplots(figsize=(5, 3))
            sns.histplot(df, x=col, hue="target", kde=True, ax=ax)
            st.pyplot(fig)

    # ===========================
    # Pairplot
    # ===========================

    elif visual_choice == "Pairplot":
        st.subheader("Pairplot")

        if "target" not in selected_cols:
            st.warning("⚠️ Sélectionnez 'target' pour afficher le pairplot.")
            return   
        
        # Vérifier qu'il y a au moins 1 feature en plus de target
        if len(selected_cols) == 1:  # only target selected
            st.warning("⚠️ Le pairplot nécessite au moins une autre variable numérique en plus de 'target'.")
            return

        if st.button("Générez Pairplot"):
            pairplot_cols = feature_cols[:10] + ["target"]
            fig = sns.pairplot(df[pairplot_cols], hue="target")
            st.pyplot(fig)
        
        else:
            st.info("Cliquez pour afficher le pairplot, peut être lent ⏳")
  
    # ===========================
    # Matrice de corrélation
    # ===========================
    elif visual_choice == "Matrice de corrélation":
        st.subheader("Matrice de corrélation")
        
        num_df = df_selected.drop(columns=["target"], axis=1) #, errors="ignore"
        
        # Vérifier qu'il y a au moins 2 variables numériques
        numeric_cols = num_df.select_dtypes(include="number")

        if numeric_cols.shape[1] < 2:
            st.warning("⚠️ La matrice de corrélation nécessite au moins deux variables numériques.")
            return
        
        fig, ax = plt.subplots(figsize=(10,5))
        sns.heatmap(numeric_cols.corr(), annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5, cbar_kws={"shrink": .8}, annot_kws={"size": 6}, ax=ax)
        st.pyplot(fig)

