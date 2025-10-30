import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Import de nos modules

from modules.exploration import run_exploration
from modules.preprocessing import run_preprocessing
from modules.machine_learning import run_machine_learning
from modules.evaluation import run_evaluation




# Configuration de la page
st.set_page_config(
    page_title="Le vin Français",
    page_icon="🍷",
    layout="wide"
)

# Chargement des données
@st.cache_data
def load_data():
    """Charge les données une seule fois"""
    df = pd.read_csv('vin.csv')
    # Nettoyer la colonne Unnamed: 0 si elle existe
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])
    return df



# --- SIDEBAR (Navigation) ---
st.sidebar.title("🍷 Menu principal")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Choisissez une section :",
    ["🏠 Accueil",  "📊 Jeu de données", "🔍 Exploration", "⚙️ Nettoyage", "🤖 Entraînement", "📈 Évaluation"]
)
  
# Chargement des données
df = load_data()


# --- CONTENU PRINCIPAL ---

if page == "🏠 Accueil":
    # Header principal avec style
    st.markdown("""
    <style>
    .main-title {
        font-size: 3rem;
        color: #8B0000;
        text-align: center;
        margin-bottom: 1rem;
        font-family: 'Georgia', serif;
    }
    .welcome-text {
        font-size: 1.2rem;
        line-height: 1.6;
        text-align: center;
        color: #5D4037;
        margin-bottom: 2rem;
    }
    .feature-card {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #8B0000;
        margin: 1rem 0;
    }
    .stats-card {
        background: linear-gradient(135deg, #8B0000, #5D4037);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        margin: 0.5rem;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Titre principal
    st.markdown('<h1 class="main-title">🍷 Le vin Français</h1>', unsafe_allow_html=True)
    st.markdown('<p class="welcome-text">Cher visiteur, bienvenue dans notre application de Machine Learning dédiée à l\'analyse et la classification des vins Français.</p>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Section Mission
    st.markdown("## 🎯 Mission du projet")
    
    st.markdown("""
    <div style='background-color: #fff5f5; padding: 2rem; border-radius: 10px; border-left: 5px solid #8B0000;'>
    <p style='font-size: 1.1rem; line-height: 1.6;'>
    <strong>Le Vin Français</strong> est une plateforme innovante qui marie l'art ancestral de la viticulture 
    avec la puissance de l'Intelligence Artificielle moderne. Notre mission : décrypter les secrets chimiques 
    qui font l'identité unique de chaque vin français.
    </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Section Pipeline
    st.markdown("## 🔬 Notre Approche Scientifique")
    
    st.markdown("### ⚡ Pipeline IA Haute Performance")
    st.markdown("Notre pipeline complet de Machine Learning transforme les données chimiques brutes en insights actionnables :")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <h4>🔍 EXPLORATION</h4>
            <p>Analyse descriptive, visualisations avancées, matrice de corrélation</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <h4>⚙️ PRÉPARATION</h4>
            <p>Nettoyage, normalisation, encodage, feature engineering</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="feature-card">
            <h4>🤖 ENTRAÎNEMENT</h4>
            <p>Algorithmes Random Forest, validation croisée, optimisation</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="feature-card">
            <h4>📊 ÉVALUATION</h4>
            <p>Métriques précises, matrices de confusion, rapports détaillés</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Section Données
    st.markdown("## 📊 Données d'excellence")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="stats-card">
            <h3>🍇</h3>
            <h4>178 Crus</h4>
            <p>Soigneusement analysés</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="stats-card">
            <h3>🔬</h3>
            <h4>13 Paramètres</h4>
            <p>Chimiques mesurés</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="stats-card">
            <h3>🏷️</h3>
            <h4>3 Catégories</h4>
            <p>Distinctes de vins</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="stats-card">
            <h3>✅</h3>
            <h4>Données complètes</h4>
            <p>Sans valeurs manquantes</p>
        </div>
        """, unsafe_allow_html=True)
    

    # Caractéristiques des vins
    st.markdown("## 📈 Notre cépage de données")
    
    st.markdown("""
    Comme un vigneron sélectionne ses cépages, nous analysons méticuleusement chaque caractéristique :
    
    - **🍷 Alcool** - Le corps et la chaleur du vin
    - **🍋 Acidité** - La fraîcheur et la vivacité caractéristiques  
    - **🌿 Phénols** - La structure, les tanins et l'astringence
    - **🎨 Couleur** - La robe, l'intensité et la profondeur
    - **⭐ Proline** - Marqueur de qualité et de complexité
    """)
    
    # Résultats et questions
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("## 🎯 Résultats Tangibles")
        st.markdown("""
        - ⚡ **Analyse en temps réel** de nouveaux vins
        - 🔍 **Transparence totale** sur les décisions de l'IA
        - 🎓 **Pédagogie intégrée** pour comprendre l'analyse
        """)
    
    with col2:
        st.markdown("## ❓ Questions explorées")
        st.markdown("""
        - Pourquoi certains vins sont-ils plus alcoolisés ?
        - Comment l'acidité influence le caractère d'un vin ?
        - Quels paramètres déterminent la catégorie d'un vin ?
        - L'IA peut-elle rivaliser avec un œnologue humain ?
        """)
    
    # Call to Action
    st.markdown("---")
    st.markdown("## 🚀 Commencer l'Exploration")
    
    st.markdown("""
    <div style='text-align: center; padding: 2rem; background: linear-gradient(135deg, #8B0000, #5D4037); color: white; border-radius: 15px;'>
        <h3 style='color: white;'>Prêt à découvrir les secrets de nos vins ?</h3>
        <p style='font-size: 1.1rem;'>Naviguez à travers les différentes étapes de notre pipeline IA</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Navigation rapide
    st.info("💡 **Utilisez la sidebar pour naviguer entre les différentes sections de l'application**")

elif page == "📊 Jeu de données":
    st.title("📊 Aperçu du Dataset des Vins")
    
    # Vérification des colonnes disponibles
    st.header("🔍 Structure du Dataset")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Lignes", df.shape[0])
    with col2:
        st.metric("Colonnes", df.shape[1])
    with col3:
        st.metric("Valeurs manquantes", df.isnull().sum().sum())
    with col4:
        st.metric("Mémoire", f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    # Afficher les premières lignes
    st.header("📋 Aperçu des données")
    
    tab1, tab2, tab3 = st.tabs(["Données brutes", "Types de données", "Statistiques"])
    
    with tab1:
        st.subheader("Premières lignes")
        st.dataframe(df.head(10), use_container_width=True)
        
        st.subheader("Dernières lignes")
        st.dataframe(df.tail(10), use_container_width=True)
    
    with tab2:
        st.subheader("Types de données")
        info_df = pd.DataFrame({
            'Colonne': df.columns,
            'Type': df.dtypes,
            'Valeurs uniques': [df[col].nunique() for col in df.columns],
            'Valeurs manquantes': df.isnull().sum().values
        })
        st.dataframe(info_df, use_container_width=True)
    
    with tab3:
        st.subheader("Statistiques descriptives")
        st.dataframe(df.describe(), use_container_width=True)
    
    # Distribution des variables
    st.header("📈 Distribution des variables")
    
    selected_col = st.selectbox("Choisir une variable à analyser:", df.columns)
    
    if selected_col in df.columns:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader(f"Statistiques - {selected_col}")
            if df[selected_col].dtype in ['int64', 'float64']:
                stats = df[selected_col].describe()
                st.dataframe(pd.DataFrame(stats).T, use_container_width=True)
            else:
                value_counts = df[selected_col].value_counts()
                st.dataframe(pd.DataFrame({
                    'Valeur': value_counts.index,
                    'Count': value_counts.values,
                    'Pourcentage': (value_counts.values / len(df) * 100).round(2)
                }), use_container_width=True)
        
        with col2:
            st.subheader(f"Visualisation - {selected_col}")
            fig, ax = plt.subplots(figsize=(10, 6))
            
            if df[selected_col].dtype in ['int64', 'float64']:
                # Histogramme pour les variables numériques
                df[selected_col].hist(bins=30, ax=ax, alpha=0.7, color='skyblue', edgecolor='black')
                ax.set_title(f'Distribution de {selected_col}')
                ax.set_xlabel(selected_col)
                ax.set_ylabel('Fréquence')
            else:
                # Bar plot pour les variables catégorielles
                value_counts = df[selected_col].value_counts().head(10)
                value_counts.plot(kind='bar', ax=ax, color='lightcoral')
                ax.set_title(f'Distribution de {selected_col}')
                ax.set_xlabel(selected_col)
                ax.set_ylabel('Count')
                ax.tick_params(axis='x', rotation=45)
            
            st.pyplot(fig)
    
    # Matrice de corrélation
    st.header("🔗 Matrice de corrélation")
    
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    
    if len(numeric_cols) > 1:
        # Option pour filtrer les colonnes
        selected_corr_cols = st.multiselect(
            "Choisir les colonnes pour la corrélation:",
            options=numeric_cols.tolist(),
            default=numeric_cols.tolist()[:min(8, len(numeric_cols))]
        )
        
        if len(selected_corr_cols) > 1:
            corr_matrix = df[selected_corr_cols].corr()
            
            fig, ax = plt.subplots(figsize=(12, 8))
            sns.heatmap(corr_matrix, 
                       annot=True, 
                       cmap='coolwarm', 
                       center=0,
                       fmt='.2f',
                       ax=ax,
                       square=True)
            ax.set_title('Matrice de Corrélation')
            st.pyplot(fig)
            
            # Top des corrélations
            st.subheader("Corrélations les plus fortes")
            corr_pairs = corr_matrix.unstack().sort_values(ascending=False)
            corr_pairs = corr_pairs[corr_pairs < 0.999]  # Exclure l'auto-corrélation
            
            top_corr_df = pd.DataFrame({
                'Variable 1': [pair[0] for pair in corr_pairs.head(10).index],
                'Variable 2': [pair[1] for pair in corr_pairs.head(10).index],
                'Corrélation': corr_pairs.head(10).values
            })
            st.dataframe(top_corr_df, use_container_width=True)
        else:
            st.warning("Sélectionnez au moins 2 colonnes numériques")
    else:
        st.warning("Pas assez de colonnes numériques pour la corrélation")    


elif page == "🔍 Exploration":
    run_exploration(df)

elif page == "⚙️ Nettoyage":
    data_processed = run_preprocessing(df)
    if data_processed is not None:
        st.session_state.data_processed = data_processed

elif page == "🤖 Entraînement":
    if 'data_processed' in st.session_state:
        model, results = run_machine_learning(st.session_state.data_processed)
        if model is not None:
            st.session_state.model = model
            st.session_state.results = results
    else:
        st.warning("⚠️ Veuillez d'abord prétraiter les données")

elif page == "📈 Évaluation":
    if 'model' in st.session_state:
        run_evaluation(st.session_state.model, st.session_state.results)
    else:
        st.warning("⚠️ Veuillez d'abord entraîner un modèle")
    
elif page == "🔍 Exploration":
    run_exploration(df)

elif page == "⚙️ Nettoyage":
    data_processed = run_preprocessing(df)
    if data_processed is not None:
        st.session_state.data_processed = data_processed

elif page == "🤖 Entraînement":
    if 'data_processed' in st.session_state:
        model, results = run_machine_learning(st.session_state.data_processed)
        if model is not None:
            st.session_state.model = model
            st.session_state.results = results
    else:
        st.warning("⚠️ Veuillez d'abord prétraiter les données")

elif page == "📈 Évaluation":
    if 'model' in st.session_state:
        run_evaluation(st.session_state.model, st.session_state.results)
    else:
        st.warning("⚠️ Veuillez d'abord entraîner un modèle")
