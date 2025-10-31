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
    ["🏠 Accueil",  "📊 Jeu de données", "🔍 Exploration", "⚙️ Préparation", "🤖 Entraînement", "📈 Évaluation"]
)
  
# Chargement des données
df = load_data()


# --- CONTENU PRINCIPAL ---

#--------------------------------------------------------
#----------------- Accueil --------------------------
#--------------------------------------------------------

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
    st.markdown("## Mission du projet")
    
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
    st.markdown("## Notre approche scientifique")
    
    st.markdown("###    Pipeline IA Haute Performance")
    st.markdown("Notre pipeline complet de Machine Learning transforme les données chimiques brutes en insights actionnables :")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <h4> EXPLORATION</h4>
            <p>Analyse descriptive, visualisations avancées, matrice de corrélation</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <h4> PRÉPARATION</h4>
            <p>Nettoyage, normalisation, encodage, feature engineering</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="feature-card">
            <h4> ENTRAÎNEMENT</h4>
            <p>Algorithmes Random Forest, validation croisée, optimisation</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="feature-card">
            <h4> ÉVALUATION</h4>
            <p>Métriques précises, matrices de confusion, rapports détaillés</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Section Données
    st.markdown("## Données d'excellence")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="stats-card">
       <h4>178 Crus</h4>
            <p>Soigneusement analysés</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="stats-card">
         <h4>13 Paramètres</h4>
            <p>Chimiques mesurés</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="stats-card">
          <h4>3 Catégories</h4>
            <p>Distinctes de vins</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="stats-card">
         <h4>Données complètes</h4>
            <p>Sans valeurs manquantes</p>
        </div>
        """, unsafe_allow_html=True)
    

    # Caractéristiques des vins
    st.markdown("## Notre cépage de données")
    
    st.markdown("""
    Comme un vigneron sélectionne ses cépages, nous analysons méticuleusement chaque caractéristique :
    
    - ** Alcool** - Le corps et la chaleur du vin
    - ** Acidité** - La fraîcheur et la vivacité caractéristiques  
    - ** Phénols** - La structure, les tanins et l'astringence
    - ** Couleur** - La robe, l'intensité et la profondeur
    - ** Proline** - Marqueur de qualité et de complexité
    """)
    
 
    
    # Call to Action
    st.markdown("---")
    st.markdown("##  Commencer l'exploration")
    
    st.markdown("""
    <div style='text-align: center; padding: 2rem; background: linear-gradient(135deg, #8B0000, #5D4037); color: white; border-radius: 15px;'>
        <h3 style='color: white;'>Prêt à découvrir les secrets de nos vins ?</h3>
        <p style='font-size: 1.1rem;'>Naviguez à travers les différentes étapes de notre pipeline IA</p>
    </div>
    """, unsafe_allow_html=True)
    
#--------------------------------------------------------
#----------------- Jeu de données -----------------------
#--------------------------------------------------------

elif page == "📊 Jeu de données":
    st.title("Aperçu du dataset des vins")
    
    # Vérification des colonnes disponibles
    st.header("Structure du dataset")
    
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
    st.header("Aperçu des données")
    
    tab1, tab2, tab3 = st.tabs(["Données brutes", "Types de données", "Statistiques"])
    
    with tab1:
        st.subheader("Premières lignes")
        st.dataframe(df.head(10), width="stretch")
        
        st.subheader("Dernières lignes")
        st.dataframe(df.tail(10), width="stretch")
    
    with tab2:
        st.subheader("Types de données")
        info_df = pd.DataFrame({
            'Colonne': df.columns,
            'Type': df.dtypes,
            'Valeurs uniques': [df[col].nunique() for col in df.columns],
            'Valeurs manquantes': df.isnull().sum().values
        })
        st.dataframe(info_df, width="stretch")
    
    with tab3:
        st.subheader("Statistiques descriptives")
        st.dataframe(df.describe(), width="stretch")

#--------------------------------------------------------
#----------------- Exploration -----------------------
#--------------------------------------------------------

elif page == "🔍 Exploration":
    run_exploration(df)

#--------------------------------------------------------
#----------------- Préparation -----------------------
#--------------------------------------------------------

elif page == "⚙️ Préparation":
    data_processed = run_preprocessing(df)
    if data_processed is not None:
        st.session_state.data_processed = data_processed

#--------------------------------------------------------
#----------------- Entraînement -----------------------
#--------------------------------------------------------

elif page == "🤖 Entraînement":
    if 'data_processed' in st.session_state:
        output = run_machine_learning(st.session_state.data_processed)

        if output is not None:
            model, results = output
            st.session_state.model = model
            st.session_state.results = results

    else:
        st.warning("⚠️ Veuillez d'abord préparer les données")

#--------------------------------------------------------
#----------------- Évaluation -----------------------
#--------------------------------------------------------

elif page == "📈 Évaluation":
    if 'model' in st.session_state:
        run_evaluation(st.session_state.model, st.session_state.results)
    else:
        st.warning("⚠️ Veuillez d'abord entraîner un modèle")
    

