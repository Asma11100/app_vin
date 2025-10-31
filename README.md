# 🍷 Projet d'Analyse de Vin avec Machine Learning
<p>
  <img src="https://img.shields.io/badge/Streamlit-App-red?logo=streamlit"> 
  <img src="https://img.shields.io/badge/Python-3.9+-blue?logo=python"> 
  <img src="https://img.shields.io/badge/Scikit--Learn-ML-green?logo=scikitlearn"> 
</p>


## 📌 Description

Ce projet est une application web interactive, développée avec Streamlit, permet d’analyser un dataset de vins à travers un pipeline complet de Machine Learning. Elle offre une interface utilisateur claire et structurée couvrant l’ensemble du processus, de l’analyse exploratoire des données à l’évaluation finale des modèles.

## ✨ Fonctionnalités

- **📊 Jeu de données** : Chargement et visualisation du jeu de données de vins (Aperçu, types des données & analyse descriptive).
- **🔍 Exploration des données** : Visualisation interactive du dataset de vins (Histogrammes, Histogrammes segmentés par catégorie, Pairplot & Matrice de corrélation).
- **⚙️ Préparation des données** : Gestion des valeurs manquantes, normalisation, encodage et sélection de features.
- **🤖 Entraînement** : Choix entre deux algorithmes de classification (Random Forest & Régression Logistique).
- **📈 Évaluation** : Métriques de performance et visualisations des résultats (Matrice de confusion & Rapport de classification).

## 👨‍💻 Installation & Exécution

1. **Clonez le repository** :

    ```bash
    git clone https://github.com/Asma11100/app_vin.git
    ```

2. **Allez dans le dossier app_vin** :

    ```bash
    cd app_vin
    ```

3. **Installez les dépendances** :

    ```bash
    pip install -r requirements.txt
    ```

4. **Lancez l'application Streamlit** :

    ```bash
    streamlit run app.py OU python -m streamlit run app.py
    ```


## 🗂 Structure du projet
```plaintext
app_vin/
│── app.py                    # Script principal Streamlit
│── vin.csv                   # Jeu de données des vins
│── requirements.txt          # Dépendances Python
│── README.md                 # Documentation du projet
├── modules/
│   ├── __init__.py           
│   ├── exploration.py        # Exploration des données
│   ├── preprocessing.py      # Préparation des données
│   ├── machine_learning.py   # Modélisation des données
│   └── evaluation.py         # Prédiction & Evaluation des modèles 
└── .gitignore        

