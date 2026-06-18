# 💶 Détection de faux billets par Machine Learning

## 🎯 Objectif

Développer un système de détection automatique de faux billets à partir de caractéristiques physiques mesurées sur des billets en euros.

## 📌 Contexte

Projet réalisé dans le cadre de la formation **Data Analyst OpenClassrooms**.

L'Organisation Nationale de Lutte contre le Faux-Monnayage (ONCFM) souhaite automatiser l'identification des billets contrefaits à l'aide d'un modèle de Machine Learning.

L'objectif est de construire une application capable de prédire si un billet est authentique ou falsifié à partir de ses caractéristiques géométriques.

## ❓ Problématique

Comment utiliser les techniques de Machine Learning afin de distinguer efficacement les vrais billets des faux billets à partir de données mesurées lors du scan des billets ?

## 📂 Données utilisées

Le jeu de données comprend :

* 1 500 billets scannés
* 1 000 billets authentiques
* 500 billets contrefaits

### Variables étudiées

* Longueur
* Hauteur
* Largeur
* Autres caractéristiques géométriques du billet

## 🛠️ Méthodologie

### Préparation des données

* Analyse exploratoire
* Vérification de la qualité des données
* Détection d'éventuelles anomalies
* Préparation des variables pour les modèles

### Modélisation

Comparaison de plusieurs algorithmes de classification :

* K-Means
* Régression Logistique
* K-Nearest Neighbors (KNN)
* Random Forest

### Évaluation

Analyse des performances à l'aide de :

* Accuracy
* Precision
* Recall
* F1-Score
* Matrice de confusion

## 📊 Analyses réalisées

### Exploration des données

* Distribution des variables
* Comparaison vrais billets / faux billets
* Analyse des corrélations

### Comparaison des modèles

* Entraînement sur données d'apprentissage
* Validation sur données de test
* Comparaison des performances

### Sélection du modèle final

Choix du modèle offrant le meilleur compromis entre :

* Performance
* Robustesse
* Interprétabilité

## 🤖 Application de prédiction

Développement d'une application permettant :

* La saisie des caractéristiques d'un billet
* La prédiction automatique du type de billet
* L'affichage de la probabilité associée à la prédiction

## 🧰 Technologies utilisées

* Python
* Pandas
* NumPy
* Scikit-Learn
* Matplotlib
* Seaborn
* Jupyter Notebook

## 📁 Structure du projet

```text
data/
│
├── billets.csv

notebooks/
│
├── analyse_billets.ipynb

models/
│
├── logistic_regression.pkl
├── random_forest.pkl

app/
│
├── prediction_billet.py

README.md
```

## ✅ Compétences développées

* Machine Learning supervisé
* Classification binaire
* Préparation de données
* Évaluation de modèles
* Feature Engineering
* Validation de modèles
* Interprétation des résultats
* Développement d'une application prédictive

## 💡 Valeur ajoutée métier

Cette solution permet d'accélérer le contrôle des billets et d'assister les équipes de lutte contre la contrefaçon en fournissant une aide à la décision automatisée.

## 👨‍💻 Auteur

**Mohamed Zaidi**
