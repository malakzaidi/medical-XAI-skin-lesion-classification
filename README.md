# IA Explicable Multi-Modal pour la Classification Dermatologique

## Aperçu du Projet
Ce projet propose une solution d'intelligence artificielle explicable et multi-modale pour la classification automatisée de huit types de lésions cutanées à partir du dataset ISIC 2019. Face aux défis posés par le fort déséquilibre des classes et la complexité visuelle des lésions, une architecture hybride a été développée, combinant un réseau de neurones convolutif (CNN) basé sur DenseNet121 avec des données cliniques (âge, sexe, localisation).

<img width="703" height="752" alt="image" src="https://github.com/user-attachments/assets/d8bcb627-f860-4ab4-a838-f81738fce741" />

L'approche repose sur un pipeline de prétraitement rigoureux (élimination des poils, normalisation colorimétrique) et une stratégie d'entraînement avancée intégrant le Transfer Learning, des fonctions de perte adaptées (Focal Loss) et des techniques d'augmentation de données (MixUp, CutMix) pour améliorer la généralisation. Une attention particulière a été portée à la transparence du modèle via des techniques d'IA Explicable (XAI), notamment Grad-CAM et l'Occlusion Sensitivity.

Les résultats démontrent une amélioration significative du rappel (Recall) sur les classes critiques et fournissent des visualisations validant la pertinence clinique des zones analysées par le modèle.

**Mots-clés :** Classification d'images médicales, ISIC 2019, DenseNet121, Transfer Learning, Fusion Multimodale, XAI, Déséquilibre de classes.

## Résultat de la classification 

<img width="780" height="412" alt="image" src="https://github.com/user-attachments/assets/eb1f4a3e-a4d4-4075-95be-41c25e6e517f" />


## Interface du dashboard multi techniques 

<img width="761" height="526" alt="image" src="https://github.com/user-attachments/assets/125d0c95-82d3-49df-9e27-91c22ed22227" />

## Corrélation entre les méthodes XAI 

<img width="707" height="798" alt="image" src="https://github.com/user-attachments/assets/47e07923-f215-417c-a900-c25d9aaf8881" />

## EXemple de Rapport généré 

<img width="783" height="639" alt="image" src="https://github.com/user-attachments/assets/48a02b7c-b270-47bc-98f7-18e703cfecb2" />

## Fonctionnalités Principales
- **Classification Multi-Modale :** Fusion d'images dermoscopiques et de métadonnées cliniques (âge, sexe, localisation) pour une précision accrue.
- **Optimisation pour Déséquilibre :** Utilisation de Focal Loss et d'augmentations avancées (MixUp, CutMix) pour booster le rappel sur les classes rares comme le mélanome.
- **Explicabilité (XAI) :** Intégration de Grad-CAM, Occlusion Sensitivity, Integrated Gradients et LIME pour des visualisations interactives (cartes de chaleur 3D via Plotly) expliquant les décisions du modèle.
- **Pipeline de Prétraitement :** Élimination des poils, normalisation colorimétrique, standardisation géométrique pour gérer l'hétérogénéité des données ISIC.
- **Interface Utilisateur :** Application web Flask avec frontend pour soumission d'images, visualisation des résultats et génération de rapports PDF incluant avertissements légaux.
- **Score de Confiance Composite :** Agrégation de la probabilité du modèle et de la cohérence des explications XAI pour des alertes cliniques (ex. : "Confiance Limite - Biopsie nécessaire").
- **MLOps Intégré :** Utilisation de DVC pour le versioning des données, MLflow pour le tracking des expériences, et automatisation des rapports.

## Technologies Utilisées
- **IA/ML :** TensorFlow/Keras (DenseNet121 pré-entraîné sur ImageNet), OpenCV (prétraitement), Scikit-learn (métriques, clustering).
- **XAI :** Grad-CAM, Occlusion Sensitivity, LIME, Integrated Gradients.
- **Backend :** Flask (API et contrôleur MVC).
- **Frontend :** HTML/CSS/JS avec Plotly pour visualisations interactives 3D.
- **Rapports :** ReportLab pour génération PDF.
- **MLOps :** MLflow (tracking), DVC (versioning des données et modèles).
- **Autres :** Pandas/NumPy (analyse exploratoire), Matplotlib/Seaborn (visualisations EDA).

## Architecture Globale du projet 

<img width="844" height="461" alt="image" src="https://github.com/user-attachments/assets/ce7cc742-03dc-403c-8e8c-7e03f58d1783" />


## Architecture du Modèle

L'architecture est hybride et multi-input, articulée autour de deux branches :

<img width="821" height="727" alt="image" src="https://github.com/user-attachments/assets/53950e94-c62e-4b05-9351-fdefe5b85879" />

- **Branche Visuelle :** DenseNet121 pour extraction de caractéristiques des images (redimensionnées à 224x224), suivi d'un Global Average Pooling.
- **Branche Métadonnées :** MLP (couches denses avec Dropout) pour traiter les variables cliniques encodées (11 features après one-hot encoding).
- **Fusion :** Concaténation des sorties des branches, suivie de couches denses pour classification finale (8 classes).

Le backbone DenseNet121 est initialement gelé pour le Transfer Learning, puis fine-tuné sélectivement.

## Analyse Exploratoire des Données (EDA)
- **Dataset ISIC 2019 :** 25 331 images de 8 classes de lésions cutanées, avec métadonnées cliniques.
- **Déséquilibre :** Classes dominantes (ex. : Nævus ~50%) vs. rares (ex. : Mélanome ~18%).
- **Visualisations :** Analyse colorimétrique (signatures spectrales), t-SNE pour structure visuelle, PCA/K-Means pour profilage démographique.

## Processus de Prétraitement
- **Images :** Redimensionnement, élimination des poils (algorithme DullRazor), normalisation colorimétrique.
  
  <img width="800" height="522" alt="image" src="https://github.com/user-attachments/assets/886d2d4d-8698-4e2a-88e1-76093d8d1cb6" />
  
- **Métadonnées :** Encodage one-hot pour sexe et localisation, normalisation pour âge, gestion des valeurs manquantes.

  <img width="772" height="434" alt="image" src="https://github.com/user-attachments/assets/f98c879f-fee3-4098-b0a2-1c2083cc4c4b" />


## Stratégie d'Entraînement
- **Phases :** Gel initial du backbone, puis fine-tuning progressif.
- **Perte :** Focal Loss pour focaliser sur les classes difficiles.
- **Augmentation :** Rotations, flips, MixUp/CutMix pour robustesse.
- **Optimisation :** Adam avec scheduler (ReduceLROnPlateau), Early Stopping.

  <img width="803" height="439" alt="image" src="https://github.com/user-attachments/assets/95be1c9f-3dec-41da-b58a-8644d924aecd" />

## Architecture Applicative et Interface (Frontend & XAI)

  <img width="594" height="549" alt="image" src="https://github.com/user-attachments/assets/40506bcd-6d56-4353-b69f-0fa2a5c10dd4" />

## Explicabilité (XAI)
- **Méthodes :** Grad-CAM pour heatmaps, Occlusion Sensitivity pour masquage, LIME pour attributions locales.
- **Visualisations :** Cartes de chaleur interactives 3D, analyse qualitative des succès/échecs (ex. : focus sur bordures irrégulières pour mélanome).
  
<img width="1077" height="829" alt="image" src="https://github.com/user-attachments/assets/771f592e-74cc-4cf6-8a7c-d7ab24435ab4" />


## Résultats
Les résultats montrent une précision globale de 89,2 %, avec un rappel amélioré sur les classes critiques (ex. : Mélanome).

| Métrique | Valeur Globale | Exemple (Mélanome) |
|----------|----------------|---------------------|
| Précision | 89.2%         | -                   |
| Rappel (Recall) | -           | Amélioration significative |
| F1-Score | -             | -                   |

Analyse des erreurs via XAI révèle des confusions visuelles (ex. : nævus vs. vasculaire dues à couleurs similaires).

<img width="745" height="564" alt="image" src="https://github.com/user-attachments/assets/619c14fd-7f2f-4ff1-b985-9b1bb060a058" />

<img width="787" height="378" alt="image" src="https://github.com/user-attachments/assets/df4b2a1c-2b9a-40f0-b277-ca6c94359927" />

## Gestion du Projet
Méthodologie Scrumban : Sprints pour modélisation, Kanban pour tâches EDA et XAI. Outils : Git, Trello, Google Colab.

<img width="864" height="501" alt="image" src="https://github.com/user-attachments/assets/59d35d3d-d8cc-4e90-92a8-ecd950b89702" />


## Diagramme de Gantt

<img width="883" height="460" alt="image" src="https://github.com/user-attachments/assets/b0be79bd-f75c-47f7-8f85-1a8ce47908bb" />


## Installation et Exécution
1. Cloner le repository : `git clone https://github.com/votre-repo/projet15-ia-explicable-dermatologie.git`
2. Installer les dépendances : `pip install -r requirements.txt` (TensorFlow, Keras, Flask, Plotly, etc.)
3. Lancer MLflow : `mlflow ui --port 5000`
4. Démarrer l'application Flask : `python app.py`
5. Accéder à l'interface : `http://localhost:5000`

**Prérequis :** Python 3.8+, GPU recommandé pour entraînement, dataset ISIC 2019 (disponible sur Kaggle).

## Auteurs
- ZAIDI Malak
- ELHABTI Fatiha

**Encadrant :** Pr. HAMIDA Soufiane

**Date :** Décembre 2025

Ce projet est réalisé dans le cadre du Master SDIA – Module IA Avancée (Année 2025-2026). Contributions bienvenues via pull requests !
