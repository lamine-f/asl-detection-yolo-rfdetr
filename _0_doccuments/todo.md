Énoncé du Projet : Détection d’Objets avec YOLO et RF-DETR

Contexte

La détection d’objets est une branche de la vision par ordinateur permettant d’identifier et de localiser automatiquement des objets présents dans une image ou une vidéo.
Contrairement à la classification simple, qui attribue uniquement une classe à une image, la détection d’objets fournit également les coordonnées de localisation de chaque objet détecté.

Deux approches principales existent :

* Détection d’objets (Object Detection) :
le modèle prédit :
* la classe de l’objet ;
* les coordonnées de sa boîte englobante (bounding box).
* Segmentation d’images (Segmentation) :
le modèle identifie précisément chaque pixel appartenant à un objet afin de délimiter sa forme exacte.

Dans ce projet, l’accent sera mis principalement sur la détection d’objets à l’aide des modèles modernes YOLO et RF-DETR.

⸻

Objectif Général

L’objectif du projet est de réaliser une étude comparative entre les modèles YOLO et RF-DETR pour une tâche de détection d’objets, en utilisant un dataset adapté, puis de déployer le meilleur modèle obtenu.

⸻

Travail Demandé

1. Choix et Préparation du Dataset

* Choisir un dataset de détection d’objets adapté au problème étudié.
* Justifier le choix du dataset.
* Analyser :
* le nombre de classes ;
* le nombre d’images ;
* le format des annotations ;
* la diversité des données.
* Préparer les données :
* séparation entraînement / validation / test ;
* prétraitement ;
* augmentation de données (data augmentation).

Exemples de datasets possibles :

* COCO
* Pascal VOC
* Roboflow datasets
* Dataset personnalisé

⸻

2. Fine-Tuning des Modèles

Effectuer le fine-tuning des modèles suivants sur le dataset choisi :

* YOLO
* RF-DETR

Le travail devra inclure :

* la configuration des modèles ;
* le choix des hyperparamètres ;
* l’entraînement ;
* le suivi des performances ;
* l’évaluation sur les données de test.

⸻

3. Étude Comparative des Modèles

Comparer les modèles YOLO et RF-DETR selon plusieurs critères :

a) Précision

* mAP (mean Average Precision)
* précision
* rappel (recall)
* F1-score

b) Complexité

* temps d’entraînement ;
* temps d’inférence ;
* taille du modèle ;
* consommation mémoire.

c) Robustesse

* performance sur différentes conditions :
* variations de luminosité ;
* objets partiellement cachés ;
* bruit ;
* changements d’échelle.

Une analyse critique des résultats devra être réalisée afin d’identifier les avantages et limites de chaque approche.

⸻

4. Déploiement du Modèle

Déployer le modèle retenu sous forme d’application utilisable.

Le déploiement pourra être réalisé sous différentes formes :

* API REST ;
* application web ;
* application temps réel avec webcam ;
* intégration dans une interface utilisateur.

L’application devra permettre :

* l’envoi d’images ou de vidéos ;
* l’affichage des objets détectés ;
* la visualisation des résultats en temps réel.

⸻

Technologies Possibles

* Python
* PyTorch
* Ultralytics YOLO
* Transformers / RF-DETR
* OpenCV
* FastAPI ou Flask
* Next.js (pour une interface web)
* Docker (optionnel)

⸻

Résultats Attendus

À la fin du projet, l’étudiant devra être capable de :

* comprendre le fonctionnement de la détection d’objets ;
* entraîner et ajuster des modèles modernes ;
* comparer scientifiquement plusieurs approches ;
* déployer une solution fonctionnelle de vision par ordinateur.