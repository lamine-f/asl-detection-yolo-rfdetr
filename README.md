# Détection d'Objets en Temps Réel : YOLO vs RF-DETR

Étude comparative entre **YOLO11n** et **RF-DETR-S** pour la détection d'objets en temps réel, appliquée à la **reconnaissance de la langue des signes américaine (ASL)**, avec déploiement sous forme d'application web interactive utilisant la webcam.

## Cas d'usage : Accessibilité & Inclusion

Application de reconnaissance des 26 lettres de l'alphabet ASL pour faciliter la communication entre personnes sourdes/malentendantes et personnes entendantes. La démonstration permet d'épeler des mots en direct via webcam.

## Stack Technique

| Couche | Outil |
|---|---|
| Modèles | Ultralytics (YOLO11n) + Roboflow (RF-DETR-S) |
| Données | American Sign Language Letters (Roboflow Universe) |
| Augmentation & Robustesse | Albumentations, OpenCV |
| Évaluation | Supervision, pycocotools, matplotlib |
| Application | Gradio (webcam temps réel) |
| Compute | Kaggle (T4/P100 GPU) |
| Rapport | LaTeX (template IEEE) |

## Structure du projet

```
projet1/
├── _0_doccuments/
│   ├── todo.md                      # Énoncé du projet
│   └── rapport/                     # Rapport scientifique LaTeX
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_yolo_training.ipynb       # Notebook Kaggle
│   ├── 03_rfdetr_training.ipynb     # Notebook Kaggle
│   ├── 04_evaluation_comparative.ipynb
│   └── 05_robustness_study.ipynb
├── src/
│   ├── data/                        # Préparation, augmentation, robustesse
│   ├── training/                    # Configs et helpers
│   ├── evaluation/                  # Métriques et figures
│   └── inference/                   # Wrapper unifié des modèles
├── app/
│   └── gradio_app.py                # Application webcam temps réel
├── models/                          # Poids fine-tunés (gitignored)
├── data/                            # Datasets (gitignored)
└── requirements.txt
```

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Lancer l'application

```bash
python app/gradio_app.py
# → http://127.0.0.1:7860
```

## Reproduire les résultats

1. Télécharger le dataset (notebook `01_data_exploration.ipynb`)
2. Entraîner YOLO sur Kaggle (`02_yolo_training.ipynb`)
3. Entraîner RF-DETR sur Kaggle (`03_rfdetr_training.ipynb`)
4. Évaluation comparative (`04_evaluation_comparative.ipynb`)
5. Étude de robustesse (`05_robustness_study.ipynb`)

## Auteurs

Projet réalisé en groupe — DIC3 Deep Learning, 2026 :

- **Mouhamed Lamine Faye**
- **Pape Moussa Mbengue**
- **Mouhamadou Wally Ndour**
