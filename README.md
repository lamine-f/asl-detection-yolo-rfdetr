# Détection d'Objets en Temps Réel : YOLO vs RF-DETR

[![GitHub](https://img.shields.io/badge/GitHub-lamine--f%2Fasl--detection-181717?logo=github)](https://github.com/lamine-f/asl-detection-yolo-rfdetr)
[![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-lamine--f--edu%2Fasl--yolo--rfdetr-yellow)](https://huggingface.co/lamine-f-edu/asl-yolo-rfdetr)

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

## Téléchargement des modèles fine-tunés

Les poids fine-tunés sont publiés sur **Hugging Face Hub** :

→ https://huggingface.co/lamine-f-edu/asl-yolo-rfdetr

```bash
# Récupère best.pt (YOLO) et checkpoint_best_total.pth (RF-DETR)
./scripts/download_models.sh
```

Ou manuellement :

```python
from huggingface_hub import hf_hub_download
hf_hub_download("lamine-f-edu/asl-yolo-rfdetr", "yolo/best.pt", local_dir="models/yolo")
hf_hub_download("lamine-f-edu/asl-yolo-rfdetr", "rfdetr/checkpoint_best_total.pth", local_dir="models/rfdetr")
```

## Lancer l'application

### En local (développement)

```bash
python app/gradio_app.py
# → http://127.0.0.1:7860
```

### Avec Docker (recommandé pour VPS)

Le projet inclut un `Dockerfile` et un `docker-compose.yml` prêts à l'emploi.

**Prérequis sur le VPS** : Docker Engine + Docker Compose plugin, 4 GB RAM minimum, port 7860 ouvert.

```bash
# 1. Cloner le repo sur le VPS
git clone <url-du-repo> asl-detection
cd asl-detection

# 2. Vérifier que les modèles fine-tunés sont bien présents :
#    models/yolo/best.pt
#    models/rfdetr/checkpoint_best_total.pth
#    (sinon, transfère-les en SCP depuis ta machine locale)

# 3. Build & run
docker compose up -d --build

# 4. Vérifier
docker compose ps
docker compose logs -f asl-detection
```

L'application est accessible sur `http://<ip-vps>:7860`.

**Mise derrière un reverse proxy (Nginx, Caddy, Traefik)** : décommente `GRADIO_ROOT_PATH=/asl` dans `docker-compose.yml` pour servir l'app sous un sous-chemin (`https://mondomaine.fr/asl`).

**Arrêter** :
```bash
docker compose down
```

## Reproduire les résultats

1. Télécharger le dataset (notebook `01_data_exploration.ipynb`)
2. Entraîner YOLO sur Kaggle (`02_yolo_training.ipynb`)
3. Entraîner RF-DETR sur Kaggle (`03_rfdetr_training.ipynb`)
4. Évaluation comparative (`04_evaluation_comparative.ipynb`)
5. Étude de robustesse (`05_robustness_study.ipynb`)

## Auteurs

Projet de Deep Learning réalisé en groupe par :

- **Mouhamed Lamine Faye**
- **Pape Moussa Mbengue**
- **Mouhamadou Wally Ndour**

DIC3 — Département Génie Informatique
École Supérieure Polytechnique
Université Cheikh Anta Diop de Dakar — 2026
