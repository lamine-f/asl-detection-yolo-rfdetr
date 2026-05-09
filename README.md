# Détection d'Objets en Temps Réel : YOLO vs RF-DETR

[![Live demo](https://img.shields.io/badge/🚀%20Démo%20live-asl--detection--yolo--rfdetr.duckdns.org-2ea44f)](https://asl-detection-yolo-rfdetr.duckdns.org/)
[![GitHub](https://img.shields.io/badge/GitHub-lamine--f%2Fasl--detection-181717?logo=github)](https://github.com/lamine-f/asl-detection-yolo-rfdetr)
[![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-lamine--f--edu%2Fasl--yolo--rfdetr-yellow)](https://huggingface.co/lamine-f-edu/asl-yolo-rfdetr)

> **🌐 Démo en ligne (HTTPS, webcam fonctionnelle)** : https://asl-detection-yolo-rfdetr.duckdns.org/

Étude comparative entre **YOLO11n** et **RF-DETR-S** pour la détection d'objets en temps réel, appliquée à la **reconnaissance de la langue des signes américaine (ASL)**, avec déploiement sous forme d'application web interactive utilisant la webcam.

Projet de Deep Learning, **DIC3 — Génie Informatique — ESP/UCAD — 2026**.

## Cas d'usage : Accessibilité & Inclusion

Application de reconnaissance des 26 lettres de l'alphabet ASL pour faciliter la communication entre personnes sourdes/malentendantes et personnes entendantes. La démonstration permet d'**épeler des mots en direct** via webcam.

## Résultats clés

### Performance — test set (72 images, GPU T4)

| Métrique | YOLO11n | RF-DETR-S |
|---|---|---|
| mAP@0.5 | **0.904** | 0.901 |
| mAP@0.5:0.95 | 0.875 | **0.881** |
| Latence | **36.4 ms** | 132.8 ms |
| FPS | **27.5** | 7.5 |
| Taille modèle | **5.21 MB** | 121.8 MB |

→ Précision quasi identique en nominal, **YOLO ~3.6× plus rapide et 23× plus léger**.

### Robustesse — 5 conditions dégradées

| Condition | YOLO Δ% | RF-DETR-S Δ% |
|---|---|---|
| γ = 1.5 | +1.3% | +0.8% |
| γ = 0.75 | −1.0% | −0.8% |
| γ = 0.5 | −3.5% | −0.8% |
| σ = 15 (bruit modéré) | −2.1% | +0.1% |
| σ = 35 (bruit fort) | **−76.8%** | −1.1% |
| **Chute moyenne \|Δ\|** | **16.9%** | **0.7%** |

→ **RF-DETR-S est dramatiquement plus robuste**. YOLO s'effondre à 0.21 mAP sous bruit fort tandis que RF-DETR reste à 0.89.

→ Recommandation : YOLO pour edge / temps réel en environnement maîtrisé, RF-DETR pour environnement bruité ou non maîtrisé.

## Stack technique

| Couche | Outil |
|---|---|
| Modèles | Ultralytics (YOLO11n) + Roboflow (RF-DETR-S) |
| Données | American Sign Language Letters (Roboflow Universe) |
| Augmentation & Robustesse | Albumentations, OpenCV |
| Évaluation | Supervision, pycocotools, matplotlib |
| Application | Gradio (4 onglets : Alphabet, Comparaison, Webcam, Épellation) |
| Compute | Google Colab (GPU T4) |
| Déploiement | Docker + Caddy (HTTPS via Let's Encrypt) sur VPS |
| Rapport & slides | LaTeX (IEEE conf + Beamer metropolis) |

## Structure du projet

```
projet1/
├── _0_doccuments/
│   ├── todo.md                       # Énoncé du projet
│   ├── rapport/                      # Rapport scientifique IEEE
│   │   ├── main.tex
│   │   └── main.pdf                  # 5 pages, livrable final
│   └── slides/                       # Slides de soutenance Beamer
│       ├── slides.tex
│       └── slides.pdf                # 20 pages thème metropolis
├── notebooks/
│   ├── 02_yolo_training.ipynb        # Notebook Colab YOLO
│   ├── 03_rfdetr_training.ipynb      # Notebook Colab RF-DETR
│   ├── 04_evaluation_and_robustness.ipynb   # Éval + robustesse
│   └── from-google-collab/           # Mêmes notebooks avec sorties d'exécution
├── src/
│   ├── data/                         # Augmentation, robustesse, build alphabet
│   ├── inference/                    # Wrapper unifié YOLO + RF-DETR
│   └── evaluation/                   # Helpers métriques + injection rapport
├── app/
│   ├── gradio_app.py                 # Application web 4 onglets
│   └── assets/asl_alphabet.png       # Grille de référence ASL
├── scripts/
│   └── download_models.sh            # Récupère les poids depuis Hugging Face
├── data/results/                     # Résultats bruts (JSON + CSV)
├── models/                           # Poids fine-tunés (gitignored, sur HF)
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## Installation

```bash
git clone https://github.com/lamine-f/asl-detection-yolo-rfdetr
cd asl-detection-yolo-rfdetr
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Téléchargement des modèles fine-tunés

Les poids sont publiés sur **Hugging Face Hub** :
→ https://huggingface.co/lamine-f-edu/asl-yolo-rfdetr

```bash
# Script tout-en-un (nécessite : pip install -U "huggingface_hub[cli]")
./scripts/download_models.sh
```

Ou en Python :

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

### Avec Docker (production / VPS)

```bash
docker compose up -d --build
# → http://localhost:7861 (host) → 7860 (container)
```

**Prérequis VPS** : Docker Engine + Docker Compose, 4 GB RAM minimum.

**Reverse proxy HTTPS** : recommandé pour la webcam (les navigateurs bloquent `getUserMedia` sur HTTP non-localhost). La démo live tourne derrière **Caddy + Let's Encrypt** sur un sous-domaine **DuckDNS**, exemple de `Caddyfile` :

```caddy
asl-detection-yolo-rfdetr.duckdns.org {
    reverse_proxy 127.0.0.1:7861
}
```

**Déploiement détaillé** : voir [DEPLOYMENT.md](DEPLOYMENT.md) pour la configuration complète du VPS (specs, Caddy, Docker, optimisations OpenVINO, métriques de performance).

## Reproduire les résultats

1. **Entraîner YOLO** sur Colab : ouvrir `notebooks/02_yolo_training.ipynb` (Runtime T4 GPU, ~1h30)
2. **Entraîner RF-DETR** sur Colab : ouvrir `notebooks/03_rfdetr_training.ipynb` (Runtime T4 GPU, ~2h30)
3. **Évaluer et étude de robustesse** : ouvrir `notebooks/04_evaluation_and_robustness.ipynb` (~30 min)

Les résultats finaux (JSON, CSV, figures) sont versionnés dans `data/results/`. Les notebooks avec leurs sorties d'exécution sont dans `notebooks/from-google-collab/`.

## Documents livrables

- **[Rapport scientifique](_0_doccuments/rapport/main.pdf)** : 5 pages au format IEEE conference
- **[Slides de soutenance](_0_doccuments/slides/slides.pdf)** : 20 pages Beamer metropolis

## Auteurs

Projet de Deep Learning réalisé en groupe par :

- **Mouhamed Lamine Faye**
- **Pape Moussa Mbengue**
- **Mouhamadou Wally Ndour**

DIC3 — Département Génie Informatique
École Supérieure Polytechnique
Université Cheikh Anta Diop de Dakar — 2026
