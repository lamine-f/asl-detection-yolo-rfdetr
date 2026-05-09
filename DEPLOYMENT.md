# Déploiement — ASL Detection (YOLO vs RF-DETR)

Document de référence du déploiement de la démo Gradio sur VPS.

- **URL publique** : https://asl-detection-yolo-rfdetr.duckdns.org
- **Date de mise en service** : 9 mai 2026

## 1. Infrastructure (VPS)

| Élément | Spécification |
|---|---|
| Système | Ubuntu 24.04.4 LTS (kernel 6.8.0-106) |
| CPU | AMD EPYC, 6 vCPU x86_64 (1 thread/cœur) |
| RAM | 11 GB + 4 GB swap |
| Stockage | 193 GB SSD |
| GPU | aucun — inférence **CPU uniquement** |
| IP publique | 75.119.147.191 |
| DNS | DuckDNS — `asl-detection-yolo-rfdetr.duckdns.org` (enregistrement A) |

## 2. Reverse proxy & TLS

| Élément | Détail |
|---|---|
| Serveur HTTP | Caddy 2.11.2 (apt repo officiel Cloudsmith) |
| Certificat | Let's Encrypt (challenge ACME HTTP-01), renouvellement automatique |
| Redirection | HTTP → HTTPS (308 permanent) |
| Protocoles | HTTP/2 + HTTP/3 (QUIC sur port 443) |
| Compression | `zstd`, `gzip` |
| Fichier de config | `/etc/caddy/Caddyfile` |

Caddyfile :

```caddy
{
    email mouhamedlaminefaye.zeitune@gmail.com
}

asl-detection-yolo-rfdetr.duckdns.org {
    encode zstd gzip
    reverse_proxy localhost:7861
}
```

## 3. Conteneurisation

| Élément | Détail |
|---|---|
| Runtime | Docker Engine 29.4.3 + Docker Compose v5.1.3 |
| Image de base | `python:3.12-slim` |
| Politique de redémarrage | `unless-stopped` |
| Healthcheck | `curl http://localhost:7860/` toutes les 30 s |
| Port | 7861 (host) → 7860 (container) |
| Limites | 6 CPU max, 10 GB RAM, 2 GB `/dev/shm` |
| Réservations | 4 CPU, 4 GB RAM |

Variables d'environnement passées au container :

```
GRADIO_SERVER_NAME=0.0.0.0
GRADIO_SERVER_PORT=7860
OMP_NUM_THREADS=6
MKL_NUM_THREADS=6
OPENBLAS_NUM_THREADS=6
NUMEXPR_NUM_THREADS=6
TORCH_NUM_THREADS=6
```

## 4. Stack applicative

| Composant | Version |
|---|---|
| Python | 3.12.13 |
| PyTorch | 2.11.0+cpu |
| Ultralytics (YOLO) | 8.4.48 |
| OpenVINO Runtime | 2026.1.0 |
| Gradio | 6.14.0 |
| OpenCV (headless) | 4.13.0 |

## 5. Modèles déployés

| Modèle | Format | Taille | Source |
|---|---|---|---|
| YOLO11n | OpenVINO IR (FP32, imgsz=640) | 10 MB | export depuis `best.pt` (5,3 MB) |
| RF-DETR-S | PyTorch checkpoint | 122 MB | `checkpoint_best_total.pth` |

Les poids fine-tunés sont publiés sur Hugging Face Hub : `lamine-f-edu/asl-yolo-rfdetr`.

Téléchargement (depuis la racine du repo) :

```bash
python3 -m venv .venv-dl
.venv-dl/bin/pip install "huggingface_hub[cli]"
PATH="$PWD/.venv-dl/bin:$PATH" bash scripts/download_models.sh
```

L'export OpenVINO peut être régénéré à partir du `.pt` :

```bash
docker exec asl-detection python -c "\
from ultralytics import YOLO; \
YOLO('/app/models/yolo/best.pt').export(format='openvino', imgsz=640, half=False)"
docker cp asl-detection:/app/models/yolo/best_openvino_model models/yolo/
```

## 6. Optimisations appliquées

- **YOLO converti en OpenVINO IR** (mode latency, batch=1) — gain ~1,4× vs PyTorch CPU.
- **Threads parallèles** plafonnés à 6 (= vCPU) sur PyTorch / OpenMP / MKL / OpenBLAS / NumExpr.
- **OpenCV-headless** au lieu de `opencv-python` GUI, pour éviter les dépendances X11 et économiser ~30 MB de paquets.
- **`/dev/shm` à 2 GB** (Gradio écrit des fichiers temporaires de stream).
- **4 GB de swap** ajoutés au système pour absorber les pics mémoire au chargement RF-DETR.
- Correctif Dockerfile : `pip uninstall -y opencv-python opencv-python-headless && pip install --force-reinstall opencv-python-headless` pour éviter que la désinstallation de `opencv-python` ne supprime aussi les fichiers `cv2/` partagés.

## 7. Performance mesurée

CPU 6 cœurs, image 640×640, moyenne sur 30 inférences (synthétiques) :

| Backend | Latence (ms) | FPS |
|---|---|---|
| YOLO PyTorch (baseline) | 119 | 8,4 |
| YOLO ONNX Runtime | 153 | 6,5 |
| **YOLO OpenVINO (déployé)** | **86–107** | **9–12** |
| YOLO OpenVINO @ 416×416 *(non activé)* | 45 | 22 |

RF-DETR-S reste sensiblement plus lent sur CPU (transformer non quantifié, pas d'export OpenVINO direct via Ultralytics).

## 8. Architecture réseau

```
Client navigateur
        │ HTTPS (443) ─ TLS Let's Encrypt
        ▼
   ┌─────────────┐
   │   Caddy     │  systemd, /etc/caddy/Caddyfile
   │   :80, :443 │  HTTP/2 + HTTP/3
   └─────┬───────┘
         │ reverse_proxy localhost:7861
         ▼
   ┌─────────────────────┐
   │ container Docker    │  asl-detection
   │ Gradio :7860 → 7861 │  python:3.12-slim
   │ + UnifiedDetector   │  YOLO (OpenVINO) + RF-DETR (PyTorch)
   └─────────────────────┘
```

## 9. Opérations

### Démarrer / arrêter / mettre à jour

```bash
cd /home/lord/deploy/asl-detection-yolo-rfdetr

# Démarrer / recréer après modification de docker-compose.yml
docker compose up -d --build

# Arrêter
docker compose down

# Logs en direct
docker compose logs -f asl-detection

# État
docker compose ps
```

### Caddy

```bash
sudo systemctl reload caddy        # après modification du Caddyfile
sudo systemctl status caddy
sudo journalctl -u caddy -f        # logs
```

### Vérifier la santé du déploiement

```bash
curl -I https://asl-detection-yolo-rfdetr.duckdns.org/
# attendu : HTTP/2 200, server: uvicorn (via Caddy)
```

### Renouvellement TLS

Caddy gère le renouvellement automatiquement (~30 jours avant expiration). Aucune action manuelle requise.

## 10. Limites connues & pistes d'amélioration

**Limites** :
- Inférence **CPU uniquement** : webcam plafonnée autour de **10 fps** pour YOLO et nettement moins pour RF-DETR.
- L'accès webcam navigateur **exige HTTPS** — c'est garanti ici, mais à savoir si le déploiement migre.

**Pistes pour gagner en performance sans GPU** :
1. **`imgsz=416`** sur YOLO → ~22 fps, mais perte ~1-3 mAP.
2. **Quantization INT8 OpenVINO** (avec dataset de calibration de ~100 images ASL) → 2-3× supplémentaires.
3. **Limiter la démo webcam à YOLO**, garder RF-DETR pour les comparaisons hors-ligne sur images uploadées.

**Pour passer à 30+ fps en RF-DETR** : migration sur un VPS GPU (T4 / RTX 3060), installation du runtime CUDA et reconstruction de l'image Docker avec PyTorch GPU.
