#!/usr/bin/env bash
# Télécharge les poids fine-tunés depuis Hugging Face.
# Nécessite : pip install -U "huggingface_hub[cli]"

set -euo pipefail

HF_REPO="lamine-f-edu/asl-yolo-rfdetr"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "[1/2] Téléchargement de YOLO11n..."
hf download "$HF_REPO" yolo/best.pt --local-dir "$ROOT/models/yolo" --quiet
mv "$ROOT/models/yolo/yolo/best.pt" "$ROOT/models/yolo/best.pt"
rmdir "$ROOT/models/yolo/yolo" 2>/dev/null || true

echo "[2/2] Téléchargement de RF-DETR-S..."
hf download "$HF_REPO" rfdetr/checkpoint_best_total.pth --local-dir "$ROOT/models/rfdetr" --quiet
mv "$ROOT/models/rfdetr/rfdetr/checkpoint_best_total.pth" "$ROOT/models/rfdetr/checkpoint_best_total.pth"
rmdir "$ROOT/models/rfdetr/rfdetr" 2>/dev/null || true

echo
echo "Modèles téléchargés :"
ls -lh "$ROOT/models/yolo/best.pt" "$ROOT/models/rfdetr/checkpoint_best_total.pth"
