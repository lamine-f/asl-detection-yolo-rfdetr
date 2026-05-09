# syntax=docker/dockerfile:1.6
FROM python:3.12-slim AS base

# Dépendances système nécessaires à OpenCV / Pillow / albumentations
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        ffmpeg \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 1) Dépendances Python en premier (meilleure mise en cache Docker)
#    Note: opencv-python-headless est plus petit que opencv-python en server
COPY requirements.txt /tmp/requirements.txt

RUN pip install --no-cache-dir --upgrade pip wheel && \
    pip install --no-cache-dir \
        torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r /tmp/requirements.txt && \
    # ultralytics tire opencv-python (avec GUI). En serveur on veut la version headless.
    # Astuce: --force-reinstall --no-deps réécrit les fichiers cv2/ partagés sans toucher
    # aux autres dépendances ni laisser le paquet "vidé" comme un uninstall direct le ferait.
    pip install --no-cache-dir --force-reinstall --no-deps opencv-python-headless && \
    python -c "import cv2; print('cv2 OK', cv2.__version__)"

# 2) Code applicatif et assets
COPY app/        /app/app/
COPY src/        /app/src/
COPY models/     /app/models/
COPY README.md   /app/README.md

# 3) Variables d'environnement
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=7860 \
    HOME=/tmp

EXPOSE 7860

# Pas de USER non-root par défaut : Gradio crée des fichiers temporaires
# (HOME pointe vers /tmp pour rester compatible avec un --read-only éventuel)

CMD ["python", "app/gradio_app.py"]
