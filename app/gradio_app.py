"""Application Gradio — Détection ASL en temps réel.

3 onglets :
  1. Upload : comparaison YOLO vs RF-DETR sur une image
  2. Webcam : détection temps réel
  3. Épellation : accumule les lettres détectées pour former un mot

Lancement :
    python app/gradio_app.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import gradio as gr
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.inference.unified import UnifiedDetector, annotate_image

YOLO_WEIGHTS = ROOT / "models" / "yolo" / "best.pt"
RFDETR_WEIGHTS = ROOT / "models" / "rfdetr" / "checkpoint_best_total.pth"

ASL_CLASSES = [chr(ord("A") + i) for i in range(26)]

# Chargement paresseux des modèles (évite de bloquer le démarrage si un weight manque)
_detectors: dict[str, UnifiedDetector] = {}


def get_detector(backend: str, threshold: float) -> UnifiedDetector | None:
    key = backend
    if key in _detectors:
        _detectors[key].confidence_threshold = threshold
        return _detectors[key]

    weights = YOLO_WEIGHTS if backend == "yolo" else RFDETR_WEIGHTS
    if not weights.exists():
        return None

    try:
        det = UnifiedDetector(
            backend=backend,
            weights_path=weights,
            class_names=ASL_CLASSES,
            confidence_threshold=threshold,
        )
        _detectors[key] = det
        return det
    except Exception as e:
        print(f"[WARN] Impossible de charger {backend}: {e}")
        return None


def predict_image(image: np.ndarray, threshold: float):
    """Renvoie 2 images annotées (YOLO, RF-DETR) + résumé."""
    if image is None:
        return None, None, "Aucune image."

    yolo = get_detector("yolo", threshold)
    rfdetr = get_detector("rfdetr", threshold)

    summary_lines = []

    if yolo is None:
        yolo_img = image
        summary_lines.append("YOLO : modèle non chargé (poids manquants).")
    else:
        dets, ms = yolo.predict_with_timing(image)
        yolo_img = annotate_image(image, dets, title=f"YOLO11n - {ms:.1f} ms")
        letters = ", ".join(sorted({d.class_name for d in dets}))
        summary_lines.append(f"YOLO : {len(dets)} détection(s) en {ms:.1f} ms - {letters or 'rien'}")

    if rfdetr is None:
        rfdetr_img = image
        summary_lines.append("RF-DETR : modèle non chargé (poids manquants).")
    else:
        dets, ms = rfdetr.predict_with_timing(image)
        rfdetr_img = annotate_image(image, dets, title=f"RF-DETR-S - {ms:.1f} ms")
        letters = ", ".join(sorted({d.class_name for d in dets}))
        summary_lines.append(f"RF-DETR : {len(dets)} détection(s) en {ms:.1f} ms - {letters or 'rien'}")

    return yolo_img, rfdetr_img, "\n".join(summary_lines)


def predict_webcam(image: np.ndarray, backend: str, threshold: float):
    if image is None:
        return None, "En attente..."

    det = get_detector(backend, threshold)
    if det is None:
        return image, f"Modèle {backend} non chargé."

    dets, ms = det.predict_with_timing(image)
    annotated = annotate_image(image, dets, title=f"{backend.upper()} - {ms:.1f} ms - {1000/ms:.0f} FPS")
    if dets:
        top = max(dets, key=lambda d: d.confidence)
        info = f"Lettre détectée : **{top.class_name}**  (confiance {top.confidence:.2f})"
    else:
        info = "Aucun signe détecté."
    return annotated, info


# État partagé pour l'onglet épellation
class SpellingState:
    def __init__(self):
        self.word = ""
        self.last_letter = None
        self.last_time = 0.0
        self.stable_since = 0.0

    def reset(self):
        self.word = ""
        self.last_letter = None
        self.last_time = 0.0
        self.stable_since = 0.0


_spelling = SpellingState()


def predict_spelling(image: np.ndarray, backend: str, threshold: float, hold_seconds: float):
    """Accumule la lettre dans le mot si elle est détectée de manière stable
    pendant `hold_seconds` secondes."""
    if image is None:
        return None, _spelling.word, "En attente..."

    det = get_detector(backend, threshold)
    if det is None:
        return image, _spelling.word, f"Modèle {backend} non chargé."

    dets, ms = det.predict_with_timing(image)
    annotated = annotate_image(image, dets, title=f"{backend.upper()} - {ms:.1f} ms")

    now = time.time()
    if dets:
        top = max(dets, key=lambda d: d.confidence)
        letter = top.class_name
        if letter == _spelling.last_letter:
            elapsed = now - _spelling.stable_since
            if elapsed >= hold_seconds and (now - _spelling.last_time) > hold_seconds:
                _spelling.word += letter
                _spelling.last_time = now
                _spelling.stable_since = now + 1e9  # bloque jusqu'au changement
                status = f"Ajouté : {letter} - mot : {_spelling.word}"
            else:
                remaining = max(0, hold_seconds - elapsed)
                status = f"Maintenir {letter} encore {remaining:.1f}s"
        else:
            _spelling.last_letter = letter
            _spelling.stable_since = now
            status = f"Lettre détectée : {letter} (maintenir {hold_seconds:.1f}s)"
    else:
        _spelling.last_letter = None
        status = "Aucun signe détecté."

    return annotated, _spelling.word, status


def reset_word():
    _spelling.reset()
    return "", "Mot réinitialisé."


def add_space():
    _spelling.word += " "
    return _spelling.word, "Espace ajouté."


def backspace():
    _spelling.word = _spelling.word[:-1]
    return _spelling.word, "Dernière lettre supprimée."


def build_ui():
    alphabet_path = ROOT / "app" / "assets" / "asl_alphabet.png"

    with gr.Blocks(title="ASL Detection - YOLO vs RF-DETR") as demo:
        gr.Markdown(
            """
            # Détection en Temps Réel de la Langue des Signes (ASL)
            Étude comparative entre **YOLO11n** et **RF-DETR-S** sur les 26 lettres de l'alphabet ASL.
            """
        )

        with gr.Tab("Alphabet ASL"):
            gr.Markdown(
                """
                ## Référence visuelle de l'alphabet ASL

                Chaque tuile montre un échantillon réel du dataset d'entraînement, recadré
                autour de la main, avec la lettre cible en bas. Utilise cette grille pour
                **t'entraîner aux signes** avant de passer à l'onglet *Webcam* ou
                *Épellation*.

                **Astuce démo** : commence par les lettres aux signes très distincts
                (A, B, C, L, O, Y) avant les plus subtiles (M, N, T).
                """
            )
            if alphabet_path.exists():
                gr.Image(str(alphabet_path), label="26 lettres ASL", show_label=False,
                         container=True)
            else:
                gr.Markdown("*(Grille de référence non générée — relance `python -c \"...\"` pour la créer)*")

        with gr.Tab("1. Comparaison sur image"):
            with gr.Row():
                with gr.Column(scale=1):
                    img_input = gr.Image(label="Image d'entrée", type="numpy", sources=["upload"])
                    threshold1 = gr.Slider(0.05, 0.95, value=0.25, step=0.05, label="Seuil de confiance")
                    btn = gr.Button("Détecter avec les deux modèles", variant="primary")
                with gr.Column(scale=2):
                    with gr.Row():
                        yolo_out = gr.Image(label="YOLO11n", type="numpy")
                        rfdetr_out = gr.Image(label="RF-DETR-S", type="numpy")
                    summary1 = gr.Textbox(label="Résumé", lines=3)
            btn.click(predict_image, inputs=[img_input, threshold1],
                      outputs=[yolo_out, rfdetr_out, summary1])

        with gr.Tab("2. Webcam temps réel"):
            with gr.Row():
                with gr.Column(scale=1):
                    backend2 = gr.Radio(["yolo", "rfdetr"], value="yolo", label="Modèle")
                    threshold2 = gr.Slider(0.05, 0.95, value=0.4, step=0.05, label="Seuil de confiance")
                    info2 = gr.Markdown("En attente du flux webcam...")
                with gr.Column(scale=2):
                    webcam_in = gr.Image(label="Webcam", sources=["webcam"], streaming=True, type="numpy")
                    webcam_out = gr.Image(label="Détection", type="numpy")
            webcam_in.stream(predict_webcam, inputs=[webcam_in, backend2, threshold2],
                             outputs=[webcam_out, info2], stream_every=0.2)

        with gr.Tab("3. Épellation en direct"):
            gr.Markdown("Maintiens une lettre devant la caméra pour l'ajouter au mot.")
            with gr.Row():
                with gr.Column(scale=1):
                    backend3 = gr.Radio(["yolo", "rfdetr"], value="yolo", label="Modèle")
                    threshold3 = gr.Slider(0.3, 0.95, value=0.55, step=0.05, label="Seuil de confiance")
                    hold = gr.Slider(0.5, 3.0, value=1.5, step=0.1, label="Durée de stabilité (secondes)")
                    word_out = gr.Textbox(label="Mot épelé", value="")
                    status3 = gr.Textbox(label="Statut", lines=2)
                    with gr.Row():
                        space_btn = gr.Button("Espace")
                        bksp_btn = gr.Button("Effacer dernière")
                        reset_btn = gr.Button("Réinitialiser", variant="stop")
                with gr.Column(scale=2):
                    spell_in = gr.Image(label="Webcam", sources=["webcam"], streaming=True, type="numpy")
                    spell_out = gr.Image(label="Détection", type="numpy")
            spell_in.stream(predict_spelling, inputs=[spell_in, backend3, threshold3, hold],
                            outputs=[spell_out, word_out, status3], stream_every=0.2)
            space_btn.click(add_space, outputs=[word_out, status3])
            bksp_btn.click(backspace, outputs=[word_out, status3])
            reset_btn.click(reset_word, outputs=[word_out, status3])

        gr.Markdown(
            """
            ---
            *Projet académique - DIC3 Deep Learning - 2026*
            """
        )
    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.queue().launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True,
        theme=gr.themes.Soft(),
    )
