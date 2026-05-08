"""Wrapper unifié pour YOLO11 et RF-DETR.

Expose une API commune `predict(image) -> list[Detection]` pour faciliter
la comparaison équitable et l'intégration dans l'application Gradio.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from PIL import Image


@dataclass
class Detection:
    class_id: int
    class_name: str
    confidence: float
    bbox_xyxy: tuple[float, float, float, float]


class UnifiedDetector:
    """Interface commune. Charge soit un YOLO, soit un RF-DETR."""

    def __init__(
        self,
        backend: Literal["yolo", "rfdetr"],
        weights_path: str | Path,
        class_names: list[str] | None = None,
        confidence_threshold: float = 0.25,
    ):
        self.backend = backend
        self.weights_path = str(weights_path)
        self.confidence_threshold = confidence_threshold
        self._class_names = class_names

        if backend == "yolo":
            from ultralytics import YOLO
            self._model = YOLO(self.weights_path)
            if class_names is None:
                self._class_names = list(self._model.names.values())
        elif backend == "rfdetr":
            from rfdetr import RFDETRSmall
            self._model = RFDETRSmall(pretrain_weights=self.weights_path)
            if class_names is None:
                self._class_names = self._infer_rfdetr_classes()
        else:
            raise ValueError(f"backend inconnu: {backend}")

    def _infer_rfdetr_classes(self) -> list[str]:
        """26 lettres ASL par défaut si pas inférable du modèle."""
        return [chr(ord("A") + i) for i in range(26)]

    def predict(self, image: Image.Image | np.ndarray) -> list[Detection]:
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        if self.backend == "yolo":
            return self._predict_yolo(image)
        return self._predict_rfdetr(image)

    def _predict_yolo(self, image: Image.Image) -> list[Detection]:
        results = self._model.predict(
            image, conf=self.confidence_threshold, verbose=False
        )
        out = []
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                cls_id = int(box.cls.item())
                conf = float(box.conf.item())
                xyxy = tuple(float(x) for x in box.xyxy[0].cpu().numpy())
                out.append(Detection(
                    class_id=cls_id,
                    class_name=self._class_names[cls_id],
                    confidence=conf,
                    bbox_xyxy=xyxy,
                ))
        return out

    def _predict_rfdetr(self, image: Image.Image) -> list[Detection]:
        detections = self._model.predict(image, threshold=self.confidence_threshold)
        out = []
        # rfdetr renvoie un sv.Detections
        if hasattr(detections, "xyxy"):
            for i in range(len(detections.xyxy)):
                cls_id = int(detections.class_id[i])
                conf = float(detections.confidence[i])
                xyxy = tuple(float(x) for x in detections.xyxy[i])
                name = self._class_names[cls_id] if cls_id < len(self._class_names) else str(cls_id)
                out.append(Detection(
                    class_id=cls_id,
                    class_name=name,
                    confidence=conf,
                    bbox_xyxy=xyxy,
                ))
        return out

    def predict_with_timing(self, image) -> tuple[list[Detection], float]:
        start = time.perf_counter()
        dets = self.predict(image)
        return dets, (time.perf_counter() - start) * 1000.0


def annotate_image(
    image: Image.Image | np.ndarray,
    detections: list[Detection],
    title: str | None = None,
) -> np.ndarray:
    """Dessine les bounding boxes et labels sur l'image."""
    import cv2

    if isinstance(image, Image.Image):
        img = np.array(image.convert("RGB"))
    else:
        img = image.copy()
    if img.dtype != np.uint8:
        img = img.astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det.bbox_xyxy]
        color = (0, 200, 0)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        label = f"{det.class_name} {det.confidence:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(img, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        cv2.putText(img, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    if title:
        cv2.rectangle(img, (0, 0), (img.shape[1], 40), (0, 0, 0), -1)
        cv2.putText(img, title, (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
