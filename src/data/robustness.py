"""Génération de test sets dégradés pour l'étude de robustesse.

Conditions évaluées :
  - Luminosité (gamma) : 0.5 (sombre), 0.75 (légèrement sombre), 1.5 (clair)
  - Bruit gaussien : sigma=15 (modéré), sigma=35 (fort)

Pour chaque condition, on régénère le test set sans toucher aux annotations
(les bbox restent valides puisque la géométrie est préservée).
"""
from __future__ import annotations

import shutil
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


def adjust_gamma(image: np.ndarray, gamma: float) -> np.ndarray:
    inv = 1.0 / max(gamma, 1e-3)
    table = np.array([((i / 255.0) ** inv) * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(image, table)


def add_gaussian_noise(image: np.ndarray, sigma: float) -> np.ndarray:
    noise = np.random.normal(0, sigma, image.shape).astype(np.float32)
    noisy = image.astype(np.float32) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)


CONDITIONS: dict[str, callable] = {
    "lum_dark":   lambda img: adjust_gamma(img, 0.5),
    "lum_dim":    lambda img: adjust_gamma(img, 0.75),
    "lum_bright": lambda img: adjust_gamma(img, 1.5),
    "noise_med":  lambda img: add_gaussian_noise(img, 15),
    "noise_high": lambda img: add_gaussian_noise(img, 35),
}


def generate_robustness_sets_yolo(test_images_dir: str | Path,
                                  test_labels_dir: str | Path,
                                  output_root: str | Path,
                                  conditions: dict[str, callable] = None) -> dict[str, Path]:
    """Génère un dossier par condition au format YOLO (images + labels).

    Structure produite :
      output_root/
        lum_dark/images/*.jpg
        lum_dark/labels/*.txt
        lum_dim/...
        ...
    """
    test_images_dir = Path(test_images_dir)
    test_labels_dir = Path(test_labels_dir)
    output_root = Path(output_root)
    conditions = conditions or CONDITIONS

    image_files = sorted(test_images_dir.glob("*.jpg")) + sorted(test_images_dir.glob("*.png"))
    if not image_files:
        raise FileNotFoundError(f"Aucune image dans {test_images_dir}")

    out_paths = {}
    for cond_name, fn in conditions.items():
        out_img = output_root / cond_name / "images"
        out_lbl = output_root / cond_name / "labels"
        out_img.mkdir(parents=True, exist_ok=True)
        out_lbl.mkdir(parents=True, exist_ok=True)

        for img_path in tqdm(image_files, desc=f"  {cond_name}"):
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            degraded = fn(img)
            cv2.imwrite(str(out_img / img_path.name), degraded)

            # Copie le label correspondant
            label_path = test_labels_dir / (img_path.stem + ".txt")
            if label_path.exists():
                shutil.copy(label_path, out_lbl / label_path.name)

        out_paths[cond_name] = output_root / cond_name
    return out_paths


def make_data_yaml_for_condition(condition_dir: Path,
                                 base_data_yaml: Path,
                                 output_yaml: Path) -> Path:
    """Crée un data.yaml pointant vers le test set dégradé, en gardant
    train/val depuis le data.yaml original (utile pour `model.val(split=...)`)."""
    import yaml
    with open(base_data_yaml) as f:
        cfg = yaml.safe_load(f)

    cfg["test"] = str((condition_dir / "images").resolve())
    output_yaml.parent.mkdir(parents=True, exist_ok=True)
    with open(output_yaml, "w") as f:
        yaml.safe_dump(cfg, f)
    return output_yaml


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--test_images", required=True)
    parser.add_argument("--test_labels", required=True)
    parser.add_argument("--output", default="data/robustness")
    args = parser.parse_args()

    paths = generate_robustness_sets_yolo(args.test_images, args.test_labels, args.output)
    for k, v in paths.items():
        print(f"  {k} -> {v}")
