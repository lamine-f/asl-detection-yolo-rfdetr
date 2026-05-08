"""Génère la grille de référence ASL pour l'app Gradio.

Pour chaque lettre A-Z, prend la première image du train set qui la contient,
crop autour de la bbox et compose une grille 7×4 sauvegardée dans
`app/assets/asl_alphabet.png`.

Usage :
    python src/data/build_alphabet_grid.py [--ds_dir data/asl_yolo]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml
from PIL import Image, ImageDraw, ImageFont


def build(ds_dir: Path, out_path: Path):
    train = ds_dir / "train"
    with open(ds_dir / "data.yaml") as f:
        cfg = yaml.safe_load(f)
    classes = cfg["names"]

    samples: dict[str, Image.Image] = {}
    for img_path in sorted((train / "images").glob("*.jpg")):
        if len(samples) >= 26:
            break
        label_path = train / "labels" / (img_path.stem + ".txt")
        if not label_path.exists():
            continue
        with open(label_path) as f:
            first = f.readline().split()
        if not first:
            continue
        cls = classes[int(first[0])]
        if cls in samples:
            continue
        x_c, y_c, w, h = map(float, first[1:5])
        img = Image.open(img_path).convert("RGB")
        W, H = img.size
        cx, cy, bw, bh = x_c * W, y_c * H, w * W, h * H
        side = int(max(bw, bh) * 1.4)
        side = min(side, min(W, H))
        x1 = max(0, int(cx - side / 2))
        y1 = max(0, int(cy - side / 2))
        x2 = min(W, x1 + side)
        y2 = min(H, y1 + side)
        samples[cls] = img.crop((x1, y1, x2, y2)).resize((180, 180))

    TILE = 180
    LABEL_H = 28
    COLS, ROWS = 7, 4
    PAD = 8
    HEADER = 50
    W = COLS * (TILE + PAD) + PAD
    H = ROWS * (TILE + LABEL_H + PAD) + PAD + HEADER

    grid = Image.new("RGB", (W, H), color="white")
    draw = ImageDraw.Draw(grid)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22)
        title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 28)
    except OSError:
        font = ImageFont.load_default()
        title_font = font

    draw.text((PAD, 8), "Alphabet de la Langue des Signes Americaine (ASL)",
              fill=(20, 20, 60), font=title_font)

    letters = [chr(ord("A") + i) for i in range(26)]
    for idx, letter in enumerate(letters):
        col = idx % COLS
        row = idx // COLS
        x = PAD + col * (TILE + PAD)
        y = HEADER + PAD + row * (TILE + LABEL_H + PAD)

        if letter in samples:
            grid.paste(samples[letter], (x, y))
        else:
            draw.rectangle([x, y, x + TILE, y + TILE], fill=(220, 220, 220), outline=(150, 150, 150))

        draw.rectangle([x, y + TILE, x + TILE, y + TILE + LABEL_H], fill=(40, 80, 140))
        bbox = draw.textbbox((0, 0), letter, font=font)
        tw = bbox[2] - bbox[0]
        draw.text((x + (TILE - tw) // 2, y + TILE + 3),
                  letter, fill=(255, 255, 255), font=font)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(out_path)
    print(f"Saved: {out_path} ({out_path.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ds_dir", default="data/asl_yolo")
    parser.add_argument("--out", default="app/assets/asl_alphabet.png")
    args = parser.parse_args()
    build(Path(args.ds_dir), Path(args.out))
