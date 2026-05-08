"""Imprime les valeurs à recopier dans le rapport LaTeX.

Usage:
    python src/evaluation/print_report_values.py path/to/all_results.json

Le fichier all_results.json est généré par notebooks/04_evaluation_and_robustness.ipynb.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def fmt(x, digits=4):
    if x is None or (isinstance(x, float) and (x != x)):
        return "—"
    if isinstance(x, (int, float)):
        return f"{x:.{digits}f}"
    return str(x)


def main(json_path: str):
    with open(json_path) as f:
        data = json.load(f)

    base_y = data["baseline"]["YOLO11n"]
    base_r = data["baseline"]["RF-DETR-S"]
    rob_y = data["robustness"]["YOLO11n"]
    rob_r = data["robustness"]["RF-DETR-S"]

    print("=" * 70)
    print("  VALEURS À INJECTER DANS LE RAPPORT")
    print("  Recherche les marqueurs \\TODO{...} (rouge dans le PDF) et remplace.")
    print("=" * 70)

    print("\n--- 1. ABSTRACT (recherche \\TODO{YOLO/RF-DETR}, \\TODO{mAP=XX}, etc.) ---")
    winner_map = "YOLO11n" if base_y.get("mAP50", 0) > base_r.get("mAP50", 0) else "RF-DETR-S"
    other = "RF-DETR-S" if winner_map == "YOLO11n" else "YOLO11n"
    best_map = max(base_y.get("mAP50", 0), base_r.get("mAP50", 0))
    print(f"  Modèle gagnant en mAP : {winner_map}")
    print(f"  mAP@0.5 du gagnant    : {fmt(best_map)}")
    print(f"  Autre modèle          : {other}")

    print("\n--- 2. TABLEAU \\label{tab:results} ---")
    print(f"  YOLO mAP@0.5         : {fmt(base_y.get('mAP50'))}")
    print(f"  RF-DETR mAP@0.5      : {fmt(base_r.get('mAP50'))}")
    print(f"  YOLO mAP@0.5:0.95    : {fmt(base_y.get('mAP50-95'))}")
    print(f"  RF-DETR mAP@0.5:0.95 : {fmt(base_r.get('mAP50-95'))}")
    print(f"  YOLO précision       : {fmt(base_y.get('precision'))}")
    print(f"  RF-DETR précision    : {fmt(base_r.get('precision'))}")
    print(f"  YOLO rappel          : {fmt(base_y.get('recall'))}")
    print(f"  RF-DETR rappel       : {fmt(base_r.get('recall'))}")
    print(f"  YOLO F1              : {fmt(base_y.get('F1'))}")
    print(f"  RF-DETR F1           : {fmt(base_r.get('F1'))}")
    print(f"  YOLO latence (ms)    : {fmt(base_y.get('inference_ms'), 1)}")
    print(f"  RF-DETR latence (ms) : {fmt(base_r.get('inference_ms'), 1)}")
    print(f"  YOLO FPS             : {fmt(base_y.get('fps'), 0)}")
    print(f"  RF-DETR FPS          : {fmt(base_r.get('fps'), 0)}")
    print(f"  YOLO taille (MB)     : {fmt(base_y.get('size_MB'), 1)}")
    print(f"  RF-DETR taille (MB)  : {fmt(base_r.get('size_MB'), 1)}")

    print("\n--- 3. TABLEAU \\label{tab:robustness} (mAP@0.5 et chute relative %) ---")
    base_y_map = base_y.get("mAP50", 0)
    base_r_map = base_r.get("mAP50", 0)

    def drop_pct(cur, base):
        if base == 0:
            return 0.0
        return (cur - base) / base * 100.0

    print(f"  Baseline       : YOLO {fmt(base_y_map, 3)} / RF-DETR {fmt(base_r_map, 3)}  (Δ% = 0)")
    for cond_label, cond_key in [
        ("γ=1.5",    "lum_bright"),
        ("γ=0.75",   "lum_dim"),
        ("γ=0.5",    "lum_dark"),
        ("σ=15",     "noise_med"),
        ("σ=35",     "noise_high"),
    ]:
        ymap = rob_y.get(cond_key, {}).get("mAP50")
        rmap = rob_r.get(cond_key, {}).get("mAP50")
        ydrop = drop_pct(ymap, base_y_map) if ymap is not None else None
        rdrop = drop_pct(rmap, base_r_map) if rmap is not None else None
        print(f"  {cond_label:8s}     : YOLO {fmt(ymap, 3)} (Δ {fmt(ydrop, 1)}%) / "
              f"RF-DETR {fmt(rmap, 3)} (Δ {fmt(rdrop, 1)}%)")

    print("\n--- 4. ANALYSE QUALITATIVE (à intégrer en section Résultats) ---")
    if base_y_map > base_r_map:
        print(f"  YOLO gagne en nominal (+{(base_y_map-base_r_map)*100:.1f} pts mAP)")
    else:
        print(f"  RF-DETR gagne en nominal (+{(base_r_map-base_y_map)*100:.1f} pts mAP)")

    print("\n  Lecture robustesse (chute moyenne en valeur absolue) :")
    yolo_drops = [abs(drop_pct(rob_y[k]["mAP50"], base_y_map))
                  for k in rob_y if k != "baseline" and rob_y[k].get("mAP50") is not None]
    rfdetr_drops = [abs(drop_pct(rob_r[k]["mAP50"], base_r_map))
                    for k in rob_r if k != "baseline" and rob_r[k].get("mAP50") is not None]
    if yolo_drops and rfdetr_drops:
        print(f"  YOLO    : chute moyenne = {sum(yolo_drops)/len(yolo_drops):.1f}%")
        print(f"  RF-DETR : chute moyenne = {sum(rfdetr_drops)/len(rfdetr_drops):.1f}%")
        if sum(yolo_drops)/len(yolo_drops) < sum(rfdetr_drops)/len(rfdetr_drops):
            print("  -> YOLO est plus robuste aux dégradations")
        else:
            print("  -> RF-DETR est plus robuste aux dégradations")

    print("\n" + "=" * 70)
    print("  Recopie ces valeurs dans _0_doccuments/rapport/main.tex")
    print("  puis recompile : pdflatex main.tex (deux fois pour les références)")
    print("=" * 70)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1])
