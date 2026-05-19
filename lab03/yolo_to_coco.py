"""
Konwerter YOLO (txt label per obrazek) → COCO JSON.
Obsługuje dwa układy:

  A) Roboflow YOLOv8 export:
        <root>/<split>/images/*.jpg
        <root>/<split>/labels/*.txt
        <root>/data.yaml
     Zapisuje:  <root>/<split>/_annotations.coco.json
     (zgodne z układem "Roboflow COCO" w benchmark.py — żadnych zmian w benchmark.py nie trzeba)

  B) "Sidecar" YOLO:
        <root>/images/<split>/*.jpg
        <root>/labels/<split>/*.txt
        <root>/data.yaml
     Zapisuje:  <root>/<split>_annotations.json

Użycie:
    python yolo_to_coco.py --root /sciezka/do/lab03 --split test
    python yolo_to_coco.py --root /sciezka/do/lab03 --split valid
    python yolo_to_coco.py --root /sciezka/do/lab03 --split train
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import yaml


def detect_layout(root: Path, split: str):
    """Zwraca (images_dir, labels_dir, layout_name)."""
    rf_img = root / split / "images"
    rf_lbl = root / split / "labels"
    if rf_img.is_dir() and rf_lbl.is_dir():
        return rf_img, rf_lbl, "roboflow"
    old_img = root / "images" / split
    old_lbl = root / "labels" / split
    if old_img.is_dir() and old_lbl.is_dir():
        return old_img, old_lbl, "sidecar"
    raise FileNotFoundError(
        f"Nie znaleziono układu YOLO dla split={split!r} w {root}.\n"
        f"Oczekiwano jednego z:\n"
        f"  • {rf_img} + {rf_lbl}  (Roboflow YOLOv8)\n"
        f"  • {old_img} + {old_lbl}  (sidecar)")


def load_class_names(root: Path) -> list[str]:
    for cand in ["data.yaml", "dataset.yaml"]:
        p = root / cand
        if p.exists():
            cfg = yaml.safe_load(p.read_text())
            names = cfg["names"]
            if isinstance(names, dict):
                names = [names[i] for i in sorted(names.keys())]
            return list(names)
    raise FileNotFoundError(f"Brak data.yaml / dataset.yaml w {root}")


def convert(root: Path, split: str) -> Path:
    img_dir, lbl_dir, layout = detect_layout(root, split)
    names = load_class_names(root)
    categories = [{"id": i + 1, "name": n} for i, n in enumerate(names)]

    coco = {"images": [], "annotations": [], "categories": categories}
    img_id = 0
    ann_id = 0
    skipped = 0

    for img_path in sorted(img_dir.iterdir()):
        if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png", ".bmp"):
            continue
        img_id += 1
        img = cv2.imread(str(img_path))
        if img is None:
            skipped += 1
            continue
        h, w = img.shape[:2]

        # file_name: dla Roboflow wskazuje "images/X.jpg" względem katalogu split,
        # dla sidecar — sama nazwa pliku.
        file_name = f"images/{img_path.name}" if layout == "roboflow" else img_path.name

        coco["images"].append({
            "id": img_id,
            "file_name": file_name,
            "width": w,
            "height": h,
        })

        lbl = lbl_dir / f"{img_path.stem}.txt"
        if not lbl.exists():
            continue
        for line in lbl.read_text().splitlines():
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls = int(float(parts[0]))
            cx = float(parts[1]) * w
            cy = float(parts[2]) * h
            bw = float(parts[3]) * w
            bh = float(parts[4]) * h
            x, y = cx - bw / 2, cy - bh / 2
            ann_id += 1
            coco["annotations"].append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": cls + 1,
                "bbox": [x, y, bw, bh],
                "area": bw * bh,
                "iscrowd": 0,
            })

    out_file = (root / split / "_annotations.coco.json" if layout == "roboflow"
                else root / f"{split}_annotations.json")
    out_file.write_text(json.dumps(coco))

    # statystyki rozmiarów obiektów (very_tiny/tiny/small/medium/large)
    import math
    very_tiny = tiny = small = medium = large = 0
    for a in coco["annotations"]:
        side = math.sqrt(a["area"])
        if side < 8:     very_tiny += 1
        elif side < 16:  tiny += 1
        elif side < 32:  small += 1
        elif side < 96:  medium += 1
        else:            large += 1

    print(f"[{layout}] split={split}: {len(coco['images'])} obrazów "
          f"(pominięto {skipped}), {len(coco['annotations'])} adnotacji, "
          f"{len(categories)} klas")
    print(f"  rozmiary: very_tiny={very_tiny}, tiny={tiny}, small={small}, "
          f"medium={medium}, large={large}")
    print(f"  → {out_file}")
    return out_file


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="korzeń datasetu (z data.yaml)")
    ap.add_argument("--split", default="test")
    args = ap.parse_args()
    convert(Path(args.root), args.split)


if __name__ == "__main__":
    main()