"""
Benchmark detekcji małych obiektów — wszystkie podejścia + metody filtrowania.
=============================================================================

Implementuje:
  * Podejścia detekcji:
      A) baseline       — detekcja na obrazie przeskalowanym do 640
      B) focus_detect   — niskorozdzielczy podgląd → ROI → wycinki pełnej rozdz.
      C) sliding_window — okno przesuwne 640x640 ze stride

  * Metody filtrowania:
      NMS — klasyczny per-klasa
      OBS — Overlapping Box Suppression (preferuje większe fragmenty na granicach okien)
      OBM — Overlapping Box Merging   (łączy fragmenty w jeden union bbox)

  * Ewaluacja: COCO API z niestandardowymi przedziałami area:
      AP, AP50, AP75, AP_vt (very tiny), AP_t (tiny), AP_s (small)

  * Pomiary czasu: osobno faza ROI i faza detekcji, FPS

  * Wizualizacje: maski ROI nałożone na obraz + bbox detekcji vs ground truth

  * Wyniki: results/summary.csv + 3 wykresy + przykładowe wizualizacje

Użycie:
    python benchmark.py --data data/synthetic --model yolov8n.pt
    python benchmark.py --data data/synthetic --model runs/train/.../best.pt --max-images 30
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from ultralytics import YOLO


# ============================================================================
# KONFIGURACJA EKSPERYMENTU
# ============================================================================
@dataclass
class ExperimentConfig:
    name: str
    dataset_name: str
    images_dir: Path
    annotations_file: Path
    model_path: str
    approach: str                                  # baseline | focus_detect | sliding_window
    filter_method: str = "nms"                     # nms | obs | obm
    roi_method: str = "yolo"                       # yolo | saliency | tracking
    roi_resolution: Tuple[int, int] = (640, 360)
    detector_resolution: int = 640
    sliding_window_size: int = 640
    sliding_stride: int = 480
    conf_threshold: float = 0.25
    iou_threshold: float = 0.5
    roi_conf: float = 0.10
    roi_expand: float = 0.30
    # tracking-only:
    track_max_age: int = 5
    track_match_iou: float = 0.2
    track_expand: float = 0.5
    # saliency-only:
    saliency_method: str = "fine"                  # fine | spectral
    saliency_threshold: float = 0.35

    def __post_init__(self):
        self.images_dir = Path(self.images_dir)
        self.annotations_file = Path(self.annotations_file)


# ============================================================================
# PREPROCESSING — generowanie ROI
# ============================================================================
def sliding_windows(img_w: int, img_h: int, win: int = 640,
                    stride: int = 480) -> List[Tuple[int, int, int, int]]:
    """Okna (x, y, w, h) pokrywające obraz (Approach C)."""
    if img_w <= win and img_h <= win:
        return [(0, 0, img_w, img_h)]
    xs = list(range(0, max(1, img_w - win), stride)) + [max(0, img_w - win)]
    ys = list(range(0, max(1, img_h - win), stride)) + [max(0, img_h - win)]
    xs = sorted(set(xs))
    ys = sorted(set(ys))
    return [(x, y, min(win, img_w - x), min(win, img_h - y))
            for y in ys for x in xs]


def _iou_xyxy(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def merge_overlapping_rois(boxes_xywh, iou_thr: float = 0.7):
    """Łączenie nakładających się ROI — redukuje powtórne inference."""
    if not boxes_xywh:
        return []
    boxes = np.array([(x, y, x + w, y + h) for (x, y, w, h) in boxes_xywh], dtype=float)
    used = np.zeros(len(boxes), dtype=bool)
    keep = []
    for i in range(len(boxes)):
        if used[i]:
            continue
        group = [i]
        for j in range(i + 1, len(boxes)):
            if not used[j] and _iou_xyxy(boxes[i], boxes[j]) > iou_thr:
                group.append(j)
                used[j] = True
        used[i] = True
        x1 = min(boxes[g][0] for g in group)
        y1 = min(boxes[g][1] for g in group)
        x2 = max(boxes[g][2] for g in group)
        y2 = max(boxes[g][3] for g in group)
        keep.append((int(x1), int(y1), int(x2 - x1), int(y2 - y1)))
    return keep


def yolo_preview_rois(model: YOLO, image: np.ndarray, roi_res: Tuple[int, int],
                      conf: float, expand: float,
                      win: int = 640) -> List[Tuple[int, int, int, int]]:
    """ROI method "yolo": niskorozdzielczy podgląd YOLO → ROI o boku ≥ `win` wokół detekcji.

    Uproszczenie podejścia Focus-and-Detect: zamiast dedykowanego segmentera
    używamy detektora w niskiej rozdzielczości jako szybkiego heurystycznego
    estymatora "gdzie mogą być obiekty".
    """
    h, w = image.shape[:2]
    rw, rh = roi_res
    low = cv2.resize(image, (rw, rh))
    res = model.predict(low, conf=conf, verbose=False)[0]

    if res.boxes is None or len(res.boxes) == 0:
        cx, cy = w // 2, h // 2
        x = max(0, cx - win // 2)
        y = max(0, cy - win // 2)
        return [(x, y, min(win, w - x), min(win, h - y))]

    sx, sy = w / rw, h / rh
    rois: List[Tuple[int, int, int, int]] = []
    for x1, y1, x2, y2 in res.boxes.xyxy.cpu().numpy():
        x1, y1, x2, y2 = x1 * sx, y1 * sy, x2 * sx, y2 * sy
        bw, bh = x2 - x1, y2 - y1
        size = max(win, int(bw * (1 + expand)), int(bh * (1 + expand)))
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        rx1 = max(0, int(cx - size / 2))
        ry1 = max(0, int(cy - size / 2))
        rx2 = min(w, rx1 + size)
        ry2 = min(h, ry1 + size)
        rois.append((rx1, ry1, rx2 - rx1, ry2 - ry1))
    return merge_overlapping_rois(rois, iou_thr=0.7)


# Wstecznie kompatybilny alias (używane w testach jednostkowych)
estimate_rois = yolo_preview_rois


def saliency_rois(image: np.ndarray, roi_res: Tuple[int, int],
                  threshold: float = 0.35,
                  min_area_frac: float = 1e-5,
                  expand: float = 0.30,
                  win: int = 640,
                  method: str = "fine") -> List[Tuple[int, int, int, int]]:
    """ROI method "saliency": dedykowany "segmenter" ROI bez treningu.

    Pipeline (zgodny duchem z SegTrackDetect, tylko że segmenter jest klasyczny
    a nie wytrenowany U-Net):
      1. Downscale obrazu do `roi_res` (szybko)
      2. Mapa saliency (OpenCV: FineGrained albo SpectralResidual)
      3. Próg binarny + morfologiczne domknięcie
      4. Connected components → bounding box per komponent
      5. Każdy bbox rozszerzony do okna ≥ `win` × `win`, mapowany do rozdz. oryginalnej
      6. Łączenie silnie nakładających się ROI

    Saliency jest "predykatem foreground/background", więc obejmuje regiony,
    których YOLO-podgląd by przegapił (np. małe ciemne plamki na jasnej wodzie).

    `method`:
        "fine"     — StaticSaliencyFineGrained  (per-pikselowe szczegóły, wolniejsze)
        "spectral" — StaticSaliencySpectralResidual (szybkie, mniej dokładne)
    """
    h, w = image.shape[:2]
    rw, rh = roi_res
    low = cv2.resize(image, (rw, rh))

    if method == "spectral":
        sal = cv2.saliency.StaticSaliencySpectralResidual_create()
    else:
        sal = cv2.saliency.StaticSaliencyFineGrained_create()

    ok, salmap = sal.computeSaliency(low)
    if not ok:
        # fallback — central crop
        cx, cy = w // 2, h // 2
        return [(max(0, cx - win // 2), max(0, cy - win // 2),
                 min(win, w), min(win, h))]

    # do 8-bit, próg, morfologia
    salmap = (salmap * 255).astype(np.uint8)
    _, mask = cv2.threshold(salmap, int(threshold * 255), 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))

    # connected components
    n_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    sx, sy = w / rw, h / rh
    min_area_px = max(4, int(rw * rh * min_area_frac))

    rois: List[Tuple[int, int, int, int]] = []
    for i in range(1, n_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_area_px:
            continue
        # bbox w skali niskorozdzielczej → skalowanie do oryginału
        bx = stats[i, cv2.CC_STAT_LEFT] * sx
        by = stats[i, cv2.CC_STAT_TOP]  * sy
        bw = stats[i, cv2.CC_STAT_WIDTH]  * sx
        bh = stats[i, cv2.CC_STAT_HEIGHT] * sy
        cx, cy = bx + bw / 2, by + bh / 2
        size = max(win, int(bw * (1 + expand)), int(bh * (1 + expand)))
        rx1 = max(0, int(cx - size / 2))
        ry1 = max(0, int(cy - size / 2))
        rx2 = min(w, rx1 + size)
        ry2 = min(h, ry1 + size)
        rois.append((rx1, ry1, rx2 - rx1, ry2 - ry1))

    if not rois:
        # nic nie wykryto — fallback do centralnego crop
        cx, cy = w // 2, h // 2
        return [(max(0, cx - win // 2), max(0, cy - win // 2),
                 min(win, w), min(win, h))]

    return merge_overlapping_rois(rois, iou_thr=0.5)


class TrackingROIEstimator:
    """ROI method "tracking": stateful estymator dla sekwencji klatek.

    Trzyma listę aktywnych tracków (bbox + wiek). Dla każdej nowej klatki:
      1. Każdy aktualny track rozszerzany do okna ≥ `win` (większy growth dla starszych)
      2. Wynikowe ROI używane do detekcji
      3. Po detekcji: tracki, które nie mają dopasowania w detekcjach, są
         postarzane. Po `max_age` klatkach bez dopasowania — usuwane.
      4. Nowe detekcje stają się nowymi trackami

    Dla pierwszej klatki sekwencji (lub gdy brak tracków) — fallback do
    funkcji `bootstrap_fn(image)` zwracającej startowe ROI.
    """

    def __init__(self, expand: float = 0.5, win: int = 640,
                 max_age: int = 5, match_iou: float = 0.2):
        self.expand = expand
        self.win = win
        self.max_age = max_age
        self.match_iou = match_iou
        self.tracks: List[List] = []   # list of [x1,y1,x2,y2,age]

    def reset(self) -> None:
        self.tracks = []

    def predict_rois(self, image_shape: Tuple[int, int],
                     bootstrap_fn=None,
                     image: Optional[np.ndarray] = None
                     ) -> List[Tuple[int, int, int, int]]:
        h, w = image_shape[:2]
        if not self.tracks:
            if bootstrap_fn is not None and image is not None:
                return bootstrap_fn(image)
            # bezpieczny fallback — sliding window
            return sliding_windows(w, h, self.win, self.win // 2)

        rois: List[Tuple[int, int, int, int]] = []
        for x1, y1, x2, y2, age in self.tracks:
            bw, bh = x2 - x1, y2 - y1
            # rosnący growth dla starszych tracków (większa niepewność lokalizacji)
            growth = 1.0 + self.expand * (1.0 + age * 0.5)
            size = max(self.win, int(bw * growth), int(bh * growth))
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            rx1 = max(0, int(cx - size / 2))
            ry1 = max(0, int(cy - size / 2))
            rx2 = min(w, rx1 + size)
            ry2 = min(h, ry1 + size)
            rois.append((rx1, ry1, rx2 - rx1, ry2 - ry1))
        return merge_overlapping_rois(rois, iou_thr=0.6)

    def update(self, detections: np.ndarray) -> None:
        """Aktualizuj listę tracków na bazie nowych detekcji (po filtrowaniu)."""
        new_tracks: List[List] = []
        matched_old = set()

        # nowe detekcje → świeże tracki (wiek 0)
        for d in detections:
            x1, y1, x2, y2 = float(d[0]), float(d[1]), float(d[2]), float(d[3])
            new_tracks.append([x1, y1, x2, y2, 0])
            # zaznacz stare tracki które się nakładają — będą zastąpione
            for j, (ox1, oy1, ox2, oy2, _age) in enumerate(self.tracks):
                if _iou_xyxy((x1, y1, x2, y2), (ox1, oy1, ox2, oy2)) > self.match_iou:
                    matched_old.add(j)

        # niesparowane stare tracki — postarzaj i zachowaj jeśli nie za stare
        for j, (x1, y1, x2, y2, age) in enumerate(self.tracks):
            if j not in matched_old and age + 1 < self.max_age:
                new_tracks.append([x1, y1, x2, y2, age + 1])

        self.tracks = new_tracks


# ============================================================================
# WYKRYWANIE SEKWENCJI (dla tracking-based ROI)
# ============================================================================
import re

_FRAME_RE = re.compile(r"^(.*?)(\d+)(\.[^.]+)$")


def extract_sequence_key(filename: str) -> Tuple[str, int]:
    """Wydziela (sequence_id, frame_number) z nazwy pliku.

    Przykłady:
        "seq01_frame_000123.jpg"   -> ("seq01_frame_", 123)
        "lake_constance/000456.png" -> ("lake_constance/", 456)
        "img001.jpg"               -> ("img", 1)
        "image_no_digits.jpg"      -> ("image_no_digits.jpg", 0)

    Bez końcowych cyfr → cała nazwa jest "sekwencją", frame=0 (efektywnie
    pojedyncza klatka, tracking nie ma czego predykować).
    """
    m = _FRAME_RE.match(filename)
    if m:
        return m.group(1), int(m.group(2))
    return filename, 0


def order_images_by_sequence(coco_imgs: dict, img_ids: List[int]) -> List[int]:
    """Sortuje img_ids tak, by klatki tej samej sekwencji szły po sobie chronologicznie."""
    keyed = [(extract_sequence_key(coco_imgs[i]["file_name"]), i) for i in img_ids]
    keyed.sort(key=lambda x: (x[0][0], x[0][1]))
    return [i for (_, i) in keyed]


# ============================================================================
# DETEKCJA NA WYCINKACH
# ============================================================================
def detect_full(model: YOLO, image: np.ndarray,
                det_res: int, conf: float) -> np.ndarray:
    """Approach A — predykcja na obrazie. Zwraca tablicę [x1,y1,x2,y2,score,cls,win_idx]."""
    res = model.predict(image, imgsz=det_res, conf=conf, verbose=False)[0]
    if res.boxes is None or len(res.boxes) == 0:
        return np.empty((0, 7))
    n = len(res.boxes)
    return np.column_stack([
        res.boxes.xyxy.cpu().numpy(),
        res.boxes.conf.cpu().numpy(),
        res.boxes.cls.cpu().numpy(),
        np.zeros(n),    # window index (jedno "okno" = pełny obraz)
    ])


def detect_on_crops(model: YOLO, image: np.ndarray,
                    crops: List[Tuple[int, int, int, int]],
                    det_res: int, conf: float) -> np.ndarray:
    """Detekcja na wycinkach, przesunięcie do układu obrazu, dopisanie window_idx."""
    out: List[np.ndarray] = []
    for w_idx, (x, y, w, h) in enumerate(crops):
        if w <= 0 or h <= 0:
            continue
        crop = image[y:y + h, x:x + w]
        if crop.size == 0:
            continue
        res = model.predict(crop, imgsz=det_res, conf=conf, verbose=False)[0]
        if res.boxes is None or len(res.boxes) == 0:
            continue
        xyxy = res.boxes.xyxy.cpu().numpy().copy()
        xyxy[:, [0, 2]] += x
        xyxy[:, [1, 3]] += y
        n = len(xyxy)
        out.append(np.column_stack([
            xyxy,
            res.boxes.conf.cpu().numpy(),
            res.boxes.cls.cpu().numpy(),
            np.full(n, w_idx, dtype=float),
        ]))
    return np.vstack(out) if out else np.empty((0, 7))


# ============================================================================
# FILTROWANIE DETEKCJI — NMS / OBS / OBM
# ============================================================================
def nms_per_class(dets: np.ndarray, iou_thr: float) -> np.ndarray:
    """Klasyczny NMS per klasa. cv2.NMSBoxes oczekuje [x, y, w, h]."""
    if len(dets) == 0:
        return dets[:, :6] if dets.shape[1] >= 6 else dets
    keep: List[np.ndarray] = []
    for cls in np.unique(dets[:, 5]):
        sub = dets[dets[:, 5] == cls]
        boxes_xywh = np.column_stack([
            sub[:, 0], sub[:, 1],
            sub[:, 2] - sub[:, 0], sub[:, 3] - sub[:, 1]
        ]).tolist()
        idxs = cv2.dnn.NMSBoxes(boxes_xywh, sub[:, 4].tolist(), 0.0, iou_thr)
        if len(idxs) > 0:
            keep.append(sub[np.array(idxs).flatten()])
    return np.vstack(keep)[:, :6] if keep else np.empty((0, 6))


def _box_at_window_edge(box, window, tol: float = 4.0) -> bool:
    """Czy bbox dotyka krawędzi okna (z marginesem tol pikseli)?"""
    bx1, by1, bx2, by2 = box
    wx, wy, ww, wh = window
    wx2, wy2 = wx + ww, wy + wh
    return (abs(bx1 - wx) < tol or abs(bx2 - wx2) < tol
            or abs(by1 - wy) < tol or abs(by2 - wy2) < tol)


def obs_filter(dets: np.ndarray, windows: List[Tuple[int, int, int, int]],
               iou_thr: float = 0.5, edge_iou: float = 0.1) -> np.ndarray:
    """Overlapping Box Suppression.

    Krok 1: dla wysoko nakładających się detekcji (IoU > iou_thr) — klasyczny NMS.
    Krok 2: dla średnio nakładających się (edge_iou < IoU ≤ iou_thr), które
            pochodzą z RÓŻNYCH okien i każda dotyka krawędzi swojego okna,
            potraktuj jako fragmenty pojedynczego obiektu i zostaw tę o większej
            powierzchni (bardziej "kompletny" fragment).
    """
    if len(dets) == 0:
        return dets[:, :6] if dets.shape[1] >= 6 else dets

    # Sortuj malejąco po score
    order = np.argsort(-dets[:, 4])
    sorted_dets = dets[order]
    keep_mask = np.ones(len(sorted_dets), dtype=bool)

    for i in range(len(sorted_dets)):
        if not keep_mask[i]:
            continue
        for j in range(i + 1, len(sorted_dets)):
            if not keep_mask[j]:
                continue
            if sorted_dets[i, 5] != sorted_dets[j, 5]:
                continue
            iou = _iou_xyxy(sorted_dets[i, :4], sorted_dets[j, :4])
            if iou > iou_thr:
                keep_mask[j] = False
            elif iou > edge_iou:
                wi = int(sorted_dets[i, 6])
                wj = int(sorted_dets[j, 6])
                if (wi != wj and 0 <= wi < len(windows) and 0 <= wj < len(windows)
                        and _box_at_window_edge(sorted_dets[i, :4], windows[wi])
                        and _box_at_window_edge(sorted_dets[j, :4], windows[wj])):
                    ai = ((sorted_dets[i, 2] - sorted_dets[i, 0])
                          * (sorted_dets[i, 3] - sorted_dets[i, 1]))
                    aj = ((sorted_dets[j, 2] - sorted_dets[j, 0])
                          * (sorted_dets[j, 3] - sorted_dets[j, 1]))
                    if ai >= aj:
                        keep_mask[j] = False
                    else:
                        keep_mask[i] = False
                        break
    return sorted_dets[keep_mask][:, :6]


def obm_filter(dets: np.ndarray, windows: List[Tuple[int, int, int, int]],
               iou_thr: float = 0.5, edge_iou: float = 0.1) -> np.ndarray:
    """Overlapping Box Merging — fragmenty są łączone w bounding box unijny."""
    if len(dets) == 0:
        return dets[:, :6] if dets.shape[1] >= 6 else dets

    n = len(dets)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i in range(n):
        for j in range(i + 1, n):
            if dets[i, 5] != dets[j, 5]:
                continue
            iou = _iou_xyxy(dets[i, :4], dets[j, :4])
            if iou > iou_thr:
                union(i, j)
            elif iou > edge_iou:
                wi = int(dets[i, 6])
                wj = int(dets[j, 6])
                if (wi != wj and 0 <= wi < len(windows) and 0 <= wj < len(windows)
                        and _box_at_window_edge(dets[i, :4], windows[wi])
                        and _box_at_window_edge(dets[j, :4], windows[wj])):
                    union(i, j)

    groups: Dict[int, List[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)

    merged: List[np.ndarray] = []
    for idxs in groups.values():
        if len(idxs) == 1:
            merged.append(dets[idxs[0], :6])
        else:
            sub = dets[idxs]
            x1, y1 = sub[:, 0].min(), sub[:, 1].min()
            x2, y2 = sub[:, 2].max(), sub[:, 3].max()
            score = sub[:, 4].max()
            cls = sub[0, 5]
            merged.append(np.array([x1, y1, x2, y2, score, cls]))
    return np.array(merged) if merged else np.empty((0, 6))


def apply_filter(method: str, dets: np.ndarray,
                 windows: List[Tuple[int, int, int, int]],
                 iou_thr: float) -> np.ndarray:
    if method == "nms":
        return nms_per_class(dets, iou_thr)
    if method == "obs":
        return obs_filter(dets, windows, iou_thr=iou_thr)
    if method == "obm":
        return obm_filter(dets, windows, iou_thr=iou_thr)
    raise ValueError(f"Nieznana metoda filtrowania: {method}")


# ============================================================================
# EWALUACJA — COCOeval z niestandardowymi przedziałami area
# ============================================================================
AREA_RANGES = [
    [0,      1e10],     # all
    [0,      8 ** 2],   # very tiny
    [8 ** 2, 16 ** 2],  # tiny
    [16 ** 2, 32 ** 2], # small
]
AREA_LABELS = ["all", "very_tiny", "tiny", "small"]


def coco_evaluate(gt_file: Path, dt_file: Path,
                  img_ids: List[int]) -> Dict[str, float]:
    coco_gt = COCO(str(gt_file))
    coco_dt = coco_gt.loadRes(str(dt_file))
    ev = COCOeval(coco_gt, coco_dt, iouType="bbox")
    ev.params.imgIds = img_ids
    ev.params.areaRng = AREA_RANGES
    ev.params.areaRngLbl = AREA_LABELS
    ev.evaluate()
    ev.accumulate()

    prec = ev.eval["precision"]   # [T, R, K, A, M]
    rec = ev.eval["recall"]       # [T, K, A, M]
    p = ev.params
    md_idx = int(np.argmax(p.maxDets))

    def ap(iou=None, area_idx=0):
        s = prec
        if iou is not None:
            t = np.where(np.isclose(p.iouThrs, iou))[0]
            s = s[t]
        s = s[..., area_idx, md_idx]
        s = s[s > -1]
        return float(np.mean(s)) if s.size else float("nan")

    def recall(area_idx=0):
        s = rec[..., area_idx, md_idx]
        s = s[s > -1]
        return float(np.mean(s)) if s.size else float("nan")

    return {
        "AP":     ap(area_idx=0),
        "AP50":   ap(iou=0.50, area_idx=0),
        "AP75":   ap(iou=0.75, area_idx=0),
        "AP_vt":  ap(area_idx=1),
        "AP_t":   ap(area_idx=2),
        "AP_s":   ap(area_idx=3),
        "AR":     recall(area_idx=0),
        "AR_vt":  recall(area_idx=1),
        "AR_t":   recall(area_idx=2),
        "AR_s":   recall(area_idx=3),
    }


# ============================================================================
# WIZUALIZACJE
# ============================================================================
def visualize_sample(image: np.ndarray, windows: List[Tuple[int, int, int, int]],
                     dets: np.ndarray, gt_anns: List[dict],
                     title: str, save_path: Path) -> None:
    """Maska ROI + ground truth (zielone) + detekcje (czerwone). Zapisuje PNG."""
    vis = image.copy()

    # 1) maska ROI (półprzezroczysta na żółto)
    if windows:
        mask = np.zeros(vis.shape[:2], dtype=np.uint8)
        for (x, y, w, h) in windows:
            cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)
        overlay = vis.copy()
        overlay[mask > 0] = (0, 255, 255)
        vis = cv2.addWeighted(overlay, 0.15, vis, 0.85, 0)
        # i ramki okien
        for (x, y, w, h) in windows:
            cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 200, 200), 1)

    # 2) ground truth — zielone
    for ann in gt_anns:
        x, y, w, h = ann["bbox"]
        cv2.rectangle(vis, (int(x), int(y)),
                      (int(x + w), int(y + h)), (0, 220, 0), 2)

    # 3) detekcje — czerwone
    for d in dets:
        x1, y1, x2, y2, score, cls = d[:6]
        cv2.rectangle(vis, (int(x1), int(y1)),
                      (int(x2), int(y2)), (0, 0, 255), 2)
        label = f"{int(cls)}:{score:.2f}"
        cv2.putText(vis, label, (int(x1), max(10, int(y1) - 3)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

    # 4) tytuł
    cv2.rectangle(vis, (0, 0), (vis.shape[1], 30), (0, 0, 0), -1)
    cv2.putText(vis, title, (10, 22), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (255, 255, 255), 2)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(save_path), vis)


# ============================================================================
# POJEDYNCZY EKSPERYMENT
# ============================================================================
def run_experiment(cfg: ExperimentConfig, out_dir: Path,
                   max_images: Optional[int] = None,
                   save_visualizations: int = 3) -> Dict:
    print(f"\n=== {cfg.name} ===")
    coco_gt = COCO(str(cfg.annotations_file))
    model = YOLO(cfg.model_path)

    img_ids = list(coco_gt.imgs.keys())
    if max_images:
        img_ids = img_ids[:max_images]

    # Tracking wymaga uporządkowania klatek wewnątrz sekwencji
    using_tracking = (cfg.approach == "focus_detect"
                      and cfg.roi_method == "tracking")
    if using_tracking:
        img_ids = order_images_by_sequence(coco_gt.imgs, img_ids)

    # Stateful estymator dla trackingu (jeden instance na cały run, reset między sekwencjami)
    tracker = TrackingROIEstimator(
        expand=cfg.track_expand, win=cfg.sliding_window_size,
        max_age=cfg.track_max_age, match_iou=cfg.track_match_iou,
    ) if using_tracking else None
    prev_sequence_key = None

    # Bootstrap dla pierwszej klatki sekwencji (gdy roi_method="tracking")
    def bootstrap_fn(img):
        return yolo_preview_rois(model, img, cfg.roi_resolution,
                                 cfg.roi_conf, cfg.roi_expand,
                                 cfg.sliding_window_size)

    predictions: List[Dict] = []
    roi_time = 0.0
    det_time = 0.0
    n_rois_per_image: List[int] = []
    viz_saved = 0

    for k, img_id in enumerate(img_ids):
        info = coco_gt.imgs[img_id]
        img_path = cfg.images_dir / info["file_name"]
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"  [WARN] brak obrazu: {img_path}")
            continue

        # Reset trackera przy zmianie sekwencji
        if using_tracking:
            seq_key = extract_sequence_key(info["file_name"])[0]
            if seq_key != prev_sequence_key:
                tracker.reset()
                prev_sequence_key = seq_key

        # --- ROI generation ---
        t0 = time.perf_counter()
        if cfg.approach == "baseline":
            rois = [(0, 0, image.shape[1], image.shape[0])]
        elif cfg.approach == "focus_detect":
            if cfg.roi_method == "yolo":
                rois = yolo_preview_rois(model, image, cfg.roi_resolution,
                                         cfg.roi_conf, cfg.roi_expand,
                                         cfg.sliding_window_size)
            elif cfg.roi_method == "saliency":
                rois = saliency_rois(image, cfg.roi_resolution,
                                     threshold=cfg.saliency_threshold,
                                     expand=cfg.roi_expand,
                                     win=cfg.sliding_window_size,
                                     method=cfg.saliency_method)
            elif cfg.roi_method == "tracking":
                rois = tracker.predict_rois(image.shape,
                                            bootstrap_fn=bootstrap_fn,
                                            image=image)
            else:
                raise ValueError(f"Nieznane roi_method: {cfg.roi_method}")
        elif cfg.approach == "sliding_window":
            rois = sliding_windows(image.shape[1], image.shape[0],
                                   cfg.sliding_window_size, cfg.sliding_stride)
        else:
            raise ValueError(cfg.approach)
        roi_time += time.perf_counter() - t0
        n_rois_per_image.append(len(rois))

        # --- detection ---
        t0 = time.perf_counter()
        if cfg.approach == "baseline":
            dets = detect_full(model, image, cfg.detector_resolution, cfg.conf_threshold)
            dets_filtered = dets[:, :6]
        else:
            dets = detect_on_crops(model, image, rois,
                                   cfg.detector_resolution, cfg.conf_threshold)
            dets_filtered = apply_filter(cfg.filter_method, dets, rois,
                                         cfg.iou_threshold)
        det_time += time.perf_counter() - t0

        # Aktualizacja trackera ZAWSZE po detekcji (na bazie wyfiltrowanych detekcji)
        if using_tracking:
            tracker.update(dets_filtered)

        # do formatu COCO
        for d in dets_filtered:
            x1, y1, x2, y2, score, cls = d
            predictions.append({
                "image_id": int(img_id),
                "category_id": int(cls) + 1,
                "bbox": [float(x1), float(y1),
                         float(x2 - x1), float(y2 - y1)],
                "score": float(score),
            })

        # wizualizacje (kilka pierwszych obrazów)
        if viz_saved < save_visualizations:
            ann_ids = coco_gt.getAnnIds(imgIds=[img_id])
            gt_anns = coco_gt.loadAnns(ann_ids)
            viz_path = out_dir / "viz" / f"{cfg.name}_img{img_id:05d}.jpg"
            visualize_sample(
                image,
                rois if cfg.approach != "baseline" else [],
                dets_filtered, gt_anns,
                title=f"{cfg.name} | dets={len(dets_filtered)} gt={len(gt_anns)} ROIs={len(rois)}",
                save_path=viz_path,
            )
            viz_saved += 1

        if (k + 1) % 25 == 0:
            print(f"  {k + 1}/{len(img_ids)} obrazów")

    total_time = roi_time + det_time
    fps = len(img_ids) / total_time if total_time > 0 else 0.0

    out_dir.mkdir(parents=True, exist_ok=True)
    pred_file = out_dir / "preds" / f"preds_{cfg.name}.json"
    pred_file.parent.mkdir(parents=True, exist_ok=True)
    pred_file.write_text(json.dumps(predictions))

    metrics = {"AP": 0.0, "AP50": 0.0, "AP75": 0.0,
               "AP_vt": 0.0, "AP_t": 0.0, "AP_s": 0.0,
               "AR": 0.0, "AR_vt": 0.0, "AR_t": 0.0, "AR_s": 0.0}
    if predictions:
        metrics = coco_evaluate(cfg.annotations_file, pred_file, img_ids)

    result = {
        "name": cfg.name,
        "dataset": cfg.dataset_name,
        "approach": cfg.approach,
        "filter": cfg.filter_method,
        "roi_method": cfg.roi_method if cfg.approach == "focus_detect" else "-",
        "model": Path(cfg.model_path).stem,
        "roi_res": f"{cfg.roi_resolution[0]}x{cfg.roi_resolution[1]}",
        "det_res": cfg.detector_resolution,
        "n_images": len(img_ids),
        "avg_rois": round(float(np.mean(n_rois_per_image)) if n_rois_per_image else 0, 2),
        "fps": round(fps, 2),
        "roi_time_s": round(roi_time, 2),
        "det_time_s": round(det_time, 2),
        **{k: round(v, 4) for k, v in metrics.items()},
    }
    print(f"  AP={result['AP']:.3f}  AP50={result['AP50']:.3f}  "
          f"AP_vt={result['AP_vt']:.3f} AP_t={result['AP_t']:.3f} AP_s={result['AP_s']:.3f}  "
          f"FPS={result['fps']:.1f}  ROIs/img={result['avg_rois']:.1f}")
    return result


# ============================================================================
# WYNIKI — CSV i wykresy
# ============================================================================
def save_results(results: List[Dict], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv(out_dir / "summary.csv", index=False)
    print(f"\nZapisano: {out_dir/'summary.csv'}")

    plt.style.use("default")

    # 1) AP vs rozdzielczość ROI (focus_detect / roi=yolo / NMS)
    fd = df[(df["approach"] == "focus_detect")
            & (df["filter"] == "nms")
            & (df["roi_method"] == "yolo")]
    if not fd.empty:
        fig, ax = plt.subplots(figsize=(8, 5))
        for (ds, mdl), sub in fd.groupby(["dataset", "model"]):
            sub = sub.sort_values("roi_res")
            ax.plot(sub["roi_res"], sub["AP"], marker="o",
                    label=f"AP — {ds}/{mdl}", linewidth=2)
            ax.plot(sub["roi_res"], sub["AP_s"], marker="s", linestyle="--",
                    label=f"AP_s — {ds}/{mdl}", linewidth=2, alpha=0.7)
        ax.set_xlabel("Rozdzielczość ROI estimatora")
        ax.set_ylabel("AP / AP_s")
        ax.set_title("Wpływ rozdzielczości ROI estimatora na precyzję detekcji")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9)
        fig.tight_layout()
        fig.savefig(out_dir / "fig1_ap_vs_roi_resolution.png", dpi=130)
        plt.close(fig)

    # 2) AP_s wg podejścia × model
    fig, ax = plt.subplots(figsize=(10, 5))
    nms_only = df[df["filter"] == "nms"]
    pivot = nms_only.pivot_table(index=["dataset", "model"],
                                 columns="approach", values="AP_s",
                                 aggfunc="max")
    pivot.plot(kind="bar", ax=ax, edgecolor="black")
    ax.set_ylabel("AP_s (małe obiekty, area 16²–32²)")
    ax.set_title("Porównanie podejść detekcji — AP_s")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(title="Podejście")
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
    fig.tight_layout()
    fig.savefig(out_dir / "fig2_ap_small_by_approach.png", dpi=130)
    plt.close(fig)

    # 3) Czas przetwarzania vs liczba ROI
    relevant = df[df["approach"].isin(["focus_detect", "sliding_window"])]
    if not relevant.empty:
        fig, ax = plt.subplots(figsize=(8, 5))
        for app, sub in relevant.groupby("approach"):
            ax.scatter(sub["avg_rois"], sub["det_time_s"], s=100,
                       label=app, alpha=0.75, edgecolor="black")
        ax.set_xlabel("Średnia liczba ROI / obraz")
        ax.set_ylabel("Łączny czas detekcji [s]")
        ax.set_title("Czas przetwarzania w zależności od liczby ROI")
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / "fig3_time_vs_rois.png", dpi=130)
        plt.close(fig)

    # 4) Porównanie filtrów NMS/OBS/OBM (dla sliding_window i focus_detect)
    fil = df[df["approach"].isin(["focus_detect", "sliding_window"])]
    if not fil.empty and fil["filter"].nunique() > 1:
        fig, ax = plt.subplots(figsize=(10, 5))
        pivot = fil.pivot_table(index=["approach", "model"],
                                columns="filter", values="AP_s",
                                aggfunc="max")
        pivot.plot(kind="bar", ax=ax, edgecolor="black")
        ax.set_ylabel("AP_s")
        ax.set_title("Porównanie metod filtrowania detekcji (NMS / OBS / OBM)")
        ax.grid(axis="y", alpha=0.3)
        ax.legend(title="Filtr")
        plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
        fig.tight_layout()
        fig.savefig(out_dir / "fig4_filtering_comparison.png", dpi=130)
        plt.close(fig)

    # 5) FPS vs AP (kompromis szybkość/dokładność)
    fig, ax = plt.subplots(figsize=(8, 6))
    markers = {"baseline": "o", "focus_detect": "s", "sliding_window": "^"}
    for app, sub in df.groupby("approach"):
        ax.scatter(sub["fps"], sub["AP"], s=120,
                   marker=markers.get(app, "o"), label=app,
                   alpha=0.75, edgecolor="black")
    ax.set_xlabel("FPS")
    ax.set_ylabel("AP")
    ax.set_title("Kompromis szybkość / dokładność")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "fig5_fps_vs_ap.png", dpi=130)
    plt.close(fig)

    # 6) Porównanie metod ROI estimation (yolo vs saliency vs tracking)
    fb = df[(df["approach"] == "focus_detect") & (df["filter"] == "nms")]
    if not fb.empty and fb["roi_method"].nunique() > 1:
        # bierzemy najlepszy wynik per (roi_method, model) — uśredniamy po rozdzielczościach
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        for ax, metric, title in [(axes[0], "AP_s", "AP_s (małe obiekty)"),
                                   (axes[1], "fps", "FPS")]:
            pivot = fb.pivot_table(index=["dataset", "model"],
                                    columns="roi_method", values=metric,
                                    aggfunc="max")
            pivot.plot(kind="bar", ax=ax, edgecolor="black")
            ax.set_ylabel(title)
            ax.set_title(f"Porównanie metod ROI — {title}")
            ax.grid(axis="y", alpha=0.3)
            ax.legend(title="roi_method")
            plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
        fig.tight_layout()
        fig.savefig(out_dir / "fig6_roi_method_comparison.png", dpi=130)
        plt.close(fig)

    print(f"Zapisano wykresy: {out_dir}/fig*.png")
    print(f"Zapisano wizualizacje: {out_dir}/viz/*.jpg")


# ============================================================================
# MACIERZ EKSPERYMENTÓW
# ============================================================================
def resolve_dataset_paths(dataset_root: Path, split: str) -> Tuple[Path, Path]:
    """Zwraca (images_dir, annotations_file). Próbuje dwóch układów:

    1. Układ "Roboflow" (typowy dla pobrań z Roboflow Universe):
           <root>/<split>/_annotations.coco.json
           <root>/<split>/*.jpg

    2. Układ "split + sidecar JSON" (jak generator syntetyczny):
           <root>/images/<split>/*.jpg
           <root>/<split>_annotations.json
    """
    # Roboflow
    rf_ann = dataset_root / split / "_annotations.coco.json"
    if rf_ann.exists():
        return dataset_root / split, rf_ann
    # sidecar JSON
    sidecar_ann = dataset_root / f"{split}_annotations.json"
    sidecar_img = dataset_root / "images" / split
    if sidecar_ann.exists():
        return sidecar_img, sidecar_ann
    raise FileNotFoundError(
        f"Nie znaleziono adnotacji w {dataset_root} dla split={split!r}. "
        f"Oczekiwano jednego z:\n"
        f"  • {rf_ann}  (układ Roboflow)\n"
        f"  • {sidecar_ann}  (układ sidecar JSON)\n"
        f"Sprawdź ścieżkę --data oraz --split."
    )


def build_experiments(dataset_root: Path, dataset_name: str,
                      models: Dict[str, str],
                      split: str = "test",
                      include_tracking: bool = True) -> List[ExperimentConfig]:
    """Pełna macierz eksperymentów.

    Dla każdego modelu:
      - Approach A (baseline)
      - Approach B (focus_detect) × roi_method ∈ {yolo, saliency, tracking}
            * yolo: 3 rozdzielczości (320×180, 640×360, 960×540) × NMS
                    + 640×360 × {OBS, OBM}
            * saliency: 3 rozdzielczości × NMS
            * tracking: 1 rozdzielczość (640×360) × NMS  (działa sensownie tylko na MOT)
      - Approach C (sliding_window) × {NMS, OBS, OBM}
    """
    img_dir, ann = resolve_dataset_paths(dataset_root, split)

    exps: List[ExperimentConfig] = []
    for model_label, model_path in models.items():
        # A — baseline
        exps.append(ExperimentConfig(
            name=f"{dataset_name}_{model_label}_baseline",
            dataset_name=dataset_name, images_dir=img_dir,
            annotations_file=ann, model_path=model_path,
            approach="baseline", filter_method="nms",
        ))

        # B — focus_detect / roi_method=yolo, 3 rozdzielczości × NMS
        for roi_res in [(320, 180), (640, 360), (960, 540)]:
            exps.append(ExperimentConfig(
                name=f"{dataset_name}_{model_label}_focus_yolo_{roi_res[0]}x{roi_res[1]}_nms",
                dataset_name=dataset_name, images_dir=img_dir,
                annotations_file=ann, model_path=model_path,
                approach="focus_detect", filter_method="nms",
                roi_method="yolo", roi_resolution=roi_res,
            ))

        # B — focus_detect / roi_method=yolo, 640×360, OBS i OBM
        for filt in ["obs", "obm"]:
            exps.append(ExperimentConfig(
                name=f"{dataset_name}_{model_label}_focus_yolo_640x360_{filt}",
                dataset_name=dataset_name, images_dir=img_dir,
                annotations_file=ann, model_path=model_path,
                approach="focus_detect", filter_method=filt,
                roi_method="yolo", roi_resolution=(640, 360),
            ))

        # B — focus_detect / roi_method=saliency, 3 rozdzielczości × NMS
        for roi_res in [(320, 180), (640, 360), (960, 540)]:
            exps.append(ExperimentConfig(
                name=f"{dataset_name}_{model_label}_focus_saliency_{roi_res[0]}x{roi_res[1]}_nms",
                dataset_name=dataset_name, images_dir=img_dir,
                annotations_file=ann, model_path=model_path,
                approach="focus_detect", filter_method="nms",
                roi_method="saliency", roi_resolution=roi_res,
                saliency_method="fine", saliency_threshold=0.35,
            ))

        # B — focus_detect / roi_method=tracking (tylko jeśli mamy sekwencje)
        if include_tracking:
            exps.append(ExperimentConfig(
                name=f"{dataset_name}_{model_label}_focus_tracking_640x360_nms",
                dataset_name=dataset_name, images_dir=img_dir,
                annotations_file=ann, model_path=model_path,
                approach="focus_detect", filter_method="nms",
                roi_method="tracking", roi_resolution=(640, 360),
            ))

        # C — sliding_window × {NMS, OBS, OBM}
        for filt in ["nms", "obs", "obm"]:
            exps.append(ExperimentConfig(
                name=f"{dataset_name}_{model_label}_sliding_{filt}",
                dataset_name=dataset_name, images_dir=img_dir,
                annotations_file=ann, model_path=model_path,
                approach="sliding_window", filter_method=filt,
            ))
    return exps


# ============================================================================
# MAIN
# ============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True,
                    help="katalog datasetu (układ Roboflow: <root>/<split>/_annotations.coco.json, "
                         "albo sidecar: <root>/<split>_annotations.json + images/<split>/)")
    ap.add_argument("--name", default="dataset", help="nazwa datasetu (do wyników)")
    ap.add_argument("--split", default="test", help="train/val/test")
    ap.add_argument("--model", default="yolov8n.pt",
                    help="ścieżka wag (tiny/baseline)")
    ap.add_argument("--model-advanced", default=None,
                    help="ścieżka wag zaawansowanej architektury (opcjonalnie)")
    ap.add_argument("--out", default="results")
    ap.add_argument("--max-images", type=int, default=None,
                    help="ogranicznik liczby obrazów (do szybkich testów)")
    ap.add_argument("--quick", action="store_true",
                    help="skrócony run: baseline + focus 640x360 (yolo/saliency/tracking) + sliding, wszystko z NMS")
    ap.add_argument("--no-tracking", action="store_true",
                    help="pomiń eksperymenty z roi_method=tracking (gdy dataset nie ma sekwencji)")
    args = ap.parse_args()

    models: Dict[str, str] = {"tiny": args.model}
    if args.model_advanced:
        models["advanced"] = args.model_advanced

    experiments = build_experiments(Path(args.data), args.name, models,
                                    args.split,
                                    include_tracking=not args.no_tracking)

    if args.quick:
        experiments = [e for e in experiments
                       if e.filter_method == "nms"
                       and (e.approach == "baseline"
                            or (e.approach == "focus_detect"
                                and e.roi_resolution == (640, 360))
                            or e.approach == "sliding_window")]

    out_dir = Path(args.out)
    results: List[Dict] = []
    for cfg in experiments:
        try:
            r = run_experiment(cfg, out_dir, max_images=args.max_images)
            results.append(r)
        except Exception as e:
            print(f"[ERROR] {cfg.name}: {e}")
            import traceback
            traceback.print_exc()

    if results:
        save_results(results, out_dir)
    else:
        print("Brak wyników.")


if __name__ == "__main__":
    main()