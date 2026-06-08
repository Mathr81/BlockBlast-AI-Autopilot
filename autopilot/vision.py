"""
vision.py — Extraction OpenCV : grille 8x8 et pièces des 3 slots.
Aucune dépendance aux autres modules autopilot.
"""
import cv2
import numpy as np
from autopilot.config import (
    GRID_BRIGHTNESS_THRESHOLD, COLOR_DIFF_THRESHOLD, MINI_BLOCK_SIZE,
    GRID_X, GRID_Y, GRID_W, GRID_H,
    SLOT_W, SLOT_H, SLOT_Y, SLOT_X_POSITIONS,
)

# ─── Formes valides (pour filtrer les artefacts de détection) ─────────
VALID_SHAPES_TUPLES = {
    ((1,),),
    ((1,1),),((1,),(1,)),
    ((1,1,1),),((1,),(1,),(1,)),
    ((1,1,1,1),),((1,),(1,),(1,),(1,)),
    ((1,1,1,1,1),),((1,),(1,),(1,),(1,),(1,)),
    ((1,1),(1,1)),
    ((1,1,1),(1,1,1),(1,1,1)),
    ((1,1),(1,0)),((1,1),(0,1)),((1,0),(1,1)),((0,1),(1,1)),
    ((1,1,1),(1,0,0),(1,0,0)),((1,1,1),(0,0,1),(0,0,1)),
    ((1,0,0),(1,0,0),(1,1,1)),((0,0,1),(0,0,1),(1,1,1)),
    ((1,0),(1,0),(1,1)),((0,1),(0,1),(1,1)),
    ((1,1),(1,0),(1,0)),((1,1),(0,1),(0,1)),
    ((1,1,1),(1,0,0)),((1,1,1),(0,0,1)),
    ((1,0,0),(1,1,1)),((0,0,1),(1,1,1)),
    ((1,1,1),(0,1,0)),((0,1,0),(1,1,1)),
    ((1,0),(1,1),(1,0)),((0,1),(1,1),(0,1)),
    ((0,1,1),(1,1,0)),((1,1,0),(0,1,1)),
    ((1,0),(1,1),(0,1)),((0,1),(1,1),(1,0)),
    ((1,1,1),(1,1,1)),((1,1),(1,1),(1,1)),
    ((0,1,0),(1,1,1),(0,1,0)),
    ((1,0,1),(1,1,1)),((1,1,1),(1,0,1)),
    ((1,1),(1,0),(1,1)),((1,1),(0,1),(1,1)),
    ((1,0),(0,1)),((0,1),(1,0)),
    ((1,0,0),(0,1,0),(0,0,1)),((0,0,1),(0,1,0),(1,0,0)),
}


def get_trimmed_shape_form(shape_5x5):
    """Retourne la sous-matrice minimale active d'un bloc 5x5."""
    if not np.any(shape_5x5):
        return None
    ar = np.any(shape_5x5, axis=1)
    ac = np.any(shape_5x5, axis=0)
    return shape_5x5[:np.where(ar)[0][-1]+1, :np.where(ac)[0][-1]+1].tolist()


def is_valid_block_blast_shape(form_list):
    """Vérifie qu'une forme fait partie du répertoire Block Blast."""
    return bool(form_list) and tuple(tuple(r) for r in form_list) in VALID_SHAPES_TUPLES


def extract_grid(grid_img, threshold=GRID_BRIGHTNESS_THRESHOLD):
    """Analyse de la grille 8x8 par luminosité."""
    h, w, _ = grid_img.shape
    cw, ch = w / 8, h / 8
    gm = np.zeros((8, 8), dtype=np.int8)
    gray = cv2.cvtColor(grid_img, cv2.COLOR_BGR2GRAY)
    for r in range(8):
        for c in range(8):
            x1, y1 = int(c*cw),      int(r*ch)
            x2, y2 = int((c+1)*cw),  int((r+1)*ch)
            mx, my = int((x2-x1)*0.35), int((y2-y1)*0.35)
            patch = gray[y1+my:y2-my, x1+mx:x2-mx]
            if patch.size > 0 and np.mean(patch) > threshold:
                gm[r, c] = 1
    return gm


def extract_shape_from_slot(slot_img,
                             mini_block_size=MINI_BLOCK_SIZE,
                             threshold=COLOR_DIFF_THRESHOLD):
    """Analyse dynamique d'un slot par distance de couleur au fond."""
    margin = 15
    hs, ws, _ = slot_img.shape
    if hs <= margin*2 or ws <= margin*2:
        return np.zeros((5, 5), dtype=np.int8), None
    inner = slot_img[margin:hs-margin, margin:ws-margin]
    bg = np.mean(inner[3:8, 3:8], axis=(0, 1))
    dist = np.sqrt(np.sum((inner.astype(np.float32) - bg)**2, axis=2))
    th = (dist > threshold).astype(np.uint8) * 255
    pts = cv2.findNonZero(th)
    if pts is None:
        return np.zeros((5, 5), dtype=np.int8), None
    x, y, w, h = cv2.boundingRect(pts)
    if w < 6 or h < 6:
        return np.zeros((5, 5), dtype=np.int8), None
    vc = (margin + x + w/2, margin + y + h/2)
    cols = max(1, min(5, int(round(w / mini_block_size))))
    rows = max(1, min(5, int(round(h / mini_block_size))))
    sm = np.zeros((rows, cols), dtype=np.int8)
    cw_, ch_ = w / cols, h / rows
    for r in range(rows):
        for c in range(cols):
            p = th[int(y+r*ch_):int(y+(r+1)*ch_), int(x+c*cw_):int(x+(c+1)*cw_)]
            if p.size > 0 and np.mean(p) > 120:
                sm[r, c] = 1
    f = np.zeros((5, 5), dtype=np.int8)
    f[:rows, :cols] = sm
    return f, vc


def capture_and_extract(screenshot):
    """
    Extrait grille + pièces depuis un screenshot déjà redimensionné (1084x2412).
    Retourne (grid_np, shapes_list, centers_list).
    """
    grid_crop = screenshot[GRID_Y:GRID_Y+GRID_H, GRID_X:GRID_X+GRID_W]
    grid = extract_grid(grid_crop)
    shapes, centers = [], []
    for xp in SLOT_X_POSITIONS:
        m, ctr = extract_shape_from_slot(screenshot[SLOT_Y:SLOT_Y+SLOT_H, xp:xp+SLOT_W])
        shapes.append(m)
        centers.append(ctr)
    return grid, shapes, centers


def shapes_are_valid(shapes):
    """Retourne True si toutes les formes actives sont valides et qu'au moins une existe."""
    if not any(np.any(s) for s in shapes):
        return False
    return all(
        not np.any(s) or is_valid_block_blast_shape(get_trimmed_shape_form(s))
        for s in shapes
    )