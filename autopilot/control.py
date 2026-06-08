"""
control.py — Contrôle scrcpy : tap, drag fluide, coordonnées grille.
Aucune dépendance aux modules vision/planner.
"""
import math
import time
import numpy as np
from autopilot.config import (
    DRAG_GAIN_X, DRAG_GAIN_Y, BASE_VERTICAL_OFFSET,
    GRID_X, GRID_Y, GRID_W, GRID_H,
    SLOT_W, SLOT_H, SLOT_Y, SLOT_X_POSITIONS,
    REVIVE_PX1_X, REVIVE_PX1_Y, REVIVE_PX2_X, REVIVE_PX2_Y,
    REVIVE_GREEN_BGR, REVIVE_COLOR_THRESHOLD,
    REVIVE_CLICK_X, REVIVE_CLICK_Y,
    SETTINGS_BTN_X, SETTINGS_BTN_Y,
    DEFAULT_SKIN_BTN_X, DEFAULT_SKIN_BTN_Y,
    CLOSE_SETTINGS_X, CLOSE_SETTINGS_Y,
    SKIN_CHECK_X, SKIN_CHECK_Y, EXPECTED_BG_BGR, SKIN_COLOR_THRESHOLD,
)

_SCREEN_W = 1084
_SCREEN_H = 2412


def _scale(client, x, y):
    sw, sh = client.resolution
    return int(round(x * sw / _SCREEN_W)), int(round(y * sh / _SCREEN_H))


def send_tap(client, x, y):
    """Tap simple (ACTION_DOWN + ACTION_UP)."""
    import scrcpy
    sx, sy = _scale(client, x, y)
    client.control.touch(sx, sy, scrcpy.ACTION_DOWN)
    time.sleep(0.08)
    client.control.touch(sx, sy, scrcpy.ACTION_UP)


def send_drag(client, x1, y1, x2, y2):
    """
    Glissement fluide avec trajectoire de Bézier cubique + légère courbure aléatoire.
    Simule un geste humain à ~120 Hz.
    """
    import scrcpy
    sx1, sy1 = _scale(client, x1, y1)
    sx2, sy2 = _scale(client, x2, y2)
    dx, dy = sx2 - sx1, sy2 - sy1
    dist = math.hypot(dx, dy)
    if dist > 0:
        px, py = -dy / dist, dx / dist
        arc = dist * np.random.uniform(0.02, 0.05) * (1 if np.random.rand() > 0.5 else -1)
    else:
        px = py = arc = 0

    client.control.touch(sx1, sy1, scrcpy.ACTION_DOWN)
    time.sleep(0.08)
    t0, dur = time.perf_counter(), 0.28
    while True:
        t = (time.perf_counter() - t0) / dur
        if t >= 1.0:
            break
        r = 4*t**3 if t < 0.5 else 1 - (-2*t + 2)**3 / 2
        a = arc * math.sin(t * math.pi)
        client.control.touch(
            int(round(sx1 + dx*r + px*a + np.random.uniform(-0.5, 0.5))),
            int(round(sy1 + dy*r + py*a + np.random.uniform(-0.5, 0.5))),
            scrcpy.ACTION_MOVE)
        time.sleep(0.006)
    client.control.touch(sx2, sy2, scrcpy.ACTION_MOVE)
    time.sleep(0.18)
    client.control.touch(sx2, sy2, scrcpy.ACTION_UP)


def get_grid_screen_coords(row, col, shape_rows, shape_cols):
    """Coordonnées écran (centre de la pièce posée) avec offset vertical tactile."""
    cw, ch = GRID_W / 8, GRID_H / 8
    tx = GRID_X + (col + (shape_cols - 1) / 2.0) * cw + cw / 2
    ty = GRID_Y + (row + (shape_rows - 1) / 2.0) * ch + ch / 2 + BASE_VERTICAL_OFFSET
    return int(tx), int(ty)


def execute_move(client, shape_idx, row, col, form, shape_centers):
    """
    Exécute un coup : drag depuis le centre visuel du slot vers la grille.
    Applique la compensation du gain de glissement tactile.
    """
    sr, sc_ = len(form), len(form[0])
    if shape_centers[shape_idx] is not None:
        vx, vy = shape_centers[shape_idx]
        sx = SLOT_X_POSITIONS[shape_idx] + int(vx)
        sy = SLOT_Y + int(vy)
    else:
        sx = SLOT_X_POSITIONS[shape_idx] + SLOT_W // 2
        sy = SLOT_Y + SLOT_H // 2

    ex, ey = get_grid_screen_coords(row, col, sr, sc_)
    # Compensation du gain tactile : le doigt va plus loin que la cible visuelle
    cx = sx + (ex - sx) / DRAG_GAIN_X
    cy = sy + (ey - sy) / DRAG_GAIN_Y
    send_drag(client, sx, sy, cx, cy)


# ─────────────────────────────────────────────────────────────────────
# DÉTECTION REVIVE & SKIN
# ─────────────────────────────────────────────────────────────────────
def check_revive(screenshot):
    """Retourne True si le bouton Revive vert est visible."""
    d1 = np.linalg.norm(
        screenshot[REVIVE_PX1_Y, REVIVE_PX1_X].astype(np.float32)
        - np.array(REVIVE_GREEN_BGR, np.float32))
    d2 = np.linalg.norm(
        screenshot[REVIVE_PX2_Y, REVIVE_PX2_X].astype(np.float32)
        - np.array(REVIVE_GREEN_BGR, np.float32))
    return d1 < REVIVE_COLOR_THRESHOLD and d2 < REVIVE_COLOR_THRESHOLD


def handle_revive(client):
    """Clique sur le bouton Revive."""
    send_tap(client, REVIVE_CLICK_X, REVIVE_CLICK_Y)


def check_skin(screenshot):
    """Retourne True si un skin alternatif est détecté (fond inattendu)."""
    px = screenshot[SKIN_CHECK_Y, SKIN_CHECK_X]
    return np.linalg.norm(
        px.astype(np.float32) - np.array(EXPECTED_BG_BGR, np.float32)
    ) > SKIN_COLOR_THRESHOLD


def reset_skin(client):
    """Navigue dans les paramètres pour revenir au skin par défaut."""
    send_tap(client, SETTINGS_BTN_X, SETTINGS_BTN_Y);    time.sleep(0.9)
    send_tap(client, DEFAULT_SKIN_BTN_X, DEFAULT_SKIN_BTN_Y); time.sleep(0.7)
    send_tap(client, CLOSE_SETTINGS_X, CLOSE_SETTINGS_Y); time.sleep(1.2)