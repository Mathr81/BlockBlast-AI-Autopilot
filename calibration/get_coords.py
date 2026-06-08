import cv2
import os
import ctypes

# -----------------------------
# Récupération taille écran (Windows)
# -----------------------------
user32 = ctypes.windll.user32
screen_w = user32.GetSystemMetrics(0)
screen_h = user32.GetSystemMetrics(1)

MAX_W = int(screen_w * 0.9)
MAX_H = int(screen_h * 0.9)

# Variables globales
scale = 1.0
img = None
img_display = None

def click_event(event, x, y, flags, params):
    """S'exécute à chaque clic gauche de la souris sur l'image affichée."""
    global img, scale, img_display

    if event == cv2.EVENT_LBUTTONDOWN:
        # conversion coordonnées écran -> image originale
        x_orig = int(x / scale)
        y_orig = int(y / scale)

        print(f"[Clic détecté] -> X = {x_orig} | Y = {y_orig}")
        print(f"Color at ({x_orig}, {y_orig}): {img[y_orig, x_orig]}")

        # dessin sur image affichée (pas l'originale)
        cv2.circle(img_display, (x, y), 4, (0, 0, 255), -1)
        cv2.putText(img_display, f"({x_orig},{y_orig})", (x + 10, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

        cv2.imshow("Obtenir Coordonnees", img_display)


screenshot_path = "screenshot.png"

if not os.path.exists(screenshot_path):
    print(f"[-] Erreur : image '{screenshot_path}' introuvable.")
else:
    img = cv2.imread(screenshot_path)
    h, w = img.shape[:2]

    # -----------------------------
    # Calcul du scale pour fit écran
    # -----------------------------
    scale = min(MAX_W / w, MAX_H / h, 1.0)

    new_w = int(w * scale)
    new_h = int(h * scale)

    img_display = cv2.resize(img, (new_w, new_h))

    print("[*] Fenêtre adaptée à l'écran.")
    print("[*] Clique pour obtenir les coordonnées (origine image).")
    print("[*] Appuie sur une touche pour quitter.")

    cv2.namedWindow("Obtenir Coordonnees", cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback("Obtenir Coordonnees", click_event)

    cv2.imshow("Obtenir Coordonnees", img_display)
    cv2.waitKey(0)
    cv2.destroyAllWindows()