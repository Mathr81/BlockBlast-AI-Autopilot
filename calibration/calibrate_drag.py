import os
import sys
import time
import subprocess
import cv2
import numpy as np
import re

# =====================================================================
# CONFIGURATION DE BASE DE L'ÉCRAN (Issue de run_autopilot.py)
# =====================================================================
GRID_X = 63
GRID_Y = 584
GRID_W = 956
GRID_H = 956

SLOT_W = 264
SLOT_H = 264
SLOT_Y = 1690
SLOT_X_POSITIONS = [95, 410, 725]

MINI_BLOCK_SIZE = 50  
COLOR_DIFF_THRESHOLD = 40  

def get_android_logical_size():
    """Interroge le gestionnaire de fenêtres d'Android pour récupérer la résolution tactile."""
    try:
        result = subprocess.run("adb shell wm size", shell=True, capture_output=True, text=True)
        sizes = re.findall(r"(\d+)x(\d+)", result.stdout)
        if sizes:
            w, h = map(int, sizes[-1])
            return w, h
    except Exception as e:
        print(f"[-] Erreur lors de la détection de la taille d'affichage ADB : {e}")
    return None, None

def capture_screen_to_ram():
    """Prend une capture d'écran du téléphone directement en RAM."""
    cmd = ["adb", "exec-out", "screencap", "-p"]
    try:
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, _ = process.communicate()
        if not stdout:
            return None
        image_array = np.frombuffer(stdout, dtype=np.uint8)
        return cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    except Exception as e:
        print(f"[-] Erreur lors de la capture d'écran ADB : {e}")
        return None

def extract_shape_from_slot(slot_img, mini_block_size=MINI_BLOCK_SIZE, threshold=COLOR_DIFF_THRESHOLD):
    """Analyse dynamique par distance de couleur pour identifier les blocs."""
    margin = 15
    h_slot, w_slot, _ = slot_img.shape
    if h_slot <= margin * 2 or w_slot <= margin * 2:
        return np.zeros((5, 5), dtype=np.int8), None
        
    inner_slot = slot_img[margin : h_slot - margin, margin : w_slot - margin]
    
    bg_patch = inner_slot[3:8, 3:8]
    bg_color = np.mean(bg_patch, axis=(0, 1))
    
    diff = np.abs(inner_slot.astype(np.float32) - bg_color)
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    
    thresh = np.zeros(dist.shape, dtype=np.uint8)
    thresh[dist > threshold] = 255
    
    pts = cv2.findNonZero(thresh)
    if pts is None:
        return np.zeros((5, 5), dtype=np.int8), None
        
    x, y, w, h = cv2.boundingRect(pts)
    if w < 6 or h < 6:
        return np.zeros((5, 5), dtype=np.int8), None
        
    visual_center = (margin + x + (w / 2), margin + y + (h / 2))
        
    cols = int(round(w / mini_block_size))
    rows = int(round(h / mini_block_size))
    
    cols = max(1, min(5, cols))
    rows = max(1, min(5, rows))
    
    shape_matrix = np.zeros((rows, cols), dtype=np.int8)
    cell_w = w / cols
    cell_h = h / rows
    
    for r in range(rows):
        for c in range(cols):
            cx1 = int(x + c * cell_w)
            cy1 = int(y + r * cell_h)
            cx2 = int(x + (c + 1) * cell_w)
            cy2 = int(y + (r + 1) * cell_h)
            
            cell_patch = thresh[cy1:cy2, cx1:cx2]
            if cell_patch.size > 0 and np.mean(cell_patch) > 120:
                shape_matrix[r, c] = 1
                
    final_5x5 = np.zeros((5, 5), dtype=np.int8)
    final_5x5[:rows, :cols] = shape_matrix
    
    return final_5x5, visual_center

def get_compensated_coords(row, col, shape_rows, shape_cols, start_x, start_y, gain_x, gain_y, offset_y):
    """Calcule les coordonnées physiques du doigt compensées par le gain et l'offset."""
    cell_w = GRID_W / 8
    cell_h = GRID_H / 8
    
    offset_r = (shape_rows - 1) / 2.0
    offset_c = (shape_cols - 1) / 2.0
    
    target_r = row + offset_r
    target_c = col + offset_c
    
    # Coordonnées théoriques cibles sur la grille (avec décalage offset_y)
    end_x = GRID_X + (target_c * cell_w) + (cell_w / 2)
    end_y = GRID_Y + (target_r * cell_h) + (cell_h / 2) + offset_y
    
    # Application de la formule de compensation
    compensated_x = start_x + (end_x - start_x) / gain_x
    compensated_y = start_y + (end_y - start_y) / gain_y
    
    return int(round(compensated_x)), int(round(compensated_y))

def execute_adb_motion(event_type, x, y, scale_x, scale_y):
    """Envoie un événement tactile individuel via adb input motionevent."""
    adb_x = int(round(x * scale_x))
    adb_y = int(round(y * scale_y))
    cmd = f"adb shell input motionevent {event_type} {adb_x} {adb_y}"
    subprocess.run(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def main():
    print("=" * 60)
    print("      TESTEUR INTERACTIF DE SÉQUENCE DE CALIBRATION (DRAG)")
    print("=" * 60)
    print("[*] Connexion ADB en cours...")
    
    result = subprocess.run("adb devices", shell=True, capture_output=True, text=True)
    if "device" not in result.stdout.split('\n')[1]:
        print("[!] Erreur : Aucun appareil Android détecté.")
        return
        
    logical_w, logical_h = get_android_logical_size()
    if logical_w is None:
        print("[!] Impossible de récupérer wm size. Utilisation de la résolution par défaut 1084x2412.")
        logical_w, logical_h = 1084, 2412

    print("[*] Analyse de l'écran pour trouver une pièce...")
    screenshot = capture_screen_to_ram()
    if screenshot is None:
        print("[!] Erreur : Impossible de prendre une capture d'écran.")
        return

    h_img, w_img, _ = screenshot.shape
    scale_x = logical_w / w_img
    scale_y = logical_h / h_img

    # Recherche de la première pièce active
    active_idx = -1
    detected_shape = None
    visual_center = None

    for idx, x_pos in enumerate(SLOT_X_POSITIONS):
        slot_crop = screenshot[SLOT_Y : SLOT_Y + SLOT_H, x_pos : x_pos + SLOT_W]
        shape_matrix, center = extract_shape_from_slot(slot_crop)
        if np.any(shape_matrix):
            active_idx = idx
            detected_shape = shape_matrix
            visual_center = center
            break

    if active_idx == -1:
        print("[!] Aucune pièce active détectée dans les emplacements du bas.")
        print("[*] Veuillez lancer une partie et vous assurer qu'au moins un bloc est présent.")
        return

    # Extraction de la taille réelle du bloc identifié
    active_rows = np.any(detected_shape, axis=1)
    active_cols = np.any(detected_shape, axis=0)
    max_row = np.where(active_rows)[0][-1] + 1
    max_col = np.where(active_cols)[0][-1] + 1
    shape_rows = max_row
    shape_cols = max_col

    print(f"[+] Pièce détectée dans le Slot {active_idx + 1} (Taille : {shape_rows}x{shape_cols})")
    
    # Calcul de l'ancrage de départ
    start_x = SLOT_X_POSITIONS[active_idx] + int(visual_center[0])
    start_y = SLOT_Y + int(visual_center[1])

    # Valeurs initiales à tester
    gain_x = 1.4
    gain_y = 1.4
    offset_y = 270

    while True:
        print("\n" + "-"*50)
        print("  CONFIGURATION DU TEST DE SÉQUENCE :")
        print(f"  [1] DRAG_GAIN_X      : {gain_x}")
        print(f"  [2] DRAG_GAIN_Y      : {gain_y}")
        print(f"  [3] VERTICAL_OFFSET  : {offset_y}")
        print("-"*50)
        
        user_input = input("Appuyez sur ENTRER pour lancer le test de mouvement, ou saisissez de nouvelles valeurs (ex: 1.4 1.4 250) : ").strip()
        
        if user_input:
            try:
                parts = user_input.split()
                if len(parts) == 3:
                    gain_x = float(parts[0])
                    gain_y = float(parts[1])
                    offset_y = int(parts[2])
                    print(f"[+] Valeurs mises à jour : Gain X = {gain_x}, Gain Y = {gain_y}, Offset = {offset_y}")
                else:
                    print("[!] Format invalide. Utilisation des valeurs précédentes.")
            except ValueError:
                print("[!] Saisie non valide. Utilisation des valeurs précédentes.")

        # Calcul des 4 coins cibles compensés
        corners = [
            ("Haut-Gauche (0, 0)", 0, 0),
            ("Haut-Droite (0, Max)", 0, 8 - shape_cols),
            ("Bas-Droite (Max, Max)", 8 - shape_rows, 8 - shape_cols),
            ("Bas-Gauche (Max, 0)", 8 - shape_rows, 0)
        ]

        print("\n[*] Lancement de la simulation physique sur le téléphone...")
        print("[!] Regardez l'écran de votre téléphone pour évaluer l'alignement.")
        
        try:
            # 1. DOWN : On clique sur le centre du bloc
            execute_adb_motion("DOWN", start_x, start_y, scale_x, scale_y)
            time.sleep(0.6)  # Attente de prise en compte par le jeu

            # 2. MOVE à chaque angle
            for name, r, c in corners:
                comp_x, comp_y = get_compensated_coords(r, c, shape_rows, shape_cols, start_x, start_y, gain_x, gain_y, offset_y)
                print(f"  -> Déplacement vers {name}...")
                execute_adb_motion("MOVE", comp_x, comp_y, scale_x, scale_y)
                time.sleep(1.8)  # Temps d'observation par l'utilisateur à chaque angle

            # 3. Retour au point de départ pour éviter de jouer la pièce accidentellement
            print("  -> Retour au slot d'origine (Sécurité anti-pose)...")
            execute_adb_motion("MOVE", start_x, start_y, scale_x, scale_y)
            time.sleep(0.8)

            # 4. Relâchement
            execute_adb_motion("UP", start_x, start_y, scale_x, scale_y)
            print("[+] Séquence terminée. La pièce a été reposée sans être validée.")

        except Exception as e:
            # En cas de plantage intermédiaire, on tente d'envoyer un UP par sécurité
            execute_adb_motion("UP", start_x, start_y, scale_x, scale_y)
            print(f"[-] Une erreur s'est produite lors de la transmission tactile : {e}")

        # Choix de poursuivre ou de s'arrêter
        choice = input("\nVoulez-vous modifier les valeurs et relancer un test ? (o/n) : ").strip().lower()
        if choice not in ['o', 'oui', 'y', 'yes']:
            print("\n[*] Fin de la session de calibration.")
            print(f"Valeurs finales retenues : DRAG_GAIN_X = {gain_x} | DRAG_GAIN_Y = {gain_y} | BASE_VERTICAL_OFFSET = {offset_y}")
            break

if __name__ == "__main__":
    main()