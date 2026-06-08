import os
import sys
import cv2
import numpy as np

# =====================================================================
# VARIABLES DE CALIBRATION (À ajuster selon la résolution de votre Scrcpy/Téléphone)
# =====================================================================
# Coordonnées (X, Y, Largeur, Hauteur) de la grille 8x8 sur votre capture d'écran
GRID_X = 63
GRID_Y = 584
GRID_W = 956
GRID_H = 956

# Coordonnées et taille des 3 emplacements de pièces en bas
SLOT_W = 264
SLOT_H = 264
SLOT_Y = 1690
SLOT_X_POSITIONS = [95, 410, 725] # X de départ pour Slot 1, Slot 2, Slot 3

# Taille moyenne en pixels d'un seul mini-bloc de forme dans l'emplacement de prévisualisation
MINI_BLOCK_SIZE = 50  

# Seuils de luminosité (0-255)
GRID_BRIGHTNESS_THRESHOLD = 70  # Si la luminosité moyenne centrale de la case dépasse ce seuil, elle est pleine
SHAPE_BRIGHTNESS_THRESHOLD = 90 # Seuil pour binariser la forme en bas
# =====================================================================


def extract_grid(grid_img, threshold=GRID_BRIGHTNESS_THRESHOLD):
    """Analyse la grille 8x8 et extrait la matrice binaire."""
    h, w, _ = grid_img.shape
    cell_w = w / 8
    cell_h = h / 8
    
    grid_matrix = np.zeros((8, 8), dtype=np.int8)
    gray = cv2.cvtColor(grid_img, cv2.COLOR_BGR2GRAY)
    
    for r in range(8):
        for c in range(8):
            # Coordonnées de la cellule
            x1 = int(c * cell_w)
            y1 = int(r * cell_h)
            x2 = int((c + 1) * cell_w)
            y2 = int((r + 1) * cell_h)
            
            # Échantillonner uniquement le CENTRE de la cellule (les 30% centraux)
            # Cela permet d'éviter les bordures de case, les reflets et les lignes de grille
            margin_x = int((x2 - x1) * 0.35)
            margin_y = int((y2 - y1) * 0.35)
            
            patch = gray[y1 + margin_y : y2 - margin_y, x1 + margin_x : x2 - margin_x]
            
            if patch.size > 0 and np.mean(patch) > threshold:
                grid_matrix[r, c] = 1
                
    return grid_matrix


def extract_shape_from_slot(slot_img, mini_block_size=MINI_BLOCK_SIZE, threshold=SHAPE_BRIGHTNESS_THRESHOLD):
    """Analyse un emplacement de pièce et reconstruit sa matrice 5x5 de manière dynamique."""
    gray = cv2.cvtColor(slot_img, cv2.COLOR_BGR2GRAY)
    
    # Seuillage pour séparer le fond noir des blocs de couleur clairs
    _, thresh = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    
    # Trouver les coordonnées de tous les pixels clairs
    pts = cv2.findNonZero(thresh)
    if pts is None:
        # Aucun pixel clair détecté : l'emplacement est vide
        return np.zeros((5, 5), dtype=np.int8)
        
    # Extraire la boîte englobante entourant la forme
    x, y, w, h = cv2.boundingRect(pts)
    
    # Si la boîte est microscopique, c'est du bruit numérique
    if w < 6 or h < 6:
        return np.zeros((5, 5), dtype=np.int8)
        
    # Calculer le nombre théorique de lignes et colonnes de blocs
    cols = int(round(w / mini_block_size))
    rows = int(round(h / mini_block_size))
    
    # S'assurer de rester dans les limites admissibles (1 à 5)
    cols = max(1, min(5, cols))
    rows = max(1, min(5, rows))
    
    shape_matrix = np.zeros((rows, cols), dtype=np.int8)
    
    cell_w = w / cols
    cell_h = h / rows
    
    # Échantillonner la grille de la boîte englobante
    for r in range(rows):
        for c in range(cols):
            cx1 = int(x + c * cell_w)
            cy1 = int(y + r * cell_h)
            cx2 = int(x + (c + 1) * cell_w)
            cy2 = int(y + (r + 1) * cell_h)
            
            cell_patch = thresh[cy1:cy2, cx1:cx2]
            if cell_patch.size > 0 and np.mean(cell_patch) > 120:
                shape_matrix[r, c] = 1
                
    # Centrer/Pad au format 5x5 attendu par l'IA (aligné en haut à gauche)
    final_5x5 = np.zeros((5, 5), dtype=np.int8)
    final_5x5[:rows, :cols] = shape_matrix
    return final_5x5


def main():
    screenshot_path = "screenshot.png"
    
    if not os.path.exists(screenshot_path):
        print(f"[-] Erreur : Enregistrez une capture d'écran nommée '{screenshot_path}' à la racine du projet.")
        print("[*] Astuce : Utilisez Scrcpy, faites une capture, renommez-la, et relancez ce script.")
        return
        
    # Charger l'image originale
    img = cv2.imread(screenshot_path)
    h_img, w_img, _ = img.shape
    print(f"[+] Image chargée : {w_img}x{h_img} pixels")
    
    # Faire une copie pour dessiner les repères de calibration de couleur rouge
    annotated_img = img.copy()
    
    # 1. Dessiner le cadre de la grille de jeu
    cv2.rectangle(annotated_img, (GRID_X, GRID_Y), (GRID_X + GRID_W, GRID_Y + GRID_H), (0, 0, 255), 3)
    
    # 2. Dessiner les cadres des 3 emplacements de formes en bas
    for idx, x_pos in enumerate(SLOT_X_POSITIONS):
        cv2.rectangle(annotated_img, (x_pos, SLOT_Y), (x_pos + SLOT_W, SLOT_Y + SLOT_H), (255, 0, 0), 2)
        cv2.putText(annotated_img, f"Slot {idx+1}", (x_pos, SLOT_Y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
    # --- PROCESSUS D'ANALYSE ---
    # Découper la grille et analyser
    grid_crop = img[GRID_Y : GRID_Y + GRID_H, GRID_X : GRID_X + GRID_W]
    detected_grid = extract_grid(grid_crop)
    
    # Découper et analyser les 3 formes
    detected_shapes = []
    for x_pos in SLOT_X_POSITIONS:
        slot_crop = img[SLOT_Y : SLOT_Y + SLOT_H, x_pos : x_pos + SLOT_W]
        shape_matrix = extract_shape_from_slot(slot_crop)
        detected_shapes.append(shape_matrix)
        
    # --- AFFICHAGE DES RÉSULTATS DANS LA CONSOLE ---
    print("\n" + "=" * 30)
    print("   GRILLE DETECTEE (8x8) :")
    print("=" * 30)
    for r in detected_grid:
        print("  " + " ".join(["█" if cell else "·" for cell in r]))
    print("=" * 30)
    
    print("\n" + "=" * 30)
    print("   PIÈCES DETECTEES (5x5) :")
    print("=" * 30)
    for idx, shape in enumerate(detected_shapes):
        print(f" Pièce {idx+1} :")
        for r in shape[:4]:  # Afficher les 4 premières lignes pour la compacité
            row_str = "".join(["█" if cell else "·" for cell in r[:4]])
            if np.any(r):
                print(f"   {row_str}")
        if not np.any(shape):
            print("   [Vide]")
        print("-" * 30)
        
    # Sauvegarder l'image de calibration avec les rectangles rouges d'alignement
    output_path = "calibration_result.png"
    cv2.imwrite(output_path, annotated_img)
    print(f"\n[+] Image de calibration enregistrée sous : '{output_path}'")
    print("[*] Ouvrez 'calibration_result.png' pour vérifier si les rectangles encadrent parfaitement votre grille et vos formes.")
    print("[*] Ajustez les valeurs au début du fichier 'calibrate_opencv.py' si nécessaire.")


if __name__ == "__main__":
    main()