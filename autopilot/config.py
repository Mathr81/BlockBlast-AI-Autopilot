"""
config.py — Configuration centrale de l'autopilote Block Blast
À placer à la racine du projet (même dossier que run_autopilot.py).

C'est le SEUL fichier à modifier pour recalibrer le bot.
"""

# =====================================================================
# ÉCRAN CALIBRÉ (1084x2412)
# =====================================================================
GRID_X = 63;  GRID_Y = 584;  GRID_W = 956;  GRID_H = 956
SLOT_W = 264; SLOT_H = 264;  SLOT_Y = 1690
SLOT_X_POSITIONS = [95, 410, 725]
MINI_BLOCK_SIZE  = 50

GRID_BRIGHTNESS_THRESHOLD = 70
COLOR_DIFF_THRESHOLD      = 40

# ─── DRAG ─────────────────────────────────────────────────────────────
DRAG_GAIN_X          = 1.4
DRAG_GAIN_Y          = 1.4
BASE_VERTICAL_OFFSET = 270

# ─── TIMING ───────────────────────────────────────────────────────────
PREVIEW_DELAY          = 0      # Délai avant d'exécuter un coup (0 = instantané)
DELAY_AFTER_SWIPE      = 0.15   # Pause en fin de tour
SHORT_INTER_MOVE_DELAY = 0.05   # Pause entre deux pièces sans efface

# =====================================================================
# ROBUSTESSE & RESCAN
# =====================================================================
GRID_RESCAN_FREQUENCY = 1       # Re-scan OpenCV à chaque N tours (1 = tous les tours)

ANIMATION_DELAY_ALL_CLEAR = 3.0 # Attente après un All Clear
ANIMATION_DELAY_CLEARED   = 2.0 # Attente après un efface normal
ANIMATION_DELAY_NORMAL    = 1.0 # Attente entre deux tours sans efface

MAX_DETECTION_RETRIES = 3
DETECTION_RETRY_DELAY = 0.4

# =====================================================================
# LOGGING
# =====================================================================
ENABLE_LOGGING = True
LOG_FILENAME   = "blockblast_autopilot.log"

# =====================================================================
# SKIN (reset automatique si thème alternatif détecté)
# =====================================================================
SKIN_CHECK_X = 25;  SKIN_CHECK_Y = 1565
EXPECTED_BG_BGR      = [148, 83, 57]
SKIN_COLOR_THRESHOLD = 30

SETTINGS_BTN_X    = 982;  SETTINGS_BTN_Y    = 203
DEFAULT_SKIN_BTN_X = 541; DEFAULT_SKIN_BTN_Y = 1772
CLOSE_SETTINGS_X  = 932;  CLOSE_SETTINGS_Y  = 513

# =====================================================================
# REVIVE (détection automatique du bouton vert)
# =====================================================================
REVIVE_PX1_X = 245; REVIVE_PX1_Y = 1312
REVIVE_PX2_X = 823; REVIVE_PX2_Y = 1409
REVIVE_GREEN_BGR       = [25, 202, 66]
REVIVE_COLOR_THRESHOLD = 60
REVIVE_CLICK_X = 540; REVIVE_CLICK_Y = 1370

# =====================================================================
# PLANIFICATEUR — BEAM WIDTHS
#
# Grille vide (zen < 10 blocs) :
#   Les positions sont quasi-équivalentes → beam réduit suffit
#   → ~400ms par tour
# Grille tactique (10-28 blocs) :
#   Plus de variance stratégique → beam moyen
# Grille crisis (> 28 blocs) :
#   Positions rares → beam plus large mais naturellement limité
#
# Augmenter ces valeurs = plus intelligent mais plus lent (O(beam²))
# =====================================================================
BEAM_WIDTH_ZEN      = 12
BEAM_WIDTH_TACTICAL = 18
BEAM_WIDTH_CRISIS   = 22

# =====================================================================
# COMBO ANDROID
#
# current_combo = cumul de LIGNES effacées consécutivement (+=cleared par pièce)
#   → c'est le multiplicateur affiché dans Block Blast
#   → ex: effacer 2 lignes d'un coup → combo += 2
#
# mslc = moves_since_last_clear (blocs posés sans effacer depuis dernier efface)
#   → transmis entre les tours, incrémenté par PIÈCE
#   → fenêtre Android = 4 blocs avant bris du combo
# =====================================================================
ANDROID_COMBO_WINDOW = 4

# =====================================================================
# VALIDATION DE GRILLE
#
# Après chaque re-scan OpenCV, compare la grille extraite avec
# l'état simulé du tour précédent.
# Si l'écart dépasse ce seuil, une re-capture est effectuée
# automatiquement (0 = désactiver la validation).
# =====================================================================
GRID_VALIDATION_MAX_MISMATCH = 8  # cellules sur 64

# =====================================================================
# AFFICHAGE VIRTUEL (option --virtual-display)
#
# Permet de jouer avec l'écran du téléphone éteint.
# Nécessite Android 10+ et scrcpy-server 2.x.
# =====================================================================
VIRTUAL_DISPLAY_W   = 1080            # Largeur de l'affichage virtuel
VIRTUAL_DISPLAY_H   = 2408            # Hauteur (doit correspondre au calibrage)
VIRTUAL_DISPLAY_DPI = 420
BLOCK_BLAST_PACKAGE = "com.block.juggle"