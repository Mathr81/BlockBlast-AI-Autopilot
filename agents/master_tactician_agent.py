import itertools
from collections import deque
import numpy as np

# ==============================================================================
# CONFIGURATION V8.0
# ==============================================================================
CONFIG = {
    # --- Phases (blocs occupés) ---
    "THRESHOLD_ZEN":    10,   # < 10 blocs : expansion libre
    "THRESHOLD_CRISIS": 28,   # > 28 blocs : survie pure

    # --- Densité ---
    # Cible : 25% = 16 blocs sur 64
    "DENSITY_TARGET":         16,
    "DENSITY_PENALTY_BASE":    3.0,
    "DENSITY_PENALTY_CRISIS":  9.0,
    "DENSITY_EXCESS_EXP":      1.9,

    # --- Topologie (Flood-Fill) ---
    "PENALTY_HOLE_1":  240.0,
    "PENALTY_HOLE_2":  150.0,
    "PENALTY_HOLE_3":   95.0,
    "PENALTY_HOLE_4":   58.0,
    "PENALTY_HOLE_5":   32.0,
    "PENALTY_HOLE_6":   16.0,

    # --- Fragmentation ---
    "PENALTY_FRAG_TACTICAL":  65.0,
    "PENALTY_FRAG_CRISIS":   150.0,

    # --- Espace vital ---
    "PENALTY_NO_3X3":  280.0,  # Zone 3x3 libre indispensable
    "PENALTY_NO_2X2":  320.0,  # Zone 2x2 libre = quasi game over

    # --- Clustering ---
    "PENALTY_ISOLATED":       95.0,
    "PENALTY_SEMI_ISOLATED":  34.0,

    # --- Pièces tueuses ---
    "PENALTY_KILLER_3X3":  300.0,
    "PENALTY_KILLER_1X5":  220.0,
    "PENALTY_KILLER_2X3":  185.0,

    # --- Lanes propres ---
    "BONUS_CLEAR_LANE": 38.0,

    # --- Priming (lignes/colonnes presque pleines) ---
    "BONUS_PRIMING_7": 60.0,   # 7/8 remplis → 1 pièce suffit pour effacer
    "BONUS_PRIMING_6": 20.0,
    "BONUS_PRIMING_5":  4.0,

    # ═══════════════════════════════════════════════════════════════
    # COMBO — Modèle Android corrigé
    #
    # current_combo = COMPTEUR DE LIGNES effacées en cumul consécutif
    #   → incrémenté de `cleared` (pas de 1) à chaque pièce qui efface
    #   → ex: effacer 3 lignes d'un coup → combo += 3
    #   → le multiplicateur en jeu = f(current_combo)
    #
    # mslc = moves_since_last_clear (blocs posés sans effacer)
    #   → fenêtre Android = 4 blocs avant bris
    #   → transmis entre tours
    # ═══════════════════════════════════════════════════════════════

    # Pénalité de bris de combo (proportionnelle au combo actuel)
    "COMBO_BREAK_BASE":           600.0,
    "COMBO_BREAK_PER_LEVEL":       15.0,   # Combo 0-30
    "COMBO_BREAK_PER_LEVEL_HIGH":  40.0,   # Combo > 30

    # Bonus de maintien (par niveau de combo maintenu ce tour)
    "COMBO_MAINTAIN_BONUS":        70.0,

    # Bonus pour démarrer un combo
    "COMBO_START_BONUS":           55.0,

    # Urgence : bonus supplémentaire d'effacer quand MSLC est critique
    "COMBO_URGENCY_MSLC2":        150.0,   # MSLC entrant = 2 → 2 blocs restants
    "COMBO_URGENCY_MSLC3":        450.0,   # MSLC entrant = 3 → grâce Android

    # Stratégie "3ème mouvement" : effacer avec la DERNIÈRE pièce du tour
    # repart le MSLC à 0, donne la pleine fenêtre au tour suivant
    "COMBO_BONUS_LAST_PIECE":      90.0,

    # Phase "Slow Burn" : combo < 30 → préférer 1 ligne à la fois (économiser les blocs)
    # Phase "Harvest"   : combo > 80 → chercher les doubles/triples
    "COMBO_SLOW_BURN_THRESHOLD":   30,
    "COMBO_HARVEST_THRESHOLD":     80,
    "COMBO_HARVEST_MULTI_BONUS":  200.0,   # Bonus pour ≥2 lignes d'un coup en phase harvest

    # --- Nettoyage super-linéaire (reflète le scoring exponentiel du jeu) ---
    "BONUS_CLEAR_1_LINE":    42.0,
    "BONUS_CLEAR_2_LINES":  130.0,
    "BONUS_CLEAR_3_LINES":  300.0,
    "BONUS_CLEAR_4P_LINES": 600.0,
    "BONUS_ALL_CLEAR":     3000.0,

    # --- DFI ---
    "DFI_WEIGHT_CRISIS": 300.0,
    "DFI_WEIGHT_NORMAL": 170.0,

    # --- Rugosité ---
    "ROUGHNESS_PENALTY": 2.5,

    # --- Centre (occuper le centre = fragmenter l'espace) ---
    "CENTER_CORE_PENALTY": 3.5,
    "CENTER_RING_PENALTY": 1.5,
}

# Poids DFI pondérés par dangerosité
_DFI_SHAPE_WEIGHTS = [
    0.5,  # 1x1
    0.7,  # 1x2
    0.7,  # 2x1
    1.0,  # 1x3
    1.0,  # 3x1
    1.2,  # 2x2
    2.5,  # 3x3 ← pièce la plus contraignante
    1.8,  # 1x5
    1.8,  # 5x1
    1.1,  # L 2x2
    2.0,  # L 3x3
    1.3,  # Z
    1.3,  # S
    1.2,  # T
    1.5,  # 1x4
    1.5,  # 4x1
    1.6,  # 2x3
    1.6,  # 3x2
    1.4,  # Croix
    0.6,  # Diag 2x2 /
    0.6,  # Diag 2x2 \
    1.0,  # Diag 3x3 /
    1.0,  # Diag 3x3 \
]
_DFI_TOTAL_WEIGHT = sum(_DFI_SHAPE_WEIGHTS)

ANDROID_COMBO_WINDOW = 4


class MasterTacticianAgent:
    """
    Agent Block Blast v8.0 — Modèle combo Android exact + optimisations vitesse.

    Corrections v8 :
    ────────────────
    1. current_combo = compteur de LIGNES effacées (+=cleared, pas +=1 par tour)
       → reflète le vrai multiplicateur affiché dans Block Blast
    2. Phases "Slow Burn" (combo<30) et "Harvest" (combo>80) intégrées
    3. quick_rate allégé (~2x plus rapide) pour beam élargis sans coût CPU
    """

    UNIQUE_CANONICAL_SHAPES = [
        [[1]],
        [[1,1]],          [[1],[1]],
        [[1,1,1]],        [[1],[1],[1]],
        [[1,1],[1,1]],
        [[1,1,1],[1,1,1],[1,1,1]],
        [[1,1,1,1,1]],    [[1],[1],[1],[1],[1]],
        [[1,1],[1,0]],
        [[1,1,1],[1,0,0],[1,0,0]],
        [[1,1,0],[0,1,1]], [[0,1,1],[1,1,0]],
        [[1,1,1],[0,1,0]],
        [[1,1,1,1]],      [[1],[1],[1],[1]],
        [[1,1,1],[1,1,1]], [[1,1],[1,1],[1,1]],
        [[0,1,0],[1,1,1],[0,1,0]],
        [[0,1],[1,0]],    [[1,0],[0,1]],
        [[0,0,1],[0,1,0],[1,0,0]],
        [[1,0,0],[0,1,0],[0,0,1]],
    ]

    def __init__(self, sample_size=16, action_masking=True):
        self.sample_size    = sample_size
        self.action_masking = action_masking
        self.THRESHOLD_CRISIS = CONFIG["THRESHOLD_CRISIS"]
        self.THRESHOLD_ZEN    = CONFIG["THRESHOLD_ZEN"]

    # ─────────────────────────────────────────────────────────────────────
    # POINT D'ENTRÉE PRINCIPAL (appelé par run_autopilot)
    # ─────────────────────────────────────────────────────────────────────
    def evaluate_grid_ultimate_v4(self, grid, current_combo, clears_history, game_phase,
                                   mslc_entrant=0, n_pieces_in_turn=3):
        return self._score(grid, current_combo, clears_history, game_phase,
                           mslc_entrant, n_pieces_in_turn)

    # ─────────────────────────────────────────────────────────────────────
    # ÉVALUATION PRINCIPALE
    # ─────────────────────────────────────────────────────────────────────
    def _score(self, grid, current_combo, clears_history, game_phase,
               mslc_entrant=0, n_pieces_in_turn=3):

        block_count   = sum(row.count(1) for row in grid)
        total_cleared = sum(clears_history)

        # ── 1. DENSITÉ (cible 25% = 16 blocs) ───────────────────────────
        target = CONFIG["DENSITY_TARGET"]
        if game_phase == "crisis":
            density_penalty = CONFIG["DENSITY_PENALTY_CRISIS"] * block_count
            if block_count > 40:
                density_penalty += 50.0 * (block_count - 40) ** CONFIG["DENSITY_EXCESS_EXP"]
        else:
            excess = block_count - target
            if excess < 0:
                density_penalty = 0.5 * abs(excess)
            else:
                density_penalty = CONFIG["DENSITY_PENALTY_BASE"] * excess
                if block_count > 36:
                    density_penalty += 20.0 * (block_count - 36) ** CONFIG["DENSITY_EXCESS_EXP"]

        # ── 2. TOPOLOGIE ─────────────────────────────────────────────────
        pockets = self._analyze_pockets(grid)
        frag_coef = (CONFIG["PENALTY_FRAG_CRISIS"] if game_phase == "crisis"
                     else CONFIG["PENALTY_FRAG_TACTICAL"])
        fragmentation_penalty = max(0, len(pockets) - 1) * frag_coef

        _pt = {1:240.,2:150.,3:95.,4:58.,5:32.,6:16.}
        small_pocket_penalty = sum(_pt[s] for s in pockets if s <= 6)

        # ── 3. ESPACE VITAL ──────────────────────────────────────────────
        max_pocket = max(pockets) if pockets else 0
        living_space_penalty = 0.0
        if max_pocket < 9:
            living_space_penalty += CONFIG["PENALTY_NO_3X3"]
        if not self._has_2x2_empty_space(grid):
            living_space_penalty += CONFIG["PENALTY_NO_2X2"]

        # ── 4. CLUSTERING ────────────────────────────────────────────────
        clustering_penalty = self._calculate_clustering_penalty(grid)

        # ── 5. PRIMING ───────────────────────────────────────────────────
        priming_bonus = 0.0
        if game_phase != "crisis":
            for r in range(8):
                s = sum(grid[r])
                if   s == 7: priming_bonus += CONFIG["BONUS_PRIMING_7"]
                elif s == 6: priming_bonus += CONFIG["BONUS_PRIMING_6"]
                elif s == 5: priming_bonus += CONFIG["BONUS_PRIMING_5"]
            for c in range(8):
                s = sum(grid[ri][c] for ri in range(8))
                if   s == 7: priming_bonus += CONFIG["BONUS_PRIMING_7"]
                elif s == 6: priming_bonus += CONFIG["BONUS_PRIMING_6"]
                elif s == 5: priming_bonus += CONFIG["BONUS_PRIMING_5"]

        # ── 6. LANES PROPRES ─────────────────────────────────────────────
        clear_lanes_bonus = 0.0
        for r in range(8):
            if sum(grid[r]) <= 1:
                clear_lanes_bonus += CONFIG["BONUS_CLEAR_LANE"]
        for c in range(8):
            if sum(grid[ri][c] for ri in range(8)) <= 1:
                clear_lanes_bonus += CONFIG["BONUS_CLEAR_LANE"]

        # ── 7. CENTRE ────────────────────────────────────────────────────
        center_penalty = 0.0
        if game_phase != "crisis":
            for r in range(2, 6):
                for c in range(2, 6):
                    if grid[r][c]: center_penalty += CONFIG["CENTER_RING_PENALTY"]
            for r in range(3, 5):
                for c in range(3, 5):
                    if grid[r][c]: center_penalty += CONFIG["CENTER_CORE_PENALTY"]

        # ── 8. RUGOSITÉ ──────────────────────────────────────────────────
        heights = [next((8-r for r in range(8) if grid[r][c]),0) for c in range(8)]
        roughness_penalty = CONFIG["ROUGHNESS_PENALTY"] * sum(
            abs(heights[i]-heights[i+1]) for i in range(7))

        # ── 9. PIÈCES TUEUSES ────────────────────────────────────────────
        killer_penalty = 0.0
        if not self._can_fit_shape(grid, [[1,1,1],[1,1,1],[1,1,1]]):
            killer_penalty += CONFIG["PENALTY_KILLER_3X3"]
        if (not self._can_fit_shape(grid, [[1,1,1,1,1]]) and
                not self._can_fit_shape(grid, [[1],[1],[1],[1],[1]])):
            killer_penalty += CONFIG["PENALTY_KILLER_1X5"]
        if (not self._can_fit_shape(grid, [[1,1,1],[1,1,1]]) and
                not self._can_fit_shape(grid, [[1,1],[1,1],[1,1]])):
            killer_penalty += CONFIG["PENALTY_KILLER_2X3"]

        # ── 10. NETTOYAGE SUPER-LINÉAIRE ─────────────────────────────────
        clearing_bonus = 0.0
        for cl in clears_history:
            if   cl == 1: clearing_bonus += CONFIG["BONUS_CLEAR_1_LINE"]
            elif cl == 2: clearing_bonus += CONFIG["BONUS_CLEAR_2_LINES"]
            elif cl == 3: clearing_bonus += CONFIG["BONUS_CLEAR_3_LINES"]
            elif cl >= 4: clearing_bonus += CONFIG["BONUS_CLEAR_4P_LINES"]
        if all(cell == 0 for row in grid for cell in row):
            clearing_bonus += CONFIG["BONUS_ALL_CLEAR"]
        if game_phase == "crisis":
            clearing_bonus *= 2.5

        # ── 11. COMBO — MODÈLE ANDROID CORRIGÉ ──────────────────────────
        #
        # current_combo = somme de TOUTES les lignes effacées consécutivement
        #   (incrémenté de cleared à chaque pièce, pas de 1 par tour)
        # clears_history = liste de cleared par pièce ce tour
        # mslc_entrant = blocs posés sans efface en entrée du tour
        #
        combo_delta = 0.0

        # Simuler le MSLC et le combo pièce par pièce
        mslc_sim          = mslc_entrant
        combo_broken      = False
        last_piece_cleared = False
        lines_this_turn   = 0

        for piece_idx, cleared in enumerate(clears_history):
            mslc_sim += 1
            if cleared > 0:
                mslc_sim = 0
                lines_this_turn += cleared
                last_piece_cleared = (piece_idx == len(clears_history) - 1)
            if mslc_sim >= ANDROID_COMBO_WINDOW and current_combo > 0:
                combo_broken = True
                break

        if combo_broken and current_combo > 0:
            # Pénalité graduelle
            if current_combo <= 30:
                penalty = CONFIG["COMBO_BREAK_BASE"] + current_combo * CONFIG["COMBO_BREAK_PER_LEVEL"]
            else:
                penalty = (CONFIG["COMBO_BREAK_BASE"]
                           + 30 * CONFIG["COMBO_BREAK_PER_LEVEL"]
                           + (current_combo - 30) * CONFIG["COMBO_BREAK_PER_LEVEL_HIGH"])
            combo_delta -= penalty

        elif not combo_broken and game_phase != "crisis":
            if lines_this_turn > 0:
                if current_combo > 0:
                    combo_delta += CONFIG["COMBO_MAINTAIN_BONUS"] * min(current_combo // 10 + 1, 5)
                else:
                    combo_delta += CONFIG["COMBO_START_BONUS"]

                # Urgence : bonus si le MSLC entrant était critique
                if mslc_entrant == 2:
                    combo_delta += CONFIG["COMBO_URGENCY_MSLC2"]
                elif mslc_entrant == 3:
                    combo_delta += CONFIG["COMBO_URGENCY_MSLC3"]

                # 3ème mouvement cycle
                if last_piece_cleared:
                    combo_delta += CONFIG["COMBO_BONUS_LAST_PIECE"]

                # Phase "Harvest" : chercher les multi-effaces à haut combo
                if current_combo >= CONFIG["COMBO_HARVEST_THRESHOLD"]:
                    max_cl = max(clears_history)
                    if max_cl >= 2:
                        combo_delta += CONFIG["COMBO_HARVEST_MULTI_BONUS"] * (max_cl - 1)

                # Phase "Slow Burn" : à bas combo, 1 ligne suffit — ne pas gaspiller
                # (pas de bonus/malus spécifique ici, la densité gère déjà ça)

        elif lines_this_turn > 0 and current_combo == 0:
            combo_delta += CONFIG["COMBO_START_BONUS"]

        # ── 12. DFI PONDÉRÉ ──────────────────────────────────────────────
        dfi_raw    = self._evaluate_future_resilience_weighted(grid)
        dfi_weight = (CONFIG["DFI_WEIGHT_CRISIS"] if game_phase == "crisis"
                      else CONFIG["DFI_WEIGHT_NORMAL"])
        dfi_bonus  = dfi_raw * dfi_weight

        return (
            - density_penalty
            - fragmentation_penalty
            - small_pocket_penalty
            - living_space_penalty
            - clustering_penalty
            - center_penalty
            - roughness_penalty
            - killer_penalty
            + priming_bonus
            + clear_lanes_bonus
            + clearing_bonus
            + combo_delta
            + dfi_bonus
        )

    # ─────────────────────────────────────────────────────────────────────
    # PRE-FILTRE BEAM — version allégée pour vitesse
    # ─────────────────────────────────────────────────────────────────────
    def quick_rate(self, grid, cleared, mslc_after=None):
        """
        Pre-filtre rapide : ~10µs au lieu de ~20µs.
        Conserve les signaux clés : densité, trous 1x1, priming, cleared.
        Supprime les calculs coûteux (trous 2x1, isolation) du hot path.
        """
        blocks = 0
        holes  = 0
        priming = 0.0

        for r in range(8):
            s = sum(grid[r])
            blocks += s
            if   s == 7: priming += 42.0
            elif s == 6: priming += 14.0

        for c in range(8):
            s = sum(grid[r][c] for r in range(8))
            if   s == 7: priming += 42.0
            elif s == 6: priming += 14.0

        for r in range(8):
            for c in range(8):
                if grid[r][c] == 0:
                    if ((r == 0 or grid[r-1][c]) and (r == 7 or grid[r+1][c]) and
                            (c == 0 or grid[r][c-1]) and (c == 7 or grid[r][c+1])):
                        holes += 1

        score = -4.0*blocks - 85.0*holes + 40.0*cleared + priming

        # Bonus d'urgence combo au niveau du filtre
        if mslc_after is not None and cleared > 0 and mslc_after == 0:
            score += 300.0   # On a effacé sur un coup d'urgence → garder ce candidat

        return score

    # ─────────────────────────────────────────────────────────────────────
    # HELPERS
    # ─────────────────────────────────────────────────────────────────────
    def _analyze_pockets(self, grid):
        visited = [[False]*8 for _ in range(8)]
        pockets = []
        for r in range(8):
            for c in range(8):
                if grid[r][c] == 0 and not visited[r][c]:
                    size = 0
                    q = deque([(r,c)])
                    visited[r][c] = True
                    while q:
                        cr, cc = q.popleft()
                        size += 1
                        for dr, dc in ((-1,0),(1,0),(0,-1),(0,1)):
                            nr, nc = cr+dr, cc+dc
                            if 0<=nr<8 and 0<=nc<8 and grid[nr][nc]==0 and not visited[nr][nc]:
                                visited[nr][nc] = True
                                q.append((nr,nc))
                    pockets.append(size)
        return pockets

    def _calculate_clustering_penalty(self, grid):
        penalty = 0.0
        for r in range(8):
            for c in range(8):
                if grid[r][c]:
                    n = sum(1 for dr,dc in ((-1,0),(1,0),(0,-1),(0,1))
                            if 0<=r+dr<8 and 0<=c+dc<8 and grid[r+dr][c+dc])
                    if   n == 0: penalty += CONFIG["PENALTY_ISOLATED"]
                    elif n == 1: penalty += CONFIG["PENALTY_SEMI_ISOLATED"]
        return penalty

    def _has_2x2_empty_space(self, grid):
        for r in range(7):
            for c in range(7):
                if (grid[r][c]==0 and grid[r+1][c]==0
                        and grid[r][c+1]==0 and grid[r+1][c+1]==0):
                    return True
        return False

    def _can_fit_shape(self, grid, shape):
        h, w = len(shape), len(shape[0])
        for r in range(8-h+1):
            for c in range(8-w+1):
                if all(not(shape[i][j] and grid[r+i][c+j])
                       for i in range(h) for j in range(w)):
                    return True
        return False

    def _evaluate_future_resilience_weighted(self, grid):
        score = sum(w for shape, w in zip(self.UNIQUE_CANONICAL_SHAPES, _DFI_SHAPE_WEIGHTS)
                    if self._can_fit_shape(grid, shape))
        return score / _DFI_TOTAL_WEIGHT

    def evaluate_grid_health(self, grid):
        bc = sum(row.count(1) for row in grid)
        phase = ("crisis" if bc > self.THRESHOLD_CRISIS
                 else "zen" if bc < self.THRESHOLD_ZEN else "tactical")
        return self._score(grid, 0, [0], phase, 0, 1)