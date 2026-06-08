import itertools
import random
from collections import deque
import numpy as np


class HeuristicSearchAgent:
    """Agent de recherche prospective ultra-rapide et stratégique pour Block Blast.

    Intègre un élagage par faisceau (Beam Search) pour garantir des performances temps réel,
    une adaptation dynamique à la densité du plateau et une quantification fine de l'espace
    pour anticiper les blocs critiques (3x3 et lignes de 5).
    """

    # Formes complexes de Block Blast pour la simulation Monte Carlo
    SHAPE_POOL = [
        [[1]],  # 1x1
        [[1, 1]],  # 1x2
        [[1], [1]],  # 2x1
        [[1, 1, 1]],  # 1x3
        [[1], [1], [1]],  # 3x1
        [[1, 1], [1, 1]],  # 2x2
        [[1, 1, 1, 1]],  # 1x4
        [[1], [1], [1], [1]],  # 4x1
        [[1, 1, 1, 1, 1]],  # 1x5 (Critique)
        [[1], [1], [1], [1], [1]],  # 5x1 (Critique)
        [[1, 1, 1], [1, 1, 1], [1, 1, 1]],  # 3x3 (Super Critique)
        [[1, 1, 1], [1, 0, 0], [1, 0, 0]],  # L 3x3
        [[1, 1, 1], [0, 0, 1], [0, 0, 1]],  # L 3x3 inversé
        [[1, 1], [1, 0]],  # L 2x2
        [[1, 1], [0, 1]],  # L 2x2 inversé
        [[1, 0], [1, 1]],  # L 2x2 alternatif
        [[0, 1], [1, 1]],  # L 2x2 alternatif 2
        [[1, 1, 0], [0, 1, 1]],  # Z horizontal
        [[0, 1, 1], [1, 1, 0]],  # S horizontal
    ]

    def __init__(self, sample_size=12, action_masking=True, beam_width=6):
        self.sample_size = sample_size
        self.action_masking = action_masking
        self.beam_width = beam_width  # Largeur du faisceau d'élagage pour le DFS

    def predict(
        self, observation, state=None, episode_start=None, deterministic=True, **kwargs
    ):
        """Décode l'observation Gym et retourne l'action optimale."""
        grid = observation["grid"]
        shapes = observation["shapes"]

        is_batched = len(grid.shape) == 3
        if is_batched:
            grid = grid[0]
            shapes = shapes[0]

        # 1. Extraction des pièces actives de la main actuelle
        active_shapes = []
        for idx in range(3):
            slice_5x5 = shapes[idx]
            if np.any(slice_5x5):
                active_rows = np.any(slice_5x5, axis=1)
                active_cols = np.any(slice_5x5, axis=0)

                max_row = np.where(active_rows)[0][-1] + 1
                max_col = np.where(active_cols)[0][-1] + 1

                shape_form = slice_5x5[:max_row, :max_col].tolist()
                active_shapes.append((idx, shape_form))

        if not active_shapes:
            return np.array([0]) if is_batched else 0, None

        # 2. Extraction du combo actuel
        current_combo = 0
        if "combo" in observation:
            combo_val = observation["combo"]
            if isinstance(combo_val, (np.ndarray, list)):
                current_combo = int(combo_val[0][0]) if len(combo_val) > 0 and isinstance(combo_val[0], (np.ndarray, list)) else int(combo_val[0]) if len(combo_val) > 0 else 0
            else:
                current_combo = int(combo_val)

        grid_list = [[int(cell) for cell in row] for row in grid]

        # 3. Recherche prospective avec élagage par faisceau (Beam Search)
        all_paths = self._explore_all_placements(grid_list, active_shapes)

        if not all_paths:
            # En cas d'impasse totale (Game Over)
            return np.array([0]) if is_batched else 0, None

        # 4. Évaluation stratégique et dynamique des trajectoires
        for path in all_paths:
            path['simulated_score'] = self.simulate_path_score(path, current_combo)
            if 'incomplete_penalty' in path:
                path['simulated_score'] -= path['incomplete_penalty']

        # Tri et extraction des 6 meilleures solutions candidates
        all_paths.sort(key=lambda x: x['simulated_score'], reverse=True)
        top_candidates = all_paths[:6]

        # 5. Évaluation Monte Carlo rapide sur les candidats restants
        future_samples = self._sample_future_shapes(self.sample_size)
        
        best_overall_score = -float("inf")
        best_move = None

        for candidate in top_candidates:
            resilience = self._evaluate_future_resilience(candidate['grid'], future_samples)
            total_score = candidate['simulated_score'] + (resilience * 100.0)

            if total_score > best_overall_score:
                best_overall_score = total_score
                best_move = candidate['first_move']

        # 6. Encodage de l'action choisie
        if best_move is not None:
            shape_idx, r, c = best_move
            action = shape_idx * 64 + r * 8 + c
        else:
            action = 0

        if is_batched:
            return np.array([action]), None
        return action, None

    def evaluate_grid(self, grid):
        """Méthode de compatibilité descendante avec run_autopilot.py."""
        return self.evaluate_grid_advanced(grid)

    def evaluate_grid_advanced(self, grid):
        """Évalue finement la structure géométrique et la santé de la grille 8x8."""
        block_count = sum(1 for r in range(8) for c in range(8) if grid[r][c])
        density = block_count / 64.0

        # Adaptation dynamique des pénalités selon le taux d'encombrement du plateau
        if density > 0.42:
            # Mode Survie : Le nettoyage des trous et la compacité priment sur tout le reste
            hole_mult = 2.2
            fragmentation_penalty_weight = 30.0
            survival_weight = 200.0
        else:
            # Mode Normal : Jeu structuré, équilibre entre placement et construction
            hole_mult = 1.0
            fragmentation_penalty_weight = 15.0
            survival_weight = 120.0

        # A. Analyse topologique (Flood-Fill) pour détecter la fragmentation et les zones fermées
        visited = [[False] * 8 for _ in range(8)]
        small_empty_penalty = 0.0
        large_openings = 0
        
        for r in range(8):
            for c in range(8):
                if not grid[r][c] and not visited[r][c]:
                    # BFS pour mesurer la taille de la zone vide contiguë
                    comp_size = 0
                    queue = deque([(r, c)])
                    visited[r][c] = True
                    while queue:
                        curr_r, curr_c = queue.popleft()
                        comp_size += 1
                        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                            nr, nc = curr_r + dr, curr_c + dc
                            if 0 <= nr < 8 and 0 <= nc < 8:
                                if not grid[nr][nc] and not visited[nr][nc]:
                                    visited[nr][nc] = True
                                    queue.append((nr, nc))
                    
                    if comp_size < 4:
                        # Pénalisation des micro-trous (extrêmement difficiles à combler)
                        small_empty_penalty += (4 - comp_size) * 55.0 * hole_mult
                    else:
                        large_openings += 1

        # Pénalité de fragmentation : favorise un seul grand espace uni plutôt que des zones éparpillées
        fragmentation_penalty = max(0, large_openings - 1) * fragmentation_penalty_weight

        # B. Détection des trous isolés de taille 1 (entourés de 4 blocs)
        holes = 0
        for r in range(8):
            for c in range(8):
                if not grid[r][c]:
                    filled_neighbors = 0
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = r + dr, c + dc
                        if nr < 0 or nr >= 8 or nc < 0 or nc >= 8 or grid[nr][nc]:
                            filled_neighbors += 1
                    if filled_neighbors == 4:
                        holes += 1

        # C. Analyse de la rugosité verticale (Bumpiness)
        heights = []
        for c in range(8):
            height = 0
            for r in range(8):
                if grid[r][c]:
                    height = 8 - r
                    break
            heights.append(height)
        roughness = sum(abs(heights[i] - heights[i + 1]) for i in range(7))

        # D. Quantification de la liberté de placement pour les blocs critiques (Anticipation)
        freedom_3x3 = self._count_placements(grid, [[1, 1, 1], [1, 1, 1], [1, 1, 1]])
        freedom_5x1 = self._count_placements(grid, [[1], [1], [1], [1], [1]])
        freedom_1x5 = self._count_placements(grid, [[1, 1, 1, 1, 1]])
        freedom_2x2 = self._count_placements(grid, [[1, 1], [1, 1]])

        survival_penalty = 0.0
        # Sanction sévère si aucun emplacement n'est disponible pour les pièces critiques
        if freedom_3x3 == 0:
            survival_penalty += survival_weight
        elif freedom_3x3 == 1:
            survival_penalty += survival_weight * 0.4

        if (freedom_5x1 + freedom_1x5) == 0:
            survival_penalty += survival_weight * 0.8
        elif (freedom_5x1 + freedom_1x5) == 1:
            survival_penalty += survival_weight * 0.3

        if freedom_2x2 == 0:
            survival_penalty += 50.0

        # Encouragement de la liberté d'espace (bonus modéré pour chaque emplacement critique disponible)
        freedom_bonus = (
            min(freedom_3x3, 4) * 6.0 +
            min(freedom_5x1 + freedom_1x5, 4) * 4.0 +
            min(freedom_2x2, 5) * 2.0
        )

        # E. Bonus d'alignement (Lignes presque pleines à 6 ou 7 blocs)
        almost_cleared_bonus = 0.0
        for r in range(8):
            row_sum = sum(grid[r])
            if row_sum in [6, 7]:
                almost_cleared_bonus += (row_sum - 5) * 6.0
        for c in range(8):
            col_sum = sum(grid[r][c] for r in range(8))
            if col_sum in [6, 7]:
                almost_cleared_bonus += (col_sum - 5) * 6.0

        # Score final consolidé
        score = (
            - 5.0 * block_count
            - 28.0 * holes * hole_mult
            - small_empty_penalty
            - fragmentation_penalty
            - survival_penalty
            - 1.8 * roughness
            + freedom_bonus
            + almost_cleared_bonus
        )
        return score

    def simulate_path_score(self, path, current_combo):
        """Estime l'impact d'une séquence complète de 3 pièces sur le score et le combo."""
        grid = path['grid']
        clears_per_step = path['clears_per_step']
        total_cleared = sum(clears_per_step)
        cleared_any = total_cleared > 0

        # Densité du plateau final pour adapter les coefficients
        block_count = sum(1 for r in range(8) for c in range(8) if grid[r][c])
        density = block_count / 64.0

        # Évaluation géométrique fine de l'état final du plateau
        board_score = self.evaluate_grid_advanced(grid)

        points_gained = 0
        temp_combo = current_combo

        # Simulation du score réel accumulé pas-à-pas
        for cleared in clears_per_step:
            if cleared > 0:
                temp_combo += 1
                points_gained += (cleared * 10) + (temp_combo * 15)

        # Gestion intelligente du combo selon la situation de crise
        combo_bonus = 0.0
        if current_combo > 0:
            if not cleared_any:
                # En situation critique (density > 0.40), on accepte de casser le combo pour rester en vie!
                # Si le plateau est propre, on protège le combo agressivement.
                penalty_factor = 25.0 if density > 0.40 else 95.0
                combo_bonus -= (current_combo * penalty_factor)
            else:
                # Bonus réduit si le plateau est trop encombré (la survie prime)
                scale = 0.4 if density > 0.40 else 1.0
                combo_bonus += (temp_combo * 22.0) * scale
        else:
            if cleared_any:
                combo_bonus += 12.0

        return board_score + points_gained + combo_bonus

    def _explore_all_placements(self, grid, active_shapes):
        """Explore toutes les séquences possibles de placement."""
        results = []
        permutations = list(itertools.permutations(active_shapes))

        for perm in permutations:
            self._dfs_backtrack(grid, perm, 0, None, [], results)

        return results

    def _dfs_backtrack(self, grid, perm, depth, first_move, clears_history, results):
        if depth == len(perm):
            results.append({
                'grid': [row[:] for row in grid],
                'first_move': first_move,
                'clears_per_step': list(clears_history),
                'total_cleared': sum(clears_history)
            })
            return

        shape_idx, form = perm[depth]
        h, w = len(form), len(form[0])

        # 1. Évaluation rapide de toutes les coordonnées physiques valides
        candidates = []
        for r in range(8 - h + 1):
            for c in range(8 - w + 1):
                # Vérification rapide de collision
                can_place = True
                for i in range(h):
                    for j in range(w):
                        if form[i][j] and grid[r + i][c + j]:
                            can_place = False
                            break
                    if not can_place:
                        break

                if can_place:
                    # Simulation ultra-rapide
                    next_grid, cleared = self._simulate_placement_fast(grid, form, r, c)
                    # Heuristique rapide : densité finale + lignes vidées
                    occupied = sum(1 for ri in range(8) for ci in range(8) if next_grid[ri][ci])
                    quick_rating = -4.0 * occupied + (cleared * 20.0)
                    candidates.append((quick_rating, r, c, next_grid, cleared))

        if not candidates:
            # Gestion d'un blocage de la branche (Game Over partiel)
            unplaced_count = len(perm) - depth
            results.append({
                'grid': [row[:] for row in grid],
                'first_move': first_move if first_move is not None else (shape_idx, 0, 0),
                'clears_per_step': list(clears_history) + [0] * unplaced_count,
                'total_cleared': sum(clears_history),
                'incomplete_penalty': unplaced_count * 250.0
            })
            return

        # 2. Élagage par faisceau (Beam Search) : on ne trie et n'explore que les 'beam_width' meilleures options
        candidates.sort(key=lambda x: x[0], reverse=True)
        top_candidates = candidates[:self.beam_width]

        # 3. Récursion sur le faisceau retenu
        for _, r, c, next_grid, cleared in top_candidates:
            next_first_move = first_move if first_move is not None else (shape_idx, r, c)
            clears_history.append(cleared)
            self._dfs_backtrack(
                next_grid, perm, depth + 1, next_first_move, clears_history, results
            )
            clears_history.pop()

    def _simulate_placement_fast(self, grid, shape_form, r, c):
        """Simule le placement d'un bloc et renvoie la grille mise à jour et le nombre de lignes nettoyées."""
        new_grid = [row[:] for row in grid]
        h, w = len(shape_form), len(shape_form[0])

        for i in range(h):
            for j in range(w):
                if shape_form[i][j]:
                    new_grid[r + i][c + j] = 1

        rows_to_clear = [i for i in range(8) if all(new_grid[i])]
        cols_to_clear = [j for j in range(8) if all(new_grid[i][j] for i in range(8))]

        for row in rows_to_clear:
            for col in range(8):
                new_grid[row][col] = 0

        for col in cols_to_clear:
            for row in range(8):
                new_grid[row][col] = 0

        return new_grid, len(rows_to_clear) + len(cols_to_clear)

    def _count_placements(self, grid, shape_form):
        """Compte le nombre total de positions d'accueil valides pour un bloc donné."""
        h, w = len(shape_form), len(shape_form[0])
        count = 0
        for r in range(8 - h + 1):
            for c in range(8 - w + 1):
                fits = True
                for i in range(h):
                    for j in range(w):
                        if shape_form[i][j] and grid[r + i][c + j]:
                            fits = False
                            break
                    if not fits:
                        break
                if fits:
                    count += 1
        return count

    def _sample_future_shapes(self, n):
        """Prélève un échantillon aléatoire de n formes de pièces."""
        try:
            from blockblast_game.game_state import BlockGameState
            pool = [var for form in BlockGameState.FORMS for var in form]
        except ImportError:
            pool = self.SHAPE_POOL
            
        return random.choices(pool, k=n)

    def _evaluate_future_resilience(self, grid, future_shapes):
        """Calcule la probabilité qu'un ensemble de blocs futurs s'insère sur la grille."""
        fits = 0
        for shape in future_shapes:
            # Si le bloc peut s'insérer au moins une fois
            if self._count_placements(grid, shape) > 0:
                fits += 1
        return fits / len(future_shapes)