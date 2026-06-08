import itertools
from collections import deque
import numpy as np


class EliteSearchAgent:
    """Agent d'élite basé sur l'évaluation de contiguïté par Flood-Fill

    et l'anticipation préventive des formes de blocage (Killer Shapes).
    """

    def __init__(self, action_masking=True):
        self.action_masking = action_masking

    def predict(
        self, observation, state=None, episode_start=None, deterministic=True, action_masks=None, **kwargs
    ):
        """Méthode de prédiction compatible avec Stable-Baselines3 et le visualiseur.

        Décode l'observation Gym pour trouver le coup optimal de l'arborescence.
        """
        grid = observation["grid"]
        shapes = observation["shapes"]

        is_batched = len(grid.shape) == 3
        if is_batched:
            grid = grid[0]
            shapes = shapes[0]

        # 1. Extraction des pièces en cours de jeu
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
            return (np.array([0]), None) if is_batched else (0, None)

        # 2. Recherche prospective récursive (Lookahead DFS)
        grid_list = [[int(cell) for cell in row] for row in grid]
        state_tracker = {
            "best_score": -float("inf"),
            "best_first_move": None,
            "max_depth_reached": 0,
        }

        permutations = list(itertools.permutations(active_shapes))

        for perm in permutations:
            self._dfs(grid_list, perm, 0, None, 0, state_tracker)

        best_move = state_tracker["best_first_move"]

        if best_move is not None:
            shape_idx, r, c = best_move
            action = shape_idx * 64 + r * 8 + c
        else:
            action = 0

        if is_batched:
            return np.array([action]), None
        return action, None

    def _can_fit_shape(self, grid, shape_form):
        """Vérifie si une forme spécifique peut s'insérer sur la grille."""
        h, w = len(shape_form), len(shape_form[0])
        for r in range(8 - h + 1):
            for c in range(8 - w + 1):
                fits = True
                for i in range(h):
                    for j in range(w):
                        if shape_form[i][j] and grid[r+i][c+j]:
                            fits = False
                            break
                    if not fits:
                        break
                if fits:
                    return True
        return False

    def evaluate_grid(self, grid, lines_cleared_total=0):
        """Évalue la santé structurelle de la grille.

        Un score élevé indique un plateau sain, aéré, sans fragmentation et sans blocages.
        """
        block_count = sum(row.count(1) for row in grid)

        # Base de score pénalisant la densité de blocs occupés
        score = -5.5 * block_count

        # 1. Analyse de la contiguïté via Flood-Fill (Bitmasks)
        grid_mask = 0
        for r in range(8):
            for c in range(8):
                if grid[r][c]:
                    grid_mask |= 1 << (r * 8 + c)

        visited = grid_mask
        pockets = []

        for idx in range(64):
            if not (visited & (1 << idx)):
                size = 0
                queue = deque([idx])
                visited |= 1 << idx
                while queue:
                    curr = queue.popleft()
                    size += 1
                    curr_r = curr // 8
                    curr_c = curr % 8

                    # Exploration des 4 voisins directes
                    if curr_r > 0:
                        neighbor = curr - 8
                        if not (visited & (1 << neighbor)):
                            visited |= 1 << neighbor
                            queue.append(neighbor)
                    if curr_r < 7:
                        neighbor = curr + 8
                        if not (visited & (1 << neighbor)):
                            visited |= 1 << neighbor
                            queue.append(neighbor)
                    if curr_c > 0:
                        neighbor = curr - 1
                        if not (visited & (1 << neighbor)):
                            visited |= 1 << neighbor
                            queue.append(neighbor)
                    if curr_c < 7:
                        neighbor = curr + 1
                        if not (visited & (1 << neighbor)):
                            visited |= 1 << neighbor
                            queue.append(neighbor)
                pockets.append(size)

        # Pénalité si l'espace est fragmenté en plusieurs petites poches isolées
        if len(pockets) > 1:
            score -= 25.0 * (len(pockets) - 1)

        # Pénalité sévère pour les micro-pockets de taille inutilisable (trous de 1, 2 ou 3)
        for size in pockets:
            if size == 1:
                score -= 50.0
            elif size == 2:
                score -= 30.0
            elif size == 3:
                score -= 12.0

        # 2. Préservation d'espace pour les pièces critiques du jeu (Killer Shapes)
        killer_shapes = [
            [[1, 1, 1], [1, 1, 1], [1, 1, 1]],  # Carré 3x3
            [[1, 1, 1, 1, 1]],  # Barre 1x5
            [[1], [1], [1], [1], [1]],  # Barre 5x1
            [[1, 0, 0], [1, 0, 0], [1, 1, 1]],  # L 3x3
        ]

        for shape in killer_shapes:
            if not self._can_fit_shape(grid, shape):
                score -= 90.0

        # 3. Analyse du relief (hauteur des colonnes et rugosité)
        heights = []
        for c in range(8):
            col_height = 0
            for r in range(8):
                if grid[r][c]:
                    col_height = 8 - r
                    break
            heights.append(col_height)

        roughness = sum(abs(heights[i] - heights[i + 1]) for i in range(7))
        max_height = max(heights)

        score -= 2.5 * roughness
        score -= 3.5 * max_height

        # 4. Pénalité pour l'obstruction des 4 coins
        corners = [(0, 0), (0, 7), (7, 0), (7, 7)]
        filled_corners = sum(1 for r, c in corners if grid[r][c])
        score -= 4.0 * filled_corners

        # 5. Récompense de nettoyage adaptative en fonction de la congestion
        if block_count > 24:
            score += lines_cleared_total * 45.0
        else:
            score += lines_cleared_total * 15.0

        return score

    def _simulate_placement_fast(self, grid, shape_form, r, c):
        """Simule le placement et calcule les lignes éliminées."""
        new_grid = [row[:] for row in grid]
        h, w = len(shape_form), len(shape_form[0])

        for i in range(h):
            for j in range(w):
                if shape_form[i][j]:
                    new_grid[r + i][c + j] = 1

        rows_to_clear = [i for i in range(8) if all(new_grid[i])]
        cols_to_clear = [
            j for j in range(8) if all(new_grid[i][j] for i in range(8))
        ]

        for row in rows_to_clear:
            for col in range(8):
                new_grid[row][col] = 0

        for col in cols_to_clear:
            for row in range(8):
                new_grid[row][col] = 0

        lines_cleared = len(rows_to_clear) + len(cols_to_clear)
        return new_grid, lines_cleared

    def _dfs(self, grid, perm, depth, first_move, lines_cleared_total, state_tracker):
        if depth == len(perm):
            heuristic_val = self.evaluate_grid(grid, lines_cleared_total)

            if heuristic_val > state_tracker["best_score"]:
                state_tracker["best_score"] = heuristic_val
                state_tracker["best_first_move"] = first_move
                state_tracker["max_depth_reached"] = depth
            return

        idx, form = perm[depth]
        h, w = len(form), len(form[0])

        placed_any = False
        for r in range(8 - h + 1):
            for c in range(8 - w + 1):
                can_place = True
                for i in range(h):
                    for j in range(w):
                        if form[i][j] and grid[r + i][c + j]:
                            can_place = False
                            break
                    if not can_place:
                        break

                if can_place:
                    placed_any = True
                    next_grid, cleared = self._simulate_placement_fast(
                        grid, form, r, c
                    )
                    move_to_record = (
                        first_move if first_move is not None else (idx, r, c)
                    )

                    self._dfs(
                        next_grid,
                        perm,
                        depth + 1,
                        move_to_record,
                        lines_cleared_total + cleared,
                        state_tracker,
                    )

        if not placed_any:
            heuristic_val = self.evaluate_grid(grid, lines_cleared_total)
            total_score = heuristic_val - 150.0 * (len(perm) - depth)

            if depth > state_tracker["max_depth_reached"] or (
                depth == state_tracker["max_depth_reached"]
                and total_score > state_tracker["best_score"]
            ):
                state_tracker["best_score"] = total_score
                state_tracker["best_first_move"] = first_move
                state_tracker["max_depth_reached"] = depth