import os
import random
import itertools
import numpy as np
import torch

from sb3_contrib import MaskablePPO
from blockblast_game.game_state import BlockGameState

try:
    from stable_baselines3.common.policies import obs_as_tensor
except ImportError:
    from stable_baselines3.common.utils import obs_as_tensor


class HybridAlphaZeroAgent:
    """
    Agent Hybride de niveau industriel.
    Combine la recherche déterministe (DFS), l'évaluation statistique (Monte Carlo)
    et l'intuition spatiale d'un réseau de neurones MaskablePPO (Value Critic).
    """
    def __init__(self, model_path=None, sample_size=12, action_masking=True):
        self.sample_size = sample_size
        self.action_masking = action_masking
        self.all_possible_forms = BlockGameState.FORMS
        self.model = None

        # Tentative de chargement du modèle de Deep RL
        if model_path and os.path.isfile(model_path):
            try:
                print(f"[Hybrid Engine] Chargement de l'évaluation neurale depuis : {model_path}")
                self.model = MaskablePPO.load(model_path)
                self.model.policy.eval() # Mode évaluation pour PyTorch
            except Exception as e:
                print(f"[Hybrid Engine] Attention: Impossible de charger le modèle PPO ({e}). Utilisation du mode heuristique pur.")
        else:
            print("[Hybrid Engine] Aucun modèle RL spécifié ou trouvé. Mode de secours activé.")

    def predict(self, observation, state=None, episode_start=None, deterministic=True, action_masks=None, **kwargs):
        """
        Méthode de prédiction compatible avec l'arborescence et le visualiseur.
        """
        grid = observation["grid"]
        shapes = observation["shapes"]
        
        is_batched = len(grid.shape) == 3
        if is_batched:
            grid = grid[0]
            shapes = shapes[0]
            
        # 1. Extraction des pièces en cours
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

        grid_list = [[int(cell) for cell in row] for row in grid]
        
        # Recherche prospective systématique pour le tour en cours
        candidate_final_states = []
        permutations = list(itertools.permutations(active_shapes))
        self._find_all_paths(grid_list, permutations, 0, None, 0, candidate_final_states)
        
        if not candidate_final_states:
            # En cas de blocage inévitable (Game Over)
            return (np.array([0]), None) if is_batched else (0, None)

        # 2. Élagage et filtrage : on garde les 8 meilleurs plateaux candidats d'après l'heuristique rapide
        candidate_final_states.sort(key=lambda x: x['base_heuristic'], reverse=True)
        top_candidates = candidate_final_states[:8]

        # 3. Échantillonnage Monte Carlo des pièces futures
        random_future_shapes = self._sample_random_shapes(self.sample_size)

        best_score = -float('inf')
        best_first_move = None

        for candidate in top_candidates:
            # A. Évaluation Monte Carlo de la résilience à court terme
            resilience_score = self._evaluate_future_resilience(candidate['grid'], random_future_shapes)
            
            # B. Évaluation par le réseau de neurones PPO (Value Head)
            ppo_value = 0.0
            if self.model is not None:
                ppo_value = self._get_ppo_state_value(candidate['grid'], candidate['lines_cleared'])

            # C. Calcul de la note combinée
            # ppo_value (ajustement de l'intuition) + résilience Monte Carlo + bonus immédiat de lignes complétées
            total_score = ppo_value + (resilience_score * 45.0) + (candidate['lines_cleared'] * 15.0)
            
            if total_score > best_score:
                best_score = total_score
                best_first_move = candidate['first_move']

        if best_first_move is not None:
            shape_idx, r, c = best_first_move
            action = shape_idx * 64 + r * 8 + c
        else:
            action = 0
            
        if is_batched:
            return np.array([action]), None
        return action, None

    def _get_ppo_state_value(self, grid, lines_cleared):
        """
        Convertit l'état simulé sous un format compatible avec PyTorch/SB3
        et interroge le Critic Head du réseau de neurones pour estimer la valeur de l'état.
        """
        # Reconstruction d'une observation compatible avec le réseau de neurones
        # On simule le fait que les 3 pièces ont été jouées (donc le sous-espace 'shapes' est vide)
        simulated_obs = {
            "grid": np.array([grid], dtype=np.int8),
            "shapes": np.zeros((1, 3, 5, 5), dtype=np.int8),
            "score": np.array([[float(lines_cleared * 10)]], dtype=np.float32),
            "combo": np.array([[int(lines_cleared > 0)]], dtype=np.int8)
        }
        
        try:
            with torch.no_grad():
                # Envoi des données sur le device d'évaluation (CPU/GPU) du modèle
                obs_tensor = obs_as_tensor(simulated_obs, device=self.model.policy.device)
                value_tensor = self.model.policy.predict_values(obs_tensor)
                return value_tensor.item()
        except Exception:
            return 0.0

    def _sample_random_shapes(self, n):
        sampled = []
        for _ in range(n):
            form_idx = random.randint(0, len(self.all_possible_forms) - 1)
            var_idx = random.randint(0, len(self.all_possible_forms[form_idx]) - 1)
            sampled.append(self.all_possible_forms[form_idx][var_idx])
        return sampled

    def _evaluate_future_resilience(self, grid, future_shapes):
        possible_placements = 0
        for shape in future_shapes:
            if self._can_fit_somewhere(grid, shape):
                possible_placements += 1
        return possible_placements / len(future_shapes)

    def _can_fit_somewhere(self, grid, shape_form):
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

    def evaluate_grid_health_fast(self, grid):
        """
        Heuristique statique ultra-rapide servant de premier filtre
        pour ne retenir que les plateaux prometteurs.
        """
        block_count = sum(1 for r in range(8) for c in range(8) if grid[r][c])
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
                        
        # Mesure de rugosité des colonnes
        heights = []
        for c in range(8):
            height = 0
            for r in range(8):
                if grid[r][c]:
                    height = 8 - r
                    break
            heights.append(height)
        roughness = sum(abs(heights[i] - heights[i+1]) for i in range(7))

        return -4.0 * block_count - 15.0 * holes - 1.2 * roughness

    def _simulate_placement_fast(self, grid, shape_form, r, c):
        new_grid = [row[:] for row in grid]
        h, w = len(shape_form), len(shape_form[0])
        for i in range(h):
            for j in range(w):
                if shape_form[i][j]:
                    new_grid[r+i][c+j] = 1
                    
        rows_to_clear = [i for i in range(8) if all(new_grid[i])]
        cols_to_clear = [j for j in range(8) if all(new_grid[i][j] for i in range(8))]
        for row in rows_to_clear:
            for col in range(8):
                new_grid[row][col] = 0
        for col in cols_to_clear:
            for row in range(8):
                new_grid[row][col] = 0
        return new_grid, len(rows_to_clear) + len(cols_to_clear)

    def _find_all_paths(self, grid, permutations, perm_idx, first_move, lines_cleared, results):
        if perm_idx >= len(permutations):
            return
            
        current_perm = permutations[perm_idx]
        
        def _backtrack(g, depth, f_move, l_cleared):
            if depth == len(current_perm):
                results.append({
                    'grid': g,
                    'first_move': f_move,
                    'lines_cleared': l_cleared,
                    'base_heuristic': self.evaluate_grid_health_fast(g)
                })
                return

            idx, form = current_perm[depth]
            h, w = len(form), len(form[0])
            for r in range(8 - h + 1):
                for c in range(8 - w + 1):
                    can_place = True
                    for i in range(h):
                        for j in range(w):
                            if form[i][j] and g[r+i][c+j]:
                                can_place = False
                                break
                        if not can_place:
                            break
                    if can_place:
                        next_g, cleared = self._simulate_placement_fast(g, form, r, c)
                        _backtrack(
                            next_g, 
                            depth + 1, 
                            f_move if f_move is not None else (idx, r, c), 
                            l_cleared + cleared
                        )

        _backtrack(grid, 0, first_move, lines_cleared)
        self._find_all_paths(grid, permutations, perm_idx + 1, first_move, lines_cleared, results)