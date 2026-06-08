"""
planner.py — Planificateur DFS + beam search.

Contient :
  - simulate_local_placement  : simule un placement et retourne les lignes effacées
  - plan_entire_turn_sequence : DFS sur toutes les permutations de pièces
  - evaluate_grid_with_agent  : dispatcher d'évaluation vers l'agent choisi
  - _quick_rate_fallback      : pre-filtre beam utilisé si l'agent n'en a pas

Ce module est indépendant de la détection OpenCV et du contrôle scrcpy.
"""
import itertools
import numpy as np
from autopilot.config import (
    BEAM_WIDTH_ZEN, BEAM_WIDTH_TACTICAL, BEAM_WIDTH_CRISIS,
    ANDROID_COMBO_WINDOW,
)


# ─────────────────────────────────────────────────────────────────────
# SIMULATION
# ─────────────────────────────────────────────────────────────────────
def simulate_local_placement(grid_list, form, r, c):
    """
    Pose `form` à (r, c) sur une copie de `grid_list`.
    Efface les lignes et colonnes complètes.
    Retourne (nouvelle_grille, nb_lignes_effacées).
    """
    g = [row[:] for row in grid_list]
    h, w = len(form), len(form[0])
    for i in range(h):
        for j in range(w):
            if form[i][j]:
                g[r+i][c+j] = 1
    rc = [i for i in range(8) if all(g[i])]
    cc = [j for j in range(8) if all(g[i][j] for i in range(8))]
    for ri in rc:
        for ci in range(8): g[ri][ci] = 0
    for ci in cc:
        for ri in range(8): g[ri][ci] = 0
    return g, len(rc) + len(cc)


# ─────────────────────────────────────────────────────────────────────
# ÉVALUATION (dispatcher vers l'agent)
# ─────────────────────────────────────────────────────────────────────
def evaluate_grid_with_agent(grid_list, clears_history, agent, agent_name,
                              current_combo=0, mslc_entrant=0, n_pieces=3):
    """
    Appelle la fonction d'évaluation de l'agent avec les bons paramètres.
    Gère les fallbacks pour les agents qui n'ont pas evaluate_grid_ultimate_v4.
    """
    bc = sum(row.count(1) for row in grid_list)
    tc = getattr(agent, "THRESHOLD_CRISIS", 28)
    tz = getattr(agent, "THRESHOLD_ZEN", 10)
    phase = "crisis" if bc > tc else "zen" if bc < tz else "tactical"

    if agent_name == "Tacticien Personnalisé" and hasattr(agent, "evaluate_grid_ultimate_v4"):
        return agent.evaluate_grid_ultimate_v4(
            grid_list, current_combo, clears_history, phase,
            mslc_entrant=mslc_entrant, n_pieces_in_turn=n_pieces)

    lc = sum(clears_history)
    if   agent_name == "Élite DFS":
        return agent.evaluate_grid(grid_list, lc)
    elif agent_name == "Heuristique":
        return agent.evaluate_grid(grid_list) + lc * 25.0
    elif agent_name == "Monte Carlo":
        return agent.evaluate_grid_health(grid_list) + lc * 25.0
    elif agent_name == "Hybride AlphaZero":
        return (agent._get_ppo_state_value(grid_list, lc)
                + agent.evaluate_grid_health_fast(grid_list)
                + lc * 15.0)
    return -5.0 * bc + lc * 25.0


# ─────────────────────────────────────────────────────────────────────
# PRE-FILTRE BEAM (fallback si l'agent n'a pas quick_rate)
# ─────────────────────────────────────────────────────────────────────
def _quick_rate_fallback(g, cleared, **kw):
    blocks = holes = 0
    for r in range(8):
        blocks += g[r].count(1)
        for c in range(8):
            if g[r][c] == 0:
                if ((r == 0 or g[r-1][c]) and (r == 7 or g[r+1][c]) and
                        (c == 0 or g[r][c-1]) and (c == 7 or g[r][c+1])):
                    holes += 1
    return -4.0*blocks - 70.0*holes + 35.0*cleared


# ─────────────────────────────────────────────────────────────────────
# PLANIFICATEUR PRINCIPAL
# ─────────────────────────────────────────────────────────────────────
def plan_entire_turn_sequence(grid, shapes, agent, agent_name,
                              current_combo=0, mslc_entrant=0):
    """
    DFS + beam search sur toutes les permutations de pièces actives.

    Paramètres :
        grid          : np.ndarray (8,8) ou liste 8×8
        shapes        : liste de 3 np.ndarray (5,5)
        agent         : instance de l'agent
        agent_name    : str (pour le dispatcher d'évaluation)
        current_combo : cumul de lignes effacées consécutivement (+=cleared par pièce)
        mslc_entrant  : blocs posés sans effacer en entrée du tour

    Retourne :
        Liste de tuples (shape_idx, row, col, form) dans l'ordre d'exécution.
        Liste vide si aucun placement possible.

    Modèle combo Android :
        - mslc est incrémenté de 1 par pièce posée
        - remis à 0 si la pièce efface au moins 1 ligne
        - combo brisé si mslc atteint ANDROID_COMBO_WINDOW (4)
        - le planificateur pousse les coups d'urgence (cleared > 0 quand mslc critique)
          en tête du beam avec un bonus +15000

    Vitesse :
        - beam_width adaptatif selon la densité de la grille (12/18/22)
        - beam élargi à tactical+8 si mslc_entrant >= 2 et combo actif
        - quick_rate de l'agent (~10µs) utilisé si disponible, sinon fallback
    """
    # Extraire les formes actives
    active = []
    for idx, s5 in enumerate(shapes):
        if np.any(s5):
            ar = np.any(s5, axis=1); ac = np.any(s5, axis=0)
            mr = np.where(ar)[0][-1] + 1; mc = np.where(ac)[0][-1] + 1
            active.append((idx, s5[:mr, :mc].tolist()))
    if not active:
        return []

    grid_list = [[int(c) for c in row] for row in grid]
    n_pieces  = len(active)

    # Beam width selon phase
    bc = sum(row.count(1) for row in grid_list)
    tc = getattr(agent, "THRESHOLD_CRISIS", 28)
    tz = getattr(agent, "THRESHOLD_ZEN",    10)
    if bc > tc:   bw = BEAM_WIDTH_CRISIS
    elif bc < tz: bw = BEAM_WIDTH_ZEN
    else:         bw = BEAM_WIDTH_TACTICAL

    # Urgence combo → beam élargi
    if current_combo > 0 and mslc_entrant >= 2:
        bw = max(bw, BEAM_WIDTH_TACTICAL + 8)

    # Sélection du pre-filtre
    _qr = agent.quick_rate if hasattr(agent, "quick_rate") else _quick_rate_fallback

    best_seq   = []
    best_score = -float('inf')
    max_depth  = 0
    perms      = list(itertools.permutations(active))

    def backtrack(g, perm, depth, seq, ch, mslc_sim):
        nonlocal best_seq, best_score, max_depth

        if depth == len(perm):
            score = evaluate_grid_with_agent(
                g, ch, agent, agent_name,
                current_combo=current_combo,
                mslc_entrant=mslc_entrant,
                n_pieces=n_pieces)
            if score > best_score:
                best_score = score
                best_seq   = list(seq)
                max_depth  = depth
            return

        idx, form = perm[depth]
        h, w = len(form), len(form[0])
        mslc_after = mslc_sim + 1
        urgent = (mslc_after >= ANDROID_COMBO_WINDOW and current_combo > 0)

        cands = []
        for r in range(8 - h + 1):
            for c in range(8 - w + 1):
                if all(not (form[i][j] and g[r+i][c+j])
                       for i in range(h) for j in range(w)):
                    ng, cleared = simulate_local_placement(g, form, r, c)
                    new_mslc = 0 if cleared > 0 else mslc_after
                    q = _qr(ng, cleared, mslc_after=new_mslc)
                    # Coup salvateur en urgence : priorité absolue dans le beam
                    if urgent and cleared > 0:
                        q += 15000.0
                    cands.append((q, r, c, ng, cleared, new_mslc))

        if not cands:
            penalty = -100_000.0 - 12_000.0 * (len(perm) - depth)
            if depth > max_depth or (depth == max_depth and penalty > best_score):
                best_score = penalty
                best_seq   = list(seq)
                max_depth  = depth
            return

        cands.sort(key=lambda x: x[0], reverse=True)
        for _, r, c, ng, cleared, new_mslc in cands[:bw]:
            seq.append((idx, r, c, form))
            ch.append(cleared)
            backtrack(ng, perm, depth + 1, seq, ch, new_mslc)
            ch.pop()
            seq.pop()

    for perm in perms:
        backtrack(grid_list, perm, 0, [], [], mslc_entrant)

    return best_seq