"""
run_autopilot.py — Orchestre la boucle principale de l'autopilote Block Blast.

Structure du projet :
    run_autopilot.py        ← CE FICHIER (boucle principale, rarement modifié)
    autopilot/
        config.py           ← TOUS les paramètres à ajuster (écran, timing, beam, combo)
        vision.py           ← Extraction OpenCV (grille + pièces)
        planner.py          ← Planificateur DFS + beam search
        control.py          ← Contrôle scrcpy (drag, tap, revive, skin)
    agents/
        master_tactician_agent.py  ← Agent principal (heuristiques, évaluation)
"""
import os
import sys
import time
import cv2
import numpy as np
import pygame
import logging

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from autopilot.config import (
    ENABLE_LOGGING, LOG_FILENAME,
    GRID_RESCAN_FREQUENCY,
    ANIMATION_DELAY_ALL_CLEAR, ANIMATION_DELAY_CLEARED, ANIMATION_DELAY_NORMAL,
    MAX_DETECTION_RETRIES, DETECTION_RETRY_DELAY,
    PREVIEW_DELAY, DELAY_AFTER_SWIPE, SHORT_INTER_MOVE_DELAY,
    GRID_X, GRID_Y, GRID_W, GRID_H,
    ANDROID_COMBO_WINDOW,
)
from autopilot.vision import (
    capture_and_extract, shapes_are_valid,
    get_trimmed_shape_form, extract_grid,
)
from autopilot.planner import plan_entire_turn_sequence, simulate_local_placement
from autopilot.control import (
    check_revive, handle_revive, check_skin, reset_skin,
    execute_move,
)
from blockblast_game.game_env import BlockGameEnv

LOG_FILE_PATH = os.path.join(SCRIPT_DIR, LOG_FILENAME)


# ── Logger ───────────────────────────────────────────────────────────
def setup_logger():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers = []
    fmt = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s',
                             datefmt='%Y-%m-%d %H:%M:%S')
    if ENABLE_LOGGING:
        fh = logging.FileHandler(LOG_FILE_PATH, mode='w', encoding='utf-8')
        fh.setFormatter(fmt); logger.addHandler(fh)
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(logging.Formatter('%(message)s')); logger.addHandler(ch)


# ── Pygame ───────────────────────────────────────────────────────────
class DummyShape:
    def __init__(self, form): self.form = form; self.color = (252, 48, 28)

def sync_virtual_env_state(env, grid, shapes, preview_data=None):
    pg = [[(62, 181, 208) if c else 0 for c in row] for row in grid]
    p_idx = -1
    if preview_data:
        sf, pr, pc, p_idx = preview_data
        for i, row_f in enumerate(sf):
            for j, cell in enumerate(row_f):
                if cell and 0 <= pr+i < 8 and 0 <= pc+j < 8:
                    pg[pr+i][pc+j] = (255, 191, 0)
    env.game_state.grid = pg
    env.game_state.current_shapes = []
    for si, s5 in enumerate(shapes):
        if preview_data and si == p_idx:
            env.game_state.current_shapes.append(0); continue
        if np.any(s5):
            ar = np.any(s5, axis=1); ac = np.any(s5, axis=0)
            env.game_state.current_shapes.append(
                DummyShape(s5[:np.where(ar)[0][-1]+1,
                             :np.where(ac)[0][-1]+1].tolist()))
        else:
            env.game_state.current_shapes.append(0)

def update_title(agent_name, paused, combo, mslc):
    try:
        st = "PAUSE" if paused else "RUN"
        pygame.display.set_caption(
            f"BlockBlast v8 [{agent_name}] {st} | Combo={combo} MSLC={mslc}")
    except Exception: pass


# ── Sélection agent ──────────────────────────────────────────────────
def select_agent():
    os.system('cls' if os.name == 'nt' else 'clear')
    print("=" * 60)
    print("  BLOCK BLAST AUTOPILOT v8")
    print("=" * 60)
    print("1. Agent Heuristique")
    print("2. Agent Monte Carlo")
    print("3. Agent Hybride AlphaZero")
    print("4. Agent Elite DFS")
    print("5. Agent Tacticien v8  [RECOMMANDE]")
    print("=" * 60)
    ppo = os.path.join(SCRIPT_DIR, "agents", "models", "final_masked_ppo_model.zip")
    while True:
        c = input("\nVotre choix (1-5) : ").strip()
        if c == '1':
            from agents.heuristic_agent import HeuristicSearchAgent
            return HeuristicSearchAgent(), "Heuristique"
        elif c == '2':
            from agents.monte_carlo_agent import MonteCarloExpectimaxAgent
            return MonteCarloExpectimaxAgent(sample_size=12), "Monte Carlo"
        elif c == '3':
            from agents.hybrid_alphazero_agent import HybridAlphaZeroAgent
            return HybridAlphaZeroAgent(model_path=ppo, sample_size=12), "Hybride AlphaZero"
        elif c == '4':
            from agents.elite_search_agent import EliteSearchAgent
            return EliteSearchAgent(), "Elite DFS"
        elif c == '5':
            from agents.master_tactician_agent import MasterTacticianAgent
            return MasterTacticianAgent(sample_size=16), "Tacticien Personnalisé"
        else:
            print("Choix invalide.")


# ── Utilitaires ──────────────────────────────────────────────────────
def log_state(turn, grid, shapes, combo, mslc):
    gs = "".join("  " + " ".join("■" if c else "·" for c in row) + "\n" for row in grid)
    ss = ""
    for i, s in enumerate(shapes):
        if np.any(s):
            f = get_trimmed_shape_form(s)
            ss += f"  Slot {i+1}:\n" + "".join(
                f"    {' '.join('■' if c else '·' for c in r)}\n" for r in f)
        else:
            ss += f"  Slot {i+1}: [Vide]\n"
    logging.info(f"\n--- TOUR {turn} | Combo={combo} | MSLC={mslc} ---\n"
                 f"Grille:\n{gs}Pieces:\n{ss}" + "-"*40)

def log_defeat(grid, shapes, active_shapes):
    gs = "".join("  " + " ".join("■" if c else "·" for c in r) + "\n" for r in grid)
    lines = ["\n" + "="*50 + " GAME OVER " + "="*50, gs]
    all_blocked = True
    for si, form in active_shapes:
        h, w = len(form), len(form[0])
        fits = any(all(not (form[i][j] and grid[r+i][c+j])
                       for i in range(h) for j in range(w))
                   for r in range(8-h+1) for c in range(8-w+1))
        if fits: all_blocked = False
        lines.append(f"  Slot {si+1}: {'BLOQUEE' if not fits else 'Inserable seule'}")
    lines.append("[Conclusion] " + (
        "Blocage absolu." if all_blocked and active_shapes else "Blocage sequentiel."))
    logging.info("\n".join(lines))

def log_revive_pause():
    logging.info("\n" + "="*80)
    logging.info("[PAUSE] PUB REVIVE. Fermez-la puis ESPACE/P.")
    logging.info("="*80 + "\n")

def smart_sleep(client, delay, agent_name):
    """Attend en surveillant Revive et les touches Pygame."""
    start = time.perf_counter()
    while time.perf_counter() - start < delay:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT: raise KeyboardInterrupt
            if ev.type == pygame.KEYDOWN:
                if ev.key in (pygame.K_p, pygame.K_SPACE) or ev.unicode.lower() == 'p':
                    return "PAUSED"
        raw = client.last_frame
        if raw is not None:
            sc = cv2.resize(raw, (1084, 2412))
            if check_revive(sc):
                logging.info("[REVIVE] Detecte pendant l'attente !")
                handle_revive(client); log_revive_pause(); return "REVIVED"
        time.sleep(0.15)
    return "NORMAL"


# ── MAIN ─────────────────────────────────────────────────────────────
def main():
    setup_logger()
    agent, agent_name = select_agent()

    logging.info("[*] Connexion ADB / Scrcpy...")
    from adbutils import adb
    import scrcpy
    devices = adb.device_list()
    if not devices: logging.error("[!] Pas d'appareil Android."); return
    client = scrcpy.Client(device=devices[0], max_fps=30)
    client.start(threaded=True)
    retries = 30
    while client.last_frame is None and retries > 0:
        time.sleep(0.1); retries -= 1
    if client.last_frame is None:
        logging.error("[!] Pas de flux video."); client.stop(); return

    logging.info(f"[+] {client.device_name} {client.resolution} | Agent: {agent_name}")
    logging.info("  R = Re-scan | ESPACE/P = Pause\n")

    env = BlockGameEnv(render_mode="human")
    env.reset(); time.sleep(1.0)

    turn = 0; persisted_grid = None; manual_rescan = False; is_paused = False
    last_had_clears = False; last_had_all_clear = False
    current_combo = 0  # cumul de LIGNES effacees consecutivement (+=cleared par piece)
    mslc = 0           # blocs poses sans effacer, transmis entre tours

    try:
        while True:
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT: raise KeyboardInterrupt
                if ev.type == pygame.KEYDOWN:
                    if ev.key == pygame.K_r or ev.unicode.lower() == 'r':
                        manual_rescan = True; logging.info("[!] Re-scan force.")
                    if ev.key in (pygame.K_p, pygame.K_SPACE) or ev.unicode.lower() == 'p':
                        is_paused = not is_paused
                        logging.info("[PAUSE]" if is_paused else "[REPRISE]")

            update_title(agent_name, is_paused, current_combo, mslc)
            if is_paused: time.sleep(0.1); continue

            should_scan = (persisted_grid is None or manual_rescan or
                           (GRID_RESCAN_FREQUENCY > 0 and turn > 0
                            and turn % GRID_RESCAN_FREQUENCY == 0))

            if should_scan and turn > 1:
                delay = (ANIMATION_DELAY_ALL_CLEAR if last_had_all_clear
                         else ANIMATION_DELAY_CLEARED if last_had_clears
                         else ANIMATION_DELAY_NORMAL)
                res = smart_sleep(client, delay, agent_name)
                if res == "PAUSED": is_paused = True; continue
                elif res == "REVIVED":
                    is_paused = True; env.reset(); persisted_grid = None
                    current_combo = 0; mslc = 0; continue

            screenshot = shapes = shape_centers = None; ok = False

            for _ in range(MAX_DETECTION_RETRIES):
                raw = client.last_frame
                if raw is None: time.sleep(DETECTION_RETRY_DELAY); continue
                screenshot = cv2.resize(raw, (1084, 2412))

                if check_revive(screenshot):
                    logging.info("[REVIVE] Detecte !")
                    handle_revive(client); is_paused = True; log_revive_pause()
                    env.reset(); persisted_grid = None
                    current_combo = 0; mslc = 0; break

                if check_skin(screenshot):
                    logging.info("[!] Skin alternatif -> reset.")
                    reset_skin(client)
                    raw = client.last_frame
                    if raw is not None:
                        screenshot = cv2.resize(raw, (1084, 2412))

                _, shapes, shape_centers = capture_and_extract(screenshot)
                if not shapes_are_valid(shapes):
                    time.sleep(DETECTION_RETRY_DELAY); continue
                ok = True; break

            if is_paused: continue
            if not ok: logging.info("[*] Attente distribution valide..."); time.sleep(0.5); continue

            if should_scan:
                grid_crop = screenshot[GRID_Y:GRID_Y+GRID_H, GRID_X:GRID_X+GRID_W]
                grid = extract_grid(grid_crop)
                persisted_grid = grid; manual_rescan = False
                logging.info("[*] Grille synchronisee.")
            else:
                grid = persisted_grid

            turn += 1
            log_state(turn, grid, shapes, current_combo, mslc)
            sync_virtual_env_state(env, grid, shapes); env.render()

            t0 = time.perf_counter()
            planned = plan_entire_turn_sequence(
                grid, shapes, agent, agent_name,
                current_combo=current_combo, mslc_entrant=mslc)
            elapsed = time.perf_counter() - t0
            logging.info(f"[+] Plan: {len(planned)} coup(s) en {elapsed:.3f}s "
                         f"| Combo={current_combo} MSLC={mslc}")

            if not planned:
                logging.warning("[!] Aucun coup — recuperation...")
                rev_pau = False
                for _ in range(3):
                    res = smart_sleep(client, 1.2, agent_name)
                    if res in ("REVIVED", "PAUSED"):
                        is_paused = True
                        if res == "REVIVED":
                            env.reset(); persisted_grid = None
                            current_combo = 0; mslc = 0
                        rev_pau = True; break
                    raw = client.last_frame
                    if raw is None: continue
                    sc2 = cv2.resize(raw, (1084, 2412))
                    _, shapes2, centers2 = capture_and_extract(sc2)
                    grid2 = extract_grid(sc2[GRID_Y:GRID_Y+GRID_H, GRID_X:GRID_X+GRID_W])
                    if not shapes_are_valid(shapes2): continue
                    planned = plan_entire_turn_sequence(
                        grid2, shapes2, agent, agent_name, current_combo, mslc)
                    if planned:
                        grid = grid2; shapes = shapes2; shape_centers = centers2
                        persisted_grid = grid2
                        logging.info("[+] Plan de secours trouve !"); break
                if rev_pau: continue
                if not planned:
                    raw = client.last_frame
                    if raw is not None and check_revive(cv2.resize(raw, (1084, 2412))):
                        handle_revive(client); is_paused = True; log_revive_pause()
                        env.reset(); persisted_grid = None
                        current_combo = 0; mslc = 0; continue
                    active = [(i, get_trimmed_shape_form(s))
                              for i, s in enumerate(shapes) if np.any(s)]
                    log_defeat(grid, shapes, active)
                    logging.error("[-] Defaite."); break

            sim_grid = [row[:] for row in grid]
            turn_had_clears = False; turn_had_all_clear = False
            mslc_sim = mslc

            for step, (shape_idx, row, col, form) in enumerate(planned):
                sync_virtual_env_state(env, sim_grid, shapes,
                                       preview_data=(form, row, col, shape_idx))
                env.render()
                if PREVIEW_DELAY > 0: time.sleep(PREVIEW_DELAY)

                logging.info(f"  -> {step+1}/{len(planned)}: "
                             f"Slot{shape_idx+1}->({row},{col}) MSLC_sim={mslc_sim}")
                execute_move(client, shape_idx, row, col, form, shape_centers)

                sim_grid, cleared = simulate_local_placement(sim_grid, form, row, col)

                mslc_sim += 1
                if cleared > 0:
                    mslc_sim = 0; turn_had_clears = True
                    current_combo += cleared   # += lignes effacees (pas += 1)
                    logging.info(f"  [OK] {cleared} ligne(s) -> Combo={current_combo}")
                else:
                    if mslc_sim >= ANDROID_COMBO_WINDOW and current_combo > 0:
                        logging.info(f"  [X] Combo {current_combo} brise (MSLC={mslc_sim})")
                        current_combo = 0

                is_all_clear = all(cell == 0 for r in sim_grid for cell in r)
                if is_all_clear: turn_had_all_clear = True; logging.info("  [All Clear!]")

                shapes[shape_idx] = np.zeros((5, 5), dtype=np.int8)
                sync_virtual_env_state(env, sim_grid, shapes); env.render()

                if step < len(planned) - 1:
                    if is_all_clear:  time.sleep(ANIMATION_DELAY_ALL_CLEAR)
                    elif cleared > 0: time.sleep(0.6)
                    else:             time.sleep(SHORT_INTER_MOVE_DELAY)

            mslc = mslc_sim
            if mslc >= ANDROID_COMBO_WINDOW and current_combo > 0:
                logging.info(f"  [X] Combo {current_combo} brise en fin de tour (MSLC={mslc})")
                current_combo = 0

            last_had_clears = turn_had_clears; last_had_all_clear = turn_had_all_clear
            persisted_grid = np.array(sim_grid, dtype=np.int8)
            logging.info(f"[=] Fin tour {turn} | Combo={current_combo} | MSLC={mslc}")
            time.sleep(DELAY_AFTER_SWIPE if DELAY_AFTER_SWIPE > 0 else 0.1)

    except KeyboardInterrupt:
        logging.info("\n[-] Arrete.")
    finally:
        if 'client' in locals() and client: client.stop()
        env.close()


if __name__ == "__main__":
    main()