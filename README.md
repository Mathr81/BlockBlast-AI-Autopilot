# BlockBlast AI Autopilot

An AI autopilot that plays **Block Blast** autonomously on your Android phone via ADB and scrcpy. It captures the screen in real-time, detects the game grid and pieces using OpenCV, plans the best move sequence with various AI agents, and executes them via touch simulation.

## Features

- **Real-time screen capture** via scrcpy (30 fps)
- **OpenCV grid & piece detection** — robust to animations, color themes, and combos
- **5 AI agents** with different strategies (heuristic, Monte Carlo, AlphaZero hybrid, elite DFS, master tactician)
- **Combo & MSLC tracking** — mirrors Block Blast's internal combo system
- **Grid validation** — detects vision drift between turns and re-captures automatically
- **Auto-revive handling** — detects and clicks through the revive screen automatically
- **Skin detection** — resets to the default theme if an alternate skin is detected
- **Pygame live visualization** — mirrors the game state with move preview in real-time
- **Discord / Telegram notifications** — optional, with screenshots attached (see `.env` below)
- **CLI arguments** — skip the interactive menu, disable Pygame, adjust timing presets
- **Session stats** — printed at shutdown and sent as a notification summary

## Requirements

- Android phone with **Developer Options** and **USB Debugging** enabled
- [scrcpy](https://github.com/Genymobile/scrcpy) installed and in PATH
- Python 3.10+
- USB cable or ADB over Wi-Fi

## Installation

```bash
git clone https://github.com/Mathr81/BlockBlast-AI-Autopilot.git
cd BlockBlast-AI-Autopilot

python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux

pip install -r requirements.txt
```

## Setup & Calibration

Before the first run, calibrate the screen coordinates for your device:

```bash
# 1. Find the exact grid and piece-slot coordinates on your screen
python calibration/get_coords.py

# 2. Test that the grid is correctly extracted with OpenCV
python calibration/calibrate_opencv.py

# 3. Test that drag gestures land in the right place
python calibration/calibrate_drag.py
```

Then update **`autopilot/config.py`** with the values you found. This is the **only file you need to edit** to support a new device or screen resolution.

Key parameters in `config.py`:

| Parameter | Description |
|-----------|-------------|
| `GRID_X/Y/W/H` | Bounding box of the 8×8 grid on screen |
| `SLOT_X_POSITIONS` | X coordinates of the 3 piece slots |
| `DRAG_GAIN_X/Y` | Scaling factors for drag gestures |
| `GRID_BRIGHTNESS_THRESHOLD` | OpenCV threshold to distinguish filled vs. empty cells |
| `BEAM_WIDTH_*` | Beam search width per game phase (zen / tactical / crisis) |
| `GRID_VALIDATION_MAX_MISMATCH` | Max cell drift before a re-capture is triggered (0 = off) |

## Running the Autopilot

Connect your phone, open Block Blast, then run:

```bash
python run_autopilot.py
```

### CLI arguments

```
python run_autopilot.py [--agent 1-5] [--no-pygame] [--speed fast|normal|slow]
```

| Argument | Description |
|----------|-------------|
| `--agent 1-5` | Skip the interactive menu and load an agent directly |
| `--no-pygame` | Run without the Pygame visualization window (headless) |
| `--speed fast` | Reduced animation delays (~2× faster turns) |
| `--speed slow` | Extended delays, safer on older/slower devices |

**Interactive agent menu (default when `--agent` is omitted):**

```
============================================================
  BLOCK BLAST AUTOPILOT v8
============================================================
1. Agent Heuristique
2. Agent Monte Carlo
3. Agent Hybride AlphaZero
4. Agent Elite DFS
5. Agent Tacticien v8  [RECOMMANDE]
============================================================
```

**Controls (Pygame window):**

| Key | Action |
|-----|--------|
| `SPACE` / `P` | Pause / Resume |
| `R` | Force a full grid re-scan |
| Close window | Stop the autopilot |

## Notifications (Discord / Telegram)

Copy `.env.example` to `.env` at the project root and fill in your credentials.  
All fields are optional — leave them empty to disable that channel silently.

```env
# Discord : webhook URL from channel Settings → Integrations → Webhooks
DISCORD_WEBHOOK_URL=

# Telegram : bot token from @BotFather, chat_id from @getidsbot
TELEGRAM_BOT_TOKEN=
TELEGRAM_CHAT_ID=

# Send a stats summary every N turns (0 = disable)
NOTIFY_STATS_EVERY_N_TURNS=20
```

Notification events:

| Event | Content |
|-------|---------|
| Session start | Agent name |
| **Revive detected** | Screenshot of the revive screen — action required |
| All Clear | Screenshot captured right after the last piece lands |
| Periodic stats (every N turns) | Turn, duration, combo current/max, clearings/turn, MSLC |
| Ad / interstitial | Screenshot — manual action required |
| Game over | Turn count, duration, max combo, total clearings + screenshot |

> **Note:** `.env` is gitignored and never committed. Notifications use only Python stdlib (`urllib`, `uuid`) — no external packages required.

## Agents

| Agent | File | Strategy |
|-------|------|----------|
| **Heuristique** | `agents/heuristic_agent.py` | Fast beam search + flood-fill topology analysis |
| **Monte Carlo** | `agents/monte_carlo_agent.py` | Expectimax with sampled future piece distributions |
| **Hybride AlphaZero** | `agents/hybrid_alphazero_agent.py` | DFS + trained MaskablePPO value critic |
| **Elite DFS** | `agents/elite_search_agent.py` | DFS scored by connectivity and killer-shape avoidance |
| **Tacticien v8** ⭐ | `agents/master_tactician_agent.py` | Full heuristic suite: density, hole penalty, topology, combo/MSLC-aware |

> **Recommended:** Agent Tacticien v8 — combines the strongest heuristics with combo and MSLC awareness to maximize long streaks.

## Project Structure

```
BlockBlast-AI-Autopilot/
├── run_autopilot.py                    # Main entry point — autopilot loop
├── .env                                # Notification credentials (gitignored, create from .env.example)
├── autopilot/
│   ├── config.py                       # ← Edit this to calibrate your device
│   ├── vision.py                       # OpenCV screen capture & grid/piece extraction
│   ├── planner.py                      # DFS + beam search planning engine
│   ├── control.py                      # scrcpy touch control (drag, tap, revive)
│   └── notifier.py                     # Optional Discord/Telegram notifications
├── agents/
│   ├── master_tactician_agent.py       # Recommended agent
│   ├── heuristic_agent.py
│   ├── monte_carlo_agent.py
│   ├── hybrid_alphazero_agent.py
│   ├── elite_search_agent.py
│   └── models/
│       └── final_masked_ppo_model.zip  # Pre-trained PPO model for HybridAlphaZero
├── blockblast_game/                    # Game engine (Pygame visualization + state logic)
│   ├── game_env.py
│   ├── game_renderer.py
│   ├── game_state.py
│   └── Assets/
└── calibration/
    ├── get_coords.py                   # Interactive coordinate finder
    ├── calibrate_opencv.py             # Test grid extraction
    └── calibrate_drag.py              # Test drag gestures
```

## Troubleshooting

**No Android device detected**
- Make sure USB Debugging is enabled and the device is authorized (`adb devices`)
- Try `adb kill-server && adb start-server`

**Grid not detected / wrong coordinates**
- Run `calibration/get_coords.py` to re-measure grid and slot coordinates
- Adjust `GRID_BRIGHTNESS_THRESHOLD` in `config.py` if cells are misclassified

**Drags land in the wrong place**
- Run `calibration/calibrate_drag.py` and adjust `DRAG_GAIN_X/Y` in `config.py`

**Revive screen not dismissed**
- Adjust `REVIVE_PX1_X/Y` and `REVIVE_CLICK_X/Y` in `config.py`

**Antivirus flags `notifier.py`**
- This is a false positive: the file sends HTTP notifications using only stdlib, with no screen capture of its own. Add the project folder to your AV exclusions.

## License

MIT
