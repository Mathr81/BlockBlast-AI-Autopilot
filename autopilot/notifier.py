"""
notifier.py — Notifications optionnelles Discord / Telegram.

Configuration : éditer le fichier .env à la racine du projet.
Laisser les champs vides pour désactiver silencieusement.

Aucune dépendance externe — utilise urllib + uuid (bibliothèque standard).
Les envois sont non-bloquants (thread daemon).
Les screenshots sont passés en JPEG bytes pré-encodés par l'appelant.
"""
import json
import logging
import os
import threading
import urllib.request
import uuid


def _load_dotenv() -> None:
    """Charge les variables depuis .env (si présent) sans écraser l'environnement."""
    env_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"
    )
    if not os.path.isfile(env_path):
        return
    with open(env_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, val = line.partition("=")
                os.environ.setdefault(key.strip(), val.strip().strip("'\""))


_load_dotenv()

# =====================================================================
# CONFIG — éditer .env plutôt que ce fichier
# =====================================================================
DISCORD_WEBHOOK_URL        = os.getenv("DISCORD_WEBHOOK_URL", "")
TELEGRAM_BOT_TOKEN         = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID           = os.getenv("TELEGRAM_CHAT_ID", "")
NOTIFY_STATS_EVERY_N_TURNS = int(os.getenv("NOTIFY_STATS_EVERY_N_TURNS", "50"))

# =====================================================================
# INTERNALS
# =====================================================================
_discord_ok      = bool(DISCORD_WEBHOOK_URL)
_telegram_ok     = bool(TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID)
_last_stats_turn = 0


def _build_multipart_telegram(fields: dict, photo_bytes: bytes) -> tuple[bytes, str]:
    boundary = uuid.uuid4().hex
    parts = []
    for key, value in fields.items():
        parts.append(
            f'--{boundary}\r\n'
            f'Content-Disposition: form-data; name="{key}"\r\n\r\n'
            f'{value}\r\n'.encode()
        )
    parts.append(
        f'--{boundary}\r\n'
        f'Content-Disposition: form-data; name="photo"; filename="game.jpg"\r\n'
        f'Content-Type: image/jpeg\r\n\r\n'.encode()
        + photo_bytes
        + f'\r\n--{boundary}--\r\n'.encode()
    )
    return b''.join(parts), f'multipart/form-data; boundary={boundary}'


def _build_multipart_discord(message: str, photo_bytes: bytes) -> tuple[bytes, str]:
    boundary = uuid.uuid4().hex
    payload  = json.dumps({"content": message})
    parts = [
        f'--{boundary}\r\n'
        f'Content-Disposition: form-data; name="payload_json"\r\n\r\n'
        f'{payload}\r\n'.encode(),
        f'--{boundary}\r\n'
        f'Content-Disposition: form-data; name="files[0]"; filename="game.jpg"\r\n'
        f'Content-Type: image/jpeg\r\n\r\n'.encode()
        + photo_bytes
        + f'\r\n--{boundary}--\r\n'.encode(),
    ]
    return b''.join(parts), f'multipart/form-data; boundary={boundary}'


def _post(url: str, payload: dict) -> None:
    try:
        data = json.dumps(payload).encode()
        req  = urllib.request.Request(
            url, data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        urllib.request.urlopen(req, timeout=5)
    except Exception as exc:
        logging.debug(f"[Notifier] Erreur envoi JSON: {exc}")


def _post_bytes(url: str, body: bytes, content_type: str) -> None:
    try:
        req = urllib.request.Request(
            url, data=body,
            headers={"Content-Type": content_type},
            method="POST",
        )
        urllib.request.urlopen(req, timeout=15)
    except Exception as exc:
        logging.debug(f"[Notifier] Erreur envoi photo: {exc}")


def _send(message: str, frame_jpg: bytes | None = None) -> None:
    """Envoie message (+ JPEG optionnel) sur tous les canaux. Non-bloquant."""
    if not (_discord_ok or _telegram_ok):
        return

    def _do() -> None:
        if _discord_ok:
            if frame_jpg:
                body, ct = _build_multipart_discord(message, frame_jpg)
                _post_bytes(DISCORD_WEBHOOK_URL, body, ct)
            else:
                _post(DISCORD_WEBHOOK_URL, {"content": message})
        if _telegram_ok:
            if frame_jpg:
                body, ct = _build_multipart_telegram(
                    {"chat_id": TELEGRAM_CHAT_ID, "caption": message, "parse_mode": "Markdown"},
                    frame_jpg,
                )
                _post_bytes(
                    f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto",
                    body, ct,
                )
            else:
                _post(
                    f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
                    {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"},
                )

    threading.Thread(target=_do, daemon=True).start()


# =====================================================================
# API PUBLIQUE
# =====================================================================

def notify_session_start(agent_name: str, frame_jpg: bytes | None = None) -> None:
    _send(f"*BlockBlast Autopilot* demarre\nAgent : *{agent_name}*", frame_jpg)


def notify_game_over(turn: int, max_combo: int, total_clearings: int,
                     elapsed_s: float = 0.0, frame_jpg: bytes | None = None) -> None:
    h, rem = divmod(int(elapsed_s), 3600)
    m, s   = divmod(rem, 60)
    dur    = f"{h:02d}h{m:02d}m{s:02d}s" if h else f"{m:02d}m{s:02d}s"
    _send(
        f"*Game Over* apres *{turn}* tours ({dur})\n"
        f"Combo max : *{max_combo}* | Clearings : *{total_clearings}*",
        frame_jpg,
    )


def notify_all_clear(turn: int, current_combo: int, frame_jpg: bytes | None = None) -> None:
    _send(f"*All Clear !* (Tour {turn} | Combo={current_combo})", frame_jpg)


def notify_periodic_stats(turn: int, max_combo: int, total_clearings: int,
                          elapsed_s: float, frame_jpg: bytes | None = None) -> None:
    """Stats periodiques envoyees toutes les NOTIFY_STATS_EVERY_N_TURNS tours."""
    global _last_stats_turn
    if NOTIFY_STATS_EVERY_N_TURNS <= 0 or turn == 0:
        return
    if turn % NOTIFY_STATS_EVERY_N_TURNS == 0 and turn != _last_stats_turn:
        _last_stats_turn = turn
        h, rem = divmod(int(elapsed_s), 3600)
        m, s   = divmod(rem, 60)
        dur    = f"{h:02d}h{m:02d}m{s:02d}s" if h else f"{m:02d}m{s:02d}s"
        _send(
            f"*Stats — Tour {turn}* ({dur})\n"
            f"Combo max : *{max_combo}* | Clearings : *{total_clearings}*",
            frame_jpg,
        )


def notify_ad_detected(turn: int, frame_jpg: bytes | None = None) -> None:
    _send(
        f"*Pub / interstitiel detecte* (tour {turn})\n"
        f"Action manuelle requise.",
        frame_jpg,
    )
