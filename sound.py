"""
sound.py — Cross-Platform Sound Alert Manager
===============================================
Uses Windows winsound (built-in, no install needed).
Falls back to a silent no-op on Linux/Mac.
"""

import threading
import sys
from config import Config


class SoundAlert:
    """
    Non-blocking sound alerts for each alert type.
    Uses Windows built-in winsound — no extra files or libraries needed.
    """

    def __init__(self):
        self._playing = False
        self._enabled = Config.ENABLE_SOUND and sys.platform == "win32"

        if Config.ENABLE_SOUND and sys.platform != "win32":
            print("[SoundAlert] Sound only supported on Windows. Running silently.")

    def _beep(self, freq: int, duration: int):
        """Play a beep in a background thread so it doesn't block video."""
        if not self._enabled or self._playing:
            return

        def _play():
            self._playing = True
            try:
                import winsound
                winsound.Beep(freq, duration)
            except Exception:
                pass
            self._playing = False

        threading.Thread(target=_play, daemon=True).start()

    def alert_drowsy(self):
        """High-pitched beep for drowsiness."""
        self._beep(Config.DROWSY_BEEP_FREQ, Config.BEEP_DURATION_MS)

    def alert_yawn(self):
        """Medium-pitched beep for yawning."""
        self._beep(Config.YAWN_BEEP_FREQ, Config.BEEP_DURATION_MS)

    def alert_phone(self):
        """Low-pitched beep for phone distraction."""
        self._beep(Config.PHONE_BEEP_FREQ, Config.BEEP_DURATION_MS)