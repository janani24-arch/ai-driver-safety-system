"""
alert.py — Visual Alert Manager
=================================
Handles all on-screen alert banners and the HUD panel.
Sound is handled separately in sound.py.
"""

import cv2
from config import Config


class AlertManager:
    """Draws alert banners and the HUD overlay on video frames."""

    # ─── Drowsiness Alert ─────────────────────────────────────────────────────

    @staticmethod
    def draw_drowsiness_alert(frame):
        """Red banner — driver is drowsy."""
        h, w = frame.shape[:2]
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 65), (0, 0, 180), -1)
        cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)
        cv2.putText(frame, "!! DROWSINESS ALERT  Wake Up !!",
                    (15, 43), cv2.FONT_HERSHEY_DUPLEX,
                    Config.FONT_SCALE + 0.05, Config.ALERT_COLOR, Config.TEXT_THICKNESS)

    # ─── Yawn Alert ───────────────────────────────────────────────────────────

    @staticmethod
    def draw_yawn_alert(frame):
        """Orange banner — driver is yawning."""
        h, w = frame.shape[:2]
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 70), (w, 130), (0, 100, 200), -1)
        cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)
        cv2.putText(frame, "!! YAWNING DETECTED  Take a Break !!",
                    (15, 113), cv2.FONT_HERSHEY_DUPLEX,
                    Config.FONT_SCALE - 0.05, (0, 165, 255), Config.TEXT_THICKNESS)

    # ─── Phone Distraction Alert ──────────────────────────────────────────────

    @staticmethod
    def draw_phone_alert(frame):
        """Orange-red banner — driver is using phone."""
        h, w = frame.shape[:2]
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 135), (w, 195), (0, 60, 180), -1)
        cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)
        cv2.putText(frame, "!! PHONE DETECTED  Eyes on Road !!",
                    (15, 178), cv2.FONT_HERSHEY_DUPLEX,
                    Config.FONT_SCALE - 0.05, Config.PHONE_COLOR, Config.TEXT_THICKNESS)

    # ─── HUD Panel ────────────────────────────────────────────────────────────

    @staticmethod
    def draw_hud(frame, ear: float, mar: float,
                 blink_count: int, yawn_count: int, phone_count: int,
                 blink_frames: int, yawn_frames: int, phone_frames: int):
        """
        Draw bottom HUD with all live metrics.
        """
        h, w = frame.shape[:2]
        panel_y = h - 90

        # Dark background panel
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, panel_y), (w, h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

        # Divider line
        cv2.line(frame, (0, panel_y), (w, panel_y), (60, 60, 60), 1)

        # ── Column 1: EAR / MAR ──────────────────────────────
        ear_color = Config.ALERT_COLOR if ear < Config.EAR_THRESHOLD else Config.NORMAL_COLOR
        mar_color = (0, 165, 255) if mar > Config.MAR_THRESHOLD else Config.NORMAL_COLOR

        cv2.putText(frame, f"EAR: {ear:.3f}", (10, panel_y + 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, ear_color, 2)
        cv2.putText(frame, f"MAR: {mar:.3f}", (10, panel_y + 58),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, mar_color, 2)

        # ── Column 2: Event Counters ──────────────────────────
        cv2.putText(frame, f"Blinks : {blink_count}", (155, panel_y + 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, Config.INFO_COLOR, 1)
        cv2.putText(frame, f"Yawns  : {yawn_count}", (155, panel_y + 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, Config.INFO_COLOR, 1)
        cv2.putText(frame, f"Phone  : {phone_count}", (155, panel_y + 82),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, Config.PHONE_COLOR, 1)

        # ── Column 3: Frame Counters ──────────────────────────
        cv2.putText(frame, f"Eye closed  : {blink_frames} f", (310, panel_y + 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (180, 180, 180), 1)
        cv2.putText(frame, f"Mouth open  : {yawn_frames} f", (310, panel_y + 52),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (180, 180, 180), 1)
        cv2.putText(frame, f"Phone seen  : {phone_frames} f", (310, panel_y + 76),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (180, 180, 180), 1)

    # ─── Status Badge (top-right) ─────────────────────────────────────────────

    @staticmethod
    def draw_status(frame, drowsy: bool, yawning: bool, phone: bool):
        """Show overall system status badge in top-right corner."""
        h, w = frame.shape[:2]
        if drowsy or yawning or phone:
            status = "!! ALERT !!"
            color  = Config.ALERT_COLOR
        else:
            status = "Monitoring..."
            color  = Config.NORMAL_COLOR
        cv2.putText(frame, status, (w - 230, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.78, color, 2)