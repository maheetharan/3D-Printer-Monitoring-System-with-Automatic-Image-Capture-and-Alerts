import os
import requests
from datetime import datetime

# ========== CONFIGURE THESE ==========
TELEGRAM_TOKEN = "YOUR_BOT_TOKEN"          # e.g. "123456:ABC-DEF..."
TELEGRAM_CHAT_ID = "YOUR_CHAT_ID"           # e.g. "123456789"
ENABLE_TELEGRAM = True                     # set to False to disable
# =====================================

class AlertManager:
    def __init__(self, log_file="alerts.log"):
        self.log_file = log_file
        self.clog_active = False
        self.leak_active = False
        self.stop_active = False
        self.combo_active = False

    def _send_telegram(self, text, photo_paths=None):
        """Send a message + (optionally) photos to Telegram."""
        if not ENABLE_TELEGRAM:
            return

        # 1. Send text message
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        payload = {"chat_id": TELEGRAM_CHAT_ID, "text": text}
        try:
            requests.post(url, data=payload, timeout=10)
        except Exception as e:
            print(f"  [Telegram] Failed to send message: {e}")

        # 2. Send photos (up to 5)
        if photo_paths:
            for path in photo_paths[:5]:  # max 5 photos per alert
                if os.path.exists(path):
                    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto"
                    try:
                        with open(path, "rb") as img:
                            requests.post(url, data={"chat_id": TELEGRAM_CHAT_ID},
                                          files={"photo": img}, timeout=10)
                    except Exception as e:
                        print(f"  [Telegram] Failed to send photo {path}: {e}")

    def _log_and_print(self, msg):
        print(msg)
        with open(self.log_file, "a") as f:
            f.write(msg)

    def _format_photos(self, photo_paths):
        if not photo_paths:
            return "N/A"
        return "\n    ".join([os.path.basename(p) for p in photo_paths])

    def send_clog_alert(self, photo_paths=None):
        if self.clog_active:
            return
        self.clog_active = True
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        photos_str = self._format_photos(photo_paths) if photo_paths else "N/A"
        msg = (
            "🔴 NOZZLE CLOGGED - NO MATERIAL FLOWING\n"
            f"Time: {ts}\n"
            f"Photos: {photos_str}\n\n"
            "Action: PAUSE PRINT → Clean/replace nozzle → Resume."
        )
        self._log_and_print(msg)
        self._send_telegram(msg, photo_paths)

    def send_leak_alert(self, photo_paths=None):
        if self.leak_active:
            return
        self.leak_active = True
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        photos_str = self._format_photos(photo_paths) if photo_paths else "N/A"
        msg = (
            "🟠 MATERIAL LEAKING FROM NOZZLE\n"
            f"Time: {ts}\n"
            f"Photos: {photos_str}\n\n"
            "Action: PAUSE PRINT → Check nozzle seal → Clean drip → Resume."
        )
        self._log_and_print(msg)
        self._send_telegram(msg, photo_paths)

    def send_clog_leak_alert(self, photo_paths=None):
        if self.combo_active:
            return
        self.combo_active = True
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        photos_str = self._format_photos(photo_paths) if photo_paths else "N/A"
        msg = (
            "🔴🔴 CRITICAL: CLOGGED + LEAKING!\n"
            f"Time: {ts}\n"
            f"Photos: {photos_str}\n\n"
            "Action: STOP PRINT IMMEDIATELY → Replace nozzle → Check system."
        )
        self._log_and_print(msg)
        self._send_telegram(msg, photo_paths)

    def send_stop_alert(self, photo_paths=None):
        if self.stop_active:
            return
        self.stop_active = True
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        photos_str = self._format_photos(photo_paths) if photo_paths else "N/A"
        msg = (
            "🟡 Printer has STOPPED moving.\n"
            f"Time: {ts}\n"
            f"Photos: {photos_str}\n\n"
            "Check if print finished, paused, or mechanical failure."
        )
        self._log_and_print(msg)
        self._send_telegram(msg, photo_paths)

    def reset_all(self):
        if self.clog_active: print("  ✅ Clog cleared.")
        if self.leak_active: print("  ✅ Leak stopped.")
        if self.stop_active: print("  ✅ Printer resumed.")
        if self.combo_active: print("  ✅ Critical condition cleared.")
        self.clog_active = False
        self.leak_active = False
        self.stop_active = False
        self.combo_active = False