import cv2
import numpy as np
from collections import deque


class TripleZoneDetector:
    """
    Three-zone detection system:
      Zone A: Printer head movement
      Zone B: Extrusion gap (normal flow)
      Zone L: Nozzle body (leak/drip detection)
    
    This correctly distinguishes:
      - Normal working (head moves + extrusion flows + nozzle clean)
      - Clogged (head moves + no extrusion + nozzle clean)
      - Leaking (head moves + nozzle body shows drips forming)
      - Clogged + leaking (head moves + no extrusion + nozzle dripping)
    """

    def __init__(self,
                 threshold=15,
                 blur_kernel=(5, 5),
                 head_stop_score=0.002,
                 extrusion_stop_score=0.005,
                 leak_score=0.003,          # Score above this on nozzle body = leak forming
                 clog_confirmation=4.0,
                 leak_confirmation=3.0,      # Seconds of leak before alert
                 stop_confirmation=3.0):
        
        self.threshold = threshold
        self.blur_kernel = blur_kernel
        self.head_stop_score = head_stop_score
        self.extrusion_stop_score = extrusion_stop_score
        self.leak_score = leak_score
        self.clog_confirmation = clog_confirmation
        self.leak_confirmation = leak_confirmation
        self.stop_confirmation = stop_confirmation
        self.fps = 30

        # Three zone rectangles (fractional coordinates)
        self.zone_a_rect = (0.15, 0.05, 0.70, 0.30)   # Head movement area
        self.zone_b_rect = (0.30, 0.40, 0.40, 0.20)   # Extrusion gap
        self.zone_l_rect = (0.38, 0.10, 0.24, 0.30)   # Nozzle body (between head and gap)

        # Background models
        self.bg_a = None
        self.bg_b = None
        self.bg_l = None
        self.alpha = 0.03

        # Score histories
        self.score_a_history = deque(maxlen=90)
        self.score_b_history = deque(maxlen=90)
        self.score_l_history = deque(maxlen=90)

        # State tracking
        self.head_moving = True
        self.material_flowing = True
        self.nozzle_leaking = False
        
        # Confirmation timers
        self.clog_start_frame = None
        self.leak_start_frame = None
        self.stop_start_frame = None

        # Pixel counts
        self.pixels_a = self.pixels_b = self.pixels_l = None

    def set_zones(self, zone_a, zone_b, zone_l):
        """Set all three zone rectangles."""
        self.zone_a_rect = zone_a
        self.zone_b_rect = zone_b
        self.zone_l_rect = zone_l

    def _extract_zone(self, full_frame, zone_rect):
        """Extract a zone from frame using fractional coordinates."""
        h, w = full_frame.shape[:2]
        x = int(zone_rect[0] * w)
        y = int(zone_rect[1] * h)
        zw = int(zone_rect[2] * w)
        zh = int(zone_rect[3] * h)
        x = max(0, min(x, w-1))
        y = max(0, min(y, h-1))
        zw = max(10, min(zw, w-x))
        zh = max(10, min(zh, h-y))
        return full_frame[y:y+zh, x:x+zw], (x, y, zw, zh)

    def _compute_motion(self, gray_zone, background, total_pixels):
        """Compute motion score for a single zone."""
        if background is None:
            background = gray_zone.astype(np.float32)
            return 0.0, background, np.zeros_like(gray_zone)

        cv2.accumulateWeighted(gray_zone, background, self.alpha)
        bg_uint8 = cv2.convertScaleAbs(background)
        diff = cv2.absdiff(bg_uint8, gray_zone)
        _, thresh = cv2.threshold(diff, self.threshold, 255, cv2.THRESH_BINARY)
        changed = np.count_nonzero(thresh)
        score = changed / total_pixels if total_pixels > 0 else 0.0
        return score, background, thresh

    def process_frame(self, frame):
        """Process one frame through all three zones."""
        h, w = frame.shape[:2]
        gray_full = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_full = cv2.GaussianBlur(gray_full, self.blur_kernel, 0)

        # Extract zones
        zone_a_gray, coords_a = self._extract_zone(gray_full, self.zone_a_rect)
        zone_b_gray, coords_b = self._extract_zone(gray_full, self.zone_b_rect)
        zone_l_gray, coords_l = self._extract_zone(gray_full, self.zone_l_rect)

        # Initialize pixel counts
        if self.pixels_a is None:
            self.pixels_a = zone_a_gray.shape[0] * zone_a_gray.shape[1]
            self.pixels_b = zone_b_gray.shape[0] * zone_b_gray.shape[1]
            self.pixels_l = zone_l_gray.shape[0] * zone_l_gray.shape[1]

        # Compute motion for each zone
        score_a, self.bg_a, diff_a = self._compute_motion(zone_a_gray, self.bg_a, self.pixels_a)
        score_b, self.bg_b, diff_b = self._compute_motion(zone_b_gray, self.bg_b, self.pixels_b)
        score_l, self.bg_l, diff_l = self._compute_motion(zone_l_gray, self.bg_l, self.pixels_l)

        # Store histories
        self.score_a_history.append(score_a)
        self.score_b_history.append(score_b)
        self.score_l_history.append(score_l)

        # Rolling averages
        avg_a = np.mean(self.score_a_history) if self.score_a_history else score_a
        avg_b = np.mean(self.score_b_history) if self.score_b_history else score_b
        avg_l = np.mean(self.score_l_history) if self.score_l_history else score_l

        # Determine states
        self.head_moving = avg_a > self.head_stop_score
        self.material_flowing = avg_b > self.extrusion_stop_score
        self.nozzle_leaking = avg_l > self.leak_score

        # --- Clog detection: head moving + no extrusion + nozzle NOT leaking ---
        clog_detected = False
        if self.head_moving and not self.material_flowing:
            if self.clog_start_frame is None:
                self.clog_start_frame = len(self.score_a_history)
            else:
                duration = (len(self.score_a_history) - self.clog_start_frame) / self.fps
                if duration >= self.clog_confirmation:
                    clog_detected = True
        else:
            self.clog_start_frame = None

        # --- Leak detection: nozzle body shows changes (drip forming/growing) ---
        leak_detected = False
        if self.nozzle_leaking:
            if self.leak_start_frame is None:
                self.leak_start_frame = len(self.score_l_history)
            else:
                duration = (len(self.score_l_history) - self.leak_start_frame) / self.fps
                if duration >= self.leak_confirmation:
                    leak_detected = True
        else:
            self.leak_start_frame = None

        # --- Stop detection: head not moving ---
        stop_detected = False
        if not self.head_moving:
            if self.stop_start_frame is None:
                self.stop_start_frame = len(self.score_a_history)
            else:
                duration = (len(self.score_a_history) - self.stop_start_frame) / self.fps
                if duration >= self.stop_confirmation:
                    stop_detected = True
        else:
            self.stop_start_frame = None

        # Build result
        result = {
            "head_moving": self.head_moving,
            "material_flowing": self.material_flowing,
            "nozzle_leaking": self.nozzle_leaking,
            "score_head": score_a,
            "score_extrusion": score_b,
            "score_leak": score_l,
            "avg_head": avg_a,
            "avg_extrusion": avg_b,
            "avg_leak": avg_l,
            "clog_detected": clog_detected,
            "leak_detected": leak_detected,
            "stop_detected": stop_detected,
            "coords_a": coords_a,
            "coords_b": coords_b,
            "coords_l": coords_l,
            "diff_a": diff_a,
            "diff_b": diff_b,
            "diff_l": diff_l,
        }

        # Determine overall state
        if stop_detected:
            result["state"] = "stopped"
        elif clog_detected and leak_detected:
            result["state"] = "clogged_and_leaking"
        elif clog_detected:
            result["state"] = "clogged"
        elif leak_detected:
            result["state"] = "leaking"
        elif self.head_moving and self.material_flowing and not self.nozzle_leaking:
            result["state"] = "working"
        elif self.head_moving and not self.material_flowing:
            result["state"] = "warning_no_flow"
        elif self.nozzle_leaking:
            result["state"] = "warning_leak"
        else:
            result["state"] = "idle"

        return result

    def reset(self):
        self.bg_a = self.bg_b = self.bg_l = None
        self.score_a_history.clear()
        self.score_b_history.clear()
        self.score_l_history.clear()
        self.head_moving = True
        self.material_flowing = True
        self.nozzle_leaking = False
        self.clog_start_frame = None
        self.leak_start_frame = None
        self.stop_start_frame = None