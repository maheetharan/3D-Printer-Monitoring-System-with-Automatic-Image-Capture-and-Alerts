import cv2
import numpy as np
from collections import deque


class DualZoneDetector:
    """
    Detects TWO things independently:
      1. Is the printer head MOVING? (Zone A)
      2. Is material EXTRUDING? (Zone B - the nozzle tip area)
    
    Alert conditions:
      - Head MOVING + No extrusion = CLOG ALERT (ink dried/stuck)
      - Head STOPPED + No extrusion = PAUSE/FINISH alert
      - Head MOVING + Extrusion flowing = WORKING NORMALLY
    """

    def __init__(self,
                 threshold=15,
                 blur_kernel=(5, 5),
                 head_stop_score=0.002,      # Zone A: head movement threshold
                 extrusion_stop_score=0.005, # Zone B: extrusion threshold (higher = more sensitive to flow)
                 clog_confirmation=4.0,      # Seconds of "moving but no extrusion" to trigger clog alert
                 stop_confirmation=3.0):      # Seconds of "no head movement" to trigger stop alert
        
        # Detection parameters
        self.threshold = threshold
        self.blur_kernel = blur_kernel
        self.head_stop_score = head_stop_score
        self.extrusion_stop_score = extrusion_stop_score
        self.clog_confirmation = clog_confirmation
        self.stop_confirmation = stop_confirmation
        self.fps = 30

        # Zone rectangles (set by user or auto-detected)
        # Format: (x, y, width, height) as fractions of frame dimensions (0.0 to 1.0)
        self.zone_a_rect = (0.2, 0.05, 0.6, 0.35)   # Upper-middle: printer head area
        self.zone_b_rect = (0.35, 0.40, 0.3, 0.20)  # Below zone A: extrusion gap area

        # Background models for each zone
        self.bg_a = None  # Zone A background
        self.bg_b = None  # Zone B background
        self.alpha = 0.03

        # Motion score history for temporal smoothing
        self.score_a_history = deque(maxlen=90)  # Head movement scores
        self.score_b_history = deque(maxlen=90)  # Extrusion scores

        # State tracking
        self.head_moving = True
        self.material_flowing = True
        self.clog_start_frame = None
        self.stop_start_frame = None
        self.total_pixels_a = None
        self.total_pixels_b = None

    def set_zones(self, zone_a, zone_b):
        """
        Manually set zone rectangles.
        Each zone: (x_fraction, y_fraction, width_fraction, height_fraction)
        Example: (0.2, 0.1, 0.6, 0.3) = 20% from left, 10% from top, 60% wide, 30% tall
        """
        self.zone_a_rect = zone_a
        self.zone_b_rect = zone_b

    def _get_zone_frame(self, full_frame, zone_rect):
        """Extract a zone from the full frame based on fractional coordinates."""
        h, w = full_frame.shape[:2]
        x = int(zone_rect[0] * w)
        y = int(zone_rect[1] * h)
        zone_w = int(zone_rect[2] * w)
        zone_h = int(zone_rect[3] * h)
        
        # Clamp to frame boundaries
        x = max(0, min(x, w - 1))
        y = max(0, min(y, h - 1))
        zone_w = max(10, min(zone_w, w - x))
        zone_h = max(10, min(zone_h, h - y))
        
        return full_frame[y:y+zone_h, x:x+zone_w], (x, y, zone_w, zone_h)

    def _compute_zone_motion(self, gray_zone, background, total_pixels):
        """
        Core motion detection for a single zone.
        Returns (motion_score, updated_background, threshold_diff_image).
        """
        if background is None:
            background = gray_zone.astype(np.float32)
            return 0.0, background, np.zeros_like(gray_zone)

        # Update running background
        cv2.accumulateWeighted(gray_zone, background, self.alpha)
        bg_uint8 = cv2.convertScaleAbs(background)

        # Frame differencing
        diff = cv2.absdiff(bg_uint8, gray_zone)
        _, thresh = cv2.threshold(diff, self.threshold, 255, cv2.THRESH_BINARY)

        # Score = fraction of changed pixels
        changed = np.count_nonzero(thresh)
        score = changed / total_pixels if total_pixels > 0 else 0.0

        return score, background, thresh

    def process_frame(self, frame):
        """
        Process one frame. Returns a dict with complete state.
        """
        h, w = frame.shape[:2]
        gray_full = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_full = cv2.GaussianBlur(gray_full, self.blur_kernel, 0)

        # Extract zones
        zone_a_gray, zone_a_coords = self._get_zone_frame(gray_full, self.zone_a_rect)
        zone_b_gray, zone_b_coords = self._get_zone_frame(gray_full, self.zone_b_rect)

        if self.total_pixels_a is None:
            self.total_pixels_a = zone_a_gray.shape[0] * zone_a_gray.shape[1]
            self.total_pixels_b = zone_b_gray.shape[0] * zone_b_gray.shape[1]

        # Compute motion scores for each zone
        score_a, self.bg_a, diff_a = self._compute_zone_motion(
            zone_a_gray, self.bg_a, self.total_pixels_a
        )
        score_b, self.bg_b, diff_b = self._compute_zone_motion(
            zone_b_gray, self.bg_b, self.total_pixels_b
        )

        # Store history
        self.score_a_history.append(score_a)
        self.score_b_history.append(score_b)

        # Rolling averages
        avg_a = np.mean(self.score_a_history) if self.score_a_history else score_a
        avg_b = np.mean(self.score_b_history) if self.score_b_history else score_b

        # Determine states
        self.head_moving = avg_a > self.head_stop_score
        self.material_flowing = avg_b > self.extrusion_stop_score

        # Detect clog condition: head moving BUT no material flowing
        clog_detected = False
        if self.head_moving and not self.material_flowing:
            if self.clog_start_frame is None:
                self.clog_start_frame = len(self.score_a_history)
            else:
                clog_duration = (len(self.score_a_history) - self.clog_start_frame) / self.fps
                if clog_duration >= self.clog_confirmation:
                    clog_detected = True
        else:
            self.clog_start_frame = None

        # Detect stop condition: head not moving
        stop_detected = False
        if not self.head_moving:
            if self.stop_start_frame is None:
                self.stop_start_frame = len(self.score_a_history)
            else:
                stop_duration = (len(self.score_a_history) - self.stop_start_frame) / self.fps
                if stop_duration >= self.stop_confirmation:
                    stop_detected = True
        else:
            self.stop_start_frame = None

        # Build result
        result = {
            "head_moving": self.head_moving,
            "material_flowing": self.material_flowing,
            "score_head": score_a,
            "score_extrusion": score_b,
            "avg_head": avg_a,
            "avg_extrusion": avg_b,
            "clog_detected": clog_detected,
            "stop_detected": stop_detected,
            "state": "working",
            "zone_a_coords": zone_a_coords,
            "zone_b_coords": zone_b_coords,
            "diff_a": diff_a,
            "diff_b": diff_b,
        }

        # Determine overall state
        if clog_detected:
            result["state"] = "clogged"
        elif stop_detected:
            result["state"] = "stopped"
        elif self.head_moving and self.material_flowing:
            result["state"] = "working"
        elif self.head_moving and not self.material_flowing:
            result["state"] = "warning"  # Building up to clog
        else:
            result["state"] = "idle"

        return result

    def reset(self):
        """Reset all state."""
        self.bg_a = None
        self.bg_b = None
        self.score_a_history.clear()
        self.score_b_history.clear()
        self.head_moving = True
        self.material_flowing = True
        self.clog_start_frame = None
        self.stop_start_frame = None