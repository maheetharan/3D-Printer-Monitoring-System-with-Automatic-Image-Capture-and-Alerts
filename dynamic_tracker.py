import cv2
import numpy as np
from collections import deque


class DynamicZoneTracker:
    def __init__(self,
                 threshold=15,
                 blur_kernel=(5, 5),
                 head_stop_score=0.002,
                 extrusion_stop_score=0.005,
                 leak_score=0.003,
                 clog_confirmation=4.0,
                 leak_confirmation=3.0,
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

        # Zone A: fixed search area
        self.zone_a_rect = (0.05, 0.05, 0.90, 0.50)

        # Offsets for dynamic zones (fractions of frame)
        self.zone_l_offset = (0.0, 0.08)
        self.zone_l_size = (0.12, 0.18)
        self.zone_b_offset = (0.0, 0.28)
        self.zone_b_size = (0.15, 0.20)

        # Nozzle position tracking
        self.nozzle_x = 0.5
        self.nozzle_y = 0.3
        self.nozzle_found = False
        self.nozzle_history = deque(maxlen=10)

        # Background models for zones
        self.bg_a = None
        self.bg_l = None
        self.bg_b = None
        self.alpha = 0.03
        self.fast_alpha = 0.1

        # Background subtractor for robust motion detection
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=36, detectShadows=False
        )

        # Score histories
        self.score_a_history = deque(maxlen=90)
        self.score_b_history = deque(maxlen=90)
        self.score_l_history = deque(maxlen=90)

        # States
        self.head_moving = True
        self.material_flowing = True
        self.nozzle_leaking = False

        # Timers
        self.clog_start_frame = None
        self.leak_start_frame = None
        self.stop_start_frame = None

        self.pixels_a = None
        self.pixels_b = None
        self.pixels_l = None

    def _extract_zone(self, full_frame, zone_rect):
        h, w = full_frame.shape[:2]
        x = int(max(0, min(zone_rect[0], 1.0)) * w)
        y = int(max(0, min(zone_rect[1], 1.0)) * h)
        zw = int(max(0.01, min(zone_rect[2], 1.0)) * w)
        zh = int(max(0.01, min(zone_rect[3], 1.0)) * h)
        x = max(0, min(x, w - zw))
        y = max(0, min(y, h - zh))
        return full_frame[y:y+zh, x:x+zw], (x, y, zw, zh)

    def _compute_motion(self, gray_zone, background, total_pixels, alpha=None):
        if alpha is None:
            alpha = self.alpha
        if background is None or gray_zone.shape != background.shape[:2]:
            background = gray_zone.astype(np.float32)
            return 0.0, background, np.zeros_like(gray_zone), None
        cv2.accumulateWeighted(gray_zone, background, alpha)
        bg_uint8 = cv2.convertScaleAbs(background)
        diff = cv2.absdiff(bg_uint8, gray_zone)
        _, thresh = cv2.threshold(diff, self.threshold, 255, cv2.THRESH_BINARY)
        changed = np.count_nonzero(thresh)
        score = changed / total_pixels if total_pixels > 0 else 0.0
        return score, background, thresh, diff

    def _find_nozzle_in_frame(self, frame, coords_a, frame_h, frame_w):
        xa, ya, wa, ha = coords_a
        zone_frame = frame[ya:ya+ha, xa:xa+wa]
        if zone_frame.size == 0:
            return None

        fg_mask = self.bg_subtractor.apply(zone_frame)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        min_area = 200
        valid = [c for c in contours if cv2.contourArea(c) > min_area]
        if not valid:
            return None

        largest = max(valid, key=cv2.contourArea)
        M = cv2.moments(largest)
        if M["m00"] == 0:
            return None

        cx_in = M["m10"] / M["m00"]
        cy_in = M["m01"] / M["m00"]
        cx_full = xa + cx_in
        cy_full = ya + cy_in
        return (cx_full / frame_w, cy_full / frame_h)

    def _calculate_dynamic_zone(self, fx, fy, offset, size):
        x = fx + offset[0] - size[0]/2
        y = fy + offset[1] - size[1]/2
        x = max(0.0, min(x, 1.0 - size[0]))
        y = max(0.0, min(y, 1.0 - size[1]))
        return (x, y, size[0], size[1])

    def process_frame(self, frame):
        h, w = frame.shape[:2]
        gray_full = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_full = cv2.GaussianBlur(gray_full, self.blur_kernel, 0)

        zone_a_gray, coords_a = self._extract_zone(gray_full, self.zone_a_rect)
        if self.pixels_a is None:
            self.pixels_a = zone_a_gray.shape[0] * zone_a_gray.shape[1]

        score_a, self.bg_a, thresh_a, diff_a = self._compute_motion(
            zone_a_gray, self.bg_a, self.pixels_a
        )

        nozzle_pos = self._find_nozzle_in_frame(frame, coords_a, h, w)

        if nozzle_pos is not None:
            self.nozzle_x, self.nozzle_y = nozzle_pos
            self.nozzle_history.append(nozzle_pos)
            self.nozzle_found = True
        elif self.nozzle_found and self.nozzle_history:
            self.nozzle_x = np.mean([p[0] for p in self.nozzle_history])
            self.nozzle_y = np.mean([p[1] for p in self.nozzle_history])

        if self.nozzle_history:
            self.nozzle_x = np.mean([p[0] for p in self.nozzle_history])
            self.nozzle_y = np.mean([p[1] for p in self.nozzle_history])

        zone_l_rect = self._calculate_dynamic_zone(self.nozzle_x, self.nozzle_y,
                                                   self.zone_l_offset, self.zone_l_size)
        zone_b_rect = self._calculate_dynamic_zone(self.nozzle_x, self.nozzle_y,
                                                   self.zone_b_offset, self.zone_b_size)

        zone_l_gray, coords_l = self._extract_zone(gray_full, zone_l_rect)
        zone_b_gray, coords_b = self._extract_zone(gray_full, zone_b_rect)

        self.pixels_l = zone_l_gray.shape[0] * zone_l_gray.shape[1]
        self.pixels_b = zone_b_gray.shape[0] * zone_b_gray.shape[1]

        score_l, self.bg_l, thresh_l, _ = self._compute_motion(
            zone_l_gray, self.bg_l, self.pixels_l, alpha=self.fast_alpha)
        score_b, self.bg_b, thresh_b, _ = self._compute_motion(
            zone_b_gray, self.bg_b, self.pixels_b, alpha=self.fast_alpha)

        self.score_a_history.append(score_a)
        self.score_b_history.append(score_b)
        self.score_l_history.append(score_l)

        avg_a = np.mean(self.score_a_history) if self.score_a_history else score_a
        avg_b = np.mean(self.score_b_history) if self.score_b_history else score_b
        avg_l = np.mean(self.score_l_history) if self.score_l_history else score_l

        self.head_moving = avg_a > self.head_stop_score
        self.material_flowing = avg_b > self.extrusion_stop_score
        self.nozzle_leaking = avg_l > self.leak_score

        clog_detected = False
        if self.head_moving and not self.material_flowing:
            if self.clog_start_frame is None:
                self.clog_start_frame = len(self.score_a_history)
            elif (len(self.score_a_history) - self.clog_start_frame) / self.fps >= self.clog_confirmation:
                clog_detected = True
        else:
            self.clog_start_frame = None

        leak_detected = False
        if self.nozzle_leaking:
            if self.leak_start_frame is None:
                self.leak_start_frame = len(self.score_l_history)
            elif (len(self.score_l_history) - self.leak_start_frame) / self.fps >= self.leak_confirmation:
                leak_detected = True
        else:
            self.leak_start_frame = None

        stop_detected = False
        if not self.head_moving:
            if self.stop_start_frame is None:
                self.stop_start_frame = len(self.score_a_history)
            elif (len(self.score_a_history) - self.stop_start_frame) / self.fps >= self.stop_confirmation:
                stop_detected = True
        else:
            self.stop_start_frame = None

        result = {
            "head_moving": self.head_moving,
            "material_flowing": self.material_flowing,
            "nozzle_leaking": self.nozzle_leaking,
            "nozzle_found": self.nozzle_found,
            "nozzle_x": self.nozzle_x,
            "nozzle_y": self.nozzle_y,
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
            "zone_a_rect": self.zone_a_rect,
            "zone_l_rect": zone_l_rect,
            "zone_b_rect": zone_b_rect,
            "diff_a": diff_a,
            "diff_b": thresh_b,
            "diff_l": thresh_l,
        }

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
        self.nozzle_history.clear()
        self.head_moving = True
        self.material_flowing = True
        self.nozzle_leaking = False
        self.nozzle_found = False
        self.nozzle_x = 0.5
        self.nozzle_y = 0.3
        self.clog_start_frame = None
        self.leak_start_frame = None
        self.stop_start_frame = None
        # Reset background subtractor as well
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=36, detectShadows=False)