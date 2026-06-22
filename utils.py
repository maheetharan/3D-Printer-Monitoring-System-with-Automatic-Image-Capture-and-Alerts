import cv2
import numpy as np
import os
from datetime import datetime

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def draw_dynamic_zones(frame, result):
    """Draw the three zones: Zone A (search, dashed yellow), Zone L (leak, red), Zone B (extrusion, green)."""
    h, w = frame.shape[:2]

    # Zone A - Search area (dashed yellow)
    z = result.get("zone_a_rect", (0.05, 0.05, 0.9, 0.5))
    x, y = int(z[0]*w), int(z[1]*h)
    zw, zh = int(z[2]*w), int(z[3]*h)
    for i in range(0, zw, 15):
        cv2.line(frame, (x+i, y), (x+min(i+8, zw), y), (200, 200, 0), 1)
        cv2.line(frame, (x+i, y+zh), (x+min(i+8, zw), y+zh), (200, 200, 0), 1)
    for i in range(0, zh, 15):
        cv2.line(frame, (x, y+i), (x, y+min(i+8, zh)), (200, 200, 0), 1)
        cv2.line(frame, (x+zw, y+i), (x+zw, y+min(i+8, zh)), (200, 200, 0), 1)
    cv2.putText(frame, "A: Search", (x+5, y+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 0), 1)

    # Zone L - Leak (red)
    z = result.get("zone_l_rect")
    if z:
        x, y = int(z[0]*w), int(z[1]*h)
        zw, zh = int(z[2]*w), int(z[3]*h)
        cv2.rectangle(frame, (x, y), (x+zw, y+zh), (0, 0, 255), 2)
        cv2.putText(frame, "L: Leak", (x+5, y+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Zone B - Extrusion (green)
    z = result.get("zone_b_rect")
    if z:
        x, y = int(z[0]*w), int(z[1]*h)
        zw, zh = int(z[2]*w), int(z[3]*h)
        cv2.rectangle(frame, (x, y), (x+zw, y+zh), (0, 255, 0), 2)
        cv2.putText(frame, "B: Extrusion", (x+5, y+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Nozzle tracking dot
    if result.get("nozzle_found"):
        nx = int(result["nozzle_x"] * w)
        ny = int(result["nozzle_y"] * h)
        cv2.circle(frame, (nx, ny), 8, (0, 255, 255), -1)
        cv2.circle(frame, (nx, ny), 10, (0, 255, 255), 2)
        cv2.line(frame, (nx-15, ny), (nx+15, ny), (0, 255, 255), 1)
        cv2.line(frame, (nx, ny-15), (nx, ny+15), (0, 255, 255), 1)
        cv2.putText(frame, "Nozzle", (nx+15, ny-10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

    return frame

def draw_status_overlay(frame, result, fps, status_text=""):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h-160), (w, h), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

    state = result["state"]
    state_config = {
        "working": ("TRACKING - Working", (0, 255, 0)),
        "clogged": ("CLOGGED - No Flow", (0, 0, 255)),
        "leaking": ("LEAKING - Nozzle Drip", (0, 140, 255)),
        "clogged_and_leaking": ("CRITICAL: Clog+Leak", (0, 0, 255)),
        "warning_no_flow": ("Warning: Flow Low", (0, 200, 255)),
        "warning_leak": ("Warning: Leak Possible", (0, 200, 255)),
        "stopped": ("STOPPED", (0, 200, 255)),
        "idle": ("IDLE", (180, 180, 180)),
    }
    state_text, state_color = state_config.get(state, ("UNKNOWN", (255,255,255)))
    cv2.putText(frame, state_text, (10, h-125), cv2.FONT_HERSHEY_SIMPLEX, 0.8, state_color, 3)

    track_text = "Nozzle: LOCKED" if result.get("nozzle_found") else "Nozzle: SEARCHING..."
    track_color = (0, 255, 0) if result.get("nozzle_found") else (0, 0, 255)
    cv2.putText(frame, track_text, (10, h-100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, track_color, 1)

    # Score bars
    labels = [
        ("Head", result["avg_head"], (255, 200, 0), 0),
        ("Extrude", result["avg_extrusion"], (0, 255, 0), 1),
        ("Leak", result["avg_leak"], (0, 0, 255), 2),
    ]
    for label, score, color, idx in labels:
        y = h - 75 + idx * 22
        bar_len = int(min(score * 5000, 1.0) * (w-20))
        cv2.rectangle(frame, (10, y), (10+bar_len, y+15), color, -1)
        cv2.rectangle(frame, (10, y), (w-10, y+15), (80,80,80), 1)
        cv2.putText(frame, f"{label}: {score:.5f}", (15, y+12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)

    cv2.putText(frame, f"FPS: {fps:.1f}", (w-100, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
    if status_text:
        cv2.putText(frame, status_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 1)

    return frame

def draw_diff_overlay(diff_frame):
    if diff_frame is None or diff_frame.size == 0:
        return np.zeros((100, 100, 3), dtype=np.uint8)
    heatmap = cv2.applyColorMap(diff_frame, cv2.COLORMAP_JET)
    return heatmap