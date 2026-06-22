"""
Interactive Zone Tuner for Pen Test
====================================
Click and drag to position the three zones.
Press '1' to select Zone A, '2' for Zone B, '3' for Zone L.
Press 's' to save positions, 'q' to quit.
"""

import cv2
import numpy as np

# Zone rectangles as fractions: (x, y, w, h)
zones = {
    "A": [0.1, 0.05, 0.8, 0.30],   # Hand/Pen body
    "L": [0.35, 0.30, 0.3, 0.15],  # Pen tip
    "B": [0.1, 0.50, 0.8, 0.45],   # Paper
}
active_zone = "A"
dragging = False
drag_corner = None

def draw_zones(frame):
    h, w = frame.shape[:2]
    colors = {"A": (255, 150, 0), "L": (0, 0, 255), "B": (0, 255, 0)}
    for name, (zx, zy, zw, zh) in zones.items():
        x, y = int(zx*w), int(zy*h)
        zw_px, zh_px = int(zw*w), int(zh*h)
        color = colors[name]
        thickness = 3 if name == active_zone else 1
        cv2.rectangle(frame, (x, y), (x+zw_px, y+zh_px), color, thickness)
        cv2.putText(frame, f"Zone {name}", (x+5, y+20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return frame

def mouse_callback(event, x, y, flags, param):
    global dragging, drag_corner
    h, w = param
    if event == cv2.EVENT_LBUTTONDOWN:
        zx, zy, zw, zh = zones[active_zone]
        px, py = int(zx*w), int(zy*h)
        pzw, pzh = int(zw*w), int(zh*h)
        # Check if click is near any corner
        corners = [(px, py), (px+pzw, py), (px, py+pzh), (px+pzw, py+pzh)]
        for i, (cx, cy) in enumerate(corners):
            if abs(x-cx) < 15 and abs(y-cy) < 15:
                dragging = True
                drag_corner = i
                return
        # If not on corner, move the zone
        zones[active_zone][0] = x/w - zw/2
        zones[active_zone][1] = y/h - zh/2
    elif event == cv2.EVENT_MOUSEMOVE and dragging:
        if drag_corner in [0, 1]:
            zones[active_zone][1] = y/h
            zones[active_zone][3] = max(0.05, (py+pzh)/h - y/h)
    elif event == cv2.EVENT_LBUTTONUP:
        dragging = False

cap = cv2.VideoCapture(0)
cv2.namedWindow("Zone Tuner")
h, w = int(cap.get(4)), int(cap.get(3))
cv2.setMouseCallback("Zone Tuner", mouse_callback, (h, w))

print("Zone Tuner Controls:")
print("  Keys 1/2/3 = Select Zone A/L/B")
print("  Mouse drag = Move zone")
print("  s = Print zone positions (copy to code)")
print("  q = Quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = draw_zones(frame)
    cv2.putText(frame, f"Active: Zone {active_zone}", (10, h-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
    cv2.imshow("Zone Tuner", frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('1'):
        active_zone = "A"
        print("Active: Zone A (Hand/Pen)")
    elif key == ord('2'):
        active_zone = "B"
        print("Active: Zone B (Paper)")
    elif key == ord('3'):
        active_zone = "L"
        print("Active: Zone L (Pen Tip)")
    elif key == ord('s'):
        print("\n=== COPY THESE VALUES TO triple_zone_detector.py ===")
        for name in ["A", "L", "B"]:
            z = zones[name]
            print(f'self.zone_{name.lower()}_rect = ({z[0]:.2f}, {z[1]:.2f}, {z[2]:.2f}, {z[3]:.2f})')
        print("=====================================================\n")

cap.release()
cv2.destroyAllWindows()