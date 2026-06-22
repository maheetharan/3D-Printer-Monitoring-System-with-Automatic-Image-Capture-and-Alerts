import cv2
import time
import argparse
import numpy as np
from dynamic_tracker import DynamicZoneTracker
from photo_manager import PhotoManager
from alert_manager import AlertManager
from utils import draw_dynamic_zones, draw_status_overlay, draw_diff_overlay


def stack_images_horizontal(images, target_height=400):
    """Resize and stack images horizontally into a single image."""
    resized = []
    for img in images:
        if img is None or img.size == 0:
            img = np.zeros((target_height, target_height//2, 3), dtype=np.uint8)
        h, w = img.shape[:2]
        scale = target_height / h
        new_w = int(w * scale)
        resized.append(cv2.resize(img, (new_w, target_height)))
    return np.hstack(resized)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, default=None)
    parser.add_argument("--threshold", type=int, default=15)
    parser.add_argument("--leak-score", type=float, default=0.003)
    parser.add_argument("--leak-time", type=float, default=3.0)
    args = parser.parse_args()

    tracker = DynamicZoneTracker(
        threshold=args.threshold,
        leak_score=args.leak_score,
        leak_confirmation=args.leak_time,
    )
    photo_manager = PhotoManager(save_dir="captured_photos", interval=2, cleanup_interval=10)
    alert_manager = AlertManager(log_file="alerts.log")

    if args.video:
        cap = cv2.VideoCapture(args.video)
        tracker.fps = cap.get(cv2.CAP_PROP_FPS) or 30
        print(f"[Source] Video: {args.video}")
    else:
        cap = cv2.VideoCapture(0)
        tracker.fps = 30
        print("[Source] Webcam")

    # Set up windows
    cv2.namedWindow("Printer Monitor", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Printer Monitor", 1280, 720)   # Larger default size
    cv2.namedWindow("Diagnostics", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Diagnostics", 900, 400)

    print("\n🎯 DYNAMIC TRACKING ACTIVE")
    print("   Zones L & B follow the nozzle/pen tip")
    print("   Yellow dot = tracked nozzle position")
    print("   Yellow dashed = Zone A (search area)")
    print("   Red box = Zone L (leak)  |  Green box = Zone B (extrusion)")
    print("\n🎮 Keys: c=clog | l=leak | b=both | s=stop | t=toggle zones | f=fullscreen | r=reset | q=quit\n")

    prev_state = "working"
    frame_count = 0
    display_fps = 30
    prev_time = time.time()
    show_tracking = True
    fullscreen = False

    sim_clog = False
    sim_leak = False
    sim_stop = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        result = tracker.process_frame(frame)

        # Simulations
        if sim_stop:
            result["state"] = "stopped"; result["stop_detected"] = True; result["head_moving"] = False
        elif sim_clog and sim_leak:
            result["state"] = "clogged_and_leaking"; result["clog_detected"] = True; result["leak_detected"] = True
            result["head_moving"] = True; result["material_flowing"] = False; result["nozzle_leaking"] = True
        elif sim_clog:
            result["state"] = "clogged"; result["clog_detected"] = True
            result["head_moving"] = True; result["material_flowing"] = False
        elif sim_leak:
            result["state"] = "leaking"; result["leak_detected"] = True
            result["head_moving"] = True; result["nozzle_leaking"] = True

        # Alert transitions
        current_state = result["state"]
        if prev_state != current_state:
            if current_state in ["clogged", "leaking", "clogged_and_leaking", "stopped"]:
                photo_manager.preserve_recent(count=5)
            photo_paths = photo_manager.get_recent_photos(count=5)
            if current_state == "clogged":
                alert_manager.send_clog_alert(photo_paths)
            elif current_state == "leaking":
                alert_manager.send_leak_alert(photo_paths)
            elif current_state == "clogged_and_leaking":
                alert_manager.send_clog_leak_alert(photo_paths)
            elif current_state == "stopped":
                alert_manager.send_stop_alert(photo_paths)
            elif prev_state in ["clogged", "leaking", "clogged_and_leaking", "stopped"]:
                alert_manager.reset_all()
        prev_state = current_state

        photo_manager.capture_if_needed(frame)
        frame_count += 1
        if frame_count % (int(tracker.fps) * 10) == 0:
            photo_manager.cleanup()

        if frame_count % 30 == 0:
            now = time.time()
            display_fps = 30/(now-prev_time) if (now-prev_time) > 0 else 30
            prev_time = now

        # ---- Build main display ----
        display = frame.copy()
        if show_tracking:
            draw_dynamic_zones(display, result)

        status = ""
        if sim_clog and sim_leak: status = "[SIM: CLOG+LEAK]"
        elif sim_clog: status = "[SIM: CLOG]"
        elif sim_leak: status = "[SIM: LEAK]"
        elif sim_stop: status = "[SIM: STOP]"
        display = draw_status_overlay(display, result, display_fps, status)

        cv2.imshow("Printer Monitor", display)

        # ---- Build diagnostics panel (diff heatmaps side by side) ----
        diffs = []
        for key in ["diff_a", "diff_l", "diff_b"]:
            d = result.get(key)
            if d is None or d.size == 0:
                d = np.zeros((100, 100, 3), dtype=np.uint8)
            diffs.append(draw_diff_overlay(d))

        # Add labels on each diff
        labels = ["Zone A (Search)", "Zone L (Leak)", "Zone B (Extrusion)"]
        for i, (diff_img, label) in enumerate(zip(diffs, labels)):
            cv2.putText(diff_img, label, (5, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

        diag = stack_images_horizontal(diffs, target_height=300)
        cv2.imshow("Diagnostics", diag)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            sim_clog = not sim_clog; print(f"  Sim Clog: {'ON' if sim_clog else 'OFF'}")
        elif key == ord('l'):
            sim_leak = not sim_leak; print(f"  Sim Leak: {'ON' if sim_leak else 'OFF'}")
        elif key == ord('b'):
            sim_clog = not sim_clog; sim_leak = not sim_leak
            print(f"  Sim Clog+Leak: {'ON' if sim_clog else 'OFF'}")
        elif key == ord('s'):
            sim_stop = not sim_stop; sim_clog = sim_leak = False
            print(f"  Sim Stop: {'ON' if sim_stop else 'OFF'}")
        elif key == ord('t'):
            show_tracking = not show_tracking; print(f"  Tracking: {'ON' if show_tracking else 'OFF'}")
        elif key == ord('f'):
            fullscreen = not fullscreen
            if fullscreen:
                cv2.setWindowProperty("Printer Monitor", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            else:
                cv2.setWindowProperty("Printer Monitor", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
            print(f"  Fullscreen: {'ON' if fullscreen else 'OFF'}")
        elif key == ord('r'):
            tracker.reset(); alert_manager.reset_all()
            sim_clog = sim_leak = sim_stop = False; prev_state = "working"
            print("  [RESET] Complete.")

    cap.release()
    cv2.destroyAllWindows()
    print("\n[Exit] Monitor stopped.")


if __name__ == "__main__":
    main()