import cv2
import os
import time
from datetime import datetime
from utils import ensure_dir


class PhotoManager:
    """
    Handles periodic photo capture and automatic cleanup.
    
    - Captures a full-resolution photo every 2 seconds.
    - Deletes photos older than 5 seconds (unless preserved for an alert).
    - When a problem is detected, recent photos are preserved and collected for the alert.
    """

    def __init__(self, save_dir="captured_photos", interval=2, cleanup_interval=5):
        self.save_dir = save_dir
        self.interval = interval          # 2 seconds
        self.cleanup_interval = cleanup_interval  # 5 seconds
        self.last_capture_time = 0
        self.photos = []  # List of dicts: {"path": ..., "timestamp": ..., "preserved": False}

        ensure_dir(save_dir)

    def capture_if_needed(self, frame):
        """
        Call every frame. Takes a photo if `interval` seconds have passed.
        Returns the path of the photo taken, or None.
        """
        now = time.time()
        if now - self.last_capture_time >= self.interval:
            self.last_capture_time = now
            filename = f"photo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            filepath = os.path.join(self.save_dir, filename)
            cv2.imwrite(filepath, frame)
            self.photos.append({
                "path": filepath,
                "timestamp": now,
                "preserved": False
            })
            print(f"  [Photo] Captured: {filename}")
            return filepath
        return None

    def get_latest_photo(self):
        """Return the dict of the most recent unpreserved photo (or preserved if none)."""
        for photo in reversed(self.photos):
            if os.path.exists(photo["path"]):
                return photo
        return None

    def preserve_photo(self, photo_info):
        """Mark a single photo as preserved so cleanup won't delete it."""
        if photo_info:
            photo_info["preserved"] = True
            print(f"  [Photo] Preserved: {os.path.basename(photo_info['path'])}")

    def preserve_recent(self, count=5):
        """
        Mark the last `count` photos as preserved.
        Call this when a problem (clog/leak) is detected so we keep evidence.
        """
        preserved = 0
        for photo in reversed(self.photos):
            if preserved >= count:
                break
            if os.path.exists(photo["path"]) and not photo["preserved"]:
                photo["preserved"] = True
                preserved += 1
                print(f"  [Photo] Preserved for alert: {os.path.basename(photo['path'])}")
        return preserved

    def get_recent_photos(self, count=5):
        """
        Return a list of paths for the most recent `count` photos that exist.
        (Can be used to attach multiple images to an alert.)
        """
        recent = []
        for photo in reversed(self.photos):
            if len(recent) >= count:
                break
            if os.path.exists(photo["path"]):
                recent.append(photo["path"])
        return recent

    def cleanup(self):
        """
        Delete photos older than cleanup_interval (5 seconds), unless preserved.
        Call this periodically.
        """
        now = time.time()
        for photo in self.photos[:]:  # iterate over a copy
            if photo["preserved"]:
                continue
            if now - photo["timestamp"] > self.cleanup_interval:
                if os.path.exists(photo["path"]):
                    os.remove(photo["path"])
                    print(f"  [Cleanup] Deleted: {os.path.basename(photo['path'])}")
                self.photos.remove(photo)