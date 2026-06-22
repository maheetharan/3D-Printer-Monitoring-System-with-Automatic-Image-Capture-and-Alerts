import cv2
import numpy as np
from collections import deque


class MotionDetector:
    """
    Real-time motion detection using frame differencing.
    
    How it works:
    1. Compares each frame to the previous frame (pixel by pixel).
    2. Counts how many pixels changed significantly.
    3. Computes a "motion score" (0.0 = completely still, 1.0 = every pixel changed).
    4. Uses a rolling average over time to confirm the printer has actually stopped,
       avoiding false triggers from momentary pauses.
    """

    def __init__(self,
                 threshold=70,           # Pixel difference to count as "changed" (0-255)
                 blur_kernel=(5, 5),     # Gaussian blur size to reduce noise
                 stop_score=0.001,       # Motion score below this = stopped
                 confirmation_time=3.0,   # Seconds of continuous "stopped" to trigger alert
                 history_size=90):       # Number of frames to average (at 30fps, 90 = 3 sec)
        self.threshold = threshold
        self.blur_kernel = blur_kernel
        self.stop_score = stop_score
        self.confirmation_time = confirmation_time
        self.history_size = history_size

        self.prev_frame = None
        self.background = None
        self.alpha = 0.03  # Background model learning rate
        self.score_history = deque(maxlen=history_size)
        self.total_pixels = None

        # State tracking
        self.is_moving = True
        self.stopped_start_time = None
        self.fps = 30  # Will be updated externally

    def process_frame(self, frame):
        """
        Main function: takes a frame, returns (is_moving, motion_score, diff_frame).
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, self.blur_kernel, 0)

        if self.total_pixels is None:
            self.total_pixels = gray.shape[0] * gray.shape[1]

        # Initialize background model
        if self.background is None:
            self.background = gray.astype(np.float32)
            self.prev_frame = gray
            return True, 0.0, np.zeros_like(gray)

        # Update running background model (handles slow lighting changes)
        cv2.accumulateWeighted(gray, self.background, self.alpha)
        background_uint8 = cv2.convertScaleAbs(self.background)

        # Compute absolute difference between current frame and background
        diff = cv2.absdiff(background_uint8, gray)

        # Apply threshold to binarize
        _, thresh = cv2.threshold(diff, self.threshold, 255, cv2.THRESH_BINARY)

        # Calculate motion score (fraction of changed pixels)
        changed_pixels = np.count_nonzero(thresh)
        motion_score = changed_pixels / self.total_pixels

        # Store in history for temporal smoothing
        self.score_history.append(motion_score)

        # Compute rolling average
        avg_score = np.mean(self.score_history) if self.score_history else motion_score

        # Determine movement state
        currently_moving = avg_score > self.stop_score

        # State machine: moving <-> stopped with confirmation delay
        if currently_moving:
            self.is_moving = True
            self.stopped_start_time = None
        else:
            if self.stopped_start_time is None:
                self.stopped_start_time = len(self.score_history)  # frame count start
            else:
                stopped_duration = (len(self.score_history) - self.stopped_start_time) / self.fps
                if stopped_duration >= self.confirmation_time:
                    self.is_moving = False

        # Store current frame for next iteration
        self.prev_frame = gray

        return self.is_moving, motion_score, thresh

    def reset(self):
        """Reset all state (useful when restarting)."""
        self.prev_frame = None
        self.background = None
        self.score_history.clear()
        self.is_moving = True
        self.stopped_start_time = None