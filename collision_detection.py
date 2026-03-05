import cv2
import numpy as np
import time
from collections import deque

# ─── Config ───────────────────────────────────────────────────────────────────
MIN_CONTOUR_AREA     = 1200     # ignore small blobs (leaves, debris)
PATH_REGION_RATIO    = 0.45     # centre 45% width = vehicle path
CROSSING_FRAMES      = 5        # frames object must be in path to confirm
GROWTH_SMOOTH_WIN    = 8        # rolling window for area history
GROWTH_APPROACH_THRESH = 0.06   # steady growth → approaching
GROWTH_FAST_THRESH   = 0.20     # fast growth → imminent
PARALLEL_GROWTH_MAX  = 0.03     # below this = moving parallel, not toward camera
PERSIST_FRAMES       = 12       # frames to keep object alive after it disappears
TTC_WARN_THRESH      = 3.5      # seconds for pre-collision warning
TTC_CRITICAL_THRESH  = 1.5      # seconds for critical alert

# ─── Background subtractor ────────────────────────────────────────────────────
bg_subtractor = cv2.createBackgroundSubtractorMOG2(
    history=300, varThreshold=60, detectShadows=True
)

# ─── Single Object Tracker ────────────────────────────────────────────────────
class SingleObjectTracker:
    """
    Tracks only ONE object at a time — the largest moving contour.
    Maintains area history to compute smooth growth rate.
    Uses a persistence buffer so brief occlusions don't reset the track.
    """
    def __init__(self):
        self.centroid      = None
        self.bbox          = None
        self.area_history  = deque(maxlen=GROWTH_SMOOTH_WIN)
        self.missed_frames = 0
        self.in_path_count = 0
        self.active        = False

    def update(self, detections, path_x1, path_x2):
        """
        detections: list of (cx, cy, x, y, w, h) sorted largest first.
        Only uses the first (largest) detection.
        """
        if detections:
            cx, cy, x, y, w, h = detections[0]   # largest only
            area = w * h

            # Smooth out camera-shake jumps: reject if centroid teleports
            if self.centroid is not None:
                dist = np.hypot(cx - self.centroid[0], cy - self.centroid[1])
                if dist > 160 and self.active:
                    # Likely a different object or camera shake — soft ignore
                    self.missed_frames += 1
                    return

            self.centroid      = (cx, cy)
            self.bbox          = (x, y, w, h)
            self.area_history.append(area)
            self.missed_frames = 0
            self.active        = True

            # Track how long object has been inside path region
            if path_x1 < cx < path_x2:
                self.in_path_count += 1
            else:
                # Slowly decay when object leaves path
                self.in_path_count = max(0, self.in_path_count - 1)

        else:
            # No detection this frame
            self.missed_frames += 1
            if self.missed_frames > PERSIST_FRAMES:
                self._reset()

    def _reset(self):
        self.centroid      = None
        self.bbox          = None
        self.area_history.clear()
        self.missed_frames = 0
        self.in_path_count = 0
        self.active        = False

    def get_growth_rate(self):
        """
        Compares recent area average vs older area average.
        Positive = growing (approaching), negative = shrinking (leaving).
        """
        hist = list(self.area_history)
        if len(hist) < 4:
            return 0.0
        half   = len(hist) // 2
        older  = np.mean(hist[:half])
        recent = np.mean(hist[half:])
        if older < 1:
            return 0.0
        return (recent - older) / older

    def estimate_ttc(self, growth_rate):
        """
        Heuristic TTC: faster growth = shorter time to collision.
        Not metric — relative risk indicator only.
        """
        if growth_rate < 0.005:
            return 99.9
        ttc = 1.0 / (growth_rate * 8.0)
        return round(min(ttc, 99.9), 1)

    def is_crossing(self):
        return self.active and self.in_path_count >= CROSSING_FRAMES

    def is_parallel(self, growth_rate):
        """Object in path but not growing → moving across, not toward us."""
        return self.is_crossing() and growth_rate < PARALLEL_GROWTH_MAX


# ─── Alert state machine ──────────────────────────────────────────────────────
class AlertState:
    CLEAR      = ("CLEAR",          (30, 180, 30))
    CROSSING   = ("CROSSING",       (0, 200, 255))
    WARN       = ("PRE-COLLISION",   (0, 130, 255))
    CRITICAL   = ("!! COLLISION !!", (0, 0, 220))

    @staticmethod
    def evaluate(tracker, growth, ttc):
        if not tracker.active:
            return AlertState.CLEAR
        if growth >= GROWTH_FAST_THRESH:
            return AlertState.CRITICAL
        if tracker.is_crossing() and ttc < TTC_CRITICAL_THRESH:
            return AlertState.CRITICAL
        if tracker.is_crossing() and ttc < TTC_WARN_THRESH and growth > GROWTH_APPROACH_THRESH:
            return AlertState.WARN
        if tracker.is_crossing():
            return AlertState.CROSSING
        return AlertState.CLEAR


# ─── Draw helpers ─────────────────────────────────────────────────────────────
def draw_path_region(frame, x1, x2):
    h = frame.shape[0]
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, 0), (x2, h), (0, 255, 100), -1)
    cv2.addWeighted(overlay, 0.07, frame, 0.93, 0, frame)
    cv2.line(frame, (x1, 0), (x1, h), (0, 255, 100), 1)
    cv2.line(frame, (x2, 0), (x2, h), (0, 255, 100), 1)

def draw_object(frame, tracker, growth, ttc, alert):
    if not tracker.active or tracker.bbox is None:
        return
    x, y, w, h = tracker.bbox
    cx, cy = tracker.centroid
    _, color = alert

    # Bounding box with corner accents
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    corner = 12
    for px, py, dx, dy in [(x,y,1,1),(x+w,y,-1,1),(x,y+h,1,-1),(x+w,y+h,-1,-1)]:
        cv2.line(frame, (px, py), (px + dx * corner, py), color, 3)
        cv2.line(frame, (px, py), (px, py + dy * corner), color, 3)

    # Centroid dot
    cv2.circle(frame, (cx, cy), 5, color, -1)

    # Growth bar (right side of bbox)
    bar_h   = h
    fill    = int(np.clip(growth / GROWTH_FAST_THRESH, 0, 1) * bar_h)
    bar_x   = x + w + 6
    cv2.rectangle(frame, (bar_x, y), (bar_x + 8, y + bar_h), (60, 60, 60), -1)
    cv2.rectangle(frame, (bar_x, y + bar_h - fill), (bar_x + 8, y + bar_h), color, -1)
    cv2.putText(frame, "G", (bar_x, y - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)

    # Info label
    label = f"Growth:{growth:+.2f}"
    if ttc < 99:
        label += f"  TTC:{ttc:.1f}s"
    if tracker.is_parallel(growth):
        label += "  [PARALLEL]"
    cv2.putText(frame, label, (x, y - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.46, color, 1)

def draw_hud(frame, alert, fps, obj_active, growth):
    h, w = frame.shape[:2]
    label, color = alert

    # Top banner
    cv2.rectangle(frame, (0, 0), (w, 44), color, -1)
    cv2.putText(frame, label, (12, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.95, (255, 255, 255), 2)

    # Bottom info bar
    cv2.rectangle(frame, (0, h - 22), (w, h), (20, 20, 20), -1)
    info = f"FPS:{fps:.1f}  Object:{'TRACKED' if obj_active else 'NONE'}  " \
           f"Growth:{growth:+.3f}  IMU:SIMULATED  [Q] quit"
    cv2.putText(frame, info, (8, h - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1)


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("[ERROR] Cannot open webcam.")
        return

    tracker    = SingleObjectTracker()
    kernel     = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fps_time   = time.time()
    frame_cnt  = 0

    print("[INFO] Running — press Q to quit")
    print("[INFO] Move ONE object across the camera to test detection\n")
    print("  Conditions simulated:")
    print("  • Small debris filtered (min area threshold)")
    print("  • Camera shake rejected (centroid jump guard)")
    print("  • Parallel crossing detected (low growth rate)")
    print("  • Brief occlusion handled (persistence buffer)\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_cnt += 1
        h, w = frame.shape[:2]
        path_x1 = int(w * (0.5 - PATH_REGION_RATIO / 2))
        path_x2 = int(w * (0.5 + PATH_REGION_RATIO / 2))

        # ── Foreground mask ───────────────────────────────────────────────────
        fg = bg_subtractor.apply(frame)
        _, fg = cv2.threshold(fg, 200, 255, cv2.THRESH_BINARY)   # remove shadows
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN,  kernel)
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, kernel)
        fg = cv2.dilate(fg, kernel, iterations=2)

        # ── Contours → sort by area descending → take largest ─────────────────
        contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detections  = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < MIN_CONTOUR_AREA:
                continue
            x, y, bw, bh = cv2.boundingRect(cnt)
            cx = x + bw // 2
            cy = y + bh // 2
            detections.append((cx, cy, x, y, bw, bh))

        # Sort largest area first
        detections.sort(key=lambda d: d[4] * d[5], reverse=True)

        # ── Tracker ───────────────────────────────────────────────────────────
        tracker.update(detections, path_x1, path_x2)

        growth = tracker.get_growth_rate()
        ttc    = tracker.estimate_ttc(growth)
        alert  = AlertState.evaluate(tracker, growth, ttc)

        # ── Draw ──────────────────────────────────────────────────────────────
        draw_path_region(frame, path_x1, path_x2)
        draw_object(frame, tracker, growth, ttc, alert)

        fps = frame_cnt / (time.time() - fps_time + 1e-5)
        draw_hud(frame, alert, fps, tracker.active, growth)

        cv2.imshow("Pre-Collision Detection v2 [Single Object]", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Done.")


if __name__ == "__main__":
    main()