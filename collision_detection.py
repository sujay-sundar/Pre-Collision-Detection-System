"""
Animal Pre-Collision Tracker
=============================
Two modes — switch with TAB:

  CALIBRATE  — draw a box around the animal, set its real-world width,
               set distance, then press SAVE. Writes template + config
               to  tracker_profile.json + tracker_template.png

  TRACK      — loads saved profile, tracks the animal with LK optical
               flow, estimates distance and approach speed, shows alert

Vehicle speed — estimated from camera via background optical flow
               (sparse LK on static background points). IMU slot left
               as a clear TODO comment for Pi integration.

Controls:
  TAB         toggle CALIBRATE / TRACK
  mouse drag  (calibrate mode) draw bounding box
  S           save calibration
  R           reset / clear track
  Q           quit
"""

import cv2
import numpy as np
import json
import os
import time
from collections import deque

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG  — edit these to match your setup
# ─────────────────────────────────────────────────────────────────────────────
CAM_W, CAM_H    = 640, 480
FOCAL_PX        = 554.0          # camera focal length in pixels
                                  # (run calibration or use 554 for 640×480 ~60° FOV)
PROFILE_FILE    = "tracker_profile.json"
TEMPLATE_FILE   = "tracker_template.png"

# ─────────────────────────────────────────────────────────────────────────────
# KALMAN — smooths noisy distance → velocity
# ─────────────────────────────────────────────────────────────────────────────
class KalmanDist:
    def __init__(self):
        self.x  = np.array([[2.0], [0.0]])   # [distance, velocity]
        self.P  = np.eye(2) * 1.0
        self.H  = np.array([[1.0, 0.0]])
        self.R  = np.array([[0.002]])
        self._t = None

    def update(self, z_dist: float, q: float = 0.01):
        now = time.time()
        dt  = np.clip((now - self._t) if self._t else 0.033, 0.005, 0.2)
        self._t = now
        F  = np.array([[1.0, -dt], [0.0, 1.0]])
        Q  = np.array([[q * dt**2, q * dt], [q * dt, q]])
        xp = F @ self.x
        Pp = F @ self.P @ F.T + Q
        S  = float((self.H @ Pp @ self.H.T + self.R)[0, 0])
        K  = Pp @ self.H.T / S
        innov = z_dist - float((self.H @ xp)[0, 0])
        self.x = xp + K * innov
        self.P = (np.eye(2) - K @ self.H) @ Pp
        return max(0.01, float(self.x[0, 0])), float(self.x[1, 0])

    def reset(self, dist=2.0):
        self.x  = np.array([[dist], [0.0]])
        self.P  = np.eye(2) * 1.0
        self._t = None


# ─────────────────────────────────────────────────────────────────────────────
# TEMPLATE PROFILE  — save / load
# ─────────────────────────────────────────────────────────────────────────────
class TrackerProfile:
    """Holds template image + physical calibration data."""

    def __init__(self):
        self.template     = None   # grayscale patch
        self.real_width_m = 0.5    # animal real width in metres
        self.calib_dist_m = 2.0    # distance at calibration (metres)
        self.calib_px_w   = 100    # pixel width at calibration distance
        self.label        = "animal"

    # ── Derived ──────────────────────────────────────────────────────────────

    def px_per_metre_at_1m(self) -> float:
        """How many pixels does 1m of object width occupy at 1m distance?"""
        if self.calib_px_w <= 0 or self.real_width_m <= 0:
            return FOCAL_PX
        # pixel_width = focal * real_width / dist  →  focal = px_w * dist / real_w
        return self.calib_px_w * self.calib_dist_m / self.real_width_m

    def dist_from_px_width(self, px_w: float) -> float:
        """Estimate distance given current pixel width of tracked object."""
        if px_w <= 0:
            return 99.0
        f = self.px_per_metre_at_1m()
        return (f * self.real_width_m) / px_w

    # ── Persistence ──────────────────────────────────────────────────────────

    def save(self, template_bgr: np.ndarray, rect: tuple):
        """rect = (x,y,w,h) in frame coords."""
        x, y, w, h = rect
        gray = cv2.cvtColor(template_bgr, cv2.COLOR_BGR2GRAY)
        patch = gray[y:y+h, x:x+w]
        self.template    = patch.copy()
        self.calib_px_w  = int(w)
        cv2.imwrite(TEMPLATE_FILE, patch)
        meta = dict(
            real_width_m  = self.real_width_m,
            calib_dist_m  = self.calib_dist_m,
            calib_px_w    = int(w),
            focal_px      = self.px_per_metre_at_1m(),
            label         = self.label,
        )
        with open(PROFILE_FILE, "w") as f:
            json.dump(meta, f, indent=2)
        print(f"[SAVE] Profile → {PROFILE_FILE}  Template → {TEMPLATE_FILE}")

    def load(self) -> bool:
        if not os.path.exists(PROFILE_FILE) or not os.path.exists(TEMPLATE_FILE):
            return False
        with open(PROFILE_FILE) as f:
            meta = json.load(f)
        self.real_width_m = meta.get("real_width_m", 0.5)
        self.calib_dist_m = meta.get("calib_dist_m", 2.0)
        self.calib_px_w   = meta.get("calib_px_w",  100)
        self.label        = meta.get("label", "animal")
        self.template     = cv2.imread(TEMPLATE_FILE, cv2.IMREAD_GRAYSCALE)
        if self.template is None:
            return False
        print(f"[LOAD] {self.label}  real_w={self.real_width_m}m  "
              f"calib_dist={self.calib_dist_m}m  px_w={self.calib_px_w}px")
        return True

    @property
    def is_ready(self):
        return self.template is not None


# ─────────────────────────────────────────────────────────────────────────────
# OBJECT TRACKER  — LK optical flow + template re-verify
# ─────────────────────────────────────────────────────────────────────────────
class ObjectTracker:
    """
    Tracks a single object using Lucas-Kanade sparse optical flow.

    Seeded by either:
      (a) a user-drawn rect (calibration mode)
      (b) template match against saved profile (auto-acquire in track mode)
    """

    LK_PARAMS = dict(
        winSize  = (19, 19),
        maxLevel = 3,
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.01),
    )
    FEAT_PARAMS = dict(
        maxCorners   = 60,
        qualityLevel = 0.08,
        minDistance  = 5,
        blockSize    = 5,
    )
    MIN_CORNERS   = 5
    REVERIFY_EVERY = 15    # frames between template re-verifications
    REVERIFY_THRESH = 0.35

    def __init__(self, profile: TrackerProfile):
        self.profile   = profile
        self.kf        = KalmanDist()
        self._pts      = None     # tracked corners (N,1,2) float32
        self._prev_gray= None
        self._bbox     = None     # (x,y,w,h)
        self._frame_n  = 0
        self.active    = False
        self.dist_f    = None
        self.speed_f   = 0.0      # m/s, + = approaching
        self.dist_hist = deque(maxlen=120)
        self.spd_hist  = deque(maxlen=120)
        self.trail     = deque(maxlen=40)

    # ── Public ───────────────────────────────────────────────────────────────

    def seed(self, frame: np.ndarray, rect: tuple):
        """Start tracking from a given (x,y,w,h) rect."""
        x, y, w, h = rect
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        patch = gray[y:y+h, x:x+w]
        corners = cv2.goodFeaturesToTrack(patch, **self.FEAT_PARAMS)
        if corners is None or len(corners) < 4:
            print("[TRACKER] Not enough texture — try a different box"); return
        self._pts       = corners + np.array([[[x, y]]], dtype=np.float32)
        self._prev_gray = gray.copy()
        self._bbox      = rect
        cx = x + w // 2;  cy = y + h // 2
        self.trail.clear(); self.trail.append((cx, cy))
        dist0 = self.profile.dist_from_px_width(w)
        self.kf.reset(dist0)
        self.dist_f  = dist0
        self.speed_f = 0.0
        self.active  = True
        self._frame_n = 0
        print(f"[TRACKER] Seeded {len(self._pts)} corners  dist≈{dist0:.2f}m")

    def auto_acquire(self, frame: np.ndarray) -> bool:
        """Try to find the saved template in the frame (track-mode startup)."""
        if not self.profile.is_ready: return False
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        tmpl = self.profile.template
        th, tw = tmpl.shape
        if gray.shape[0] < th or gray.shape[1] < tw: return False
        res  = cv2.matchTemplate(gray, tmpl, cv2.TM_CCOEFF_NORMED)
        _, score, _, loc = cv2.minMaxLoc(res)
        if score < 0.45: return False
        x, y = loc
        rect = (x, y, tw, th)
        self.seed(frame, rect)
        print(f"[TRACKER] Auto-acquired  score={score:.2f}  at {loc}")
        return True

    def update(self, frame: np.ndarray):
        """Call every frame. Updates dist_f, speed_f."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self._prev_gray is None:
            self._prev_gray = gray.copy(); return
        if not self.active or self._pts is None:
            self._prev_gray = gray.copy(); return

        self._frame_n += 1

        # LK flow
        new_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            self._prev_gray, gray, self._pts, None, **self.LK_PARAMS)
        good = new_pts[status.flatten() == 1]
        if len(good) < self.MIN_CORNERS:
            self._try_redetect(gray)
            self._prev_gray = gray.copy(); return

        # Update bbox from point spread
        cx = int(np.median(good[:, 0, 0]))
        cy = int(np.median(good[:, 0, 1]))
        spread_x = max(int(np.ptp(good[:, 0, 0]) * 1.4), self.profile.calib_px_w // 2)
        spread_y = max(int(np.ptp(good[:, 0, 1]) * 1.4), self.profile.calib_px_w // 2)
        bx = max(0, cx - spread_x // 2)
        by = max(0, cy - spread_y // 2)
        bw = min(spread_x, CAM_W - bx)
        bh = min(spread_y, CAM_H - by)
        self._bbox = (bx, by, bw, bh)
        self._pts  = good.reshape(-1, 1, 2)
        self.trail.append((cx, cy))

        # Distance + speed via Kalman
        dist_raw     = self.profile.dist_from_px_width(bw)
        d, v         = self.kf.update(dist_raw)
        self.dist_f  = d
        self.speed_f = v
        self.dist_hist.append(d)
        self.spd_hist.append(max(0.0, v))

        # Periodic template re-verify
        if self._frame_n % self.REVERIFY_EVERY == 0:
            score = self._template_score(gray, cx, cy, bw, bh)
            if score < self.REVERIFY_THRESH:
                self._try_redetect(gray)

        self._prev_gray = gray.copy()

    def reset(self):
        self.active  = False
        self._pts    = None
        self._bbox   = None
        self.dist_f  = None
        self.speed_f = 0.0
        self.trail.clear()
        self.kf.reset()

    # ── Properties ───────────────────────────────────────────────────────────

    @property
    def bbox(self): return self._bbox

    @property
    def centre(self):
        if self._bbox is None: return None
        x, y, w, h = self._bbox
        return x + w // 2, y + h // 2

    def ttc(self) -> float:
        """Time-to-collision in seconds. 99 = no risk."""
        if self.dist_f and self.speed_f > 0.05:
            return min(self.dist_f / self.speed_f, 99.0)
        return 99.0

    def speed_kmh(self) -> float:
        return max(0.0, self.speed_f) * 3.6

    # ── Internal ─────────────────────────────────────────────────────────────

    def _template_score(self, gray, cx, cy, bw, bh) -> float:
        if not self.profile.is_ready: return 1.0
        tmpl = self.profile.template
        th, tw = tmpl.shape
        m  = 50
        sx = max(0, cx - m - tw // 2); sy = max(0, cy - m - th // 2)
        ex = min(CAM_W, cx + m + tw // 2); ey = min(CAM_H, cy + m + th // 2)
        roi = gray[sy:ey, sx:ex]
        if roi.shape[0] < th or roi.shape[1] < tw: return 1.0
        res = cv2.matchTemplate(roi, tmpl, cv2.TM_CCOEFF_NORMED)
        return float(np.max(res))

    def _try_redetect(self, gray):
        if self._bbox is None: self.active = False; return
        bx, by, bw, bh = self._bbox
        patch = gray[max(0,by):min(CAM_H,by+bh), max(0,bx):min(CAM_W,bx+bw)]
        if patch.size < 50: self.active = False; return
        corners = cv2.goodFeaturesToTrack(patch, **self.FEAT_PARAMS)
        if corners is None or len(corners) < self.MIN_CORNERS:
            self.active = False
            print("[TRACKER] Lost — try auto-acquire or re-calibrate"); return
        self._pts  = corners + np.array([[[bx, by]]], dtype=np.float32)
        self.active = True


# ─────────────────────────────────────────────────────────────────────────────
# VEHICLE SPEED (camera-based)
# Estimates camera/vehicle forward motion from background optical flow.
# Dense background points far from the tracked object are tracked; their
# net horizontal/vertical displacement indicates ego-motion.
#
# TODO Pi: replace with IMU integration
#   ax_g = imu.get_data()["ax"]  →  veh_speed += ax_g * 9.81 * dt
# ─────────────────────────────────────────────────────────────────────────────
class VehicleSpeed:
    """
    Sparse optical flow on background corners to estimate camera (= vehicle)
    forward speed as apparent scale change.

    Output: veh_spd_ms  (m/s, 0 when still)
    """

    def __init__(self):
        self._prev_gray  = None
        self._bg_pts     = None
        self._t          = None
        self.veh_spd_ms  = 0.0
        self._spd_hist   = deque(maxlen=10)   # short window for smoothing
        self.spd_hist    = deque(maxlen=120)  # for sparkline

    def update(self, frame: np.ndarray, obj_bbox=None):
        """
        obj_bbox: (x,y,w,h) of tracked object — excluded from background mask
                  so object motion doesn't pollute vehicle speed estimate.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        now  = time.time()
        dt   = np.clip((now - self._t) if self._t else 0.033, 0.005, 0.2)
        self._t = now

        if self._prev_gray is None or self._bg_pts is None or len(self._bg_pts) < 4:
            self._reseed(gray, obj_bbox)
            self._prev_gray = gray.copy(); return

        # Track background corners
        new_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            self._prev_gray, gray, self._bg_pts, None,
            winSize=(15, 15), maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 15, 0.03))

        good_new = new_pts[status.flatten() == 1]
        good_old = self._bg_pts[status.flatten() == 1]

        if len(good_new) < 4:
            self._reseed(gray, obj_bbox)
            self._prev_gray = gray.copy(); return

        # Mean flow vector of background points
        flow = good_new - good_old   # (N, 1, 2)
        mean_flow = np.median(flow.reshape(-1, 2), axis=0)   # [dx, dy]

        # Forward motion appears as zoom (points diverge from principal point).
        # Estimate zoom scale from displacement relative to frame centre.
        cx_f, cy_f = CAM_W / 2., CAM_H / 2.
        old_r = np.sqrt((good_old[:, 0, 0] - cx_f)**2 +
                        (good_old[:, 0, 1] - cy_f)**2) + 1e-6
        new_r = np.sqrt((good_new[:, 0, 0] - cx_f)**2 +
                        (good_new[:, 0, 1] - cy_f)**2) + 1e-6
        scale = float(np.median(new_r / old_r))  # >1 = zooming in = forward motion

        # Convert scale to approximate forward speed:
        # scale = 1 + (v * dt / dist)  →  v = (scale-1) * dist / dt
        # Use a nominal distance of 3m as the background plane estimate.
        NOMINAL_BG_DIST = 3.0
        raw_spd = max(0.0, (scale - 1.0) * NOMINAL_BG_DIST / dt)
        raw_spd = np.clip(raw_spd, 0.0, 10.0)   # cap at 36 km/h

        self._spd_hist.append(raw_spd)
        self.veh_spd_ms = float(np.median(self._spd_hist))
        self.spd_hist.append(self.veh_spd_ms)

        self._bg_pts     = good_new.reshape(-1, 1, 2)
        self._prev_gray  = gray.copy()

        # Periodically reseed to replace lost points
        if len(self._bg_pts) < 10:
            self._reseed(gray, obj_bbox)

    def _reseed(self, gray: np.ndarray, obj_bbox=None):
        mask = np.ones_like(gray)
        if obj_bbox:
            x, y, w, h = obj_bbox
            pad = 20
            mask[max(0,y-pad):min(CAM_H,y+h+pad),
                 max(0,x-pad):min(CAM_W,x+w+pad)] = 0
        corners = cv2.goodFeaturesToTrack(
            gray, maxCorners=40, qualityLevel=0.05,
            minDistance=12, blockSize=5, mask=mask)
        self._bg_pts = corners if corners is not None else None

    def kmh(self) -> float:
        return self.veh_spd_ms * 3.6


# ─────────────────────────────────────────────────────────────────────────────
# ALERT LEVEL
# ─────────────────────────────────────────────────────────────────────────────
def alert_level(ttc: float, obj_spd: float, closing_spd: float) -> tuple:
    """
    Returns (level 0-4, label, colour BGR).
    Uses fused closing speed TTC when vehicle speed available.
    """
    if ttc > 99 or not obj_spd > 0.05:
        return 0, "CLEAR", (50, 200, 50)
    if ttc > 6.0:
        return 1, "MONITORING", (0, 200, 255)
    if ttc > 3.5:
        return 2, "CAUTION", (0, 130, 255)
    if ttc > 1.8:
        return 3, "WARNING", (0, 50, 255)
    return 4, "CRITICAL", (0, 0, 220)


# ─────────────────────────────────────────────────────────────────────────────
# DRAWING HELPERS
# ─────────────────────────────────────────────────────────────────────────────
_DARK = (16, 18, 26)
_MID  = (32, 36, 48)

def sparkline(canvas, data, x, y, w, h, col, vmin=0., vmax=1.):
    arr = np.array(list(data), dtype=np.float32)
    if len(arr) < 2: return
    rng = max(vmax - vmin, 1e-6)
    xs  = np.linspace(x, x + w - 1, len(arr)).astype(int)
    ys  = np.clip((y + h - 1 - ((arr - vmin) / rng * (h - 2))).astype(int), y, y + h - 1)
    pts = np.stack([xs, ys], axis=1).reshape(-1, 1, 2)
    cv2.polylines(canvas, [pts], False, col, 1, cv2.LINE_AA)


def draw_hud(canvas, tracker: ObjectTracker, vspd: VehicleSpeed,
             profile: TrackerProfile, mode: str):
    """Bottom panel — 160px tall."""
    PANEL_TOP = CAM_H
    PH        = 160
    canvas[PANEL_TOP:, :] = _DARK
    cv2.line(canvas, (0, PANEL_TOP), (CAM_W, PANEL_TOP), (50, 55, 70), 1)

    # ── Left block — live readouts (210px wide) ───────────────────────────────
    LW = 210
    lx = 8
    def label_val(label, val, unit, row, col):
        ry = PANEL_TOP + 18 + row * 38
        cv2.putText(canvas, label, (lx, ry - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.30, (80, 88, 100), 1)
        cv2.putText(canvas, val,   (lx, ry + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, col, 2)
        cv2.putText(canvas, unit,  (lx, ry + 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.27, (65, 72, 85), 1)

    if tracker.active and tracker.dist_f:
        d = tracker.dist_f
        dc = (60,200,255) if d > 2 else (0,180,255) if d > 1 else (0,80,255)
        label_val("DISTANCE", f"{d:.2f}", "metres", 0, dc)

        spd = tracker.speed_kmh()
        sc  = (50,230,50) if spd < 3 else (0,200,255) if spd < 8 else (0,60,255)
        label_val("OBJECT", f"{spd:.1f}", "km/h", 1, sc)

        ttc = tracker.ttc()
        tc  = (50,230,50) if ttc > 6 else (0,180,255) if ttc > 3 else (0,40,255)
        ttc_str = f"{ttc:.1f}" if ttc < 99 else "---"
        label_val("TTC", ttc_str, "seconds", 2, tc)
    else:
        cv2.putText(canvas, "No target", (lx, PANEL_TOP + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (55, 60, 72), 1)
        if mode == "TRACK" and profile.is_ready:
            cv2.putText(canvas, "Searching for target...", (lx, PANEL_TOP + 62),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (60, 70, 90), 1)
        elif mode == "TRACK":
            cv2.putText(canvas, "No profile — press TAB to calibrate",
                        (lx, PANEL_TOP + 62), cv2.FONT_HERSHEY_SIMPLEX,
                        0.33, (80, 50, 50), 1)

    cv2.line(canvas, (LW, PANEL_TOP + 4), (LW, PANEL_TOP + PH - 4), (38, 42, 55), 1)

    # ── Centre block — sparklines (260px wide) ────────────────────────────────
    SX = LW + 8
    SW = 250
    sh = (PH - 20) // 3 - 2
    labels = [("OBJ km/h",  (100,255,140), tracker.spd_hist,  0., 15.),
              ("VEH km/h",  (80, 130,255), vspd.spd_hist,     0., 15.),
              ("DIST m",    (60, 200,255), tracker.dist_hist,  0., 10.)]
    for i, (lbl, col, data, lo, hi) in enumerate(labels):
        sy = PANEL_TOP + 10 + i * (sh + 2)
        cv2.rectangle(canvas, (SX, sy), (SX+SW, sy+sh), _MID, 1)
        cv2.putText(canvas, lbl, (SX + 2, sy - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.26, col, 1)
        sparkline(canvas, data, SX, sy, SW, sh, col, lo, hi)
        if data:
            cv2.putText(canvas, f"{list(data)[-1]:.1f}",
                        (SX + SW - 30, sy + sh - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.26, col, 1)

    cv2.line(canvas, (SX+SW+8, PANEL_TOP+4), (SX+SW+8, PANEL_TOP+PH-4), (38,42,55), 1)

    # ── Right block — vehicle speed + TTC gauge ───────────────────────────────
    RX = SX + SW + 16

    # Vehicle speed display
    v_kmh = vspd.kmh()
    vc    = (80,130,255) if v_kmh < 8 else (0,180,255) if v_kmh < 15 else (0,60,255)
    cv2.putText(canvas, "VEHICLE", (RX, PANEL_TOP + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.30, (80,88,100), 1)
    cv2.putText(canvas, f"{v_kmh:.1f}", (RX, PANEL_TOP + 36),
                cv2.FONT_HERSHEY_SIMPLEX, 0.72, vc, 2)
    cv2.putText(canvas, "km/h (cam)", (RX, PANEL_TOP + 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.27, (55,60,72), 1)
    cv2.putText(canvas, "TODO: +IMU", (RX, PANEL_TOP + 62),
                cv2.FONT_HERSHEY_SIMPLEX, 0.26, (40,45,55), 1)

    # TTC arc gauge
    gx = RX + 55;  gy = PANEL_TOP + 108;  gr = 40
    ttc  = tracker.ttc() if tracker.active else 99.0
    vttc = max(ttc, 0)
    cspd = tracker.speed_f + vspd.veh_spd_ms
    if cspd > 0.05 and tracker.dist_f:
        vttc = min(tracker.dist_f / cspd, 99.0)

    arc_frac = min(vttc / 8.0, 1.0)
    arc_end  = int(200 + arc_frac * 140)
    gc = (0,40,220) if vttc<2 else (0,130,255) if vttc<4 else (0,220,255) if vttc<6 else (50,220,50)
    cv2.ellipse(canvas, (gx,gy), (gr,gr), 0, 200, 340, (38,42,52), 4)
    cv2.ellipse(canvas, (gx,gy), (gr,gr), 0, 200, arc_end, gc, 4, cv2.LINE_AA)
    ttc_str = f"{vttc:.1f}" if vttc < 99 else "∞"
    cv2.putText(canvas, "TTC", (gx-13, gy-8),  cv2.FONT_HERSHEY_SIMPLEX, 0.30, (80,88,100), 1)
    cv2.putText(canvas, ttc_str, (gx-18, gy+10), cv2.FONT_HERSHEY_SIMPLEX, 0.65, gc, 2)
    cv2.putText(canvas, "sec", (gx-10, gy+24), cv2.FONT_HERSHEY_SIMPLEX, 0.28, (65,72,85), 1)

    # ── Mode + profile badge ─────────────────────────────────────────────────
    mc = (0,220,255) if mode=="CALIBRATE" else (80,200,80)
    cv2.rectangle(canvas, (0, PANEL_TOP+PH-16), (CAM_W, PANEL_TOP+PH), (18,20,26), -1)
    cv2.putText(canvas, f"[TAB] {mode}  |  profile: {profile.label if profile.is_ready else 'none'}",
                (6, PANEL_TOP+PH-4), cv2.FONT_HERSHEY_SIMPLEX, 0.30, mc, 1)


def draw_track_overlay(frame, tracker: ObjectTracker, mode: str):
    """Draws bbox, trail, corners on the camera frame."""
    if not tracker.active or tracker.bbox is None: return
    x, y, w, h = tracker.bbox
    cx_t, cy_t = tracker.centre

    lvl, lbl, col = alert_level(tracker.ttc(), tracker.speed_f,
                                 tracker.speed_f)

    # Bbox
    cv2.rectangle(frame, (x, y), (x+w, y+h), col, 2)

    # Corner brackets
    s = 10
    for px_, py_, dx, dy in [(x,y,1,1),(x+w,y,-1,1),(x,y+h,1,-1),(x+w,y+h,-1,-1)]:
        cv2.line(frame, (px_,py_), (px_+dx*s,py_), col, 2)
        cv2.line(frame, (px_,py_), (px_,py_+dy*s), col, 2)

    # Centroid dot
    cv2.circle(frame, (cx_t, cy_t), 4, col, -1)

    # Trail
    trail = list(tracker.trail)
    for i in range(1, len(trail)):
        alpha = i / len(trail)
        tc    = tuple(int(c * alpha) for c in col)
        cv2.line(frame, trail[i-1], trail[i], tc, 1, cv2.LINE_AA)

    # Label above box
    label_str = f"{tracker.profile.label}  {tracker.speed_kmh():.1f}km/h"
    cv2.putText(frame, label_str, (x, max(y-6, 14)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.44, col, 1)

    # Tracked corners (faint)
    if tracker._pts is not None:
        for pt in tracker._pts:
            cv2.circle(frame, (int(pt[0][0]), int(pt[0][1])), 2, (40,60,160), -1)


def draw_banner(frame, mode: str, level_label: str, level_col, fps: float,
                profile: TrackerProfile):
    """Top 40px banner."""
    cv2.rectangle(frame, (0,0), (CAM_W, 40), (14,16,22), -1)
    mc = (0,220,255) if mode=="CALIBRATE" else (80,200,80)
    cv2.putText(frame, mode, (8, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.70, mc, 2)
    cv2.putText(frame, level_label, (140, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.70, level_col, 2)
    cv2.putText(frame, f"FPS:{fps:.0f}", (CAM_W-70, 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.30, (80,88,100), 1)
    pname = profile.label if profile.is_ready else "no profile"
    cv2.putText(frame, pname, (CAM_W-90, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.28, (60,70,85), 1)


# ─────────────────────────────────────────────────────────────────────────────
# CALIBRATION UI
# ─────────────────────────────────────────────────────────────────────────────
class CalibUI:
    """Manages mouse-drag box drawing and parameter entry in calibrate mode."""

    def __init__(self):
        self.drawing   = False
        self.rect_start= None
        self.rect_end  = None
        self.rect      = None   # finalised (x,y,w,h)
        self.real_width= 0.5    # metres — user edits with W/w keys
        self.calib_dist= 2.0    # metres — user edits with D/d keys
        self.label     = "animal"

    def on_mouse(self, event, x, y, flags, param):
        # Clamp to camera area
        x = max(0, min(CAM_W-1, x))
        y = max(0, min(CAM_H-1, y))
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing    = True
            self.rect_start = (x, y)
            self.rect_end   = (x, y)
            self.rect       = None
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            self.rect_end   = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing    = False
            self.rect_end   = (x, y)
            self._finalise()

    def _finalise(self):
        if self.rect_start is None or self.rect_end is None: return
        x1 = min(self.rect_start[0], self.rect_end[0])
        y1 = min(self.rect_start[1], self.rect_end[1])
        x2 = max(self.rect_start[0], self.rect_end[0])
        y2 = max(self.rect_start[1], self.rect_end[1])
        if x2 - x1 < 10 or y2 - y1 < 10:
            self.rect = None; return
        self.rect = (x1, y1, x2-x1, y2-y1)

    def draw(self, frame):
        """Draw calibration overlay on frame."""
        # Active drag
        if self.drawing and self.rect_start and self.rect_end:
            cv2.rectangle(frame, self.rect_start, self.rect_end, (0,220,255), 1)
        # Finalised box
        if self.rect:
            x, y, w, h = self.rect
            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,120), 2)
            cv2.putText(frame, "TARGET BOX", (x, max(y-6,14)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0,255,120), 1)

        # Instruction overlay (bottom of camera area)
        lines = [
            f"DRAG box around animal",
            f"W / w  =  real width:  {self.real_width:.2f} m  (+/- 0.05m)",
            f"D / d  =  distance:    {self.calib_dist:.1f} m  (+/- 0.1m)",
            f"L      =  change label  [{self.label}]",
            f"S      =  SAVE profile",
        ]
        for i, line in enumerate(lines):
            col = (0,220,255) if i==0 else (180,190,200)
            cv2.putText(frame, line, (10, CAM_H - 100 + i*18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.36, col, 1)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
WIN = "Animal Pre-Collision Tracker"

def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAM_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        print("[ERROR] No camera found"); return

    profile  = TrackerProfile()
    tracker  = ObjectTracker(profile)
    vspd     = VehicleSpeed()
    calib    = CalibUI()
    mode     = "TRACK"   # start in TRACK, auto-loads profile if saved

    PANEL_H  = 160
    WIN_H    = CAM_H + PANEL_H
    DISP_W   = int(CAM_W * 1.5)
    DISP_H   = int(WIN_H * 1.5)
    canvas   = np.zeros((WIN_H, CAM_W, 3), dtype=np.uint8)

    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, DISP_W, DISP_H)

    # Mouse only active in calibrate mode — callback checks mode internally
    def on_mouse(event, wx, wy, flags, param):
        if mode != "CALIBRATE": return
        px = max(0, min(CAM_W-1, int(wx / 1.5)))
        py = max(0, min(CAM_H-1, int(wy / 1.5)))
        calib.on_mouse(event, px, py, flags, param)

    cv2.setMouseCallback(WIN, on_mouse)

    # Auto-load profile if available
    if profile.load():
        mode = "TRACK"
        print("[INFO] Profile loaded — starting in TRACK mode")
    else:
        mode = "CALIBRATE"
        print("[INFO] No profile found — starting in CALIBRATE mode")

    print("[INFO] TAB=toggle mode  S=save  R=reset  Q=quit")
    print("[INFO] Calibrate: drag box → set W/D → press S")

    fps_t  = time.time()
    fcnt   = 0
    auto_tried = False   # only attempt auto-acquire once per mode switch

    while True:
        ret, frame = cap.read()
        if not ret: break
        fcnt += 1
        fps = fcnt / (time.time() - fps_t + 1e-6)

        key = cv2.waitKey(1) & 0xFF

        # ── Keys ──────────────────────────────────────────────────────────────
        if key == ord("q"):
            break

        elif key == 9:   # TAB
            mode = "CALIBRATE" if mode == "TRACK" else "TRACK"
            tracker.reset()
            calib.rect = None
            auto_tried = False
            print(f"[MODE] → {mode}")

        elif key == ord("r"):
            tracker.reset()
            auto_tried = False
            print("[RESET] Tracker cleared")

        elif key == ord("s") and mode == "CALIBRATE":
            if calib.rect:
                profile.real_width_m = calib.real_width
                profile.calib_dist_m = calib.calib_dist
                profile.label        = calib.label
                profile.save(frame, calib.rect)
                tracker.seed(frame, calib.rect)
                mode = "TRACK"
                print("[SAVE] Switching to TRACK mode")
            else:
                print("[WARN] Draw a box first")

        elif key == ord("W"):
            calib.real_width = round(calib.real_width + 0.05, 2)
        elif key == ord("w"):
            calib.real_width = max(0.05, round(calib.real_width - 0.05, 2))
        elif key == ord("D"):
            calib.calib_dist = round(calib.calib_dist + 0.1, 1)
        elif key == ord("d"):
            calib.calib_dist = max(0.5, round(calib.calib_dist - 0.1, 1))
        elif key == ord("l") or key == ord("L"):
            # Cycle through common labels
            labels = ["animal","deer","dog","person","cat","other"]
            idx = labels.index(calib.label) if calib.label in labels else 0
            calib.label = labels[(idx + 1) % len(labels)]
            print(f"[LABEL] → {calib.label}")

        # ── Auto-acquire on entering TRACK mode ───────────────────────────────
        if mode == "TRACK" and not tracker.active and not auto_tried:
            if profile.is_ready:
                tracker.auto_acquire(frame)
            auto_tried = True

        # ── Per-mode processing ───────────────────────────────────────────────
        if mode == "CALIBRATE":
            if calib.rect:
                tracker.seed(frame, calib.rect)   # live preview in calib box
            else:
                tracker.reset()

        # Always update tracker + vehicle speed
        tracker.update(frame)
        vspd.update(frame, tracker.bbox)

        # ── Alert ─────────────────────────────────────────────────────────────
        cspd = tracker.speed_f + vspd.veh_spd_ms
        fttc = (tracker.dist_f / cspd) if (cspd > 0.05 and tracker.dist_f) else 99.0
        lvl, lbl, col = alert_level(fttc, tracker.speed_f, cspd)

        # ── Draw ─────────────────────────────────────────────────────────────
        draw_banner(frame, mode, lbl, col, fps, profile)
        if mode == "CALIBRATE":
            calib.draw(frame)
        draw_track_overlay(frame, tracker, mode)

        canvas[:CAM_H, :] = frame
        draw_hud(canvas, tracker, vspd, profile, mode)

        display = cv2.resize(canvas, (DISP_W, DISP_H), interpolation=cv2.INTER_LINEAR)
        cv2.imshow(WIN, display)

    cap.release()
    cv2.destroyAllWindows()
    print("[DONE]")


if __name__ == "__main__":
    main()