"""
Animal Pre-Collision Tracker  v2
==================================
Modes  (TAB to switch):
  CALIBRATE  — draw box, set real width + distance, press S to save
  TRACK      — auto-loads profile, tracks with LK optical flow

Keys:
  TAB       toggle mode
  S         save calibration (calibrate mode)
  R         force re-acquire from template
  Q         quit
  W/w       real width  +/- 0.05 m   (calibrate)
  D/d       calib dist  +/- 0.5 m    (calibrate)  [disabled when sensor active]
  L         cycle label
  U         cycle distance source: vision -> ultrasonic -> emulator

Distance source (DISTANCE_SOURCE config):
  "vision"      — manual D/d keys only (original behaviour)
  "ultrasonic"  — HC-SR04 on RPi TRIG=GPIO14 ECHO=GPIO15 auto-fills calib dist
  "emulator"    — on-screen trackbar slider (for PC testing without hardware)

Calibration distance tip:
  Place the animal (or a stand-in object) at a known distance.
  The on-screen 1m ruler guide at the bottom shows how wide 1m looks at
  that distance — use it to pace off the exact calibration distance.
"""

import cv2
import numpy as np
import json, os, time
from collections import deque

# ──────────────────────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────────────────────
CAM_W, CAM_H    = 640, 480
FOCAL_PX        = 554.0       # pixels — valid for 640x480 ~60deg FOV
PROFILE_FILE    = "tracker_profile.json"
TEMPLATE_FILE   = "tracker_template.png"
PATH_ZONE_X     = (int(CAM_W * 0.25), int(CAM_W * 0.75))  # vehicle path corridor

TTC_WARN        = 4.0   # s
TTC_CRIT        = 2.0   # s
CROSS_SPD_FAST  = 2.0   # m/s  lateral speed considered "fast crossing"
VEH_SPD_CONCERN = 1.5   # m/s  vehicle speed that elevates risk

# ── Bench-test motion scaling ──────────────────────────────────────────────────
# Your setup moves cm; real traffic operates in metres.
# MOTION_SCALE multiplies ALL distance/speed outputs so 10 cm feels like 1 m.
# Set to 1.0 for real deployment.
MOTION_SCALE    = 10.0

# ── Distance source ────────────────────────────────────────────────────────────
# "vision"      pixel-width estimation only  (no hardware needed)
# "ultrasonic"  HC-SR04 feeds BOTH calibration AND live tracking distance
# "emulator"    on-screen trackbar — PC testing without hardware
DISTANCE_SOURCE     = "emulator"   # change to "ultrasonic" on the Pi

ULTRASONIC_TRIG_PIN = 14   # BCM GPIO number
ULTRASONIC_ECHO_PIN = 15   # BCM GPIO number

EMULATOR_WIN        = "Distance Emulator"   # name of the trackbar window


# ──────────────────────────────────────────────────────────────────────────────
# ULTRASONIC SENSOR  (HC-SR04 via RPi.GPIO)
# TRIG=GPIO14  ECHO=GPIO15  (BCM numbering)
# Gracefully no-ops on non-Pi hardware (ImportError / RuntimeError).
# ──────────────────────────────────────────────────────────────────────────────
class UltrasonicSensor:
    _TIMEOUT = 0.04   # seconds — skip pulse if echo doesn't arrive

    def __init__(self, trig: int = ULTRASONIC_TRIG_PIN,
                 echo: int = ULTRASONIC_ECHO_PIN):
        self._ok   = False
        self._trig = trig
        self._echo = echo
        self._last = None
        try:
            import RPi.GPIO as GPIO
            self._GPIO = GPIO
            GPIO.setmode(GPIO.BCM)
            GPIO.setup(trig, GPIO.OUT)
            GPIO.setup(echo, GPIO.IN)
            GPIO.output(trig, False)
            time.sleep(0.05)   # let sensor settle
            self._ok = True
            print(f"[ULTRASONIC] OK  TRIG=GPIO{trig}  ECHO=GPIO{echo}")
        except ImportError:
            print("[ULTRASONIC] RPi.GPIO not available — sensor disabled")
        except Exception as e:
            print(f"[ULTRASONIC] Init error: {e}")

    @property
    def available(self) -> bool:
        return self._ok

    def read(self) -> float | None:
        """
        Fire a pulse and return distance in metres.
        Returns the last valid reading as fallback when the echo times out or is
        out-of-range — prevents NoneType errors in callers that do arithmetic.
        Returns None only on the very first call before any valid reading exists.
        HC-SR04 rated range: 0.02 m – 4.0 m.
        """
        if not self._ok:
            return self._last
        GPIO = self._GPIO
        try:
            # Send 10 µs trigger pulse
            GPIO.output(self._trig, True)
            time.sleep(0.00001)
            GPIO.output(self._trig, False)

            # Wait for echo rising edge
            t0 = time.time()
            while GPIO.input(self._echo) == 0:
                if time.time() - t0 > self._TIMEOUT:
                    return self._last   # timeout — return last known good value
            pulse_start = time.time()

            # Wait for echo falling edge
            while GPIO.input(self._echo) == 1:
                if time.time() - pulse_start > self._TIMEOUT:
                    return self._last
            pulse_end = time.time()

            dist_m = (pulse_end - pulse_start) * 17150 / 100   # speed-of-sound
            if 0.02 <= dist_m <= 4.0:
                self._last = round(dist_m, 3)
            # Out-of-range: keep self._last as-is, still return it
            return self._last
        except Exception:
            return self._last

    def cleanup(self):
        if self._ok:
            try:
                self._GPIO.cleanup([self._trig, self._echo])
            except Exception:
                pass


# ──────────────────────────────────────────────────────────────────────────────
# KALMAN  (2-state: distance + radial velocity)
# ──────────────────────────────────────────────────────────────────────────────
class KalmanDist:
    def __init__(self):
        self.x  = np.array([[2.0], [0.0]])
        self.P  = np.eye(2) * 1.0
        self.H  = np.array([[1.0, 0.0]])
        self.R  = np.array([[0.004]])
        self._t = None

    def update(self, z: float, q: float = 0.008, r: float = 0.004):
        """
        z  — measured distance (m)
        q  — process noise  (higher = trust measurement more, model less)
        r  — measurement noise (lower = trust this measurement more)
              vision default 0.004 | ultrasonic 0.0002 (hardware is accurate)
        """
        now = time.time()
        dt  = float(np.clip((now - self._t) if self._t else 0.033, 0.005, 0.2))
        self._t = now
        F   = np.array([[1.0, -dt], [0.0, 1.0]])
        Q   = np.array([[q*dt**2, q*dt], [q*dt, q]])
        xp  = F @ self.x
        Pp  = F @ self.P @ F.T + Q
        R_  = np.array([[r]])
        S   = float((self.H @ Pp @ self.H.T + R_)[0, 0])
        K   = Pp @ self.H.T / S
        self.x = xp + K * (z - float((self.H @ xp)[0, 0]))
        self.P = (np.eye(2) - K @ self.H) @ Pp
        return max(0.01, float(self.x[0, 0])), float(self.x[1, 0])

    def reset(self, d=2.0):
        self.x = np.array([[d], [0.0]]); self.P = np.eye(2); self._t = None


# ──────────────────────────────────────────────────────────────────────────────
# TRACKER PROFILE
# ──────────────────────────────────────────────────────────────────────────────
class TrackerProfile:
    def __init__(self):
        self.template     = None
        self.real_width_m = 0.5
        self.calib_dist_m = 2.0
        self.calib_px_w   = 100
        self.label        = "animal"

    def _focal(self):
        if self.calib_px_w <= 0 or self.real_width_m <= 0:
            return FOCAL_PX
        return self.calib_px_w * self.calib_dist_m / self.real_width_m

    def dist_from_px(self, px_w: float) -> float:
        if px_w <= 0: return 99.0
        return self._focal() * self.real_width_m / px_w * MOTION_SCALE

    def save(self, frame_bgr: np.ndarray, rect: tuple):
        x, y, w, h = rect
        gray  = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        patch = gray[y:y+h, x:x+w]
        self.template   = patch.copy()
        self.calib_px_w = int(w)
        cv2.imwrite(TEMPLATE_FILE, patch)
        with open(PROFILE_FILE, "w") as f:
            json.dump(dict(real_width_m=self.real_width_m,
                           calib_dist_m=self.calib_dist_m,
                           calib_px_w=int(w), label=self.label), f, indent=2)
        print(f"[SAVE] {self.label}  w={self.real_width_m}m  d={self.calib_dist_m}m  px={w}")

    def load(self) -> bool:
        if not os.path.exists(PROFILE_FILE) or not os.path.exists(TEMPLATE_FILE):
            return False
        with open(PROFILE_FILE) as f:
            m = json.load(f)
        self.real_width_m = m.get("real_width_m", 0.5)
        self.calib_dist_m = m.get("calib_dist_m", 2.0)
        self.calib_px_w   = m.get("calib_px_w",  100)
        self.label        = m.get("label", "animal")
        self.template     = cv2.imread(TEMPLATE_FILE, cv2.IMREAD_GRAYSCALE)
        ok = self.template is not None
        if ok:
            print(f"[LOAD] {self.label} "
                  f"w={self.real_width_m}m d={self.calib_dist_m}m px={self.calib_px_w}")
        return ok

    @property
    def is_ready(self): return self.template is not None


# ──────────────────────────────────────────────────────────────────────────────
# OBJECT TRACKER
# On loss: goes to LOST state and waits for template re-acquire.
# Does NOT re-seed corners on arbitrary texture (prevents static-object drift).
# ──────────────────────────────────────────────────────────────────────────────
class ObjectTracker:
    _LK = dict(winSize=(21, 21), maxLevel=3,
               criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 25, 0.01))
    _FT = dict(maxCorners=150, qualityLevel=0.03, minDistance=6, blockSize=7)
    MIN_PTS     = 6
    TOPUP_THR   = 25    # reseed more points inside bbox when below this count
    VERIFY_N    = 8     # template verify every N frames
    VERIFY_THR  = 0.38
    LOST_THR    = 0.30
    BW_EMA_A    = 0.25  # EMA alpha for bbox width — reduces noisy distance jumps
    TTC_EMA_A   = 0.15  # slow EMA on TTC — prevents jumps when closing speed is small

    def __init__(self, profile: TrackerProfile):
        self.profile   = profile
        self.kf        = KalmanDist()
        self._pts      = None
        self._pgray    = None
        self._bbox     = None
        self._fn       = 0
        self._state    = "IDLE"
        self._bw_ema   = None   # EMA-smoothed bbox width (pixels)
        self._ttc_ema  = 99.0  # EMA-smoothed TTC (seconds)
        self.dist_f    = None
        self.speed_f   = 0.0
        self.lateral_ms = 0.0
        self.dist_hist = deque(maxlen=120)
        self.spd_hist  = deque(maxlen=120)
        self.trail     = deque(maxlen=60)
        self._lat_hist = deque(maxlen=8)

    @property
    def active(self): return self._state == "TRACKING"

    @property
    def lost(self): return self._state == "LOST"

    @property
    def bbox(self): return self._bbox

    @property
    def centre(self):
        if self._bbox is None: return None
        x, y, w, h = self._bbox
        return x + w // 2, y + h // 2

    def ttc(self) -> float:
        """EMA-smoothed TTC — stable even when closing speed is noisy."""
        if self.dist_f and self.speed_f > 0.05:
            raw = min(self.dist_f / self.speed_f, 99.0)
        else:
            raw = 99.0
        self._ttc_ema = self.TTC_EMA_A * raw + (1.0 - self.TTC_EMA_A) * self._ttc_ema
        return self._ttc_ema

    def inject_distance(self, dist_m: float):
        """
        Feed a trusted external distance (e.g. ultrasonic) into the Kalman.
        Uses very low r so the sensor reading dominates over the model prediction.
        Call this every frame when the ultrasonic sensor is active.
        """
        d, v = self.kf.update(dist_m, q=0.005, r=0.0002)
        self.dist_f  = d
        self.speed_f = v
        self.dist_hist.append(d)
        self.spd_hist.append(max(0.0, v))

    def speed_kmh(self) -> float:
        return max(0.0, self.speed_f) * 3.6

    def lateral_kmh(self) -> float:
        return abs(self.lateral_ms) * 3.6

    def crossing_dir(self) -> str:
        if abs(self.lateral_ms) < 0.15: return ""
        return "RIGHT ->" if self.lateral_ms > 0 else "<- LEFT"

    def seed(self, frame: np.ndarray, rect: tuple) -> bool:
        x, y, w, h = [int(v) for v in rect]
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        patch = gray[max(0,y):min(CAM_H,y+h), max(0,x):min(CAM_W,x+w)]
        corners = cv2.goodFeaturesToTrack(patch, **self._FT)
        if corners is None or len(corners) < 4:
            print("[TRACKER] Not enough texture — try a different box")
            return False
        self._pts   = corners + np.array([[[x, y]]], dtype=np.float32)
        self._pgray = gray.copy()
        self._bbox  = (x, y, w, h)
        self._state   = "TRACKING"
        self._fn      = 0
        self._bw_ema  = float(w)
        self._ttc_ema = 99.0
        d0 = self.profile.dist_from_px(w)
        self.kf.reset(d0)
        self.dist_f = d0
        self.speed_f = 0.0
        self.trail.clear()
        self.trail.append((x + w // 2, y + h // 2))
        print(f"[TRACKER] Seeded  pts={len(self._pts)}  d={d0:.2f}m")
        return True

    def try_acquire(self, frame: np.ndarray, min_score=0.45) -> bool:
        """Template-match whole frame. Only accepts confident hits."""
        if not self.profile.is_ready: return False
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        tmpl = self.profile.template
        th, tw = tmpl.shape
        if gray.shape[0] < th or gray.shape[1] < tw: return False
        res = cv2.matchTemplate(gray, tmpl, cv2.TM_CCOEFF_NORMED)
        _, score, _, loc = cv2.minMaxLoc(res)
        if score < min_score: return False
        print(f"[TRACKER] Acquired  score={score:.2f}  at {loc}")
        return self.seed(frame, (loc[0], loc[1], tw, th))

    def reset(self):
        self._pts = None; self._pgray = None; self._bbox = None
        self._state = "IDLE"; self._bw_ema = None; self._ttc_ema = 99.0
        self.dist_f = None; self.speed_f = 0.0; self.lateral_ms = 0.0
        self.trail.clear(); self.kf.reset()

    def update(self, frame: np.ndarray):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self._pgray is None:
            self._pgray = gray.copy(); return
        if self._state != "TRACKING":
            self._pgray = gray.copy(); return

        self._fn += 1
        new_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            self._pgray, gray, self._pts, None, **self._LK)
        good_new = new_pts[status.flatten() == 1]
        good_old = self._pts[status.flatten() == 1]

        if len(good_new) < self.MIN_PTS:
            self._go_lost()
            self._pgray = gray.copy(); return

        # ── RANSAC outlier rejection ───────────────────────────────────────────
        # estimateAffinePartial2D (scale+rotation+translation) with RANSAC
        # removes flow vectors that don't belong to the same rigid motion.
        if len(good_new) >= 6:
            _, mask = cv2.estimateAffinePartial2D(
                good_old.reshape(-1, 1, 2),
                good_new.reshape(-1, 1, 2),
                method=cv2.RANSAC, ransacReprojThreshold=3.0)
            if mask is not None:
                inliers = mask.flatten().astype(bool)
                if inliers.sum() >= self.MIN_PTS:
                    good_new = good_new[inliers]
                    good_old = good_old[inliers]

        cx = int(np.median(good_new[:, 0, 0]))
        cy = int(np.median(good_new[:, 0, 1]))

        # Lateral velocity (px/frame -> m/s)
        flow_x = float(np.median(good_new[:, 0, 0] - good_old[:, 0, 0]))
        dt = 0.033
        px_per_m = self.profile._focal() / self.dist_f if self.dist_f else FOCAL_PX
        lat_raw = flow_x / (px_per_m * dt)
        self._lat_hist.append(lat_raw)
        self.lateral_ms = float(np.median(self._lat_hist))

        # Robust bbox width from percentile spread
        xs = good_new[:, 0, 0]; ys = good_new[:, 0, 1]
        half_w = max(int(np.percentile(xs, 85) - np.percentile(xs, 15)), 8)
        half_h = max(int(np.percentile(ys, 85) - np.percentile(ys, 15)), 8)
        min_h  = max(self.profile.calib_px_w // 4, 12)
        half_w = max(half_w, min_h); half_h = max(half_h, min_h)
        bx = max(0, cx - half_w); by = max(0, cy - half_h)
        bw = min(half_w * 2, CAM_W - bx); bh = min(half_h * 2, CAM_H - by)

        # EMA-smooth the bbox width to reduce noisy distance jumps
        if self._bw_ema is None:
            self._bw_ema = float(bw)
        else:
            self._bw_ema = self.BW_EMA_A * bw + (1.0 - self.BW_EMA_A) * self._bw_ema
        bw = max(4, int(self._bw_ema))

        self._bbox = (bx, by, bw, bh)

        # ── Point top-up ──────────────────────────────────────────────────────
        # If inlier count drops below threshold, reseed extra corners inside the
        # current bbox so tracking doesn't degrade silently before going LOST.
        if len(good_new) < self.TOPUP_THR:
            pad = 4
            rx1 = max(0, bx - pad); ry1 = max(0, by - pad)
            rx2 = min(CAM_W, bx + bw + pad); ry2 = min(CAM_H, by + bh + pad)
            patch = gray[ry1:ry2, rx1:rx2]
            if patch.size > 0:
                extra = cv2.goodFeaturesToTrack(
                    patch, maxCorners=60, qualityLevel=0.03,
                    minDistance=6, blockSize=7)
                if extra is not None:
                    extra = extra + np.array([[[rx1, ry1]]], dtype=np.float32)
                    good_new = np.vstack([good_new, extra])

        self._pts  = good_new.reshape(-1, 1, 2)
        self.trail.append((cx, cy))

        d_raw = self.profile.dist_from_px(bw)
        d, v  = self.kf.update(d_raw)
        self.dist_f = d; self.speed_f = v
        self.dist_hist.append(d); self.spd_hist.append(max(0.0, v))

        if self._fn % self.VERIFY_N == 0:
            score = self._tmpl_score(gray, cx, cy)
            if score < self.LOST_THR:
                print(f"[TRACKER] Verify failed ({score:.2f}) -> LOST")
                self._go_lost()
                self._pgray = gray.copy(); return

        self._pgray = gray.copy()

    def _go_lost(self):
        self._state = "LOST"; self._pts = None; self.speed_f = 0.0
        print("[TRACKER] LOST")

    def _tmpl_score(self, gray, cx, cy) -> float:
        if not self.profile.is_ready: return 1.0
        tmpl = self.profile.template
        th, tw = tmpl.shape
        m  = 60
        sx = max(0, cx - m - tw//2); sy = max(0, cy - m - th//2)
        ex = min(CAM_W, cx + m + tw//2); ey = min(CAM_H, cy + m + th//2)
        roi = gray[sy:ey, sx:ex]
        if roi.shape[0] < th or roi.shape[1] < tw: return 1.0
        res = cv2.matchTemplate(roi, tmpl, cv2.TM_CCOEFF_NORMED)
        return float(np.max(res))


# ──────────────────────────────────────────────────────────────────────────────
# VEHICLE SPEED  (background optical flow - vertical translation method)
#
# Why vertical flow, not zoom-scale:
#   Forward motion makes background points drift DOWNWARD in frame
#   (because road surface below the camera moves toward the bottom edge).
#   Median dy of background points is much more stable than measuring radial
#   divergence, which requires good point distribution around the full frame.
#
# Formula: v_fwd = dy_px * ref_dist / (FOCAL_PX * dt)
#   This is the inverse of the projection equation for a flat ground plane.
#
# TODO Pi IMU swap: comment out update(), uncomment imu_update() below.
# ──────────────────────────────────────────────────────────────────────────────
class VehicleSpeed:
    _LK = dict(winSize=(15, 15), maxLevel=2,
               criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 15, 0.03))

    def __init__(self):
        self._pgray   = None
        self._bg_pts  = None
        self._t       = None
        self.veh_ms   = 0.0
        self._hist    = deque(maxlen=12)
        self.spd_hist = deque(maxlen=120)

    def update(self, frame: np.ndarray, obj_bbox=None, ref_dist=3.0):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        now  = time.time()
        dt   = float(np.clip((now - self._t) if self._t else 0.033, 0.005, 0.2))
        self._t = now

        if self._pgray is None or self._bg_pts is None or len(self._bg_pts) < 4:
            self._reseed(gray, obj_bbox)
            self._pgray = gray.copy(); return

        new_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            self._pgray, gray, self._bg_pts, None, **self._LK)
        good_new = new_pts[status.flatten() == 1]
        good_old = self._bg_pts[status.flatten() == 1]

        if len(good_new) < 4:
            self._reseed(gray, obj_bbox)
            self._pgray = gray.copy(); return

        # Median downward shift of background = forward vehicle motion
        dy  = float(np.median(good_new[:, 0, 1] - good_old[:, 0, 1]))
        raw = max(0.0, dy * ref_dist / (FOCAL_PX * dt)) * MOTION_SCALE
        raw = min(raw, 8.0 * MOTION_SCALE)

        self._hist.append(raw)
        self.veh_ms = float(np.median(self._hist))
        self.spd_hist.append(self.veh_ms)

        self._bg_pts = good_new.reshape(-1, 1, 2)
        self._pgray  = gray.copy()
        if len(self._bg_pts) < 10:
            self._reseed(gray, obj_bbox)

    def _reseed(self, gray, obj_bbox):
        mask = np.ones(gray.shape, dtype=np.uint8)
        if obj_bbox:
            x, y, w, h = obj_bbox; pad = 30
            mask[max(0,y-pad):min(CAM_H,y+h+pad),
                 max(0,x-pad):min(CAM_W,x+w+pad)] = 0
        mask[:50, :]     = 0   # sky / banner
        mask[CAM_H-40:,:] = 0  # bonnet
        corners = cv2.goodFeaturesToTrack(gray, maxCorners=30, qualityLevel=0.04,
                                          minDistance=14, blockSize=5, mask=mask)
        self._bg_pts = corners

    def kmh(self) -> float: return self.veh_ms * 3.6

    # ── TODO Pi: replace update() call in main with this ──────────────────────
    # def imu_update(self, ax_g: float):
    #     dt = 0.033
    #     self.veh_ms = max(0.0, self.veh_ms + ax_g * 9.81 * dt * 0.998)
    #     self.spd_hist.append(self.veh_ms)


# ──────────────────────────────────────────────────────────────────────────────
# SITUATION ASSESSMENT
# ──────────────────────────────────────────────────────────────────────────────
def assess(tracker: ObjectTracker, vspd: VehicleSpeed) -> tuple:
    """
    Returns (level 0-5, short_label, detail_label, colour_BGR)

    Scenarios:
      Fast lateral cross             — FAST CROSS  (any distance, warns early)
      Cross + fast vehicle           — escalated by vehicle contribution to TTC
      Fused TTC < threshold          — WARNING / CRITICAL
      Both slow but object close     — SLOW+CLOSE caution
      Vehicle fast, obj stationary   — VEH RISK
      Object in path only            — MONITORING
    """
    if not tracker.active or tracker.dist_f is None:
        return 0, "CLEAR", "", (50, 200, 50)

    d      = tracker.dist_f
    v_obj  = tracker.speed_f
    v_lat  = abs(tracker.lateral_ms)
    v_veh  = vspd.veh_ms
    cspd   = max(0.0, v_obj) + v_veh
    ttc    = tracker.ttc()   # EMA-smoothed — stable under noisy closing speed

    in_path = False
    if tracker.centre:
        cx, _ = tracker.centre
        in_path = PATH_ZONE_X[0] < cx < PATH_ZONE_X[1]

    dir_s = tracker.crossing_dir()

    # Fast lateral crossing
    if v_lat > CROSS_SPD_FAST:
        detail = f"FAST CROSS {dir_s}  {v_lat*3.6:.1f}km/h"
        if d < 4.0 and in_path:
            return 4, "WARNING", detail, (0, 50, 255)
        return 3, "CAUTION", detail, (0, 130, 255)

    # Crossing + vehicle fast
    if in_path and v_lat > 0.3 and v_veh > VEH_SPD_CONCERN:
        detail = f"CROSSING {dir_s}  VEH {v_veh*3.6:.1f}km/h"
        if ttc < TTC_CRIT:
            return 5, "CRITICAL", detail, (0, 0, 220)
        if ttc < TTC_WARN:
            return 4, "WARNING", detail, (0, 50, 255)
        return 3, "CAUTION", detail, (0, 130, 255)

    # Fused TTC
    if in_path or v_obj > 0.2:
        if ttc < TTC_CRIT:
            return 5, "CRITICAL", f"TTC {ttc:.1f}s  {cspd*3.6:.0f}km/h closing", (0, 0, 220)
        if ttc < TTC_WARN:
            return 4, "WARNING", f"TTC {ttc:.1f}s", (0, 50, 255)

    # Both slow, object close in path
    if in_path and d < 3.0 and v_obj < 0.5 and v_veh < VEH_SPD_CONCERN:
        return 2, "CAUTION", f"SLOW+CLOSE  d={d:.1f}m", (0, 160, 255)

    # Vehicle fast, object stationary in path
    if in_path and v_veh > VEH_SPD_CONCERN and v_obj < 0.3:
        detail = f"VEH {v_veh*3.6:.1f}km/h  OBJ STILL"
        if ttc < TTC_WARN:
            return 4, "WARNING", detail, (0, 50, 255)
        return 3, "CAUTION", detail, (0, 130, 255)

    # Object in path, monitoring
    if in_path:
        return 1, "MONITORING", f"IN PATH  {dir_s}".strip(), (0, 200, 255)

    return 0, "CLEAR", "", (50, 200, 50)


# ──────────────────────────────────────────────────────────────────────────────
# DRAWING
# All draw functions receive a `disp` canvas that is already at display
# resolution (CAM_W*DS × WIN_H*DS). Coordinates passed in are logical
# (640×480 space); s() scales them to display pixels. Text is rendered at
# native display resolution so it is never blurry or pixelated.
# ──────────────────────────────────────────────────────────────────────────────
_DARK = (16, 18, 26)
_MID  = (30, 34, 46)
_FONT = cv2.FONT_HERSHEY_SIMPLEX

# DS is set in main() before any draw call
DS = 1.5

def s(v):
    """Scale a single logical coordinate/size to display pixels."""
    return int(round(v * DS))

def sp(pt):
    """Scale a (x,y) logical point."""
    return (s(pt[0]), s(pt[1]))

def txt(img, text, lx, ly, scale, col, thick=1, anchor="bl"):
    """
    Draw anti-aliased text at logical position (lx, ly) scaled to display res.
    anchor: 'bl' = baseline-left (default), 'tl' = top-left
    """
    fs  = scale * DS          # font scale at display resolution
    th  = max(1, int(thick * DS * 0.67))
    if anchor == "tl":
        (tw, th_px), bl = cv2.getTextSize(text, _FONT, fs, th)
        ly = ly + th_px / DS  # convert back to logical for the s() call below
    cv2.putText(img, text, (s(lx), s(ly)), _FONT, fs, col, th, cv2.LINE_AA)


def sparkline(disp, data, x, y, w, h, col, vmin=0., vmax=1.):
    """Draw sparkline; all coords are logical, scaled inside."""
    arr = np.array(list(data), dtype=np.float32)
    if len(arr) < 2: return
    rng = max(vmax - vmin, 1e-6)
    # work in display pixels
    dx, dy = s(x), s(y); dw, dh = s(w), s(h)
    xs = np.linspace(dx, dx + dw - 1, len(arr)).astype(int)
    ys = np.clip((dy + dh - 1 - ((arr - vmin) / rng * (dh - 2))).astype(int),
                 dy, dy + dh - 1)
    pts = np.stack([xs, ys], axis=1).reshape(-1, 1, 2)
    cv2.polylines(disp, [pts], False, col, 1, cv2.LINE_AA)


def draw_path_zone(disp):
    """Subtle green corridor overlay — logical coords scaled."""
    x1, x2 = PATH_ZONE_X
    ov = disp.copy()
    cv2.rectangle(ov, sp((x1, 0)), sp((x2, CAM_H)), (0, 200, 80), -1)
    cv2.addWeighted(ov, 0.06, disp, 0.94, 0, disp)
    cv2.line(disp, sp((x1, 0)), sp((x1, CAM_H)), (0, 160, 60), 1)
    cv2.line(disp, sp((x2, 0)), sp((x2, CAM_H)), (0, 160, 60), 1)


def draw_object(disp, tracker: ObjectTracker, alert_col):
    if not tracker.active or tracker.bbox is None: return
    x, y, w, h = tracker.bbox
    cx, cy = tracker.centre

    cv2.rectangle(disp, sp((x, y)), sp((x+w, y+h)), alert_col, max(1, s(2)))
    cv2.circle(disp, sp((cx, cy)), s(4), alert_col, -1)

    # Trail
    trail = list(tracker.trail)
    for i in range(1, len(trail)):
        a  = i / len(trail)
        tc = tuple(int(c * a) for c in alert_col)
        cv2.line(disp, sp(trail[i-1]), sp(trail[i]), tc, 1, cv2.LINE_AA)

    # Direction arrow
    v_lat = tracker.lateral_ms
    if abs(v_lat) > 0.15:
        arrow_len = int(min(abs(v_lat) * 28, 70))
        ax_end = sp((cx + (arrow_len if v_lat > 0 else -arrow_len), cy))
        acol   = (0, 220, 255) if abs(v_lat) < CROSS_SPD_FAST else (0, 50, 255)
        cv2.arrowedLine(disp, sp((cx, cy)), ax_end, acol,
                        max(1, s(2)), tipLength=0.30, line_type=cv2.LINE_AA)
        sym = "->" if v_lat > 0 else "<-"
        txt(disp, f"{sym} {abs(v_lat)*3.6:.1f}km/h",
            x, max(y - 18, 20), 0.36, acol)

    # Approach label
    txt(disp, f"{tracker.profile.label}  ^{tracker.speed_kmh():.1f}km/h",
        x, max(y - 4, 32), 0.40, alert_col)


def draw_lost_indicator(disp, tracker: ObjectTracker):
    if not tracker.lost: return
    if int(time.time() * 3) % 2 == 0:
        txt(disp, "TARGET LOST  searching...",
            CAM_W//2 - 120, CAM_H//2, 0.55, (0, 0, 220), thick=2)


def draw_banner(disp, mode, label, detail, col, fps):
    cv2.rectangle(disp, (0, 0), (s(CAM_W), s(44)), (14, 16, 22), -1)
    mc = (0, 220, 255) if mode == "CALIBRATE" else (80, 200, 80)
    txt(disp, mode,  8,  28, 0.65, mc,  thick=2)
    txt(disp, label, 160, 28, 0.65, col, thick=2)
    if detail:
        txt(disp, detail, 8, 42, 0.32, (160, 170, 185))
    src_col = (0, 255, 120) if DISTANCE_SOURCE == "ultrasonic" else (80, 88, 100)
    txt(disp, f"FPS:{fps:.0f}  {DISTANCE_SOURCE.upper()}  x{MOTION_SCALE:.0f}",
        CAM_W - 170, 14, 0.28, src_col)


def draw_hud(disp, tracker: ObjectTracker, vspd: VehicleSpeed,
             profile: TrackerProfile, alert_col, mode: str):
    PT = CAM_H; PH = 160
    # Fill panel area
    cv2.rectangle(disp, sp((0, PT)), sp((CAM_W, PT+PH)), _DARK, -1)
    cv2.line(disp, sp((0, PT)), sp((CAM_W, PT)), (50, 55, 70), 1)

    lx = 8
    def rv(lbl, val, unit, row, c):
        ry = PT + 18 + row * 36
        txt(disp, lbl,  lx, ry-2,  0.28, (75, 82, 96))
        txt(disp, val,  lx, ry+14, 0.62, c,            thick=2)
        txt(disp, unit, lx, ry+25, 0.26, (58, 65, 78))

    if tracker.active and tracker.dist_f:
        d  = tracker.dist_f
        dc = (60,200,255) if d>2 else (0,180,255) if d>1 else (0,80,255)
        rv("DISTANCE", f"{d:.2f}", "metres", 0, dc)
        os = tracker.speed_kmh()
        oc = (50,220,50) if os<3 else (0,200,255) if os<8 else (0,60,255)
        rv("OBJ SPD",  f"{os:.1f}", "km/h approach", 1, oc)
        lat = tracker.lateral_kmh()
        lc  = (50,220,50) if lat<5 else (0,200,255) if lat<10 else (0,60,255)
        rv("LATERAL",  f"{lat:.1f}", f"km/h {tracker.crossing_dir()}", 2, lc)
    else:
        msg = "LOST - searching" if tracker.lost else "No target"
        txt(disp, msg, lx, PT+40, 0.48, (55, 62, 75))
        if mode == "TRACK" and profile.is_ready:
            txt(disp, "Template search active...", lx, PT+62, 0.34, (50, 60, 80))

    LW = 205
    cv2.line(disp, sp((LW, PT+4)), sp((LW, PT+PH-4)), (38, 42, 55), 1)

    SX = LW+8; SW = 248; sh = (PH-18)//3 - 2
    plots = [("OBJ km/h", (100,255,140), tracker.spd_hist,  0., 15.),
             ("VEH km/h", (80,130,255),  vspd.spd_hist,     0., 15.),
             ("DIST m",   (60,200,255),  tracker.dist_hist,  0., 10.)]
    for i, (lbl, c, data, lo, hi) in enumerate(plots):
        sy = PT + 8 + i * (sh + 3)
        cv2.rectangle(disp, sp((SX, sy)), sp((SX+SW, sy+sh)), _MID, 1)
        txt(disp, lbl, SX+2, sy-2, 0.27, c)
        sparkline(disp, data, SX, sy, SW, sh, c, lo, hi)
        if data:
            txt(disp, f"{list(data)[-1]:.1f}", SX+SW-32, sy+sh-3, 0.27, c)

    cv2.line(disp, sp((SX+SW+8, PT+4)), sp((SX+SW+8, PT+PH-4)), (38, 42, 55), 1)

    RX = SX + SW + 16
    vc = (80, 130, 255)
    txt(disp, "VEHICLE",        RX, PT+14, 0.28, (75, 82, 96))
    txt(disp, f"{vspd.kmh():.1f}", RX, PT+36, 0.68, vc, thick=2)
    txt(disp, "km/h cam",       RX, PT+48, 0.26, (55, 62, 75))

    gx = RX+52; gy = PT+108; gr = 38
    ttcv = tracker.ttc()   # EMA-smoothed, same value assess() uses
    gc   = (0,40,220) if ttcv<2 else (0,130,255) if ttcv<4 else \
           (0,220,255) if ttcv<6 else (50,220,50)
    cv2.ellipse(disp, sp((gx, gy)), (s(gr), s(gr)), 0, 200, 340, (38,42,52), s(4))
    cv2.ellipse(disp, sp((gx, gy)), (s(gr), s(gr)), 0, 200,
                int(200 + min(ttcv/8, 1)*140), gc, s(4), cv2.LINE_AA)
    ttc_str = f"{ttcv:.1f}" if ttcv < 99 else "---"
    txt(disp, "TTC",    gx-13, gy-8,  0.28, (75, 82, 96))
    txt(disp, ttc_str,  gx-16, gy+10, 0.62, gc, thick=2)
    txt(disp, "sec",    gx-9,  gy+24, 0.26, (58, 65, 78))

    mc = (0,220,255) if mode=="CALIBRATE" else (80,200,80)
    cv2.rectangle(disp, sp((0, PT+PH-16)), sp((CAM_W, PT+PH)), (18,20,26), -1)
    hint = "  |  [R] re-acquire" if tracker.lost else ""
    txt(disp, f"[TAB] {mode}  |  {profile.label if profile.is_ready else 'no profile'}{hint}",
        6, PT+PH-4, 0.29, mc)


# ──────────────────────────────────────────────────────────────────────────────
# CALIBRATION UI
# ──────────────────────────────────────────────────────────────────────────────
class CalibUI:
    def __init__(self):
        self.drawing = False
        self.p0 = None; self.p1 = None
        self.rect = None
        self.real_width = 0.5
        self.calib_dist = 2.0
        self.label = "animal"
        self.sensor_dist: float | None = None   # live reading from sensor/emulator

    def on_mouse(self, ev, x, y, flags, param):
        x = max(0, min(CAM_W-1, x)); y = max(44, min(CAM_H-1, y))
        if ev == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True; self.p0 = (x,y); self.p1 = (x,y); self.rect = None
        elif ev == cv2.EVENT_MOUSEMOVE and self.drawing:
            self.p1 = (x, y)
        elif ev == cv2.EVENT_LBUTTONUP:
            self.drawing = False; self.p1 = (x,y); self._fin()

    def _fin(self):
        if not self.p0 or not self.p1: return
        x1=min(self.p0[0],self.p1[0]); y1=min(self.p0[1],self.p1[1])
        x2=max(self.p0[0],self.p1[0]); y2=max(self.p0[1],self.p1[1])
        self.rect = (x1,y1,x2-x1,y2-y1) if (x2-x1>10 and y2-y1>10) else None

    def draw(self, disp):
        # Drag preview
        if self.drawing and self.p0 and self.p1:
            cv2.rectangle(disp, sp(self.p0), sp(self.p1), (0, 220, 255), 1)

        # Finalised box
        if self.rect:
            x, y, w, h = self.rect
            cv2.rectangle(disp, sp((x,y)), sp((x+w,y+h)), (0, 255, 120), s(2))
            txt(disp, f"box:{w}x{h}px  real:{self.real_width}m  @{self.calib_dist}m",
                x, max(y-6, 56), 0.36, (0, 255, 120))

        # 1-metre ruler
        ruler_px = max(10, int(FOCAL_PX / self.calib_dist))
        rx = CAM_W//2 - ruler_px//2; ry = CAM_H - 22
        cv2.line(disp, sp((rx, ry)),  sp((rx+ruler_px, ry)),  (0,200,200), s(2))
        cv2.line(disp, sp((rx, ry-5)), sp((rx, ry+5)),         (0,200,200), s(2))
        cv2.line(disp, sp((rx+ruler_px, ry-5)), sp((rx+ruler_px, ry+5)), (0,200,200), s(2))
        txt(disp, f"<-- 1m at {self.calib_dist:.1f}m -->", rx, ry-7, 0.34, (0,200,200))

        # Sensor / emulator distance badge
        src = DISTANCE_SOURCE
        if src != "vision" and self.sensor_dist is not None:
            badge = f"  {src.upper()}: {self.sensor_dist:.3f} m  "
            bx, by = CAM_W - 160, CAM_H - 90
            (bw, _), _ = cv2.getTextSize(badge, _FONT, 0.42 * DS, max(1, int(DS)))
            cv2.rectangle(disp, (s(bx)-4, s(by)-s(14)),
                          (s(bx) + bw + 4, s(by) + 4), (0, 60, 0), -1)
            txt(disp, badge, bx, by, 0.42, (0, 255, 120), thick=2)
        elif src != "vision":
            badge_col = (0, 180, 255) if src == "emulator" else (0, 80, 220)
            txt(disp, f"  {src.upper()}: ---  ", CAM_W - 160, CAM_H - 90,
                0.38, badge_col)

        # Instructions
        dist_hint = ("sensor" if src == "ultrasonic" else
                     "slider" if src == "emulator" else
                     f"D/d keys: {self.calib_dist:.1f}m")
        lines = [
            "DRAG box around animal",
            f"W/w = real width: {self.real_width:.2f}m    dist: {self.calib_dist:.2f}m [{dist_hint}]    [{self.label}]  L=cycle",
            "TIP: use ruler above to confirm distance  |  U = cycle distance source",
        ]
        for i, ln in enumerate(lines):
            c = (0,220,255) if i==0 else (160,170,185) if i==1 else (90,100,115)
            txt(disp, ln, 8, CAM_H-54+i*18, 0.31, c)


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────
WIN = "Animal Pre-Collision Tracker"


def main():
    global DS, DISTANCE_SOURCE
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAM_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        print("[ERROR] No camera found"); return

    profile    = TrackerProfile()
    tracker    = ObjectTracker(profile)
    vspd       = VehicleSpeed()
    calib      = CalibUI()
    ultrasonic = UltrasonicSensor(ULTRASONIC_TRIG_PIN, ULTRASONIC_ECHO_PIN)

    PH    = 160
    WIN_H = CAM_H + PH
    DS    = 1.5   # display scale — update module-level so s()/sp()/txt() use it

    # Display canvas is always at full display resolution
    DDISP_W = int(CAM_W * DS)
    DDISP_H = int(WIN_H * DS)
    disp = np.zeros((DDISP_H, DDISP_W, 3), dtype=np.uint8)

    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, DDISP_W, DDISP_H)

    mode = "CALIBRATE"
    if profile.load():
        mode = "TRACK"
        print("[INFO] Profile loaded -> TRACK mode")
    else:
        print("[INFO] No profile -> CALIBRATE mode")

    auto_tried = False

    def on_mouse(ev, wx, wy, flags, param):
        if mode != "CALIBRATE": return
        # Mouse coords are in display pixels; convert to logical
        calib.on_mouse(ev, int(wx / DS), int(wy / DS), flags, param)

    cv2.setMouseCallback(WIN, on_mouse)
    print("[INFO] TAB=mode  S=save  R=re-acquire  Q=quit")
    print("[INFO] W/w=width  D/d=distance  L=label  U=dist-source")
    print(f"[DIST] source = {DISTANCE_SOURCE}")

    # ── Emulator trackbar window ───────────────────────────────────────────────
    # Range: 20..1000 → 0.20 m .. 10.00 m  (divide by 100 to get metres)
    _EMULATOR_SCALE = 100   # trackbar units per metre
    _EMULATOR_MIN_CM = 20   # 0.20 m
    _EMULATOR_MAX_CM = 1000 # 10.00 m
    _emulator_open   = False

    def _open_emulator():
        nonlocal _emulator_open
        if _emulator_open: return
        cv2.namedWindow(EMULATOR_WIN, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(EMULATOR_WIN, 420, 80)
        init_cm = max(_EMULATOR_MIN_CM,
                      min(_EMULATOR_MAX_CM, int(calib.calib_dist * _EMULATOR_SCALE)))
        cv2.createTrackbar("Distance_cm", EMULATOR_WIN,
                           init_cm, _EMULATOR_MAX_CM, lambda _: None)
        _emulator_open = True
        print("[EMULATOR] Trackbar window opened")

    def _close_emulator():
        nonlocal _emulator_open
        if not _emulator_open: return
        try:
            cv2.destroyWindow(EMULATOR_WIN)
        except Exception:
            pass
        _emulator_open = False

    if DISTANCE_SOURCE == "emulator":
        _open_emulator()

    fps_t = time.time(); fcnt = 0
    re_search_n = 30

    while True:
        ret, frame = cap.read()
        if not ret: break
        fcnt += 1
        fps = fcnt / (time.time() - fps_t + 1e-6)
        frame = cv2.flip(frame, 1)  # mirror horizontally
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"): break

        elif key == 9:  # TAB
            mode = "CALIBRATE" if mode == "TRACK" else "TRACK"
            tracker.reset(); calib.rect = None; auto_tried = False
            print(f"[MODE] -> {mode}")

        elif key == ord("r"):
            tracker.reset(); auto_tried = False
            print("[RESET] Will re-acquire")

        elif key == ord("s") and mode == "CALIBRATE":
            if calib.rect:
                profile.real_width_m = calib.real_width
                profile.calib_dist_m = calib.calib_dist
                profile.label        = calib.label
                profile.save(frame, calib.rect)
                tracker.seed(frame, calib.rect)
                mode = "TRACK"
            else:
                print("[WARN] Draw a box first")

        elif key == ord("W"): calib.real_width = round(calib.real_width + 0.05, 2)
        elif key == ord("w"): calib.real_width = max(0.05, round(calib.real_width - 0.05, 2))
        elif key == ord("D"): calib.calib_dist = round(calib.calib_dist + 0.5, 1)
        elif key == ord("d"): calib.calib_dist = max(0.5, round(calib.calib_dist - 0.5, 1))
        elif key == ord("l") or key == ord("L"):
            opts = ["animal","deer","dog","cat","person","other"]
            idx  = opts.index(calib.label) if calib.label in opts else 0
            calib.label = opts[(idx+1) % len(opts)]
            print(f"[LABEL] -> {calib.label}")

        elif key == ord("u") or key == ord("U"):
            # Simple toggle: vision <-> ultrasonic  (emulator stays as config default)
            if DISTANCE_SOURCE == "ultrasonic":
                DISTANCE_SOURCE = "vision"
                _close_emulator()
            else:
                DISTANCE_SOURCE = "ultrasonic"
                _close_emulator()
            calib.sensor_dist = None
            print(f"[DIST] source -> {DISTANCE_SOURCE}")

        # ── Read distance source every frame ──────────────────────────────────
        if DISTANCE_SOURCE == "ultrasonic":
            us_reading = ultrasonic.read()   # metres (raw, pre-scale)
            if us_reading is not None:
                us_scaled = us_reading * MOTION_SCALE
                if mode == "CALIBRATE":
                    calib.calib_dist  = us_scaled
                    calib.sensor_dist = us_scaled
                elif mode == "TRACK" and tracker.active:
                    # Ground-truth distance — overrides noisy pixel estimate
                    tracker.inject_distance(us_scaled)
        elif DISTANCE_SOURCE == "emulator":
            if _emulator_open:
                raw = cv2.getTrackbarPos("Distance_cm", EMULATOR_WIN)
                d   = max(0.20, raw / _EMULATOR_SCALE) * MOTION_SCALE
                if mode == "CALIBRATE":
                    calib.calib_dist  = round(d, 2)
                    calib.sensor_dist = calib.calib_dist

        if mode != "CALIBRATE":
            calib.sensor_dist = None   # hide badge outside calibration

        # Auto-acquire / periodic re-search
        if mode == "TRACK" and profile.is_ready:
            if not tracker.active and not auto_tried:
                tracker.try_acquire(frame); auto_tried = True
            elif tracker.lost and fcnt % re_search_n == 0:
                tracker.try_acquire(frame, min_score=0.50)

        # Calibrate: live preview
        if mode == "CALIBRATE":
            if calib.rect and not tracker.active:
                tracker.seed(frame, calib.rect)
            elif not calib.rect:
                tracker.reset()

        tracker.update(frame)
        ref_d = tracker.dist_f if (tracker.active and tracker.dist_f) else 3.0
        vspd.update(frame, tracker.bbox, ref_dist=ref_d)

        lvl, label, detail, col = assess(tracker, vspd)

        # ── Compose display ───────────────────────────────────────────────────
        # 1. Upscale raw frame (image content only, no text)
        frame_up = cv2.resize(frame, (DDISP_W, int(CAM_H * DS)),
                              interpolation=cv2.INTER_LINEAR)
        disp[:int(CAM_H * DS), :] = frame_up

        # 2. Draw all overlays + text onto the display-res canvas
        draw_path_zone(disp)
        draw_object(disp, tracker, col)
        draw_lost_indicator(disp, tracker)
        if mode == "CALIBRATE": calib.draw(disp)
        draw_banner(disp, mode, label, detail, col, fps)
        draw_hud(disp, tracker, vspd, profile, col, mode)

        cv2.imshow(WIN, disp)

    cap.release()
    _close_emulator()
    ultrasonic.cleanup()
    cv2.destroyAllWindows()
    print("[DONE]")


if __name__ == "__main__":
    main()