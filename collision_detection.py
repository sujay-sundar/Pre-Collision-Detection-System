"""
Pre-Collision Detection  —  minimal accurate tracker
======================================================
Modes (TAB to toggle):
  CALIBRATE  draw ROI box, set real width, press S to save
  TRACK      auto-loads profile, LK optical flow + Kalman distance

Keys:
  TAB    toggle mode
  S      save calibration (CALIBRATE mode)
  R      reset / force re-acquire
  Q      quit
  W / w  real width  +/- 0.05 m
  D / d  calib dist  +/- 0.5 m  (disabled when sensor active)
  L      cycle object label
  U      toggle distance source: emulator <-> ultrasonic

Distance source (DISTANCE_SOURCE config):
  "emulator"    trackbar slider — PC testing, no hardware needed
  "ultrasonic"  HC-SR04 on RPi GPIO14(TRIG) / GPIO15(ECHO)
                feeds BOTH calibration distance AND live tracking

MOTION_SCALE:
  Multiplies all distance/speed outputs.
  Set > 1 when your bench moves cm but you want to simulate real metres.
  Example: 0.10 m bench motion × 10 = 1.0 m displayed.
  Set 1.0 for real deployment.
"""

import cv2
import numpy as np
import json, os, time
from collections import deque

# ── Config ────────────────────────────────────────────────────────────────────
CAM_W, CAM_H    = 640, 480
FOCAL_PX        = 554.0          # pixels, valid for 640×480 ~60° FOV
PROFILE_FILE    = "profile.json"
TEMPLATE_FILE   = "template.png"

MOTION_SCALE    = 1.0            # set > 1 for bench-test amplification

# "emulator"    — on-screen trackbar, works on PC without hardware
# "ultrasonic"  — HC-SR04 via RPi.GPIO (gracefully no-ops on non-Pi)
DISTANCE_SOURCE = "emulator"     # change to "ultrasonic" on the Pi

TRIG_PIN = 14   # BCM GPIO
ECHO_PIN = 15   # BCM GPIO

EMUL_WIN = "Distance (cm)"      # trackbar window name

TTC_WARN = 4.0  # s
TTC_CRIT = 2.0  # s


# ── Ultrasonic Sensor (HC-SR04) ───────────────────────────────────────────────
class UltrasonicSensor:
    """
    Returns metres via read().
    Falls back to last valid reading on timeout — never returns stale None
    after the first successful ping. Returns None only before any valid echo.
    Gracefully no-ops when RPi.GPIO is not available (Windows / non-Pi).
    """
    _TIMEOUT = 0.04   # seconds

    def __init__(self, trig: int = TRIG_PIN, echo: int = ECHO_PIN):
        self._ok   = False
        self._last = None
        self._trig = trig
        self._echo = echo
        try:
            import RPi.GPIO as GPIO
            self._GPIO = GPIO
            GPIO.setmode(GPIO.BCM)
            GPIO.setup(trig, GPIO.OUT)
            GPIO.setup(echo, GPIO.IN)
            GPIO.output(trig, False)
            time.sleep(0.05)
            self._ok = True
            print(f"[US] OK  TRIG=GPIO{trig}  ECHO=GPIO{echo}")
        except ImportError:
            print("[US] RPi.GPIO not available — sensor disabled")
        except Exception as e:
            print(f"[US] init error: {e}")

    @property
    def available(self) -> bool:
        return self._ok

    def read(self) -> float | None:
        """Fire pulse, return metres. Last valid reading on error/timeout."""
        if not self._ok:
            return self._last
        GPIO = self._GPIO
        try:
            GPIO.output(self._trig, True)
            time.sleep(0.00001)
            GPIO.output(self._trig, False)

            t0 = time.time()
            while GPIO.input(self._echo) == 0:
                if time.time() - t0 > self._TIMEOUT:
                    return self._last
            t1 = time.time()

            while GPIO.input(self._echo) == 1:
                if time.time() - t1 > self._TIMEOUT:
                    return self._last
            t2 = time.time()

            d = (t2 - t1) * 17150 / 100   # speed-of-sound → metres
            if 0.02 <= d <= 4.0:
                self._last = round(d, 3)
            return self._last
        except Exception:
            return self._last

    def cleanup(self):
        if self._ok:
            try:
                self._GPIO.cleanup([self._trig, self._echo])
            except Exception:
                pass


# ── Kalman filter  (2-state: distance + radial velocity) ─────────────────────
class KalmanDist:
    """
    State: [distance_m, radial_speed_m_s]
    update(z, r):  r=0.004 for vision estimate, r=0.0002 for ultrasonic (20× trusted)
    """
    def __init__(self):
        self.x  = np.array([[2.0], [0.0]])
        self.P  = np.eye(2)
        self.H  = np.array([[1.0, 0.0]])
        self._t = None

    def update(self, z: float, r: float = 0.004) -> tuple:
        now = time.time()
        dt  = float(np.clip((now - self._t) if self._t else 0.033, 0.005, 0.2))
        self._t = now
        q   = 0.008
        F   = np.array([[1.0, -dt], [0.0, 1.0]])
        Q   = np.array([[q * dt**2, q * dt], [q * dt, q]])
        xp  = F @ self.x
        Pp  = F @ self.P @ F.T + Q
        S   = float((self.H @ Pp @ self.H.T + [[r]])[0, 0])
        K   = Pp @ self.H.T / S
        self.x = xp + K * (z - float((self.H @ xp)[0, 0]))
        self.P = (np.eye(2) - K @ self.H) @ Pp
        return max(0.01, float(self.x[0, 0])), float(self.x[1, 0])

    def reset(self, d: float = 2.0):
        self.x  = np.array([[d], [0.0]])
        self.P  = np.eye(2)
        self._t = None


# ── Object Tracker ────────────────────────────────────────────────────────────
LK_PARAMS = dict(winSize=(21, 21), maxLevel=3,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 25, 0.01))
FT_PARAMS = dict(maxCorners=150, qualityLevel=0.03, minDistance=6, blockSize=7)

class ObjectTracker:
    MIN_PTS      = 6     # go LOST when inliers drop below this
    SEED_GRACE   = 20    # skip verify for this many frames after seeding (let flow stabilise)
    VERIFY_N     = 5     # template-check every N frames after grace period
    VERIFY_THR   = 0.38  # min score to keep tracking
    VERIFY_MARGIN= 90    # px search margin around centre during verify
    ACQUIRE_THR  = 0.55  # min score for initial acquisition
    REACQ_THR    = 0.58  # min score for re-acquisition after LOST
    MAX_DRIFT_PX = 80    # max bbox centre shift per frame before LOST
    MIN_BBOX_PX  = 30    # ignore object if bbox width is below this (too far, unreliable)
    BW_ALPHA     = 0.25  # EMA alpha for bbox width → stable distance
    TTC_ALPHA    = 0.15  # slow EMA on TTC

    def __init__(self):
        self.kf        = KalmanDist()
        self._pts      = None
        self._pgray    = None
        self._bbox     = None
        self._state    = "IDLE"
        self._fn       = 0
        self._bw_ema   = None
        self._ttc_ema  = 99.0
        self._last_cx  = None   # last known centre-x (drift guard)
        self._last_cy  = None   # last known centre-y (drift guard)
        self._seed_bw  = None   # bbox width at seed time (for scale-aware verify)
        self.dist      = None
        self.speed     = 0.0
        self.trail     = deque(maxlen=40)
        self.real_w    = 0.5
        self.focal     = FOCAL_PX

    @property
    def active(self) -> bool: return self._state == "TRACKING"

    @property
    def lost(self)   -> bool: return self._state == "LOST"

    @property
    def centre(self):
        if self._bbox is None: return None
        x, y, w, h = self._bbox
        return x + w // 2, y + h // 2

    def dist_from_px(self, px_w: float) -> float:
        """Pixel-width → metres using calibrated focal length."""
        if px_w <= 0: return 99.0
        return (self.focal * self.real_w / px_w) * MOTION_SCALE

    def ttc(self) -> float:
        """EMA-smoothed time-to-collision (seconds)."""
        raw = min(self.dist / self.speed, 99.0) if (self.dist and self.speed > 0.05) else 99.0
        self._ttc_ema = self.TTC_ALPHA * raw + (1 - self.TTC_ALPHA) * self._ttc_ema
        return self._ttc_ema

    def inject_distance(self, dist_m: float):
        """Feed trusted external distance (ultrasonic/emulator) into Kalman with high trust."""
        d, v = self.kf.update(dist_m, r=0.0002)
        self.dist  = d
        self.speed = v

    def seed(self, frame: np.ndarray, rect: tuple) -> bool:
        """Start tracking from a drawn ROI."""
        x, y, w, h = [int(v) for v in rect]
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        patch = gray[max(0, y):y + h, max(0, x):x + w]
        corners = cv2.goodFeaturesToTrack(patch, **FT_PARAMS)
        if corners is None or len(corners) < 4:
            print("[TRACKER] Not enough texture — try a different box")
            return False
        self._pts   = corners + np.array([[[x, y]]], dtype=np.float32)
        self._pgray = gray.copy()
        self._bbox    = (x, y, w, h)
        self._state   = "TRACKING"
        self._fn      = 0
        self._bw_ema  = float(w)
        self._ttc_ema = 99.0
        self._seed_bw = float(w)
        self._last_cx = x + w // 2
        self._last_cy = y + h // 2
        d0 = self.dist_from_px(w)
        self.kf.reset(d0)
        self.dist  = d0
        self.speed = 0.0
        self.trail.clear()
        self.trail.append((x + w // 2, y + h // 2))
        print(f"[TRACKER] Seeded  pts={len(self._pts)}  d0={d0:.2f}m")
        return True

    def try_acquire(self, frame: np.ndarray, template: np.ndarray,
                    min_score: float = None) -> bool:
        """
        Multi-scale template search. Tries scales from 0.3× to 3× the saved
        template size so the object is found whether it is far or close.
        """
        if min_score is None:
            min_score = self.REACQ_THR if self._state == "LOST" else self.ACQUIRE_THR
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        th, tw = template.shape

        best_score = -1.0
        best_loc   = None
        best_scale = 1.0

        for scale in (0.30, 0.45, 0.60, 0.80, 1.00, 1.30, 1.60, 2.00, 2.50, 3.00):
            nw = max(4, int(tw * scale))
            nh = max(4, int(th * scale))
            if nw < self.MIN_BBOX_PX:          # object would be too far/small
                continue
            if nw > gray.shape[1] or nh > gray.shape[0]:
                continue
            scaled = cv2.resize(template, (nw, nh),
                                 interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
            res = cv2.matchTemplate(gray, scaled, cv2.TM_CCOEFF_NORMED)
            _, score, _, loc = cv2.minMaxLoc(res)
            if score > best_score:
                best_score = score
                best_loc   = loc
                best_scale = scale

        if best_score < min_score or best_loc is None:
            return False

        nw = max(4, int(tw * best_scale))
        nh = max(4, int(th * best_scale))
        print(f"[TRACKER] Acquired  score={best_score:.2f}  scale={best_scale:.2f}x  loc={best_loc}")
        return self.seed(frame, (best_loc[0], best_loc[1], nw, nh))

    def reset(self):
        self._pts = None; self._pgray = None; self._bbox = None
        self._state = "IDLE"; self._bw_ema = None; self._ttc_ema = 99.0
        self._last_cx = None; self._last_cy = None; self._seed_bw = None
        self.dist = None; self.speed = 0.0
        self.trail.clear(); self.kf.reset()

    def _go_lost(self):
        self._state = "LOST"; self._pts = None; self.speed = 0.0
        print("[TRACKER] LOST")

    def update(self, frame: np.ndarray, template: np.ndarray | None = None):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self._pgray is None:
            self._pgray = gray.copy(); return
        if self._state != "TRACKING":
            self._pgray = gray.copy(); return

        self._fn += 1
        new_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            self._pgray, gray, self._pts, None, **LK_PARAMS)
        good_new = new_pts[status.flatten() == 1]
        good_old = self._pts[status.flatten() == 1]

        if len(good_new) < self.MIN_PTS:
            self._go_lost(); self._pgray = gray.copy(); return

        # ── RANSAC: keep only points moving as one rigid body ────────────────
        if len(good_new) >= 6:
            _, mask = cv2.estimateAffinePartial2D(
                good_old.reshape(-1, 1, 2), good_new.reshape(-1, 1, 2),
                method=cv2.RANSAC, ransacReprojThreshold=2.0)
            if mask is not None:
                inliers = mask.flatten().astype(bool)
                if inliers.sum() >= self.MIN_PTS:
                    good_new = good_new[inliers]

        if len(good_new) < self.MIN_PTS:
            self._go_lost(); self._pgray = gray.copy(); return

        cx = int(np.median(good_new[:, 0, 0]))
        cy = int(np.median(good_new[:, 0, 1]))

        # ── Drift guard: centre can't jump more than MAX_DRIFT_PX ────────────
        if self._last_cx is not None:
            drift = ((cx - self._last_cx) ** 2 + (cy - self._last_cy) ** 2) ** 0.5
            if drift > self.MAX_DRIFT_PX:
                print(f"[TRACKER] Drift {drift:.0f}px > limit -> LOST")
                self._go_lost(); self._pgray = gray.copy(); return

        self._last_cx = cx; self._last_cy = cy

        # Robust bbox from inlier point spread
        xs = good_new[:, 0, 0]; ys = good_new[:, 0, 1]
        hw = max(int(np.percentile(xs, 85) - np.percentile(xs, 15)), 8)
        hh = max(int(np.percentile(ys, 85) - np.percentile(ys, 15)), 8)
        bx = max(0, cx - hw); by = max(0, cy - hh)
        bw = min(hw * 2, CAM_W - bx); bh = min(hh * 2, CAM_H - by)

        # EMA on bbox width
        self._bw_ema = self.BW_ALPHA * bw + (1 - self.BW_ALPHA) * self._bw_ema
        bw = max(4, int(self._bw_ema))

        # Too far — object too small to track reliably
        if bw < self.MIN_BBOX_PX:
            print(f"[TRACKER] bbox {bw}px < MIN ({self.MIN_BBOX_PX}px) -> LOST")
            self._go_lost(); self._pgray = gray.copy(); return

        self._bbox = (bx, by, bw, bh)
        self._pts  = good_new.reshape(-1, 1, 2)
        self.trail.append((cx, cy))

        # Vision-based distance (lower trust than sensor)
        d_raw = self.dist_from_px(bw)
        d, v  = self.kf.update(d_raw, r=0.004)
        self.dist  = d
        self.speed = v

        # ── Template verification (skip grace period, multi-scale) ──────────
        past_grace = self._fn > self.SEED_GRACE
        if template is not None and past_grace and self._fn % self.VERIFY_N == 0:
            th2, tw2 = template.shape
            base_scale = (bw / self._seed_bw) if (self._seed_bw and self._seed_bw > 0) else 1.0
            m  = self.VERIFY_MARGIN
            sx = max(0, cx - m); sy = max(0, cy - m)
            ex = min(CAM_W, cx + m); ey = min(CAM_H, cy + m)
            roi = gray[sy:ey, sx:ex]
            best_score = 0.0
            # Try base scale ±25% to tolerate EMA width noise
            for s in (base_scale * 0.75, base_scale, base_scale * 1.25):
                nw = max(4, int(tw2 * s)); nh = max(4, int(th2 * s))
                if nw > roi.shape[1] or nh > roi.shape[0]:
                    continue
                interp = cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR
                tmpl_s = cv2.resize(template, (nw, nh), interpolation=interp)
                sc = float(np.max(cv2.matchTemplate(roi, tmpl_s, cv2.TM_CCOEFF_NORMED)))
                if sc > best_score:
                    best_score = sc
            if best_score < self.VERIFY_THR:
                print(f"[TRACKER] Verify fail ({best_score:.2f}) -> LOST")
                self._go_lost(); self._pgray = gray.copy(); return

        self._pgray = gray.copy()


# ── Profile save / load ───────────────────────────────────────────────────────
def save_profile(frame: np.ndarray, rect: tuple, real_w: float,
                 calib_d: float, label: str):
    """Save template + JSON profile. Returns (template_gray, focal_px)."""
    x, y, w, h = rect
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    patch = gray[y:y + h, x:x + w]
    cv2.imwrite(TEMPLATE_FILE, patch)
    focal = w * calib_d / real_w   # calibrated focal length
    with open(PROFILE_FILE, "w") as f:
        json.dump(dict(real_w=real_w, calib_d=calib_d,
                       px_w=w, focal=focal, label=label), f, indent=2)
    print(f"[SAVE] {label}  real_w={real_w}m  calib_d={calib_d}m  "
          f"px_w={w}  focal={focal:.1f}")
    return patch, focal


def load_profile():
    """Returns (template_gray, focal, real_w, label) or None."""
    if not os.path.exists(PROFILE_FILE) or not os.path.exists(TEMPLATE_FILE):
        return None
    with open(PROFILE_FILE) as f:
        m = json.load(f)
    tmpl = cv2.imread(TEMPLATE_FILE, cv2.IMREAD_GRAYSCALE)
    if tmpl is None:
        return None
    print(f"[LOAD] {m.get('label','?')}  focal={m['focal']:.1f}  "
          f"real_w={m['real_w']}m")
    return tmpl, m["focal"], m["real_w"], m.get("label", "animal")


# ── Drawing helpers ───────────────────────────────────────────────────────────
_FONT = cv2.FONT_HERSHEY_SIMPLEX

def _t(img, text, x, y, scale, col, thick=1):
    cv2.putText(img, text, (x, y), _FONT, scale, col, thick, cv2.LINE_AA)


def draw_hud(frame: np.ndarray, tracker: ObjectTracker,
             mode: str, label: str, src: str, fps: float):
    h, w = frame.shape[:2]

    # Top banner
    cv2.rectangle(frame, (0, 0), (w, 30), (14, 16, 22), -1)
    mc = (0, 220, 255) if mode == "CALIBRATE" else (80, 200, 80)
    _t(frame, mode, 8, 22, 0.55, mc, 2)
    sc = (0, 255, 120) if src == "ultrasonic" else (80, 180, 255)
    _t(frame, src.upper(), w - 140, 22, 0.45, sc)
    _t(frame, f"FPS:{fps:.0f}", w - 72, 22, 0.38, (70, 80, 90))

    # LOST indicator
    if tracker.lost:
        if int(time.time() * 3) % 2 == 0:
            _t(frame, "TARGET LOST  searching...",
               w // 2 - 130, h // 2, 0.65, (0, 0, 220), 2)
        return

    if not tracker.active or tracker.dist is None:
        return

    # Bbox
    x, y, bw, bh = tracker._bbox
    d    = tracker.dist
    dc   = (60, 200, 255) if d > 2 else (0, 180, 255) if d > 1 else (0, 80, 255)
    cv2.rectangle(frame, (x, y), (x + bw, y + bh), dc, 2)
    _t(frame, label, x, max(y - 6, 38), 0.4, dc)

    # Trail
    trail = list(tracker.trail)
    for i in range(1, len(trail)):
        a  = i / len(trail)
        tc = tuple(int(c * a) for c in dc)
        cv2.line(frame, trail[i - 1], trail[i], tc, 1, cv2.LINE_AA)

    # Bottom info panel
    spd  = max(0.0, tracker.speed) * 3.6
    ttcv = tracker.ttc()
    tc   = (0, 40, 220) if ttcv < TTC_CRIT else \
           (0, 130, 255) if ttcv < TTC_WARN else (50, 220, 50)

    py = h - 68
    cv2.rectangle(frame, (0, py), (w, h), (12, 14, 20), -1)
    _t(frame, f"DIST: {d:.2f} m",  8, py + 20, 0.65, dc, 2)
    sc2 = (50, 220, 50) if spd < 5 else (0, 200, 255) if spd < 15 else (0, 60, 255)
    _t(frame, f"SPD:  {spd:.1f} km/h", 8, py + 42, 0.55, sc2)
    ttc_str = f"{ttcv:.1f} s" if ttcv < 99 else "---"
    _t(frame, f"TTC:  {ttc_str}", 8, py + 64, 0.65, tc, 2)

    # Warning overlay
    if ttcv < TTC_CRIT:
        if int(time.time() * 4) % 2 == 0:
            cv2.rectangle(frame, (0, 0), (w, h), (0, 0, 180), 5)
        _t(frame, "!! CRITICAL !!", w // 2 - 100, h // 2, 0.95, (0, 0, 220), 3)
    elif ttcv < TTC_WARN:
        _t(frame, "WARNING", w // 2 - 70, h // 2, 0.85, (0, 50, 255), 2)


def draw_calib_ui(frame: np.ndarray, drawing: bool, p0, p1,
                  rect, real_w: float, calib_d: float,
                  label: str, sensor_d, src: str):
    h, w = frame.shape[:2]

    # Drag preview
    if drawing and p0 and p1:
        cv2.rectangle(frame, p0, p1, (0, 220, 255), 1)

    # Finalised box
    if rect:
        rx, ry, rw, rh = rect
        cv2.rectangle(frame, (rx, ry), (rx + rw, ry + rh), (0, 255, 120), 2)
        _t(frame, f"{rw}x{rh}px  real:{real_w}m  @{calib_d:.2f}m",
           rx, max(ry - 6, 48), 0.38, (0, 255, 120))

    # 1 m ruler at bottom
    ruler_px = max(10, int(FOCAL_PX / calib_d))
    rx2 = w // 2 - ruler_px // 2; ry2 = h - 14
    cv2.line(frame, (rx2, ry2), (rx2 + ruler_px, ry2), (0, 200, 200), 2)
    cv2.line(frame, (rx2, ry2 - 5), (rx2, ry2 + 5), (0, 200, 200), 2)
    cv2.line(frame, (rx2 + ruler_px, ry2 - 5),
             (rx2 + ruler_px, ry2 + 5), (0, 200, 200), 2)
    _t(frame, f"<-- 1m @ {calib_d:.1f}m -->", rx2, ry2 - 8, 0.34, (0, 200, 200))

    # Sensor badge
    if src != "vision" and sensor_d is not None:
        bc = (0, 255, 120) if src == "ultrasonic" else (80, 180, 255)
        _t(frame, f"{src.upper()}: {sensor_d:.3f} m",
           w - 210, h - 88, 0.58, bc, 2)

    # Instruction line
    dist_hint = "sensor" if src == "ultrasonic" else "slider" if src == "emulator" \
        else f"D/d: {calib_d:.1f}m"
    _t(frame,
       f"W/w=width:{real_w:.2f}m  dist:{calib_d:.2f}m[{dist_hint}]  "
       f"[{label}]L  S=save  U=toggle",
       6, h - 4, 0.30, (160, 170, 185))


# ── Main ──────────────────────────────────────────────────────────────────────
WIN = "Pre-Collision Detector"


def main():
    global DISTANCE_SOURCE

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAM_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        print("[ERROR] No camera found"); return

    sensor  = UltrasonicSensor()
    tracker = ObjectTracker()

    template = None
    focal    = FOCAL_PX
    real_w   = 0.5
    label    = "animal"

    # Calibration UI state
    drawing  = False
    p0 = p1  = None
    rect     = None
    calib_d  = 2.0
    sensor_d = None   # live reading shown in calib badge

    mode       = "CALIBRATE"
    auto_tried = False

    # Load saved profile if available
    profile = load_profile()
    if profile:
        template, focal, real_w, label = profile
        tracker.real_w = real_w
        tracker.focal  = focal
        mode = "TRACK"
        print("[INFO] Profile loaded -> TRACK mode")
    else:
        print("[INFO] No profile -> CALIBRATE mode")

    # Emulator trackbar window
    _EMUL_MAX  = 1000   # cm  →  /100 = metres
    _emul_open = False

    def _open_emul():
        nonlocal _emul_open
        if _emul_open: return
        cv2.namedWindow(EMUL_WIN, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(EMUL_WIN, 420, 80)
        init = max(20, min(_EMUL_MAX, int(calib_d * 100)))
        cv2.createTrackbar("cm", EMUL_WIN, init, _EMUL_MAX, lambda _: None)
        _emul_open = True
        print("[EMULATOR] Trackbar opened")

    def _close_emul():
        nonlocal _emul_open
        if not _emul_open: return
        try: cv2.destroyWindow(EMUL_WIN)
        except Exception: pass
        _emul_open = False

    if DISTANCE_SOURCE == "emulator":
        _open_emul()

    cv2.namedWindow(WIN, cv2.WINDOW_AUTOSIZE)

    def on_mouse(ev, mx, my, _flags, _):
        nonlocal drawing, p0, p1, rect
        if mode != "CALIBRATE": return
        mx = max(0, min(CAM_W - 1, mx))
        my = max(30, min(CAM_H - 1, my))
        if ev == cv2.EVENT_LBUTTONDOWN:
            drawing = True; p0 = (mx, my); p1 = (mx, my); rect = None
        elif ev == cv2.EVENT_MOUSEMOVE and drawing:
            p1 = (mx, my)
        elif ev == cv2.EVENT_LBUTTONUP:
            drawing = False; p1 = (mx, my)
            x1 = min(p0[0], p1[0]); y1 = min(p0[1], p1[1])
            x2 = max(p0[0], p1[0]); y2 = max(p0[1], p1[1])
            rect = (x1, y1, x2 - x1, y2 - y1) if (x2 - x1 > 10 and y2 - y1 > 10) else None

    cv2.setMouseCallback(WIN, on_mouse)

    print(f"[DIST] source = {DISTANCE_SOURCE}")
    print("[KEYS] TAB=mode  S=save  R=reset  Q=quit  W/w=width  D/d=dist  L=label  U=toggle")

    fps_t = time.time(); fcnt = 0

    while True:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame, 1)
        fcnt += 1
        fps = fcnt / (time.time() - fps_t + 1e-6)

        key = cv2.waitKey(1) & 0xFF

        # ── Key handling ──────────────────────────────────────────────────────
        if key == ord("q"):
            break

        elif key == 9:   # TAB
            mode = "CALIBRATE" if mode == "TRACK" else "TRACK"
            tracker.reset(); rect = None; auto_tried = False
            print(f"[MODE] -> {mode}")

        elif key == ord("r") or key == ord("R"):
            tracker.reset(); auto_tried = False
            print("[RESET]")

        elif key == ord("s") and mode == "CALIBRATE":
            if rect:
                template, focal = save_profile(frame, rect, real_w, calib_d, label)
                tracker.real_w = real_w
                tracker.focal  = focal
                tracker.seed(frame, rect)
                mode = "TRACK"
            else:
                print("[WARN] Draw a box first")

        elif key == ord("W"): real_w = round(real_w + 0.05, 2)
        elif key == ord("w"): real_w = max(0.05, round(real_w - 0.05, 2))
        elif key == ord("D"): calib_d = round(calib_d + 0.5, 1)
        elif key == ord("d"): calib_d = max(0.3, round(calib_d - 0.5, 1))

        elif key == ord("l") or key == ord("L"):
            opts = ["animal", "deer", "dog", "cat", "person", "other"]
            label = opts[(opts.index(label) + 1) % len(opts)] \
                if label in opts else opts[0]
            print(f"[LABEL] -> {label}")

        elif key == ord("u") or key == ord("U"):
            DISTANCE_SOURCE = "emulator" if DISTANCE_SOURCE == "ultrasonic" \
                else "ultrasonic"
            if DISTANCE_SOURCE == "emulator": _open_emul()
            else:                             _close_emul()
            sensor_d = None
            print(f"[DIST] -> {DISTANCE_SOURCE}")

        # ── Distance source: read every frame ─────────────────────────────────
        if DISTANCE_SOURCE == "ultrasonic":
            reading = sensor.read()
            if reading is not None:
                scaled = reading * MOTION_SCALE
                if mode == "CALIBRATE":
                    calib_d  = scaled
                    sensor_d = scaled
                elif tracker.active:
                    tracker.inject_distance(scaled)

        elif DISTANCE_SOURCE == "emulator" and _emul_open:
            raw_cm = cv2.getTrackbarPos("cm", EMUL_WIN)
            d_m    = max(0.20, raw_cm / 100) * MOTION_SCALE
            if mode == "CALIBRATE":
                calib_d  = round(d_m, 2)
                sensor_d = calib_d
            elif tracker.active:
                tracker.inject_distance(d_m)

        if mode != "CALIBRATE":
            sensor_d = None   # hide badge outside calibration

        # ── Track mode: auto-acquire + periodic re-search ─────────────────────
        if mode == "TRACK" and template is not None:
            if not tracker.active and not auto_tried:
                tracker.real_w = real_w; tracker.focal = focal
                tracker.try_acquire(frame, template)  # uses ACQUIRE_THR
                auto_tried = True
            elif tracker.lost and fcnt % 30 == 0:
                tracker.try_acquire(frame, template)   # uses REACQ_THR

        # ── Calibrate mode: live preview ──────────────────────────────────────
        if mode == "CALIBRATE":
            if rect:
                if not tracker.active:
                    tracker.real_w = real_w; tracker.focal = focal
                    tracker.seed(frame, rect)
            else:
                if tracker.active:
                    tracker.reset()

        # ── Update tracker ────────────────────────────────────────────────────
        tracker.update(frame, template)

        # ── Compose and show ──────────────────────────────────────────────────
        disp = frame.copy()
        if mode == "CALIBRATE":
            draw_calib_ui(disp, drawing, p0, p1,
                          rect, real_w, calib_d, label, sensor_d, DISTANCE_SOURCE)
        draw_hud(disp, tracker, mode, label, DISTANCE_SOURCE, fps)
        cv2.imshow(WIN, disp)

    cap.release()
    _close_emul()
    sensor.cleanup()
    cv2.destroyAllWindows()
    print("[DONE]")


if __name__ == "__main__":
    main()
