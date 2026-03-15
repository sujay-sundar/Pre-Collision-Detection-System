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

TTC_WARN            = 4.0   # s
TTC_CRIT            = 2.0   # s

EMULATOR_WIN        = "Distance Emulator"
BUZZER_PIN          = 18    # BCM GPIO — PWM buzzer
PROX_WARN_M         = 0.30  # metres — triggers <30 cm danger alert + buzzer

PATH_ZONE_X     = (int(CAM_W * 0.25), int(CAM_W * 0.75))
CROSS_SPD_FAST  = 1.5   # m/s lateral crossing threshold
VEH_SPD_CONCERN = 5.0   # m/s vehicle speed concern (~18 km/h)

DS      = 1.0            # display scale (1.0 = native resolution)
PH      = 160            # HUD panel height (logical px)
WIN_H   = CAM_H + PH     # total canvas height
DDISP_W = CAM_W          # display width


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


class BuzzerAlert:
    """
    level 0 = silent
    level 1 = slow beep  1 Hz  — CAUTION
    level 2 = fast beep  4 Hz  — WARNING
    level 3 = continuous       — CRITICAL / <30 cm
    """
    def __init__(self, pin: int = BUZZER_PIN):
        self._ok = False
        self._pin = pin
        try:
            import RPi.GPIO as GPIO
            self._GPIO = GPIO
            GPIO.setmode(GPIO.BCM)
            GPIO.setup(pin, GPIO.OUT)
            self._pwm = GPIO.PWM(pin, 2000)
            self._pwm.start(0)
            self._ok = True
            print(f"[BUZZER] OK  GPIO{pin}")
        except ImportError:
            print("[BUZZER] RPi.GPIO not available — buzzer disabled")
        except Exception as e:
            print(f"[BUZZER] Init error: {e}")

    def alert(self, level: int):
        if not self._ok: return
        t = time.time()
        if level == 0:   self._pwm.ChangeDutyCycle(0)
        elif level == 1: self._pwm.ChangeDutyCycle(50 if int(t * 2) % 2 == 0 else 0)
        elif level == 2: self._pwm.ChangeDutyCycle(50 if int(t * 8) % 2 == 0 else 0)
        else:            self._pwm.ChangeDutyCycle(50)

    def cleanup(self):
        if self._ok:
            try: self._pwm.stop(); self._GPIO.cleanup([self._pin])
            except Exception: pass




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
        self._lat_ms   = 0.0
        self.trail     = deque(maxlen=40)
        self.spd_hist  = deque(maxlen=60)
        self.dist_hist = deque(maxlen=60)
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

    @property
    def bbox(self):        return self._bbox
    @property
    def dist_f(self):      return self.dist
    @property
    def speed_f(self):     return self.speed
    @property
    def lateral_ms(self):  return self._lat_ms

    def speed_kmh(self) -> float:
        return max(0.0, self.speed) * 3.6

    def lateral_kmh(self) -> float:
        return abs(self._lat_ms) * 3.6

    def crossing_dir(self) -> str:
        if self._lat_ms >  0.3: return "→"
        if self._lat_ms < -0.3: return "←"
        return ""

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

        # Lateral speed (EMA on cx change → m/s estimate)
        if self._last_cx is not None and d > 0:
            dpx = cx - self._last_cx
            lat = dpx * 30.0 * d / max(self.focal, 1.0)  # px/frame → m/s
            self._lat_ms = 0.7 * self._lat_ms + 0.3 * lat
        self.spd_hist.append(self.speed_kmh())
        self.dist_hist.append(d)

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


class TrackerProfile:
    """Wraps saved calibration data (focal, real_w, template, label)."""
    def __init__(self):
        self._tmpl  = None
        self._f     = FOCAL_PX
        self._rw    = 0.5
        self.label  = "animal"
        self._ready = False

    def load(self) -> bool:
        r = load_profile()
        if r:
            self._tmpl, self._f, self._rw, self.label = r
            self._ready = True
        return self._ready

    def save(self, frame, rect, real_w, calib_d, label):
        tmpl, f = save_profile(frame, rect, real_w, calib_d, label)
        self._tmpl = tmpl; self._f = f; self._rw = real_w
        self.label = label; self._ready = True

    @property
    def is_ready(self) -> bool:      return self._ready
    @property
    def template(self):              return self._tmpl
    @property
    def real_width_m(self) -> float: return self._rw
    def _focal(self) -> float:       return self._f


class VehicleSpeed:
    """Background optical flow → vehicle speed estimate."""
    def __init__(self):
        self.veh_ms   = 0.0
        self.spd_hist = deque(maxlen=60)
        self._pgray   = None
        self._corners = None

    def update(self, frame: np.ndarray, bbox, ref_dist: float = 3.0):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self._pgray is None:
            self._pgray = gray.copy(); return

        mask = np.ones_like(gray)
        if bbox is not None:
            x, y, bw, bh = bbox
            mask[max(0, y-10):y+bh+10, max(0, x-10):x+bw+10] = 0

        if self._corners is None or len(self._corners) < 10:
            self._corners = cv2.goodFeaturesToTrack(
                gray, maxCorners=60, qualityLevel=0.01,
                minDistance=8, blockSize=7, mask=mask)

        if self._corners is not None and len(self._corners) >= 4:
            new_c, status, _ = cv2.calcOpticalFlowPyrLK(
                self._pgray, gray, self._corners, None, **LK_PARAMS)
            ok = status.flatten() == 1
            gn, go = new_c[ok], self._corners[ok]
            if len(gn) >= 4:
                dx = gn[:, 0, 0] - go[:, 0, 0]
                dy = gn[:, 0, 1] - go[:, 0, 1]
                flow_px  = float(np.median(np.sqrt(dx**2 + dy**2)))
                px_per_m = FOCAL_PX / max(ref_dist, 0.5)
                v = max(0.0, flow_px * 30.0 / px_per_m - 0.05)
                self.veh_ms = 0.8 * self.veh_ms + 0.2 * v
                self._corners = gn.reshape(-1, 1, 2) if len(gn) >= 10 else None
            else:
                self._corners = None
        else:
            self._corners = None

        self.spd_hist.append(self.veh_ms * 3.6)
        self._pgray = gray.copy()

    def kmh(self) -> float:
        return self.veh_ms * 3.6


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
# COLLISION RISK SCORE  (0 – 100 %)
# ──────────────────────────────────────────────────────────────────────────────
def collision_risk(tracker: ObjectTracker, vspd: VehicleSpeed) -> tuple:
    """Returns (pct 0-100, label, colour_BGR) from 5 weighted factors."""
    if not tracker.active or tracker.dist_f is None:
        return 0, "SAFE", (50, 200, 50)

    d     = tracker.dist_f
    v_obj = max(0.0, tracker.speed_f)
    v_veh = vspd.veh_ms
    cspd  = v_obj + v_veh
    ttc   = tracker.ttc()

    in_path = False
    if tracker.centre:
        cx, _ = tracker.centre
        in_path = PATH_ZONE_X[0] < cx < PATH_ZONE_X[1]

    v_lat = abs(tracker.lateral_ms)

    # Factor scores (each 0–1)
    f_ttc  = max(0.0, min(1.0, (TTC_WARN * 2 - ttc) / (TTC_WARN * 2))) if ttc < 99 else 0.0
    f_cspd = min(1.0, cspd / 15.0)
    f_dist = max(0.0, min(1.0, (5.0 - d) / 5.0))
    f_path = 1.0 if in_path else 0.4
    f_lat  = 1.0 if v_lat < 0.5 else max(0.3, 1.0 - (v_lat - 0.5) / CROSS_SPD_FAST)
    if in_path and v_lat > CROSS_SPD_FAST:
        f_lat = 1.0

    if d < PROX_WARN_M:
        pct = 100
    else:
        raw = (f_ttc * 0.40 + f_cspd * 0.25 + f_dist * 0.20 + 0.15) * f_path * f_lat
        pct = int(min(100, max(0, raw * 100)))

    if pct < 20:   return pct, "SAFE",     (50, 200, 50)
    elif pct < 45: return pct, "LOW RISK", (0, 220, 180)
    elif pct < 65: return pct, "CAUTION",  (0, 190, 255)
    elif pct < 82: return pct, "WARNING",  (0, 80, 255)
    else:          return pct, "CRITICAL", (0, 20, 220)


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

def s(v):   return int(round(v * DS))
def sp(pt): return (s(pt[0]), s(pt[1]))

def txt(img, text, x, y, scale, col, thick=1):
    cv2.putText(img, text, sp((x, y)), _FONT, scale * DS, col, thick, cv2.LINE_AA)

def sparkline(img, data, x, y, w, h, col, lo, hi):
    pts = list(data)
    if len(pts) < 2: return
    rng = max(hi - lo, 0.001)
    prev = None
    for i, v in enumerate(pts):
        px = s(x) + int(i * s(w) / (len(pts) - 1))
        py = s(y + h) - int((v - lo) / rng * s(h))
        py = max(s(y), min(s(y + h), py))
        cur = (px, py)
        if prev: cv2.line(img, prev, cur, col, 1, cv2.LINE_AA)
        prev = cur


def draw_object(disp, tracker: ObjectTracker, col):
    if not tracker.active or tracker._bbox is None:
        return
    x, y, bw, bh = tracker._bbox
    cv2.rectangle(disp, sp((x, y)), sp((x+bw, y+bh)), col, max(1, s(2)))
    trail = list(tracker.trail)
    for i in range(1, len(trail)):
        a  = i / len(trail)
        tc = tuple(int(c * a) for c in col)
        cv2.line(disp, sp(trail[i-1]), sp(trail[i]), tc, 1, cv2.LINE_AA)


def draw_lost_indicator(disp, tracker: ObjectTracker):
    if not tracker.lost: return
    if int(time.time() * 3) % 2 == 0:
        txt(disp, "TARGET LOST  searching...",
            CAM_W // 2 - 130, CAM_H // 2, 0.65, (0, 0, 220), thick=2)


def draw_distance_rings(disp, profile: TrackerProfile, tracker: ObjectTracker):
    """
    Semi-elliptic distance rings from bottom-centre of frame.
    Ring nearest the live object distance pulses and highlights;
    object position is shown as a dot on that ring arc.
    """
    if not profile.is_ready:
        return
    f  = profile._focal()
    rw = profile.real_width_m
    ox = CAM_W // 2
    oy = CAM_H

    rings = [
        (4.0, (0, 140,  50), "4m"),
        (2.0, (0, 200, 120), "2m"),
        (1.0, (0, 170, 255), "1m"),
        (0.5, (0,  80, 255), "0.5m"),
        (0.3, (0,  20, 200), "30cm"),
    ]
    cur_dist = tracker.dist_f if (tracker.active and tracker.dist_f) else None

    for dist, base_col, label in rings:
        r = int(f * rw / dist)
        if r < 15 or r > CAM_W * 3:
            continue
        ry = max(r // 3, 12)

        active = cur_dist is not None and abs(cur_dist - dist) < dist * 0.55
        pulse  = active and int(time.time() * 4) % 2 == 0
        color  = tuple(min(255, int(c * 1.6)) for c in base_col) if active else base_col
        thick  = max(1, s(2)) if pulse else (2 if active else 1)

        cv2.ellipse(disp, sp((ox, oy)), (s(r), s(ry)),
                    0, 180, 360, color, thick, cv2.LINE_AA)

        lx = ox + r - 2
        if s(4) < s(lx) < s(CAM_W - 2):
            txt(disp, label, lx, oy - 4, 0.32 if active else 0.27, color)

        # Object position dot on the active ring arc
        if active and tracker.centre:
            cx_obj, _ = tracker.centre
            rel = float(np.clip((cx_obj - ox) / max(r, 1), -1.0, 1.0))
            arc_y = int(oy - ry * np.sqrt(max(0.0, 1.0 - rel * rel)))
            cv2.circle(disp, (s(cx_obj), s(arc_y)), s(5), color, -1, cv2.LINE_AA)
            cv2.circle(disp, (s(cx_obj), s(arc_y)), s(7), (255, 255, 255), 1, cv2.LINE_AA)


def draw_path_zone(disp):
    """Subtle green corridor overlay — logical coords scaled."""
    x1, x2 = PATH_ZONE_X
    ov = disp.copy()
    cv2.rectangle(ov, sp((x1, 0)), sp((x2, CAM_H)), (0, 200, 80), -1)
    cv2.addWeighted(ov, 0.06, disp, 0.94, 0, disp)
    cv2.line(disp, sp((x1, 0)), sp((x1, CAM_H)), (0, 160, 60), 1)
    cv2.line(disp, sp((x2, 0)), sp((x2, CAM_H)), (0, 160, 60), 1)
def draw_proximity_alert(disp, tracker: ObjectTracker, buzzer_lvl: int = 0):
    """Red frame flash + DANGER text when <30 cm; persistent buzzer badge."""
    # ── Buzzer badge (always visible when buzzer active) ──────────────────────
    if buzzer_lvl > 0:
        blink = int(time.time() * (2 if buzzer_lvl == 1 else 6)) % 2 == 0
        bcol  = (0, 30, 220) if buzzer_lvl >= 3 else \
                (0, 100, 255) if buzzer_lvl == 2 else (0, 180, 255)
        bg    = bcol if blink else (30, 32, 42)
        bx, by, bw, bh = CAM_W - 92, 50, 88, 22
        cv2.rectangle(disp, sp((bx, by)), sp((bx+bw, by+bh)), bg, -1)
        cv2.rectangle(disp, sp((bx, by)), sp((bx+bw, by+bh)), bcol, 1)
        lbl = ["", "BUZZ slow", "BUZZ fast", "BUZZ ON!!"][min(buzzer_lvl, 3)]
        txt(disp, lbl, bx+4, by+bh-5, 0.30,
            (255, 255, 255) if blink else bcol)

    # ── <30 cm danger overlay ─────────────────────────────────────────────────
    if not (tracker.active and tracker.dist_f is not None
            and tracker.dist_f < PROX_WARN_M):
        return
    frame_h = int(CAM_H * DS)
    if int(time.time() * 8) % 2 == 0:
        ov = disp[:frame_h].copy()
        cv2.rectangle(ov, (0, 0), (s(CAM_W), frame_h), (0, 0, 180), -1)
        cv2.addWeighted(ov, 0.35, disp[:frame_h], 0.65, 0, disp[:frame_h])
    cx = CAM_W // 2 - 145
    cy = CAM_H // 2 + 30
    txt(disp, "!! DANGER < 30 cm !!", cx,     cy,   1.0, (0,   0, 200), thick=4)
    txt(disp, "!! DANGER < 30 cm !!", cx - 1, cy-1, 1.0, (255, 255, 255), thick=1)


def draw_collision_bar(disp, risk_pct: int, risk_label: str, risk_col: tuple):
    """Full-width collision risk bar at the frame/HUD boundary."""
    BY = CAM_H
    BH = 18
    cv2.rectangle(disp, sp((0, BY)), sp((CAM_W, BY+BH)), (20, 22, 30), -1)
    bar_w = int(CAM_W * risk_pct / 100)
    if bar_w > 0:
        cv2.rectangle(disp, sp((0, BY)), sp((bar_w, BY+BH)), risk_col, -1)
    for mark in (20, 45, 65, 82):
        mx = int(CAM_W * mark / 100)
        cv2.line(disp, sp((mx, BY)), sp((mx, BY+BH)), (50, 52, 62), 1)
    cv2.rectangle(disp, sp((0, BY)), sp((CAM_W, BY+BH)), (50, 52, 62), 1)
    text_col = (255, 255, 255) if risk_pct > 44 else (180, 190, 200)
    txt(disp, f"COLLISION RISK  {risk_pct}%  {risk_label}",
        CAM_W//2 - 90, BY+BH-4, 0.32, text_col)


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
             profile: TrackerProfile, alert_col, mode: str,
             risk_pct: int = 0, risk_col: tuple = (50, 200, 50)):
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
    RW = CAM_W - RX - 4

    # Vehicle speed
    vc = (80, 130, 255)
    txt(disp, "VEHICLE",           RX, PT+14, 0.28, (75, 82, 96))
    txt(disp, f"{vspd.kmh():.1f}", RX, PT+36, 0.68, vc, thick=2)
    txt(disp, "km/h",              RX, PT+48, 0.26, (55, 62, 75))

    # TTC compact
    ttcv = tracker.ttc()
    gc   = (0,40,220) if ttcv<2 else (0,130,255) if ttcv<4 else \
           (0,220,255) if ttcv<6 else (50,220,50)
    ttc_str = f"{ttcv:.1f}s" if ttcv < 99 else "---"
    txt(disp, "TTC",    RX, PT+65, 0.27, (75, 82, 96))
    txt(disp, ttc_str,  RX, PT+80, 0.50, gc, thick=2)

    # Collision risk arc gauge
    gx = RX + RW // 2
    gy = PT + 130
    gr = 32
    cv2.ellipse(disp, sp((gx, gy)), (s(gr), s(gr)), 0, 200, 340, (38,42,52), s(5))
    cv2.ellipse(disp, sp((gx, gy)), (s(gr), s(gr)), 0, 200,
                int(200 + (risk_pct / 100) * 140), risk_col, s(5), cv2.LINE_AA)
    txt(disp, "RISK",         gx-14, gy-10, 0.27, (75, 82, 96))
    txt(disp, f"{risk_pct}%", gx-14, gy+10, 0.55, risk_col, thick=2)

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
    global DISTANCE_SOURCE

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAM_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        print("[ERROR] No camera found"); return

    profile    = TrackerProfile()
    tracker    = ObjectTracker()
    vspd       = VehicleSpeed()
    calib      = CalibUI()
    ultrasonic = UltrasonicSensor(TRIG_PIN, ECHO_PIN)
    buzzer     = BuzzerAlert()

    mode       = "CALIBRATE"
    auto_tried = False

    # Load saved profile
    if profile.load():
        tracker.real_w = profile.real_width_m
        tracker.focal  = profile._focal()
        calib.real_width = profile.real_width_m
        calib.label      = profile.label
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
        init = max(20, min(_EMUL_MAX, int(calib.calib_dist * 100)))
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

    def on_mouse(ev, mx, my, flags, param):
        if mode != "CALIBRATE": return
        calib.on_mouse(ev, mx, my, flags, param)

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
            tracker.reset(); calib.rect = None; auto_tried = False
            print(f"[MODE] -> {mode}")

        elif key == ord("r") or key == ord("R"):
            tracker.reset(); auto_tried = False
            print("[RESET]")

        elif key == ord("s") and mode == "CALIBRATE":
            if calib.rect:
                profile.save(frame, calib.rect, calib.real_width,
                             calib.calib_dist, calib.label)
                tracker.real_w = profile.real_width_m
                tracker.focal  = profile._focal()
                tracker.seed(frame, calib.rect)
                mode = "TRACK"
            else:
                print("[WARN] Draw a box first")

        elif key == ord("W"): calib.real_width = round(calib.real_width + 0.05, 2)
        elif key == ord("w"): calib.real_width = max(0.05, round(calib.real_width - 0.05, 2))
        elif key == ord("D"): calib.calib_dist = round(calib.calib_dist + 0.5, 1)
        elif key == ord("d"): calib.calib_dist = max(0.3, round(calib.calib_dist - 0.5, 1))

        elif key == ord("l") or key == ord("L"):
            opts = ["animal", "deer", "dog", "cat", "person", "other"]
            calib.label = opts[(opts.index(calib.label) + 1) % len(opts)] \
                if calib.label in opts else opts[0]
            print(f"[LABEL] -> {calib.label}")

        elif key == ord("u") or key == ord("U"):
            DISTANCE_SOURCE = "emulator" if DISTANCE_SOURCE == "ultrasonic" \
                else "ultrasonic"
            if DISTANCE_SOURCE == "emulator": _open_emul()
            else:                             _close_emul()
            calib.sensor_dist = None
            print(f"[DIST] -> {DISTANCE_SOURCE}")

        # ── Distance source: read every frame ─────────────────────────────────
        if DISTANCE_SOURCE == "ultrasonic":
            reading = ultrasonic.read()
            if reading is not None:
                scaled = reading * MOTION_SCALE
                if mode == "CALIBRATE":
                    calib.calib_dist  = scaled
                    calib.sensor_dist = scaled
                elif tracker.active:
                    tracker.inject_distance(scaled)

        elif DISTANCE_SOURCE == "emulator" and _emul_open:
            raw_cm = cv2.getTrackbarPos("cm", EMUL_WIN)
            d_m    = max(0.20, raw_cm / 100) * MOTION_SCALE
            if mode == "CALIBRATE":
                calib.calib_dist  = round(d_m, 2)
                calib.sensor_dist = calib.calib_dist
            elif tracker.active:
                tracker.inject_distance(d_m)

        if mode != "CALIBRATE":
            calib.sensor_dist = None

        # ── Track mode: auto-acquire + periodic re-search ─────────────────────
        if mode == "TRACK" and profile.is_ready:
            if not tracker.active and not auto_tried:
                tracker.real_w = profile.real_width_m
                tracker.focal  = profile._focal()
                tracker.try_acquire(frame, profile.template)
                auto_tried = True
            elif tracker.lost and fcnt % 30 == 0:
                tracker.try_acquire(frame, profile.template)

        # ── Calibrate mode: live preview ──────────────────────────────────────
        if mode == "CALIBRATE":
            if calib.rect:
                if not tracker.active:
                    tracker.real_w = calib.real_width
                    tracker.focal  = FOCAL_PX * calib.calib_dist / max(calib.real_width, 0.01)
                    tracker.seed(frame, calib.rect)
            else:
                if tracker.active:
                    tracker.reset()

        tracker.update(frame, profile.template)
        ref_d = tracker.dist if (tracker.active and tracker.dist) else 3.0
        vspd.update(frame, tracker._bbox, ref_dist=ref_d)

        lvl, label, detail, col = assess(tracker, vspd)
        risk_pct, risk_label, risk_col = collision_risk(tracker, vspd)

        # ── Buzzer level ──────────────────────────────────────────────────────
        prox = (tracker.active and tracker.dist is not None
                and tracker.dist < PROX_WARN_M)
        if prox or lvl >= 5:   buzz_lvl = 3
        elif lvl >= 4:         buzz_lvl = 2
        elif lvl >= 2:         buzz_lvl = 1
        else:                  buzz_lvl = 0
        buzzer.alert(buzz_lvl)

        # ── Compose display ───────────────────────────────────────────────────
        disp = frame.copy()
        draw_path_zone(disp)
        draw_distance_rings(disp, profile, tracker)
        draw_object(disp, tracker, col)
        draw_lost_indicator(disp, tracker)
        draw_proximity_alert(disp, tracker, buzz_lvl)
        if mode == "CALIBRATE": calib.draw(disp)
        draw_banner(disp, mode, label, detail, col, fps)
        draw_collision_bar(disp, risk_pct, risk_label, risk_col)
        draw_hud(disp, tracker, vspd, profile, col, mode, risk_pct, risk_col)
        cv2.imshow(WIN, disp)

    cap.release()
    _close_emul()
    ultrasonic.cleanup()
    buzzer.cleanup()
    cv2.destroyAllWindows()
    print("[DONE]")


if __name__ == "__main__":
    main()
