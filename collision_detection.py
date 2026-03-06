"""
ArUco Tracker — Multi-Level Warning System
==========================================
Warning levels:
  0  CLEAR            — nothing in path
  1  SAFE CROSSING    — object leaving, TTC high, growth negative/zero
  2  MONITORING       — object in path, low risk
  3  CAUTION          — moderate approach, possible conflict
  4  WARNING          — likely conflict — honk + light
  5  CRITICAL         — imminent — brake signal
  6  COLLISION        — confirmed impact — log trigger

Edge cases handled:
  • OBSTRUCTION       — object stationary inside path zone
  • RE-ENTRY          — object re-enters path after leaving (escalates faster)
  • ABORTED CROSS     — object entered then left without collision
  • LOST TRACK        — marker lost during high-risk state (hold alert)
  • RECEDING          — object was close but now moving away
"""

import cv2
import numpy as np
import time
import sys
import os
from collections import deque

# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════
# Processing resolution — NEVER change these (all CV logic uses these coords)
CAM_W, CAM_H        = 640, 480

# Panel drawn at native resolution then scaled up with canvas
PANEL_H             = 200          # taller panel = more room for bigger text
WIN_H               = CAM_H + PANEL_H

# ── Display scale ─────────────────────────────────────────────────────────────
# DISP_SCALE scales the FINAL canvas before imshow only.
# 1.0 = 640×680,  1.5 = 960×1020,  2.0 = 1280×1360
# All detection, tracking, and drawing happens at CAM_W × CAM_H — zero overhead.
DISP_SCALE          = 1.5
DISP_W              = int(CAM_W  * DISP_SCALE)
DISP_H              = int(WIN_H  * DISP_SCALE)

MARKER_REAL_SIZE_M  = 0.10
TARGET_ID           = 0
ARUCO_DICT          = cv2.aruco.DICT_4X4_50
FOCAL_PX            = 554.0

TRAIL_LEN           = 40
PREDICT_STEPS       = 30
GROWTH_WIN          = 15
DIST_HIST_LEN       = 80
PERSIST_FRAMES      = 25

# State hold time — when marker lost during high-risk, hold alert for N frames
HOLD_FRAMES         = 20

# Obstruction: in path zone for this many frames without significant movement
OBSTRUCTION_FRAMES  = 45

# Re-entry: if object left path and comes back within this many frames
REENTRY_WINDOW      = 60

# Plot colours (BGR)
PLOT_BG    = (16, 18, 26)
GRID_COL   = (32, 38, 52)
COL_DIST   = (60,  200, 255)
COL_SPEED  = (100, 255, 140)   # object speed
COL_GR     = (255, 160, 60)
COL_TTC    = (200, 80,  255)
COL_VSPD   = (80,  120, 255)   # vehicle speed (IMU)
COL_CLOSE  = (60,  230, 255)   # closing speed (fused)
COL_FTCC   = (255, 80,  180)   # fused TTC

# IMU simulation
IMU_DT          = 0.033        # 30 Hz
IMU_NOISE_ACCEL = 0.025        # g  — realistic MPU6050 noise
IMU_NOISE_GYRO  = 0.30         # °/s
IMU_BIAS_DRIFT  = 0.0003       # slow bias creep per sample

# ── Alert level definitions ────────────────────────────────────────────────────
#   Each level: (label, banner_bg_BGR, banner_fg_BGR, bar_colour, sound, gpio)
LEVELS = {
    0: dict(label="CLEAR",           bg=(8,35,8),    fg=(50,240,50),  bar=(50,200,50),  sound=False, gpio=False),
    1: dict(label="SAFE CROSSING",   bg=(8,40,20),   fg=(60,255,120), bar=(60,220,100), sound=False, gpio=False),
    2: dict(label="MONITORING",      bg=(10,35,55),  fg=(0,200,255),  bar=(0,180,220),  sound=False, gpio=False),
    3: dict(label="CAUTION",         bg=(10,45,70),  fg=(0,160,255),  bar=(0,130,255),  sound=False, gpio=True ),
    4: dict(label="!! WARNING !!",   bg=(5,25,80),   fg=(0,100,255),  bar=(0,80,255),   sound=True,  gpio=True ),
    5: dict(label="!! CRITICAL !!",  bg=(0,0,70),    fg=(0,50,255),   bar=(0,30,255),   sound=True,  gpio=True ),
    6: dict(label="COLLISION",       bg=(0,0,40),    fg=(0,0,220),    bar=(0,0,200),    sound=True,  gpio=True ),
}

# Edge case overlays (drawn over banner, do not change level number)
EDGE_LABELS = {
    "OBSTRUCTION":    (0, 160, 255),
    "RE-ENTRY":       (0, 80,  255),
    "ABORTED CROSS":  (60, 220, 100),
    "RECEDING":       (60, 255, 120),
    "LOST TRACK":     (80, 80,  180),
}

# ══════════════════════════════════════════════════════════════════════════════
# GPIO / BUZZER HOOKS  (no-op on PC, wire up on Pi)
# ══════════════════════════════════════════════════════════════════════════════
class OutputController:
    """
    On Raspberry Pi:
      import RPi.GPIO as GPIO
      GPIO.setmode(GPIO.BCM)
      GPIO.setup(HORN_PIN, GPIO.OUT)
      GPIO.setup(LIGHT_PIN, GPIO.OUT)
    Replace the stub methods below with real GPIO calls.
    """
    HORN_PIN  = 17
    LIGHT_PIN = 27

    def __init__(self):
        self._horn_on  = False
        self._light_on = False
        self._log      = []

    def set_horn(self, on: bool):
        if on != self._horn_on:
            self._horn_on = on
            # GPIO.output(self.HORN_PIN, GPIO.HIGH if on else GPIO.LOW)
            print(f"[GPIO] HORN {'ON' if on else 'OFF'}")

    def set_light(self, on: bool):
        if on != self._light_on:
            self._light_on = on
            # GPIO.output(self.LIGHT_PIN, GPIO.HIGH if on else GPIO.LOW)
            print(f"[GPIO] LIGHT {'ON' if on else 'OFF'}")

    def apply_level(self, level: int):
        cfg = LEVELS[level]
        self.set_horn(cfg["sound"])
        self.set_light(cfg["gpio"])

    def save_log(self, entries: list):
        # Placeholder — replace with file write or cloud push
        print(f"[LOG] {len(entries)} events would be saved")

    @property
    def horn_on(self): return self._horn_on

    @property
    def light_on(self): return self._light_on

# ══════════════════════════════════════════════════════════════════════════════
# KALMAN FILTER  — stabilises distance → smooth speed
# ══════════════════════════════════════════════════════════════════════════════
class KalmanDist:
    def __init__(self, q=0.01, r=0.0015):
        self.x = np.array([[1.0],[0.0]])
        self.P = np.eye(2) * 1.0
        self.R = np.array([[r]])
        self.H = np.array([[1.0, 0.0]])
        self._t = None

    def update(self, z_dist, q):
        now = time.time()
        dt  = np.clip((now - self._t) if self._t else 0.033, 0.005, 0.2)
        self._t = now
        F = np.array([[1., -dt],[0., 1.]])
        Q = np.array([[q*dt**2, q*dt],[q*dt, q]])
        xp = F @ self.x;  Pp = F @ self.P @ F.T + Q
        z_ = np.array([[float(z_dist)]])
        S  = self.H @ Pp @ self.H.T + self.R
        K  = Pp @ self.H.T @ np.linalg.inv(S)
        self.x = xp + K @ (z_ - self.H @ xp)
        self.P = (np.eye(2) - K @ self.H) @ Pp
        return max(0.01, float(self.x[0,0])), float(self.x[1,0])

    def reset(self, d=1.0):
        self.x = np.array([[d],[0.0]]); self.P = np.eye(2); self._t = None

# ══════════════════════════════════════════════════════════════════════════════
# SIMULATED IMU  — realistic MPU6050 noise model
# ══════════════════════════════════════════════════════════════════════════════
class SimulatedIMU:
    """
    Produces ax readings that mimic a real MPU6050 on a low-speed vehicle.

    Noise sources modelled:
      • White Gaussian noise       — per-sample sensor noise
      • Slow bias drift            — gyro/accel bias creep over time
      • Vibration harmonics        — road surface at ~2–8 Hz
      • Quantisation noise         — 16-bit ADC at ±2g → ~0.6mg LSB
      • Occasional spike           — cable vibration / motor interference

    Drive cycle (repeats every CYCLE seconds):
      0–5s   : accelerate  ~0 → 2.8 m/s (≈10 km/h)
      5–20s  : cruise      constant speed
      20–25s : brake       2.8 → 0 m/s
      25–30s : stationary
    """
    CYCLE = 30.0

    def __init__(self):
        self.t          = 0.0
        self.true_speed = 0.0   # m/s ground truth
        self.bias       = 0.0   # slowly drifting bias

    def step(self):
        """Advance one IMU sample. Returns raw ax in g."""
        self.t += IMU_DT
        tc = self.t % self.CYCLE

        # True acceleration profile (m/s²)
        if   tc < 5:         true_ax_ms2 =  0.56    # 0→10 km/h in 5s
        elif tc < 20:        true_ax_ms2 =  0.0
        elif tc < 25:        true_ax_ms2 = -0.56
        else:                true_ax_ms2 =  0.0

        self.true_speed = max(0.0, self.true_speed + true_ax_ms2 * IMU_DT)
        true_ax_g = true_ax_ms2 / 9.81

        # Bias drift (slow random walk)
        self.bias += np.random.normal(0, IMU_BIAS_DRIFT)
        self.bias  = np.clip(self.bias, -0.05, 0.05)

        # Road vibration (2–8 Hz harmonics)
        vib = (0.008 * np.sin(2 * np.pi * 3.1 * self.t) +
               0.005 * np.sin(2 * np.pi * 6.7 * self.t) +
               0.003 * np.sin(2 * np.pi * 8.2 * self.t))

        # White noise
        noise = np.random.normal(0, IMU_NOISE_ACCEL)

        # Occasional spike (~1% of samples)
        spike = np.random.choice([0.0, np.random.normal(0, 0.15)],
                                  p=[0.99, 0.01])

        # Quantisation (16-bit at ±2g → 1/32768 g per LSB)
        quant = round((true_ax_g + self.bias + vib + noise + spike)
                      * 32768) / 32768.0

        return quant, self.true_speed

# ══════════════════════════════════════════════════════════════════════════════
# VEHICLE VELOCITY ESTIMATOR  — integrates IMU ax with drift correction
# ══════════════════════════════════════════════════════════════════════════════
class VehicleVelocityEstimator:
    """
    Integrates calibrated forward acceleration to estimate vehicle speed.

    Drift mitigations:
      1. Low-pass filter on ax before integration
      2. Zero-velocity clamp when |ax| < noise floor for N frames
      3. Leaky integrator — gentle exponential decay
      4. Non-negative clamp (forward-only vehicle)

    On Pi: replace SimulatedIMU.step() with real MPU6050.get_data()
    and feed ax from calibrated IMU.
    """
    NOISE_FLOOR  = 0.04    # g
    STILL_FRAMES = 15
    LEAK         = 0.998
    LPF_ALPHA    = 0.20

    def __init__(self):
        self.v     = 0.0
        self.lpf   = 0.0
        self.still = 0

    def update(self, ax_raw_g):
        self.lpf = self.LPF_ALPHA * ax_raw_g + (1 - self.LPF_ALPHA) * self.lpf
        if abs(self.lpf) < self.NOISE_FLOOR:
            self.still += 1
        else:
            self.still = 0
        if self.still >= self.STILL_FRAMES:
            self.v = 0.0
            return 0.0
        self.v  = max(0.0, (self.v + self.lpf * 9.81 * IMU_DT) * self.LEAK)
        return self.v   # m/s

# ══════════════════════════════════════════════════════════════════════════════
# SENSOR FUSION ENGINE
# ══════════════════════════════════════════════════════════════════════════════
class FusionEngine:
    """
    Fuses camera-derived object approach speed with IMU-derived vehicle speed
    to compute closing speed and fused TTC.

    Closing speed = object_approach_speed + vehicle_forward_speed

    Why this matters:
      At 10 km/h vehicle speed + 5 km/h object approach speed:
        Camera-only TTC  = dist / 1.4 m/s   (underestimates risk)
        Fused TTC        = dist / 2.8 m/s   (correct)
      → Camera-only gives 2× the time — could be the difference between
        a warning and a collision.

    IMU weight slider (0–100):
      0   = camera TTC only (ignore IMU)
      50  = equal blend
      100 = full IMU contribution
      Use lower values if vehicle speed is zero or IMU is uncalibrated.
    """
    def __init__(self):
        self.closing_hist = deque(maxlen=DIST_HIST_LEN)
        self.fttc_hist    = deque(maxlen=DIST_HIST_LEN)
        self.vspd_hist    = deque(maxlen=DIST_HIST_LEN)

    def compute(self, obj_spd_ms: float, veh_spd_ms: float,
                dist_m: float, imu_weight: float) -> dict:
        """
        obj_spd_ms  : Kalman-filtered object approach speed (m/s, + = approaching)
        veh_spd_ms  : IMU-estimated vehicle forward speed (m/s)
        dist_m      : Kalman-filtered distance to object (m)
        imu_weight  : 0.0–1.0 how much vehicle speed contributes

        Returns dict with closing_speed, fused_ttc, contrib breakdown.
        """
        obj_contribution = max(0.0, obj_spd_ms)
        veh_contribution = veh_spd_ms * imu_weight

        closing_speed = obj_contribution + veh_contribution
        closing_speed = max(0.0, closing_speed)

        if closing_speed > 0.01 and dist_m:
            fused_ttc = round(min(dist_m / closing_speed, 99.9), 1)
        else:
            fused_ttc = 99.9

        self.closing_hist.append(closing_speed)
        self.vspd_hist.append(veh_spd_ms)
        self.fttc_hist.append(min(fused_ttc, 10.0))

        return dict(
            closing_speed   = closing_speed,
            fused_ttc       = fused_ttc,
            obj_contrib_ms  = obj_contribution,
            veh_contrib_ms  = veh_contribution,
        )


class ArucoDetector:
    def __init__(self):
        d = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
        p = cv2.aruco.DetectorParameters()
        self.det = cv2.aruco.ArucoDetector(d, p)
        cx,cy = CAM_W/2., CAM_H/2.
        self.K = np.array([[FOCAL_PX,0,cx],[0,FOCAL_PX,cy],[0,0,1]], np.float64)
        self.D = np.zeros((4,1))
        hs = MARKER_REAL_SIZE_M/2.
        self.obj = np.array([[-hs,hs,0],[hs,hs,0],[hs,-hs,0],[-hs,-hs,0]], np.float32)

    def detect(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cl, ids, _ = self.det.detectMarkers(gray)
        if ids is None: return None
        for i,mid in enumerate(ids.flatten()):
            if mid != TARGET_ID: continue
            corners = cl[i][0]
            x1,y1 = corners.min(axis=0).astype(int)
            x2,y2 = corners.max(axis=0).astype(int)
            ok,rvec,tvec = cv2.solvePnP(self.obj, corners.astype(np.float32), self.K, self.D)
            dist = float(np.linalg.norm(tvec)) if ok else \
                   (FOCAL_PX*MARKER_REAL_SIZE_M/max(x2-x1,1))
            return dict(cx=int((x1+x2)/2), cy=int((y1+y2)/2),
                        x=x1,y=y1,w=x2-x1,h=y2-y1,
                        dist_raw=dist, corners=corners)
        return None

# ══════════════════════════════════════════════════════════════════════════════
# TRACKER
# ══════════════════════════════════════════════════════════════════════════════
class Tracker:
    def __init__(self):
        self.kf           = KalmanDist()
        self.cx = self.cy = None
        self.bbox         = None
        self.trail        = deque(maxlen=TRAIL_LEN)
        self.vx = self.vy = 0.0

        self.dist_hist    = deque(maxlen=DIST_HIST_LEN)
        self.speed_hist   = deque(maxlen=DIST_HIST_LEN)
        self.growth_hist  = deque(maxlen=DIST_HIST_LEN)
        self.ttc_hist     = deque(maxlen=DIST_HIST_LEN)
        self.area_hist    = deque(maxlen=GROWTH_WIN)

        self.missed        = 0
        self.in_path       = 0
        self.active        = False
        self.locked        = False
        self.dist_f        = None
        self.speed_f       = 0.0

        # Edge case tracking
        self.in_path_total = 0      # cumulative frames spent in path
        self.still_frames  = 0      # frames with near-zero pixel movement
        self.left_path_at  = None   # frame counter when object left path
        self.reentry_count = 0      # how many times object re-entered path
        self.peak_growth   = 0.0    # highest growth seen this track
        self.frame_count   = 0

    def update(self, det, px1, px2, q_val):
        self.frame_count += 1
        if det:
            cx,cy = det["cx"],det["cy"]
            area  = det["w"]*det["h"]

            if self.cx is not None:
                self.vx = 0.3*(cx-self.cx)+0.7*self.vx
                self.vy = 0.3*(cy-self.cy)+0.7*self.vy
                # still detection (pixel movement < 3px)
                if abs(self.vx)<3 and abs(self.vy)<3:
                    self.still_frames += 1
                else:
                    self.still_frames = 0

            self.cx,self.cy=cx,cy; self.bbox=(det["x"],det["y"],det["w"],det["h"])
            self.trail.append((cx,cy)); self.area_hist.append(area)
            self.missed=0; self.active=True; self.locked=True

            in_path_now = px1 < cx < px2
            if in_path_now:
                if self.in_path == 0 and self.left_path_at is not None:
                    # Object re-entered path
                    self.reentry_count += 1
                    self.left_path_at = None
                self.in_path     += 1
                self.in_path_total += 1
            else:
                if self.in_path > 0:
                    self.left_path_at = self.frame_count
                self.in_path = max(0, self.in_path-1)

            self.dist_f, self.speed_f = self.kf.update(det["dist_raw"], q_val)
            gr = self.growth()
            self.peak_growth = max(self.peak_growth, gr)

            self.dist_hist.append(self.dist_f)
            self.speed_hist.append(max(0.0, self.speed_f))
            self.growth_hist.append(gr)
            self.ttc_hist.append(min(self.ttc(), 10.0))
        else:
            self.missed += 1
            if self.missed > PERSIST_FRAMES:
                self._reset()

    def _reset(self):
        self.cx=self.cy=None; self.bbox=None
        self.trail.clear(); self.area_hist.clear()
        self.dist_hist.clear(); self.speed_hist.clear()
        self.growth_hist.clear(); self.ttc_hist.clear()
        self.vx=self.vy=0.0; self.missed=0; self.in_path=0
        self.active=False; self.dist_f=None; self.speed_f=0.0
        self.in_path_total=0; self.still_frames=0
        self.left_path_at=None; self.reentry_count=0; self.peak_growth=0.0
        self.kf.reset()

    def growth(self):
        h=list(self.area_hist)
        if len(h)<4: return 0.0
        half=len(h)//2
        o=np.mean(h[:half]); r=np.mean(h[half:])
        return (r-o)/o if o>0 else 0.0

    def speed_kmh(self): return round(max(0.0, self.speed_f)*3.6, 1)

    def ttc(self):
        if self.dist_f and self.speed_f > 0.05:
            return round(min(self.dist_f/self.speed_f, 99.9), 1)
        gr=self.growth()
        return round(min(1.0/(gr*7.0), 99.9), 1) if gr>0.005 else 99.9

    def predict(self):
        if self.cx is None or (self.vx==0 and self.vy==0): return []
        return [(int(self.cx+self.vx*i), int(self.cy+self.vy*i))
                for i in range(1, PREDICT_STEPS+1)]

    def is_crossing(self):  return self.active and self.in_path >= 5
    def is_receding(self):  return self.active and self.speed_f < -0.1
    def is_still(self):     return self.still_frames >= OBSTRUCTION_FRAMES
    def is_reentry(self):   return self.reentry_count > 0 and self.in_path > 0
    def just_left_path(self):
        return (self.left_path_at is not None and
                self.frame_count - self.left_path_at < REENTRY_WINDOW and
                self.in_path == 0)

# ══════════════════════════════════════════════════════════════════════════════
# ALERT STATE MACHINE
# ══════════════════════════════════════════════════════════════════════════════
class AlertStateMachine:
    """
    Evaluates tracker state each frame and returns:
      level     : int 0–6
      edge_case : str or None

    Hysteresis: level can only drop 1 step per frame (prevents flickering).
    """
    def __init__(self):
        self.level       = 0
        self.edge_case   = None
        self.hold_frames = 0      # hold high alert when track lost briefly
        self.prev_level  = 0

    def evaluate(self, tr: Tracker, ttc_w: float, ttc_c: float, min_spd: float, fused_ttc: float = None) -> tuple:
        if not tr.active:
            if self.hold_frames > 0 and self.level >= 3:
                self.hold_frames -= 1
                self.edge_case = "LOST TRACK"
                return self.level, self.edge_case
            self.edge_case = None
            self._set(0)
            return self.level, self.edge_case

        gr  = tr.growth()
        ttc = fused_ttc if (fused_ttc is not None and fused_ttc < 99.0) else tr.ttc()
        spd = max(0.0, tr.speed_f)
        self.hold_frames = HOLD_FRAMES    # reset hold whenever we have detection
        self.edge_case   = None

        # ── Edge case classification ───────────────────────────────────────────

        # Collision confirmed: peak growth was high, object vanished OR growth
        # suddenly dropped (filled frame then gone)
        if tr.peak_growth > 0.35 and tr.missed > 3:
            self.edge_case = None   # level 6 handles this
            self._set(6)
            return self.level, self.edge_case

        # Object receding (was approaching, now moving away)
        if tr.is_receding() and gr < -0.03:
            self.edge_case = "RECEDING"
            self._set(max(1, self.level-1))   # de-escalate
            return self.level, self.edge_case

        # Object stationary inside path — obstruction, not crossing animal
        if tr.is_still() and tr.in_path > 0:
            self.edge_case = "OBSTRUCTION"
            self._set(3)
            return self.level, self.edge_case

        # Aborted crossing — entered path, now leaving, TTC was never critical
        if tr.just_left_path() and tr.peak_growth < 0.10:
            self.edge_case = "ABORTED CROSS"
            self._set(1)
            return self.level, self.edge_case

        # Re-entry — object came back, escalate faster
        reentry_boost = 1 if tr.is_reentry() else 0

        # ── Core level logic ──────────────────────────────────────────────────
        if gr > 0.30 or (spd > 2.0 and ttc < ttc_c):
            raw = 5
        elif ttc < ttc_c or gr > 0.18:
            raw = 5
        elif ttc < ttc_w and (gr > 0.06 or spd > 0.4):
            raw = 4
        elif tr.is_crossing() and (gr > 0.04 or spd > 0.2):
            raw = 3
        elif tr.is_crossing() and spd < min_spd and gr < 0.01:
            # In path but very slow → safe crossing / just passing
            raw = 1
        elif tr.active and not tr.is_crossing():
            raw = 2
        else:
            raw = 2

        raw = min(5, raw + reentry_boost)

        # Safe crossing: object in path but clearly leaving, negative growth
        if tr.is_crossing() and gr < -0.02 and spd < 0.1:
            raw = 1
            self.edge_case = None

        if tr.is_reentry():
            self.edge_case = "RE-ENTRY"

        self._set(raw)
        return self.level, self.edge_case

    def _set(self, target: int):
        """Apply hysteresis: can jump up instantly, drop only 1 level/frame."""
        if target > self.level:
            self.level = target
        elif target < self.level:
            self.level = self.level - 1    # gradual de-escalation
        self.level = int(np.clip(self.level, 0, 6))

# ══════════════════════════════════════════════════════════════════════════════
# SPARKLINE + PANEL
# ══════════════════════════════════════════════════════════════════════════════
def sparkline(canvas, data, x0, y0, pw, ph, col,
              vmin=0., vmax=1., zero_line=None):
    arr = np.array(list(data), dtype=np.float32)
    if len(arr) < 2: return
    rng = max(vmax-vmin, 1e-6)
    xs  = np.linspace(x0, x0+pw-1, len(arr)).astype(int)
    ys  = np.clip((y0+ph-1-((arr-vmin)/rng*(ph-2))).astype(int), y0, y0+ph-1)
    pts = np.stack([xs,ys],axis=1).reshape(-1,1,2)
    cv2.polylines(canvas,[pts],False,col,1,cv2.LINE_AA)
    if zero_line is not None:
        zy=int(y0+ph-1-(zero_line-vmin)/rng*(ph-2))
        cv2.line(canvas,(x0,np.clip(zy,y0,y0+ph)),(x0+pw,np.clip(zy,y0,y0+ph)),GRID_COL,1)

def draw_panel(canvas, tr, level, fusion=None):
    """
    Panel layout (3 zones across PANEL_H px):

    ┌─────────────────────────────────────────────────────────────────┐
    │  ZONE A (left 210px)   │  ZONE B (middle 290px) │  ZONE C(right140px)│
    │  Big live readouts     │  4 sparklines stacked  │  TTC gauge     │
    │  • Distance            │  • Distance (m)        │  Arc + number  │
    │  • Obj speed           │  • Object speed        │  Fused vs cam  │
    │  • Vehicle speed       │  • Vehicle speed       │                │
    │  • Closing speed       │  • Closing speed       │                │
    ├─────────────────────────────────────────────────────────────────┤
    │  ZONE D (full width, bottom 16px) — 7-step level ladder strip  │
    └─────────────────────────────────────────────────────────────────┘
    """
    y0      = CAM_H
    STRIP_H = 18          # level ladder at very bottom
    PAD     = 6
    canvas[y0:, :] = PLOT_BG
    cv2.line(canvas, (0, y0), (CAM_W, y0), (55, 60, 75), 1)

    usable_h = PANEL_H - STRIP_H - 4   # height available for zones A/B/C

    # ── Zone widths ───────────────────────────────────────────────────────────
    ZA_W = 195    # big readouts
    ZC_W = 138    # TTC gauge
    ZB_W = CAM_W - ZA_W - ZC_W   # sparklines

    ZA_X = 0
    ZB_X = ZA_W
    ZC_X = ZA_W + ZB_W

    # Dividers
    cv2.line(canvas, (ZB_X, y0), (ZB_X, y0+usable_h), (45, 50, 65), 1)
    cv2.line(canvas, (ZC_X, y0), (ZC_X, y0+usable_h), (45, 50, 65), 1)

    # ══ ZONE A — Big live readouts ════════════════════════════════════════════
    # Each readout: small label on top, big value below
    def big_readout(label, value_str, unit_str, x, y, val_col, label_col=(90,95,110)):
        cv2.putText(canvas, label,    (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, label_col, 1)
        cv2.putText(canvas, value_str,(x, y+18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.68, val_col, 2)
        cv2.putText(canvas, unit_str, (x, y+32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.30, label_col, 1)

    row_h  = usable_h // 4
    ax     = ZA_X + PAD

    # 1. Distance
    dist_val = f"{tr.dist_f:.2f}" if tr.dist_f else "---"
    dc = (60,200,255)
    big_readout("DISTANCE", dist_val, "metres", ax, y0+PAD+row_h*0+12, dc)

    # 2. Object speed
    obj_ms  = max(0.0, tr.speed_f)
    obj_kmh = obj_ms * 3.6
    oc = (50,240,50) if obj_ms<0.5 else (0,210,255) if obj_ms<1.5 else (0,80,255)
    big_readout("OBJECT SPD", f"{obj_kmh:.1f}", "km/h (approach)", ax, y0+PAD+row_h*1+12, oc)

    # 3. Vehicle speed (IMU)
    if fusion:
        veh_ms  = fusion["veh_contrib_ms"]
        veh_kmh = veh_ms * 3.6
        vc = (80,130,255)
        src_tag = "km/h  [IMU:REAL]" if fusion.get("real_imu") else "km/h  [IMU:SIM]"
        big_readout("VEHICLE SPD", f"{veh_kmh:.1f}", src_tag, ax, y0+PAD+row_h*2+12, vc)
    else:
        big_readout("VEHICLE SPD", "---", "IMU off", ax, y0+PAD+row_h*2+12, (55,60,70))

    # 4. Closing speed (fused = obj + vehicle contribution)
    if fusion:
        cspd_ms  = fusion["closing_speed"]
        cspd_kmh = cspd_ms * 3.6
        cc = (50,230,50) if cspd_ms<1.0 else (0,200,255) if cspd_ms<2.5 else (0,50,255)
        big_readout("CLOSING SPD", f"{cspd_kmh:.1f}", "km/h (obj+veh)", ax, y0+PAD+row_h*3+12, cc)
    else:
        big_readout("CLOSING SPD", f"{obj_kmh:.1f}", "km/h (cam only)", ax, y0+PAD+row_h*3+12, oc)

    # ══ ZONE B — 4 stacked sparklines with axis labels ════════════════════════
    n_plots  = 4
    plot_h   = (usable_h - PAD*2) // n_plots - 2
    plot_w   = ZB_W - PAD*2 - 28    # -28 for right-side y-axis labels
    bx       = ZB_X + PAD
    label_x  = bx + plot_w + 3

    plots = [
        ("DIST m",      COL_DIST,  tr.dist_hist,        0.,  5.,  "0","5m"),
        ("OBJ km/h",    COL_SPEED, [v*3.6 for v in tr.speed_hist],  0., 15., "0","15"),
        ("VEH km/h",    COL_VSPD,  [v*3.6 for v in (fusion["vspd_hist"] if fusion else [])], 0., 15., "0","15"),
        ("CLOSE km/h",  COL_CLOSE, [v*3.6 for v in (fusion["close_hist"] if fusion else [])],0., 20., "0","20"),
    ]

    for i,(lbl,col,data,vmin,vmax,lo_lbl,hi_lbl) in enumerate(plots):
        py  = y0 + PAD + i*(plot_h+2)
        # background
        cv2.rectangle(canvas,(bx,py),(bx+plot_w,py+plot_h),(22,25,33),-1)
        # grid lines at 1/3 and 2/3
        for frac in [0.33,0.67]:
            gy=int(py+plot_h*(1-frac))
            cv2.line(canvas,(bx,gy),(bx+plot_w,gy),(32,36,48),1)
        # border
        cv2.rectangle(canvas,(bx,py),(bx+plot_w,py+plot_h),(45,50,65),1)
        # label left of plot
        cv2.putText(canvas,lbl,(ZB_X+2,py+plot_h//2+4),
                    cv2.FONT_HERSHEY_SIMPLEX,0.26,col,1)
        # sparkline
        sparkline(canvas,data,bx,py,plot_w,plot_h,col,vmin,vmax)
        # y-axis tick labels
        cv2.putText(canvas,hi_lbl,(label_x,py+8),        cv2.FONT_HERSHEY_SIMPLEX,0.24,(70,75,90),1)
        cv2.putText(canvas,lo_lbl,(label_x,py+plot_h-2), cv2.FONT_HERSHEY_SIMPLEX,0.24,(70,75,90),1)
        # latest value — right-justified inside plot
        if data:
            last = list(data)[-1]
            vstr = f"{last:.1f}"
            tw   = len(vstr)*7
            cv2.putText(canvas,vstr,(bx+plot_w-tw-2,py+plot_h-4),
                        cv2.FONT_HERSHEY_SIMPLEX,0.32,col,1)

    # ══ ZONE C — TTC gauge (arc + large number + cam vs fused comparison) ═════
    gx   = ZC_X + ZC_W//2
    gy   = y0 + usable_h//2 - 10
    grad = 44     # gauge radius

    # Determine values
    cam_ttc   = tr.ttc()
    fused_ttc = fusion["fused_ttc"] if fusion else cam_ttc
    display_ttc = fused_ttc        # primary display value

    # Arc background (grey)
    cv2.ellipse(canvas,(gx,gy),(grad,grad),0,200,340,(40,45,55),4)

    # Arc fill — maps TTC 0–8s to angle
    # 0s (danger) = 200°, 8s (safe) = 340°
    ttc_clamped = np.clip(display_ttc, 0, 8)
    arc_end     = int(200 + (ttc_clamped/8.0)*140)
    arc_col     = (0,40,255) if display_ttc<2 else \
                  (0,120,255) if display_ttc<4 else \
                  (0,220,255) if display_ttc<6 else (50,240,80)
    cv2.ellipse(canvas,(gx,gy),(grad,grad),0,200,arc_end,arc_col,4)

    # Needle tip dot
    angle_rad = np.radians(arc_end)
    nx = int(gx + grad * np.cos(angle_rad))
    ny = int(gy + grad * np.sin(angle_rad))
    cv2.circle(canvas,(nx,ny),4,arc_col,-1)

    # Centre number
    ttc_str = f"{display_ttc:.1f}" if display_ttc<99 else "∞"
    tw      = len(ttc_str)*10
    cv2.putText(canvas,"TTC",(gx-10,gy-12),cv2.FONT_HERSHEY_SIMPLEX,0.34,(100,110,130),1)
    cv2.putText(canvas,ttc_str,(gx-tw//2,gy+10),cv2.FONT_HERSHEY_SIMPLEX,0.80,arc_col,2)
    cv2.putText(canvas,"sec",(gx-8,gy+26),cv2.FONT_HERSHEY_SIMPLEX,0.30,(80,85,100),1)

    # Cam vs Fused comparison (below gauge)
    cy2 = gy + grad + 12
    tc2 = (50,240,50) if cam_ttc>5 else (0,180,255) if cam_ttc>2 else (0,50,255)
    tf2 = (50,240,50) if fused_ttc>5 else (0,180,255) if fused_ttc>2 else (0,50,255)
    cv2.putText(canvas,f"CAM  {cam_ttc:.1f}s",  (ZC_X+PAD, cy2),   cv2.FONT_HERSHEY_SIMPLEX,0.34,tc2,1)
    cv2.putText(canvas,f"FUSE {fused_ttc:.1f}s",(ZC_X+PAD, cy2+14),cv2.FONT_HERSHEY_SIMPLEX,0.34,tf2,1)

    # Growth bar — thin vertical bar on far right of zone C
    gbx   = ZC_X + ZC_W - 10
    gr_val= tr.growth()
    gb_h  = usable_h - PAD*2
    gby   = y0 + PAD
    cv2.rectangle(canvas,(gbx,gby),(gbx+6,gby+gb_h),(28,30,38),-1)
    fill  = int(np.clip(gr_val/0.20,0,1)*gb_h)
    if fill>0:
        gcol=(50,200,50) if gr_val<0.05 else (0,160,255) if gr_val<0.12 else (0,40,255)
        cv2.rectangle(canvas,(gbx,gby+gb_h-fill),(gbx+6,gby+gb_h),gcol,-1)
    cv2.putText(canvas,"G",(gbx,gby-2),cv2.FONT_HERSHEY_SIMPLEX,0.28,(80,85,100),1)
    cv2.putText(canvas,f"{gr_val:.2f}",(gbx-14,gby+gb_h+8),cv2.FONT_HERSHEY_SIMPLEX,0.26,COL_GR,1)

    # ══ ZONE D — Level ladder strip ═══════════════════════════════════════════
    lby  = y0 + PANEL_H - STRIP_H
    step = CAM_W // 7
    for lvl, cfg in LEVELS.items():
        lx     = lvl * step
        active = (lvl == level)
        lw     = step if lvl < 6 else CAM_W - lx   # last cell takes remainder
        bg     = cfg["bg"]   if active else (18, 20, 27)
        fg     = cfg["fg"]   if active else (42, 47, 58)
        cv2.rectangle(canvas,(lx,lby),(lx+lw-1,lby+STRIP_H-1),bg,-1)
        if active:
            cv2.rectangle(canvas,(lx,lby),(lx+lw-1,lby+STRIP_H-1),cfg["fg"],1)
        short  = cfg["label"][:8]
        cv2.putText(canvas,short,(lx+3,lby+12),cv2.FONT_HERSHEY_SIMPLEX,0.26,fg,1)
        # level number
        cv2.putText(canvas,str(lvl),(lx+lw-10,lby+12),cv2.FONT_HERSHEY_SIMPLEX,0.24,
                    cfg["fg"] if active else (35,38,48),1)





# ══════════════════════════════════════════════════════════════════════════════
# CAMERA FRAME DRAWING
# ══════════════════════════════════════════════════════════════════════════════
def draw_path(frame, x1, x2):
    ov=frame.copy()
    cv2.rectangle(ov,(x1,0),(x2,CAM_H),(0,255,100),-1)
    cv2.addWeighted(ov,0.07,frame,0.93,0,frame)
    cv2.line(frame,(x1,0),(x1,CAM_H),(0,200,70),1)
    cv2.line(frame,(x2,0),(x2,CAM_H),(0,200,70),1)
    cv2.putText(frame,"PATH",(x1+4,14),cv2.FONT_HERSHEY_SIMPLEX,0.36,(0,160,55),1)

def draw_marker(frame, tr, det, level):
    if not tr.active or tr.bbox is None: return
    x,y,w,h=tr.bbox; cx,cy=tr.cx,tr.cy
    col=LEVELS[level]["fg"]

    if det and det.get("corners") is not None:
        pts=det["corners"].astype(int)
        cv2.polylines(frame,[pts],True,col,2)
        for pt in pts: cv2.circle(frame,tuple(pt),4,col,-1)

    s=12
    for px,py,dx,dy in [(x,y,1,1),(x+w,y,-1,1),(x,y+h,1,-1),(x+w,y+h,-1,-1)]:
        cv2.line(frame,(px,py),(px+dx*s,py),col,2)
        cv2.line(frame,(px,py),(px,py+dy*s),col,2)
    cv2.line(frame,(cx-8,cy),(cx+8,cy),col,1)
    cv2.line(frame,(cx,cy-8),(cx,cy+8),col,1)
    cv2.circle(frame,(cx,cy),3,col,-1)

    trail=list(tr.trail)
    for i in range(1,len(trail)):
        a=i/len(trail); tc=tuple(int(c*a) for c in col)
        cv2.line(frame,trail[i-1],trail[i],tc,1)

    pred=tr.predict()
    if pred:
        for i,pt in enumerate(pred):
            a=1-i/len(pred)
            cv2.circle(frame,pt,1,(0,int(70*a),int(200*a)),-1)
        cv2.arrowedLine(frame,(cx,cy),pred[-1],(0,55,170),1,tipLength=0.22)

    gr=tr.growth(); bx=x+w+4
    fill=int(np.clip(gr/0.20,0,1)*h)
    cv2.rectangle(frame,(bx,y),(bx+6,y+h),(30,30,30),-1)
    if fill>0:
        gc=(50,200,50) if gr<0.05 else (0,160,255) if gr<0.12 else (0,40,255)
        cv2.rectangle(frame,(bx,y+h-fill),(bx+6,y+h),gc,-1)

    dist_s=f"{tr.dist_f:.2f}m" if tr.dist_f else "---"
    cv2.putText(frame,f"ID:{TARGET_ID}  {dist_s}  {tr.speed_kmh():.1f}km/h",
                (x,max(y-5,14)),cv2.FONT_HERSHEY_SIMPLEX,0.38,col,1)

def draw_banner(frame, level, edge_case, fps, tr, gpio_ctrl):
    cfg=LEVELS[level]
    bg=cfg["bg"]; fg=cfg["fg"]
    cv2.rectangle(frame,(0,0),(CAM_W,44),bg,-1)
    cv2.putText(frame,cfg["label"],(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.85,fg,2)

    # Edge case tag
    if edge_case:
        ec_col=EDGE_LABELS.get(edge_case,(180,180,180))
        cv2.putText(frame,f"[{edge_case}]",(10,43),
                    cv2.FONT_HERSHEY_SIMPLEX,0.32,ec_col,1)

    # GPIO indicators top-right
    horn_col  =(0,80,255) if gpio_ctrl.horn_on  else (40,40,40)
    light_col =(0,200,255) if gpio_ctrl.light_on else (40,40,40)
    cv2.circle(frame,(CAM_W-20,14),7,horn_col,-1)
    cv2.putText(frame,"HORN", (CAM_W-55,18),cv2.FONT_HERSHEY_SIMPLEX,0.28,horn_col,1)
    cv2.circle(frame,(CAM_W-20,30),7,light_col,-1)
    cv2.putText(frame,"LIGHT",(CAM_W-58,34),cv2.FONT_HERSHEY_SIMPLEX,0.28,light_col,1)

    st=f"FPS:{fps:.0f} Lvl:{level} {'LOCKED' if tr.locked else 'SEARCHING'}"
    cv2.putText(frame,st,(CAM_W-200,12),cv2.FONT_HERSHEY_SIMPLEX,0.30,(120,125,135),1)

    if not tr.active and not tr.locked:
        cv2.putText(frame,f"Show ArUco ID {TARGET_ID} to camera",
                    (CAM_W//2-130,CAM_H//2),cv2.FONT_HERSHEY_SIMPLEX,0.50,(0,190,255),1)

# ══════════════════════════════════════════════════════════════════════════════
# MARKER GENERATOR
# ══════════════════════════════════════════════════════════════════════════════
def generate_marker():
    d=cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    img=cv2.aruco.generateImageMarker(d,TARGET_ID,400)
    out=cv2.copyMakeBorder(img,50,50,50,50,cv2.BORDER_CONSTANT,value=255)
    fname=f"aruco_id{TARGET_ID}.png"; cv2.imwrite(fname,out)
    print(f"[MARKER] Saved {fname}  — print at {MARKER_REAL_SIZE_M*100:.0f}×{MARKER_REAL_SIZE_M*100:.0f}cm")

# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
WIN="Pre-Collision — Multi-Level Warning"

def nothing(_): pass

def main():
    if "--gen" in sys.argv: generate_marker(); return
    if not os.path.exists(f"aruco_id{TARGET_ID}.png"): generate_marker()

    cap=cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,CAM_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,CAM_H)
    cap.set(cv2.CAP_PROP_BUFFERSIZE,1)
    if not cap.isOpened(): print("[ERROR] No camera"); return

    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, DISP_W, DISP_H + 180)   # +180 for 6 trackbar rows

    # ── Sliders ───────────────────────────────────────────────────────────────
    cv2.createTrackbar("Speed Smooth  1=smooth 100=raw",  WIN, 15, 100, nothing)
    cv2.createTrackbar("TTC Warn   (x0.1s) def=40",       WIN, 40, 100, nothing)
    cv2.createTrackbar("TTC Critical(x0.1s) def=18",      WIN, 18,  50, nothing)
    cv2.createTrackbar("Path Width %  def=40",             WIN, 40,  80, nothing)
    cv2.createTrackbar("Min Speed (x0.1m/s) def=2",       WIN,  2,  20, nothing)
    # IMU weight: how much vehicle speed contributes to closing speed / fused TTC
    #   0  = camera only (IMU ignored — use when vehicle is stationary or IMU uncalibrated)
    #  50  = half contribution
    # 100  = full fusion (recommended once IMU is calibrated on Pi)
    cv2.createTrackbar("IMU Weight  0=off 100=full",       WIN, 80, 100, nothing)

    # ── Objects ───────────────────────────────────────────────────────────────
    detector  = ArucoDetector()
    tracker   = Tracker()
    state_m   = AlertStateMachine()
    gpio_ctrl = OutputController()
    imu_sim   = SimulatedIMU()
    vel_est   = VehicleVelocityEstimator()
    fusion_eng= FusionEngine()

    # canvas stays at NATIVE resolution — all CV coordinates are correct
    canvas    = np.zeros((WIN_H, CAM_W, 3), dtype=np.uint8)
    fps_t     = time.time(); fcnt=0; last_det=None

    # Current vehicle speed (updated every frame from simulated IMU)
    veh_spd_ms = 0.0

    print(f"[INFO] Tracking ID {TARGET_ID} | 7 alert levels + sensor fusion | Q to quit")
    print(f"[INFO] Display {DISP_W}×{DISP_H}  |  Processing {CAM_W}×{CAM_H}  |  Scale {DISP_SCALE}x")
    print( "[INFO] IMU: SimulatedIMU (swap imu_sim for real MPU6050 on Pi)")
    print()

    while True:
        ret, frame = cap.read()
        if not ret: break
        fcnt += 1

        # ── Read sliders ──────────────────────────────────────────────────────
        sm     = max(1, cv2.getTrackbarPos("Speed Smooth  1=smooth 100=raw", WIN))
        ttc_w  = cv2.getTrackbarPos("TTC Warn   (x0.1s) def=40",       WIN) * 0.1
        ttc_c  = cv2.getTrackbarPos("TTC Critical(x0.1s) def=18",      WIN) * 0.1
        pw     = max(10, cv2.getTrackbarPos("Path Width %  def=40",     WIN))
        mspd   = cv2.getTrackbarPos("Min Speed (x0.1m/s) def=2",       WIN) * 0.1
        imu_w  = cv2.getTrackbarPos("IMU Weight  0=off 100=full",       WIN) / 100.0
        ttc_c  = min(ttc_c, ttc_w - 0.1)

        q_val  = 0.001 * (10 ** (sm / 50.0))
        px1    = int(CAM_W * (0.5 - pw / 200.))
        px2    = int(CAM_W * (0.5 + pw / 200.))

        # ── IMU step — one sample per camera frame (≈30 Hz) ──────────────────
        # On Pi: replace imu_sim.step() with real_imu.get_data()
        # and pass ax from calibrated MPU6050 into vel_est.update()
        ax_raw, true_spd = imu_sim.step()
        veh_spd_ms = vel_est.update(ax_raw)

        # ── Camera processing at native 640×480 ───────────────────────────────
        det = detector.detect(frame)
        tracker.update(det, px1, px2, q_val)
        if det: last_det = det

        # ── Sensor fusion ─────────────────────────────────────────────────────
        fusion_result = None
        if tracker.dist_f is not None:
            fusion_result = fusion_eng.compute(
                obj_spd_ms = tracker.speed_f,
                veh_spd_ms = veh_spd_ms,
                dist_m     = tracker.dist_f,
                imu_weight = imu_w,
            )
            # pass deque refs into result for panel sparklines
            fusion_result["vspd_hist"]  = fusion_eng.vspd_hist
            fusion_result["fttc_hist"]  = fusion_eng.fttc_hist
            fusion_result["close_hist"] = fusion_eng.closing_hist
            fusion_result["real_imu"]   = False   # flip to True on Pi

        fused_ttc = fusion_result["fused_ttc"] if fusion_result else None

        # ── Alert state machine — uses fused TTC when available ───────────────
        level, edge_case = state_m.evaluate(tracker, ttc_w, ttc_c, mspd, fused_ttc)
        gpio_ctrl.apply_level(level)

        # ── Draw ──────────────────────────────────────────────────────────────
        draw_path(frame, px1, px2)
        draw_marker(frame, tracker, last_det if tracker.active else None, level)
        fps = fcnt / (time.time() - fps_t + 1e-5)
        draw_banner(frame, level, edge_case, fps, tracker, gpio_ctrl)

        canvas[:CAM_H, :] = frame
        draw_panel(canvas, tracker, level, fusion=fusion_result)

        # ── Single upscale for display only — zero overhead on processing ─────
        if DISP_SCALE != 1.0:
            display = cv2.resize(canvas, (DISP_W, DISP_H),
                                 interpolation=cv2.INTER_LINEAR)
        else:
            display = canvas

        cv2.imshow(WIN, display)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Done.")

if __name__=="__main__":
    main()