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
PLOT_BG   = (16, 18, 26)
GRID_COL  = (32, 38, 52)
COL_DIST  = (60,  200, 255)
COL_SPEED = (100, 255, 140)
COL_GR    = (255, 160, 60)
COL_TTC   = (200, 80,  255)

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
# ARUCO DETECTOR
# ══════════════════════════════════════════════════════════════════════════════
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

    def evaluate(self, tr: Tracker, ttc_w: float, ttc_c: float, min_spd: float) -> tuple:
        if not tr.active:
            if self.hold_frames > 0 and self.level >= 3:
                self.hold_frames -= 1
                self.edge_case = "LOST TRACK"
                return self.level, self.edge_case
            self.edge_case = None
            self._set(0)
            return self.level, self.edge_case

        gr  = tr.growth()
        ttc = tr.ttc()
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

def draw_panel(canvas, tr, level):
    y0  = CAM_H
    canvas[y0:, :] = PLOT_BG
    cv2.line(canvas, (0,y0), (CAM_W,y0), (50,55,70), 1)

    # ── 4 sparkline cells ─────────────────────────────────────────────────────
    cw   = CAM_W // 4
    ph   = PANEL_H - 72    # plot area height — more space for bigger readouts
    py   = y0 + 18
    pad  = 5

    cells = [
        (0*cw+pad, "DISTANCE (m)", COL_DIST,  tr.dist_hist,   0., 5.),
        (1*cw+pad, "SPEED (m/s)",  COL_SPEED, tr.speed_hist,  0., 3.),
        (2*cw+pad, "GROWTH",       COL_GR,    tr.growth_hist,-0.1, 0.3),
        (3*cw+pad, "TTC (s ≤10)",  COL_TTC,   tr.ttc_hist,    0.,10.),
    ]
    for x0c,lbl,col,data,vmin,vmax in cells:
        pw = cw - pad*2
        cv2.rectangle(canvas,(x0c,py),(x0c+pw,py+ph),GRID_COL,1)
        for yi in [py+ph//3, py+2*ph//3]:
            cv2.line(canvas,(x0c,yi),(x0c+pw,yi),(25,28,38),1)
        # larger label
        cv2.putText(canvas, lbl, (x0c+3, py-4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, col, 1)
        sparkline(canvas, data, x0c, py, pw, ph, col, vmin, vmax,
                  zero_line=0. if vmin<0 else None)
        # larger latest value inside plot
        if data:
            cv2.putText(canvas, f"{list(data)[-1]:.2f}",
                        (x0c+3, py+ph-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.36, col, 1)

    # ── Big readout row ───────────────────────────────────────────────────────
    # Two rows so text isn't cramped
    by1 = y0 + PANEL_H - 50
    by2 = y0 + PANEL_H - 30

    dist_s = f"DIST  {tr.dist_f:.2f} m"  if tr.dist_f else "DIST  ---"
    spd_s  = f"SPD  {tr.speed_kmh():.1f} km/h"
    gr     = tr.growth()
    gr_s   = f"GROW  {gr:+.3f}"
    ttc    = tr.ttc()
    ttc_s  = f"TTC  {ttc:.1f} s"  if ttc < 99 else "TTC  ---"

    sc = (50,240,50)  if tr.speed_f < 0.5  else \
         (0,200,255)  if tr.speed_f < 1.5  else (0,80,255)
    tc = (50,240,50)  if ttc > 5    else \
         (0,180,255)  if ttc > 2    else (0,50,255)

    # Row 1 — distance + speed
    cv2.putText(canvas, dist_s, (8,       by1), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (160,185,210), 1)
    cv2.putText(canvas, spd_s,  (185,     by1), cv2.FONT_HERSHEY_SIMPLEX, 0.52, sc,            1)

    # Row 2 — growth + TTC + edge/reentry tag
    cv2.putText(canvas, gr_s,   (8,       by2), cv2.FONT_HERSHEY_SIMPLEX, 0.52, COL_GR,        1)
    cv2.putText(canvas, ttc_s,  (185,     by2), cv2.FONT_HERSHEY_SIMPLEX, 0.52, tc,            1)
    if tr.reentry_count > 0:
        cv2.putText(canvas, f"RE-ENTRY x{tr.reentry_count}",
                    (370, by2), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0,100,255), 1)

    # ── Level legend strip ────────────────────────────────────────────────────
    lby  = y0 + PANEL_H - 14
    step = CAM_W // 7
    for lvl, cfg in LEVELS.items():
        lx     = lvl * step
        active = (lvl == level)
        cv2.rectangle(canvas, (lx,lby), (lx+step-1, lby+12),
                      cfg["bg"] if active else (20,22,28), -1)
        short = cfg["label"][:8]
        col   = cfg["fg"] if active else (45,50,60)
        cv2.putText(canvas, short, (lx+2, lby+9),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.24, col, 1)

    # ── Approach speed bar ────────────────────────────────────────────────────
    bar_y = y0 + PANEL_H - 2
    sn    = np.clip(tr.speed_f / 3., 0, 1)
    bw    = int(sn * CAM_W)
    bcol  = (50,220,50) if sn<0.3 else (0,200,255) if sn<0.6 else (0,50,255)
    canvas[bar_y-4:bar_y+1, :]   = (20,22,28)
    if bw > 0:
        canvas[bar_y-4:bar_y+1, :bw] = bcol


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
    cv2.resizeWindow(WIN, DISP_W, DISP_H + 130)   # +130 for trackbar rows

    cv2.createTrackbar("Speed Smooth  1=smooth 100=raw",  WIN, 15,100,nothing)
    cv2.createTrackbar("TTC Warn   (x0.1s) def=40",       WIN, 40,100,nothing)
    cv2.createTrackbar("TTC Critical(x0.1s) def=18",      WIN, 18, 50,nothing)
    cv2.createTrackbar("Path Width %  def=40",             WIN, 40, 80,nothing)
    cv2.createTrackbar("Min Speed (x0.1m/s) def=2",       WIN,  2, 20,nothing)

    detector  = ArucoDetector()
    tracker   = Tracker()
    state_m   = AlertStateMachine()
    gpio_ctrl = OutputController()
    # canvas stays at NATIVE resolution — all processing coords are correct
    canvas    = np.zeros((WIN_H, CAM_W, 3), dtype=np.uint8)
    fps_t=time.time(); fcnt=0; last_det=None

    print(f"[INFO] Tracking ID {TARGET_ID} | 7 alert levels | Q to quit")
    print(f"[INFO] Display: {DISP_W}×{DISP_H}  |  Processing: {CAM_W}×{CAM_H}  |  Scale: {DISP_SCALE}x")
    print( "[INFO] Level guide:")
    for lvl,cfg in LEVELS.items():
        horn ="HORN " if cfg["sound"] else "     "
        light="LIGHT" if cfg["gpio"] else "     "
        print(f"  {lvl}: {cfg['label']:20s}  {horn} {light}")
    print()

    while True:
        ret,frame=cap.read()
        if not ret: break
        fcnt+=1

        # Sliders
        sm   = max(1,cv2.getTrackbarPos("Speed Smooth  1=smooth 100=raw",WIN))
        ttc_w= cv2.getTrackbarPos("TTC Warn   (x0.1s) def=40",      WIN)*0.1
        ttc_c= cv2.getTrackbarPos("TTC Critical(x0.1s) def=18",     WIN)*0.1
        pw   = max(10,cv2.getTrackbarPos("Path Width %  def=40",     WIN))
        mspd = cv2.getTrackbarPos("Min Speed (x0.1m/s) def=2",      WIN)*0.1
        ttc_c= min(ttc_c, ttc_w-0.1)

        q_val= 0.001*(10**(sm/50.0))
        px1  = int(CAM_W*(0.5-pw/200.))
        px2  = int(CAM_W*(0.5+pw/200.))

        # ── All processing at native 640×480 ──────────────────────────────────
        det = detector.detect(frame)
        tracker.update(det,px1,px2,q_val)
        if det: last_det=det

        level, edge_case = state_m.evaluate(tracker, ttc_w, ttc_c, mspd)
        gpio_ctrl.apply_level(level)

        draw_path(frame,px1,px2)
        draw_marker(frame,tracker,last_det if tracker.active else None,level)
        fps=fcnt/(time.time()-fps_t+1e-5)
        draw_banner(frame,level,edge_case,fps,tracker,gpio_ctrl)

        canvas[:CAM_H,:] = frame
        draw_panel(canvas, tracker, level)

        # ── Single upscale at display time only — zero compute cost ───────────
        if DISP_SCALE != 1.0:
            display = cv2.resize(canvas, (DISP_W, DISP_H),
                                 interpolation=cv2.INTER_LINEAR)
        else:
            display = canvas

        cv2.imshow(WIN, display)
        if cv2.waitKey(1)&0xFF==ord('q'): break

    cap.release(); cv2.destroyAllWindows()
    print("[INFO] Done.")

if __name__=="__main__":
    main()