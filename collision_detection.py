"""
ArUco Marker Tracker — Camera Only, Stabilized Speed
=====================================================
Focus: object growth + stable approach speed + TTC + collision prediction
All from camera data only. IMU removed for now.

Sliders:
  - Speed Smooth  : Kalman process noise (lower = smoother speed)
  - TTC Warn (s)  : pre-collision warning threshold
  - TTC Critical  : collision threshold
  - Path Width %  : vehicle path zone width
  - Min Speed     : minimum speed to show (noise floor)

Print marker:  python3 aruco_tracker_v2.py --gen
Run:           python3 aruco_tracker_v2.py
"""

import cv2
import numpy as np
import time
import sys
import os
from collections import deque

# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS  (not user-tunable at runtime)
# ══════════════════════════════════════════════════════════════════════════════
CAM_W, CAM_H       = 640, 480
PANEL_H            = 180          # info panel height below camera view
WIN_H              = CAM_H + PANEL_H

MARKER_REAL_SIZE_M = 0.10         # measure your printed marker (metres)
TARGET_ID          = 0
ARUCO_DICT         = cv2.aruco.DICT_4X4_50
FOCAL_PX           = 554.0        # estimate: (W/2)/tan(HFOV/2), 60° FOV webcam

TRAIL_LEN          = 40
PREDICT_STEPS      = 30
GROWTH_WIN         = 15           # rolling window for growth rate
DIST_HIST_LEN      = 60           # distance history for sparkline
PERSIST_FRAMES     = 25

# Plot layout
PLOT_BG            = (16, 18, 26)
GRID_COL           = (32, 38, 52)
COL_DIST           = (60, 200, 255)
COL_SPEED          = (100, 255, 140)
COL_GROWTH         = (255, 160, 60)
COL_TTC            = (200, 80, 255)

# Alert colours (BGR)
ALERT_BG  = [(8,35,8),   (12,35,55),  (8,35,70),   (0,0,55)  ]
ALERT_FG  = [(50,240,50),(0,210,255), (0,140,255),  (0,60,255)]
ALERT_LBL = ["CLEAR",    "DETECTED",  "PRE-COLLISION", "!! COLLISION !!"]

# ══════════════════════════════════════════════════════════════════════════════
# 1-D KALMAN FILTER  — stabilises noisy distance measurement
# ══════════════════════════════════════════════════════════════════════════════
class KalmanDist:
    """
    Scalar Kalman filter for distance (metres).

    State  x = [distance, velocity]  (velocity = approach speed in m/s)
    Obs    z = raw distance from solvePnP

    Q = process noise  (how much we trust the model)
    R = measurement noise  (how noisy the sensor is)

    Tuning:
      Lower Q  → smoother speed, slower to react to real changes
      Higher Q → faster reaction, more jitter
      R is fixed to typical ArUco distance noise (~1 cm at 1 m)
    """
    def __init__(self, q=0.01, r=0.0015):
        self.x  = np.array([[1.0], [0.0]])   # [dist_m, vel_m/s]
        self.P  = np.eye(2) * 1.0
        self.R  = np.array([[r]])
        self.H  = np.array([[1.0, 0.0]])
        self.q  = q                           # updated from slider
        self._last_t = None

    def update(self, z_dist, q_override=None):
        now = time.time()
        dt  = (now - self._last_t) if self._last_t else 0.033
        dt  = np.clip(dt, 0.005, 0.2)
        self._last_t = now

        q = q_override if q_override is not None else self.q

        # State transition
        F = np.array([[1.0, -dt],   # dist decreases as object approaches
                      [0.0,  1.0]])
        # Process noise scaled by q
        Q = np.array([[q*dt**2, q*dt],
                      [q*dt,    q   ]])

        # Predict
        x_p = F @ self.x
        P_p = F @ self.P @ F.T + Q

        # Update
        z   = np.array([[z_dist]])
        S   = self.H @ P_p @ self.H.T + self.R
        K   = P_p @ self.H.T @ np.linalg.inv(S)
        self.x = x_p + K @ (z - self.H @ x_p)
        self.P = (np.eye(2) - K @ self.H) @ P_p

        dist_filt = float(self.x[0, 0])
        vel_filt  = float(self.x[1, 0])   # positive = approaching
        return max(0.01, dist_filt), vel_filt

    def reset(self, dist):
        self.x = np.array([[dist], [0.0]])
        self.P = np.eye(2) * 1.0
        self._last_t = None

# ══════════════════════════════════════════════════════════════════════════════
# ARUCO DETECTOR
# ══════════════════════════════════════════════════════════════════════════════
class ArucoDetector:
    def __init__(self):
        d      = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
        params = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(d, params)
        cx, cy = CAM_W/2.0, CAM_H/2.0
        self.cam_mat = np.array([[FOCAL_PX,0,cx],[0,FOCAL_PX,cy],[0,0,1]], np.float64)
        self.dist_c  = np.zeros((4,1))
        hs = MARKER_REAL_SIZE_M / 2.0
        self.obj_pts = np.array([[-hs,hs,0],[hs,hs,0],[hs,-hs,0],[-hs,-hs,0]], np.float32)

    def detect(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners_list, ids, _ = self.detector.detectMarkers(gray)
        if ids is None: return None
        for i, mid in enumerate(ids.flatten()):
            if mid != TARGET_ID: continue
            corners = corners_list[i][0]
            x1,y1  = corners.min(axis=0).astype(int)
            x2,y2  = corners.max(axis=0).astype(int)
            cx = int((x1+x2)/2); cy = int((y1+y2)/2)
            ok, rvec, tvec = cv2.solvePnP(
                self.obj_pts, corners.astype(np.float32),
                self.cam_mat, self.dist_c)
            dist_raw = float(np.linalg.norm(tvec)) if ok else \
                       (FOCAL_PX * MARKER_REAL_SIZE_M) / max(x2-x1, 1)
            return dict(cx=cx, cy=cy, x=x1, y=y1, w=x2-x1, h=y2-y1,
                        dist_raw=dist_raw, corners=corners)
        return None

# ══════════════════════════════════════════════════════════════════════════════
# TRACKER
# ══════════════════════════════════════════════════════════════════════════════
class Tracker:
    def __init__(self):
        self.kf          = KalmanDist(q=0.01)
        self.cx = self.cy = None
        self.bbox        = None
        self.trail       = deque(maxlen=TRAIL_LEN)
        self.vx = self.vy = 0.0

        # Histories for plots
        self.dist_hist   = deque(maxlen=DIST_HIST_LEN)
        self.speed_hist  = deque(maxlen=DIST_HIST_LEN)
        self.growth_hist = deque(maxlen=DIST_HIST_LEN)
        self.ttc_hist    = deque(maxlen=DIST_HIST_LEN)
        self.area_hist   = deque(maxlen=GROWTH_WIN)

        self.missed      = 0
        self.in_path     = 0
        self.active      = False
        self.locked      = False

        # Current filtered values
        self.dist_f      = None
        self.speed_f     = 0.0     # m/s approach speed (Kalman output)

    def update(self, det, px1, px2, q_val):
        if det:
            cx, cy = det["cx"], det["cy"]
            area   = det["w"] * det["h"]

            # Pixel velocity (for trajectory prediction)
            if self.cx is not None:
                self.vx = 0.3*(cx-self.cx) + 0.7*self.vx
                self.vy = 0.3*(cy-self.cy) + 0.7*self.vy

            self.cx, self.cy = cx, cy
            self.bbox = (det["x"], det["y"], det["w"], det["h"])
            self.trail.append((cx, cy))
            self.area_hist.append(area)
            self.missed  = 0
            self.active  = True
            self.locked  = True
            self.in_path = self.in_path+1 if px1<cx<px2 else max(0,self.in_path-1)

            # Kalman filter on distance — this is where speed gets stabilised
            self.dist_f, self.speed_f = self.kf.update(det["dist_raw"], q_val)

            # Append to plot histories
            self.dist_hist.append(self.dist_f)
            self.speed_hist.append(max(0.0, self.speed_f))   # only show approach
            self.growth_hist.append(self.growth())
            self.ttc_hist.append(min(self.ttc(), 10.0))       # cap at 10s for plot

        else:
            self.missed += 1
            if self.missed > PERSIST_FRAMES:
                self._reset()

    def _reset(self):
        self.cx=self.cy=None; self.bbox=None
        self.trail.clear(); self.area_hist.clear()
        self.dist_hist.clear(); self.speed_hist.clear()
        self.growth_hist.clear(); self.ttc_hist.clear()
        self.vx=self.vy=0.0; self.missed=0
        self.in_path=0; self.active=False
        self.dist_f=None; self.speed_f=0.0
        self.kf.reset(1.0)

    def growth(self):
        h = list(self.area_hist)
        if len(h) < 4: return 0.0
        half = len(h)//2
        o = np.mean(h[:half]); r = np.mean(h[half:])
        return (r-o)/o if o > 0 else 0.0

    def speed_kmh(self):
        return round(max(0.0, self.speed_f) * 3.6, 1)

    def ttc(self):
        if self.dist_f and self.speed_f > 0.05:
            return round(min(self.dist_f / self.speed_f, 99.9), 1)
        gr = self.growth()
        if gr > 0.005:
            return round(min(1.0/(gr*7.0), 99.9), 1)
        return 99.9

    def predict(self):
        if self.cx is None or (self.vx==0 and self.vy==0): return []
        return [(int(self.cx+self.vx*i), int(self.cy+self.vy*i))
                for i in range(1, PREDICT_STEPS+1)]

    def crossing(self): return self.active and self.in_path >= 5
    def parallel(self, gr): return self.crossing() and abs(gr) < 0.015

# ══════════════════════════════════════════════════════════════════════════════
# ALERT
# ══════════════════════════════════════════════════════════════════════════════
def get_alert(tr, ttc_warn, ttc_crit):
    if not tr.active: return 0
    ttc = tr.ttc(); gr = tr.growth()
    if gr > 0.20 or ttc < ttc_crit: return 3
    if tr.crossing() and ttc < ttc_warn and (gr > 0.04 or tr.speed_f > 0.3): return 2
    return 1

# ══════════════════════════════════════════════════════════════════════════════
# SPARKLINE  (inline OpenCV plot)
# ══════════════════════════════════════════════════════════════════════════════
def sparkline(canvas, data, x0, y0, pw, ph, col,
              vmin=0.0, vmax=1.0, zero_line=None, thickness=1):
    arr = np.array(list(data), dtype=np.float32)
    if len(arr) < 2: return
    rng = max(vmax - vmin, 1e-6)
    xs  = np.linspace(x0, x0+pw-1, len(arr)).astype(int)
    ys  = np.clip((y0+ph-1 - (arr-vmin)/rng*(ph-2)).astype(int), y0, y0+ph-1)
    pts = np.stack([xs,ys],axis=1).reshape(-1,1,2)
    cv2.polylines(canvas, [pts], False, col, thickness, cv2.LINE_AA)
    if zero_line is not None:
        zy = int(y0+ph-1 - (zero_line-vmin)/rng*(ph-2))
        zy = np.clip(zy, y0, y0+ph-1)
        cv2.line(canvas,(x0,zy),(x0+pw,zy),GRID_COL,1)

def panel_box(canvas, x0, y0, pw, ph, label, label_col):
    """Draw a labelled plot box."""
    cv2.rectangle(canvas,(x0,y0),(x0+pw,y0+ph),GRID_COL,1)
    # subtle grid
    for yi in [y0+ph//3, y0+2*ph//3]:
        cv2.line(canvas,(x0,yi),(x0+pw,yi),(25,28,38),1)
    cv2.putText(canvas, label, (x0+3, y0-3),
                cv2.FONT_HERSHEY_SIMPLEX, 0.32, label_col, 1)

# ══════════════════════════════════════════════════════════════════════════════
# PANEL DRAWING  — 4 sparklines + live readouts
# ══════════════════════════════════════════════════════════════════════════════
def draw_panel(canvas, tr):
    y0  = CAM_H
    canvas[y0:y0+PANEL_H, :] = PLOT_BG
    cv2.line(canvas,(0,y0),(CAM_W,y0),(50,55,70),1)

    # Layout: 4 equal-width plot cells
    cw   = CAM_W // 4
    ph   = PANEL_H - 44      # plot height (leave room for labels below)
    py   = y0 + 14
    pad  = 6

    cells = [
        (0*cw+pad,  "DISTANCE (m)",     COL_DIST,   tr.dist_hist,   0.0, 5.0),
        (1*cw+pad,  "SPEED (m/s)",      COL_SPEED,  tr.speed_hist,  0.0, 3.0),
        (2*cw+pad,  "GROWTH RATE",      COL_GROWTH, tr.growth_hist,-0.1, 0.3),
        (3*cw+pad,  "TTC (s, max 10)",  COL_TTC,    tr.ttc_hist,    0.0,10.0),
    ]

    for x0c, lbl, col, data, vmin, vmax in cells:
        pw = cw - pad*2
        panel_box(canvas, x0c, py, pw, ph, lbl, col)
        sparkline(canvas, data, x0c, py, pw, ph, col, vmin, vmax,
                  zero_line=0.0 if vmin<0 else None, thickness=1)
        # Latest value
        if data:
            val = list(data)[-1]
            cv2.putText(canvas, f"{val:.2f}",
                        (x0c+3, py+ph-4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.30, col, 1)

    # ── Bottom readout row ────────────────────────────────────────────────────
    by = y0 + PANEL_H - 26

    dist_str  = f"{tr.dist_f:.2f}m"   if tr.dist_f else "---"
    spd_str   = f"{tr.speed_kmh():.1f}km/h"
    gr        = tr.growth()
    gr_str    = f"growth:{gr:+.3f}"
    ttc       = tr.ttc()
    ttc_str   = f"TTC:{ttc:.1f}s"     if ttc < 99 else "TTC:---"
    cross_str = "[IN PATH]" if tr.crossing() else ("[PARALLEL]" if tr.parallel(gr) else "")

    # colour speed by approach rate
    sc = (50,240,50) if tr.speed_f < 0.5 else \
         (0,210,255) if tr.speed_f < 1.5 else (0,100,255)
    tc = (50,240,50) if ttc > 5 else \
         (0,180,255) if ttc > 2 else (0,60,255)

    cv2.putText(canvas, f"dist: {dist_str}",   (8,      by),   cv2.FONT_HERSHEY_SIMPLEX,0.42,(180,200,220),1)
    cv2.putText(canvas, spd_str,                (8+100,  by),   cv2.FONT_HERSHEY_SIMPLEX,0.42,sc,1)
    cv2.putText(canvas, gr_str,                 (8+210,  by),   cv2.FONT_HERSHEY_SIMPLEX,0.42,COL_GROWTH,1)
    cv2.putText(canvas, ttc_str,                (8+360,  by),   cv2.FONT_HERSHEY_SIMPLEX,0.42,tc,1)
    cv2.putText(canvas, cross_str,              (8+460,  by),   cv2.FONT_HERSHEY_SIMPLEX,0.38,(150,160,170),1)

    # Approach indicator bar (full width, 6px tall)
    bar_y  = y0 + PANEL_H - 8
    spd_n  = np.clip(tr.speed_f / 3.0, 0, 1)
    bar_w  = int(spd_n * CAM_W)
    bcol   = (50,240,50) if spd_n<0.3 else (0,200,255) if spd_n<0.6 else (0,60,255)
    cv2.rectangle(canvas,(0,bar_y),(CAM_W,bar_y+6),(22,25,32),-1)
    if bar_w > 0:
        cv2.rectangle(canvas,(0,bar_y),(bar_w,bar_y+6),bcol,-1)
    cv2.putText(canvas,"APPROACH",(CAM_W-65,bar_y+5),cv2.FONT_HERSHEY_SIMPLEX,0.28,(60,65,80),1)

# ══════════════════════════════════════════════════════════════════════════════
# CAMERA FRAME DRAWING
# ══════════════════════════════════════════════════════════════════════════════
def draw_path(frame, x1, x2):
    ov = frame.copy()
    cv2.rectangle(ov,(x1,0),(x2,CAM_H),(0,255,100),-1)
    cv2.addWeighted(ov,0.07,frame,0.93,0,frame)
    cv2.line(frame,(x1,0),(x1,CAM_H),(0,200,70),1)
    cv2.line(frame,(x2,0),(x2,CAM_H),(0,200,70),1)
    cv2.putText(frame,"PATH",(x1+4,14),cv2.FONT_HERSHEY_SIMPLEX,0.36,(0,160,55),1)

def draw_marker(frame, tr, det, al):
    if not tr.active or tr.bbox is None: return
    x,y,w,h = tr.bbox; cx,cy=tr.cx,tr.cy
    col = ALERT_FG[al]

    # ArUco corner polygon
    if det and det.get("corners") is not None:
        pts = det["corners"].astype(int)
        cv2.polylines(frame,[pts],True,col,2)
        for pt in pts:
            cv2.circle(frame,tuple(pt),4,col,-1)

    # Corner brackets
    s=12
    for px,py,dx,dy in [(x,y,1,1),(x+w,y,-1,1),(x,y+h,1,-1),(x+w,y+h,-1,-1)]:
        cv2.line(frame,(px,py),(px+dx*s,py),col,2)
        cv2.line(frame,(px,py),(px,py+dy*s),col,2)

    # Crosshair
    cv2.line(frame,(cx-8,cy),(cx+8,cy),col,1)
    cv2.line(frame,(cx,cy-8),(cx,cy+8),col,1)
    cv2.circle(frame,(cx,cy),3,col,-1)

    # Trail (fading)
    trail = list(tr.trail)
    for i in range(1,len(trail)):
        a  = i/len(trail)
        tc = tuple(int(c*a) for c in col)
        cv2.line(frame,trail[i-1],trail[i],tc,1)

    # Predicted trajectory
    pred = tr.predict()
    if pred:
        for i,pt in enumerate(pred):
            a=1-i/len(pred)
            cv2.circle(frame,pt,1,(0,int(70*a),int(200*a)),-1)
        cv2.arrowedLine(frame,(cx,cy),pred[-1],(0,55,170),1,tipLength=0.22)

    # Growth bar (right of bbox)
    gr  = tr.growth()
    bx  = x+w+4
    fill= int(np.clip(gr/0.20,0,1)*h)
    cv2.rectangle(frame,(bx,y),(bx+6,y+h),(30,30,30),-1)
    if fill > 0:
        gcol=(50,200,50) if gr<0.05 else (0,180,255) if gr<0.12 else (0,60,255)
        cv2.rectangle(frame,(bx,y+h-fill),(bx+6,y+h),gcol,-1)

    # Dist + speed label above box
    dist_s = f"{tr.dist_f:.2f}m" if tr.dist_f else "---"
    spd_s  = f"{tr.speed_kmh():.1f}km/h"
    cv2.putText(frame,f"ID:{TARGET_ID}  {dist_s}  {spd_s}",
                (x, max(y-5,14)), cv2.FONT_HERSHEY_SIMPLEX, 0.40, col, 1)

def draw_banner(frame, al, fps, tr):
    bg=ALERT_BG[al]; fg=ALERT_FG[al]
    cv2.rectangle(frame,(0,0),(CAM_W,40),bg,-1)
    cv2.putText(frame, ALERT_LBL[al], (10,27),
                cv2.FONT_HERSHEY_SIMPLEX, 0.82, fg, 2)
    st = f"FPS:{fps:.0f}  {'LOCKED' if tr.locked else 'SEARCHING'}  {'TRACKED' if tr.active else f'LOST({tr.missed}f)'}"
    cv2.putText(frame,st,(CAM_W-210,27),cv2.FONT_HERSHEY_SIMPLEX,0.33,(150,155,165),1)
    if not tr.active:
        cv2.putText(frame,f"Show ArUco ID {TARGET_ID} ({MARKER_REAL_SIZE_M*100:.0f}cm marker)",
                    (CAM_W//2-155, CAM_H//2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50,(0,190,255),1)

# ══════════════════════════════════════════════════════════════════════════════
# MARKER GENERATOR
# ══════════════════════════════════════════════════════════════════════════════
def generate_marker():
    d   = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    img = cv2.aruco.generateImageMarker(d, TARGET_ID, 400)
    out = cv2.copyMakeBorder(img,50,50,50,50,cv2.BORDER_CONSTANT,value=255)
    fname = f"aruco_id{TARGET_ID}.png"
    cv2.imwrite(fname, out)
    print(f"[MARKER] Saved {fname}")
    print(f"[MARKER] Print at exactly {MARKER_REAL_SIZE_M*100:.0f}×{MARKER_REAL_SIZE_M*100:.0f} cm")
    print(f"[MARKER] Measure actual size and set MARKER_REAL_SIZE_M accordingly")

# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
WIN = "ArUco Tracker"

def nothing(_): pass

def main():
    if "--gen" in sys.argv:
        generate_marker(); return

    if not os.path.exists(f"aruco_id{TARGET_ID}.png"):
        generate_marker()

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAM_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        print("[ERROR] Cannot open camera"); return

    # ── Window + sliders ──────────────────────────────────────────────────────
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, CAM_W, WIN_H + 120)   # +120 for 5 trackbars

    # Speed smoothness: maps 1–100 → q = 0.001–0.1
    # Low value = heavy Kalman smoothing (stable but slow)
    # High value = light smoothing (reactive but jittery)
    cv2.createTrackbar("Speed Smooth  1=max 100=raw", WIN, 15, 100, nothing)

    # TTC thresholds (stored as integer × 10, so 35 = 3.5 s)
    cv2.createTrackbar("TTC Warn  (x0.1s)  default=40",    WIN, 40, 100, nothing)
    cv2.createTrackbar("TTC Critical (x0.1s) default=18",  WIN, 18,  50, nothing)

    # Path width: 20–80 %
    cv2.createTrackbar("Path Width %  default=40",          WIN, 40,  80, nothing)

    # Min speed to display (noise floor filter), 0–20 in 0.1 m/s steps
    cv2.createTrackbar("Min Speed (x0.1 m/s) default=2",   WIN,  2,  20, nothing)

    detector = ArucoDetector()
    tracker  = Tracker()
    canvas   = np.zeros((WIN_H, CAM_W, 3), dtype=np.uint8)
    fps_t    = time.time(); fcnt = 0
    last_det = None

    print(f"[INFO] Tracking ArUco ID {TARGET_ID}  |  Q to quit")
    print( "[INFO] Sliders: Speed Smooth, TTC Warn, TTC Critical, Path Width, Min Speed\n")

    while True:
        ret, frame = cap.read()
        if not ret: break
        fcnt += 1

        # ── Read sliders ──────────────────────────────────────────────────────
        sm_raw   = cv2.getTrackbarPos("Speed Smooth  1=max 100=raw",    WIN)
        ttc_w    = cv2.getTrackbarPos("TTC Warn  (x0.1s)  default=40",  WIN) * 0.1
        ttc_c    = cv2.getTrackbarPos("TTC Critical (x0.1s) default=18",WIN) * 0.1
        pw_pct   = cv2.getTrackbarPos("Path Width %  default=40",        WIN)
        min_spd  = cv2.getTrackbarPos("Min Speed (x0.1 m/s) default=2", WIN) * 0.1

        # Map slider 1–100 → q 0.001–0.10 (log scale for feel)
        sm_raw   = max(1, sm_raw)
        q_val    = 0.001 * (10 ** (sm_raw / 50.0))   # 0.001 at 0 → 0.1 at 100

        # Clamp critical < warn
        ttc_c    = min(ttc_c, ttc_w - 0.1)
        pw_pct   = max(10, pw_pct)
        px1      = int(CAM_W * (0.5 - pw_pct/200.0))
        px2      = int(CAM_W * (0.5 + pw_pct/200.0))

        # ── Detect + track ────────────────────────────────────────────────────
        det = detector.detect(frame)
        tracker.update(det, px1, px2, q_val)
        if det: last_det = det

        # Apply min speed floor (suppress display noise)
        disp_spd = tracker.speed_f if tracker.speed_f > min_spd else 0.0

        al  = get_alert(tracker, ttc_w, ttc_c)

        # ── Draw camera region ────────────────────────────────────────────────
        draw_path(frame, px1, px2)
        draw_marker(frame, tracker, last_det if tracker.active else None, al)

        fps = fcnt / (time.time() - fps_t + 1e-5)
        draw_banner(frame, al, fps, tracker)

        canvas[:CAM_H, :] = frame

        # ── Draw info panel ───────────────────────────────────────────────────
        draw_panel(canvas, tracker)

        # Overlay min_spd threshold on speed sparkline cell
        if min_spd > 0:
            spd_x0 = CAM_W//4 + 6
            spd_pw = CAM_W//4 - 12
            spd_py = CAM_H + 14
            spd_ph = PANEL_H - 44
            th_y   = int(spd_py + spd_ph - 1 - (min_spd/3.0)*(spd_ph-2))
            th_y   = np.clip(th_y, spd_py, spd_py+spd_ph)
            cv2.line(canvas,(spd_x0,th_y),(spd_x0+spd_pw,th_y),(80,80,80),1)
            cv2.putText(canvas,"floor",(spd_x0+spd_pw-28,th_y-2),
                        cv2.FONT_HERSHEY_SIMPLEX,0.25,(80,80,80),1)

        cv2.imshow(WIN, canvas)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release(); cv2.destroyAllWindows()
    print("[INFO] Done.")

if __name__ == "__main__":
    main()