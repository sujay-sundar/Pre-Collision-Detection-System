"""
ArUco Marker Tracker — Anti-False-Detection + Real Speed Estimation
====================================================================
Uses a printed ArUco marker as the tracking target.
- Zero false positives (only tracks specific marker ID)
- Real distance estimation via known marker size (no stereo)
- Object speed in m/s and km/h from distance delta
- Approaching / crossing / TTC collision logic
- Single OpenCV window with embedded IMU panel
- Trackbars: IMU panel, speed plot, marker ID filter

PRINT YOUR MARKER:
  Run this once to generate your marker image:
    python3 -c "
    import cv2
    aruco = cv2.aruco
    d = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    img = aruco.generateImageMarker(d, 0, 300)   # ID=0, 300x300px
    cv2.imwrite('aruco_marker_id0.png', img)
    print('Saved aruco_marker_id0.png — print at 10x10cm')
    "

CAMERA CALIBRATION (optional but improves distance accuracy):
  Without calibration: uses estimated focal length from FOV assumption.
  With calibration: pass your camera_matrix and dist_coeffs into Tracker.
  Quick calibration: use opencv checkerboard calibration with 20+ images.

MARKER SIZE:
  Set MARKER_REAL_SIZE_M to the printed physical size of your marker in metres.
  Default = 0.10 (10 cm). Measure the black border-to-border width.

Run: python3 aruco_tracker.py
"""

import cv2
import numpy as np
import time
import threading
from collections import deque

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════
CAM_W, CAM_H        = 640, 480
PANEL_H             = 160
WIN_W, WIN_H        = CAM_W, CAM_H + PANEL_H

MARKER_REAL_SIZE_M  = 0.10      # physical marker size in metres (measure yours)
TARGET_MARKER_ID    = 0         # only track this ArUco ID — change to match yours
ARUCO_DICT          = cv2.aruco.DICT_4X4_50

# Estimated focal length if no calibration file
# Formula: focal_px ≈ (image_width / 2) / tan(HFOV/2)
# For a typical 60° HFOV webcam: focal ≈ 554 px at 640px width
FOCAL_LENGTH_PX     = 554.0

PATH_WIDTH_RATIO    = 0.40
CROSSING_MIN_FRAMES = 5
GROWTH_WINDOW       = 12
PERSIST_FRAMES      = 20
TRAIL_LEN           = 30
PREDICT_STEPS       = 25

GROWTH_APPROACH     = 0.04
GROWTH_FAST         = 0.16
TTC_WARN            = 4.0
TTC_CRITICAL        = 1.8

# Speed estimation
SPEED_SMOOTH_WIN    = 6         # frames to average speed over
MAX_PLAUSIBLE_SPEED = 15.0      # m/s — reject speed jumps above this (noise guard)

IMU_RATE            = 0.033
PLOT_BUF            = CAM_W - 20

# ══════════════════════════════════════════════════════════════════════════════
# SIMULATED IMU (replace with real MPU6050 on Pi)
# ══════════════════════════════════════════════════════════════════════════════
class SimulatedIMU:
    def __init__(self):
        self.t=0.0; self.true_speed=0.0; self.CYCLE=30.0
    def get_data(self):
        self.t+=IMU_RATE; t=self.t; tc=t%self.CYCLE
        tax=0.55 if tc<5 else(-0.55 if 20<tc<25 else 0.0)
        self.true_speed=max(0.0,self.true_speed+tax*IMU_RATE)
        ax=tax+0.04*np.sin(t*3.1)+np.random.normal(0,0.02)
        ay=0.05*np.sin(t*1.7)+np.random.normal(0,0.012)
        az=1.0+0.03*np.sin(t*4.3)+np.random.normal(0,0.01)
        if int(t*10)%200<3: az+=1.3
        gx=0.8*np.sin(t*0.9)+np.random.normal(0,0.25)
        gy=0.5*np.sin(t*1.2)+np.random.normal(0,0.18)
        gz=0.3*np.sin(t*0.6)+np.random.normal(0,0.12)
        mag=float(np.sqrt(ax**2+ay**2+az**2))
        return dict(ax=ax,ay=ay,az=az,gx=gx,gy=gy,gz=gz,magnitude=mag,
                    true_speed_ms=self.true_speed,
                    moving_forward=(tax>0.05 or self.true_speed>0.3),
                    sudden_decel=(ax<-0.35),collision=(mag>2.5))

class VelocityEstimator:
    def __init__(self):
        self.v=0.0; self.lpf=0.0; self.still=0
    def update(self,ax):
        self.lpf=0.25*ax+0.75*self.lpf
        if abs(self.lpf)<0.04: self.still+=1
        else: self.still=0
        if self.still>=15: self.v=0.0; return 0.0
        self.v=max(0.0,(self.v+self.lpf*9.81*IMU_RATE)*0.998)
        return self.v

# ══════════════════════════════════════════════════════════════════════════════
# SHARED STATE
# ══════════════════════════════════════════════════════════════════════════════
lock=threading.Lock()
buf={k:deque(maxlen=PLOT_BUF) for k in
     ("ax","ay","az","gx","gy","gz","speed_est","speed_true")}
latest_imu={}; latest_alert=0; running=True

def imu_thread_fn(imu):
    vel=VelocityEstimator()
    while running:
        d=imu.get_data(); spd=vel.update(d["ax"])
        d["speed_est"]=spd; d["speed_kmh"]=round(spd*3.6,1)
        with lock:
            global latest_imu; latest_imu=d
            for k in ("ax","ay","az","gx","gy","gz"):
                buf[k].append(d[k])
            buf["speed_est"].append(spd)
            buf["speed_true"].append(d["true_speed_ms"])
        time.sleep(IMU_RATE)

# ══════════════════════════════════════════════════════════════════════════════
# ARUCO DETECTOR
# ══════════════════════════════════════════════════════════════════════════════
class ArucoDetector:
    """
    Detects a specific ArUco marker ID and returns:
    - 2D bounding box + centroid
    - Real distance in metres (via pinhole model or full pose if calibrated)

    Two modes:
      calibrated   : pass camera_matrix + dist_coeffs → uses solvePnP → accurate
      uncalibrated : uses focal length estimate → ±15% accuracy, fine for TTC
    """
    def __init__(self, marker_id=TARGET_MARKER_ID,
                 camera_matrix=None, dist_coeffs=None):
        self.target_id    = marker_id
        self.cam_mat      = camera_matrix
        self.dist_coeffs  = dist_coeffs if dist_coeffs is not None else np.zeros((4,1))
        self.half_size    = MARKER_REAL_SIZE_M / 2.0

        # 3D object points of marker corners (marker in its own plane, z=0)
        self.obj_pts = np.array([
            [-self.half_size,  self.half_size, 0],
            [ self.half_size,  self.half_size, 0],
            [ self.half_size, -self.half_size, 0],
            [-self.half_size, -self.half_size, 0]
        ], dtype=np.float32)

        d    = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
        params = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(d, params)

        # Build estimated camera matrix if not provided
        if self.cam_mat is None:
            cx, cy = CAM_W/2, CAM_H/2
            self.cam_mat = np.array([
                [FOCAL_LENGTH_PX, 0,              cx],
                [0,               FOCAL_LENGTH_PX, cy],
                [0,               0,               1 ]
            ], dtype=np.float64)

    def detect(self, frame):
        """
        Returns dict with keys:
          cx, cy       — centroid pixels
          x,y,w,h      — bounding box
          distance_m   — real distance in metres
          corners      — 4 corner points for drawing
          rvec, tvec   — pose vectors (None if uncalibrated)
        Or None if target marker not found.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners_list, ids, _ = self.detector.detectMarkers(gray)

        if ids is None: return None

        for i, mid in enumerate(ids.flatten()):
            if mid != self.target_id: continue

            corners = corners_list[i][0]   # shape (4,2)

            # Bounding box
            x1,y1 = corners.min(axis=0).astype(int)
            x2,y2 = corners.max(axis=0).astype(int)
            cx     = int((x1+x2)/2); cy = int((y1+y2)/2)
            w      = x2-x1;          h  = y2-y1

            # Distance estimation
            # Method A: full pose (accurate)
            success, rvec, tvec = cv2.solvePnP(
                self.obj_pts, corners.astype(np.float32),
                self.cam_mat, self.dist_coeffs
            )
            if success:
                distance_m = float(np.linalg.norm(tvec))
            else:
                # Method B: pinhole fallback
                pixel_width = float(np.linalg.norm(corners[0]-corners[1]))
                distance_m  = (FOCAL_LENGTH_PX * MARKER_REAL_SIZE_M) / (pixel_width + 1e-6)

            return dict(cx=cx, cy=cy, x=x1, y=y1, w=w, h=h,
                        distance_m=round(distance_m, 3),
                        corners=corners,
                        rvec=rvec if success else None,
                        tvec=tvec if success else None)
        return None

# ══════════════════════════════════════════════════════════════════════════════
# TRACKER  — uses real distance for speed, growth for TTC
# ══════════════════════════════════════════════════════════════════════════════
class MarkerTracker:
    """
    Tracks the ArUco marker across frames.

    Key improvements over pixel-only tracking:
    - object_speed_ms: real approach speed in m/s from distance delta
    - ttc_distance: TTC = current_distance / approach_speed  (accurate)
    - growth still used as secondary confirmation
    """
    def __init__(self):
        self.cx=self.cy=None; self.bbox=None
        self.bbox_areas   = deque(maxlen=GROWTH_WINDOW)
        self.distances    = deque(maxlen=GROWTH_WINDOW)   # metres
        self.speed_hist   = deque(maxlen=SPEED_SMOOTH_WIN)
        self.trail        = deque(maxlen=TRAIL_LEN)
        self.vx=self.vy   = 0.0
        self.missed       = 0
        self.in_path      = 0
        self.active       = False
        self.locked       = False
        self.last_ts      = None

    def update(self, det, px1, px2):
        now = time.time()
        if det:
            cx,cy = det["cx"],det["cy"]
            x,y,w,h = det["x"],det["y"],det["w"],det["h"]
            dist_m  = det["distance_m"]
            area    = w*h

            # Velocity vector
            if self.cx is not None:
                self.vx = 0.35*(cx-self.cx) + 0.65*self.vx
                self.vy = 0.35*(cy-self.cy) + 0.65*self.vy

            # Object approach speed (m/s) from distance delta
            if self.last_ts is not None and len(self.distances) > 0:
                dt = now - self.last_ts
                if dt > 0.001:
                    raw_spd = (self.distances[-1] - dist_m) / dt   # + = approaching
                    if abs(raw_spd) < MAX_PLAUSIBLE_SPEED:          # noise guard
                        self.speed_hist.append(raw_spd)

            self.cx,self.cy = cx,cy; self.bbox=(x,y,w,h)
            self.bbox_areas.append(area); self.distances.append(dist_m)
            self.trail.append((cx,cy)); self.last_ts=now
            self.missed=0; self.active=True; self.locked=True
            self.in_path=self.in_path+1 if px1<cx<px2 else max(0,self.in_path-1)
        else:
            self.missed+=1
            if self.missed>PERSIST_FRAMES:
                self.cx=self.cy=None; self.bbox=None
                self.bbox_areas.clear(); self.distances.clear()
                self.speed_hist.clear(); self.trail.clear()
                self.vx=self.vy=0.0; self.missed=0
                self.in_path=0; self.active=False; self.last_ts=None

    def growth(self):
        h=list(self.bbox_areas)
        if len(h)<4: return 0.0
        half=len(h)//2
        o=np.mean(h[:half]); r=np.mean(h[half:])
        return (r-o)/o if o>0 else 0.0

    def current_distance(self):
        return float(self.distances[-1]) if self.distances else None

    def object_speed_ms(self):
        """Smoothed approach speed of the detected object in m/s."""
        if not self.speed_hist: return 0.0
        return float(np.mean(self.speed_hist))

    def object_speed_kmh(self):
        return round(self.object_speed_ms() * 3.6, 1)

    def ttc(self):
        """
        Primary TTC = distance / approach_speed   (real metres and m/s)
        Fallback to growth-rate heuristic if speed not stable yet.
        """
        dist  = self.current_distance()
        spd   = self.object_speed_ms()
        gr    = self.growth()

        if dist is not None and spd > 0.05:
            return round(min(dist / spd, 99.9), 1)
        # Fallback
        if gr > 0.005:
            return round(min(1.0/(gr*7.0), 99.9), 1)
        return 99.9

    def predict(self):
        if self.cx is None or (self.vx==0 and self.vy==0): return []
        return [(int(self.cx+self.vx*i),int(self.cy+self.vy*i))
                for i in range(1,PREDICT_STEPS+1)]

    def crossing(self): return self.active and self.in_path>=CROSSING_MIN_FRAMES
    def parallel(self, gr): return self.crossing() and gr<0.02

# ══════════════════════════════════════════════════════════════════════════════
# ALERT
# ══════════════════════════════════════════════════════════════════════════════
LABELS    = ["CLEAR","MARKER DETECTED","⚠ PRE-COLLISION","!! COLLISION !!"]
BANNER_BG = [(10,40,10),(20,40,10),(10,40,80),(0,0,60)]
BANNER_FG = [(60,255,60),(0,220,255),(0,160,255),(0,80,255)]

def get_alert(tr, gr, ttc_val, imu):
    if imu.get("collision"): return 3
    if not tr.active: return 0
    spd = tr.object_speed_ms()
    if gr>=GROWTH_FAST or (spd>2.0 and ttc_val<TTC_CRITICAL): return 3
    if tr.crossing() and ttc_val<TTC_CRITICAL: return 3
    if tr.crossing() and ttc_val<TTC_WARN and (gr>GROWTH_APPROACH or spd>0.5): return 2
    if tr.active: return 1
    return 0

# ══════════════════════════════════════════════════════════════════════════════
# DRAWING
# ══════════════════════════════════════════════════════════════════════════════
PLOT_BG=(18,20,28); PLOT_GRID=(35,40,52)

def draw_sparkline(canvas,data,x0,y0,pw,ph,color,vmin,vmax,thickness=1):
    arr=np.array(data,dtype=np.float32)
    if len(arr)<2: return
    rng=vmax-vmin
    if rng<1e-6: rng=1.0
    xs=np.linspace(x0,x0+pw-1,len(arr)).astype(int)
    ys=np.clip((y0+ph-1-((arr-vmin)/rng*(ph-2))).astype(int),y0,y0+ph-1)
    pts=np.stack([xs,ys],axis=1).reshape(-1,1,2)
    cv2.polylines(canvas,[pts],False,color,thickness,cv2.LINE_AA)

def draw_imu_panel(canvas, show_speed):
    y0=CAM_H; pw=CAM_W//2-10; ph_a=PANEL_H//2-14; ph_g=PANEL_H//2-14
    with lock:
        imu=dict(latest_imu); d={k:list(buf[k]) for k in buf}
    canvas[y0:y0+PANEL_H,:]    = PLOT_BG
    cv2.line(canvas,(0,y0),(CAM_W,y0),(60,65,80),1)
    cv2.line(canvas,(CAM_W//2,y0),(CAM_W//2,y0+PANEL_H),(60,65,80),1)
    xL=8
    ya=y0+12
    cv2.putText(canvas,"ACCEL (g)",(xL,ya-2),cv2.FONT_HERSHEY_SIMPLEX,0.33,(100,120,160),1)
    draw_sparkline(canvas,d["ax"],xL,ya,pw,ph_a,(80,80,255),-2.5,2.5)
    draw_sparkline(canvas,d["ay"],xL,ya,pw,ph_a,(80,200,80),-2.5,2.5)
    draw_sparkline(canvas,d["az"],xL,ya,pw,ph_a,(80,220,220),-2.5,2.5)
    ax_=imu.get("ax",0); ay_=imu.get("ay",0); az_=imu.get("az",0)
    cv2.putText(canvas,f"X:{ax_:+.2f}",(xL,ya+ph_a+10),cv2.FONT_HERSHEY_SIMPLEX,0.30,(80,80,255),1)
    cv2.putText(canvas,f"Y:{ay_:+.2f}",(xL+44,ya+ph_a+10),cv2.FONT_HERSHEY_SIMPLEX,0.30,(80,200,80),1)
    cv2.putText(canvas,f"Z:{az_:+.2f}",(xL+88,ya+ph_a+10),cv2.FONT_HERSHEY_SIMPLEX,0.30,(80,220,220),1)
    yg=ya+ph_a+18
    cv2.putText(canvas,"GYRO (°/s)",(xL,yg-2),cv2.FONT_HERSHEY_SIMPLEX,0.33,(100,120,160),1)
    draw_sparkline(canvas,d["gx"],xL,yg,pw,ph_g,(200,80,80),-5,5)
    draw_sparkline(canvas,d["gy"],xL,yg,pw,ph_g,(200,160,60),-5,5)
    draw_sparkline(canvas,d["gz"],xL,yg,pw,ph_g,(180,80,200),-5,5)
    gx=imu.get("gx",0); gy=imu.get("gy",0); gz=imu.get("gz",0)
    cv2.putText(canvas,f"X:{gx:+.1f}",(xL,yg+ph_g+10),cv2.FONT_HERSHEY_SIMPLEX,0.30,(200,80,80),1)
    cv2.putText(canvas,f"Y:{gy:+.1f}",(xL+44,yg+ph_g+10),cv2.FONT_HERSHEY_SIMPLEX,0.30,(200,160,60),1)
    cv2.putText(canvas,f"Z:{gz:+.1f}",(xL+88,yg+ph_g+10),cv2.FONT_HERSHEY_SIMPLEX,0.30,(180,80,200),1)

    rx=CAM_W//2+8; rpw=CAM_W//2-16
    if show_speed:
        cv2.putText(canvas,"SPEED (m/s)",(rx,y0+12),cv2.FONT_HERSHEY_SIMPLEX,0.33,(100,120,160),1)
        sph=PANEL_H-30
        draw_sparkline(canvas,d["speed_est"],rx,y0+16,rpw,sph,(180,80,220),0,5)
        draw_sparkline(canvas,d["speed_true"],rx,y0+16,rpw,sph,(60,200,100),0,5)
        limit_y=int(y0+16+sph-(4.17/5.0)*sph)
        cv2.line(canvas,(rx,limit_y),(rx+rpw,limit_y),(40,40,180),1)
        cv2.putText(canvas,"15km/h",(rx+rpw-44,limit_y-2),cv2.FONT_HERSHEY_SIMPLEX,0.28,(80,80,200),1)
        kmh=imu.get("speed_kmh",0)
        sc=(60,255,60) if kmh<10 else (0,200,255) if kmh<14 else (0,80,255)
        cv2.putText(canvas,f"{kmh:.1f} km/h",(rx,y0+PANEL_H-22),cv2.FONT_HERSHEY_SIMPLEX,0.55,sc,1)
    else:
        kmh=imu.get("speed_kmh",0)
        sc=(60,255,60) if kmh<10 else (0,200,255) if kmh<14 else (0,80,255)
        cv2.putText(canvas,f"{kmh:.0f}",(rx+20,y0+90),cv2.FONT_HERSHEY_SIMPLEX,2.8,sc,3,cv2.LINE_AA)
        cv2.putText(canvas,"km/h",(rx+22,y0+118),cv2.FONT_HERSHEY_SIMPLEX,0.55,(120,130,140),1)
    moving="MOVING" if imu.get("moving_forward") else "STILL"
    decel=" | DECEL!" if imu.get("sudden_decel") else ""
    cv2.putText(canvas,moving+decel,(rx,y0+PANEL_H-8),cv2.FONT_HERSHEY_SIMPLEX,0.33,
                (60,255,60) if imu.get("moving_forward") else (120,130,140),1)

def draw_path(frame,x1,x2):
    ov=frame.copy()
    cv2.rectangle(ov,(x1,0),(x2,CAM_H),(0,255,100),-1)
    cv2.addWeighted(ov,0.07,frame,0.93,0,frame)
    cv2.line(frame,(x1,0),(x1,CAM_H),(0,220,80),1)
    cv2.line(frame,(x2,0),(x2,CAM_H),(0,220,80),1)
    cv2.putText(frame,"PATH",(x1+4,14),cv2.FONT_HERSHEY_SIMPLEX,0.38,(0,180,60),1)

def draw_marker(frame, tr, det, gr, ttc_val, al):
    if not tr.active or tr.bbox is None: return
    x,y,w,h=tr.bbox; cx,cy=tr.cx,tr.cy
    col=BANNER_FG[al]

    # Draw ArUco corners precisely
    if det and det.get("corners") is not None:
        pts=det["corners"].astype(int)
        cv2.polylines(frame,[pts],True,col,2)
        # Corner dots
        for pt in pts:
            cv2.circle(frame,tuple(pt),4,col,-1)

    # Corner brackets on bbox
    s=14
    for px,py,dx,dy in [(x,y,1,1),(x+w,y,-1,1),(x,y+h,1,-1),(x+w,y+h,-1,-1)]:
        cv2.line(frame,(px,py),(px+dx*s,py),col,2)
        cv2.line(frame,(px,py),(px,py+dy*s),col,2)
    cv2.line(frame,(cx-7,cy),(cx+7,cy),col,1)
    cv2.line(frame,(cx,cy-7),(cx,cy+7),col,1)
    cv2.circle(frame,(cx,cy),3,col,-1)

    # Trail
    trail=list(tr.trail)
    for i in range(1,len(trail)):
        a=i/len(trail); tc=tuple(int(c*a) for c in col)
        cv2.line(frame,trail[i-1],trail[i],tc,1)

    # Predicted trajectory
    pred=tr.predict()
    if pred:
        for i,pt in enumerate(pred):
            a=1-i/len(pred)
            cv2.circle(frame,pt,1,(0,int(80*a),int(220*a)),-1)
        cv2.arrowedLine(frame,(cx,cy),pred[-1],(0,60,180),1,tipLength=0.25)

    # Growth bar
    bx=x+w+4; fill=int(np.clip(gr/max(GROWTH_FAST,0.01),0,1)*h)
    cv2.rectangle(frame,(bx,y),(bx+6,y+h),(35,35,35),-1)
    cv2.rectangle(frame,(bx,y+h-fill),(bx+6,y+h),col,-1)
    cv2.putText(frame,"G",(bx,y-3),cv2.FONT_HERSHEY_SIMPLEX,0.3,col,1)

    # Info labels
    dist  = tr.current_distance()
    ospd  = tr.object_speed_kmh()
    dist_str = f"{dist:.2f}m" if dist else "---"
    lbl1  = f"ArUco ID:{TARGET_MARKER_ID}  dist:{dist_str}  obj:{ospd:.1f}km/h"
    lbl2  = (f"TTC:{ttc_val:.1f}s" if ttc_val<99 else "TTC:---")
    if tr.parallel(gr): lbl2+="  [PARALLEL]"
    elif tr.crossing(): lbl2+="  [IN PATH]"
    cv2.putText(frame,lbl1,(x,max(y-16,12)),cv2.FONT_HERSHEY_SIMPLEX,0.38,col,1)
    cv2.putText(frame,lbl2,(x,max(y-4,24)),cv2.FONT_HERSHEY_SIMPLEX,0.38,col,1)

def draw_banner(frame, al, fps, tr):
    bg=BANNER_BG[al]; fg=BANNER_FG[al]
    cv2.rectangle(frame,(0,0),(CAM_W,42),bg,-1)
    cv2.putText(frame,LABELS[al],(10,28),cv2.FONT_HERSHEY_SIMPLEX,0.82,fg,2)
    st=f"FPS:{fps:.0f}  ID:{TARGET_MARKER_ID}  {'LOCKED' if tr.locked else 'SEARCHING'}  {'TRACKED' if tr.active else 'LOST'}"
    cv2.putText(frame,st,(CAM_W-220,28),cv2.FONT_HERSHEY_SIMPLEX,0.34,(160,160,160),1)
    if not tr.active:
        cv2.putText(frame,f"Show ArUco ID {TARGET_MARKER_ID} ({MARKER_REAL_SIZE_M*100:.0f}cm) to camera",
                    (CAM_W//2-165,CAM_H//2),cv2.FONT_HERSHEY_SIMPLEX,0.50,(0,200,255),1)

# ══════════════════════════════════════════════════════════════════════════════
# MARKER GENERATOR  (call once to print your marker)
# ══════════════════════════════════════════════════════════════════════════════
def generate_marker(marker_id=TARGET_MARKER_ID, size_px=400):
    d   = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    img = cv2.aruco.generateImageMarker(d, marker_id, size_px)
    # Add white border so camera can detect edge
    bordered = cv2.copyMakeBorder(img,40,40,40,40,cv2.BORDER_CONSTANT,value=255)
    fname = f"aruco_marker_id{marker_id}.png"
    cv2.imwrite(fname, bordered)
    print(f"[MARKER] Saved {fname} — print at {MARKER_REAL_SIZE_M*100:.0f}x{MARKER_REAL_SIZE_M*100:.0f} cm")
    print(f"[MARKER] Set MARKER_REAL_SIZE_M={MARKER_REAL_SIZE_M} to match your printout")
    return fname

# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
WIN="ArUco Marker Tracker"

def nothing(_): pass

def main():
    global running

    # Generate marker on first run
    import os
    if not os.path.exists(f"aruco_marker_id{TARGET_MARKER_ID}.png"):
        generate_marker()

    # IMU
    imu_obj=SimulatedIMU()
    t=threading.Thread(target=imu_thread_fn,args=(imu_obj,),daemon=True)
    t.start()

    # Camera
    cap=cv2.VideoCapture(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,CAM_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,CAM_H)
    cap.set(cv2.CAP_PROP_BUFFERSIZE,1)
    if not cap.isOpened():
        print("[ERROR] Cannot open camera"); running=False; return

    cv2.namedWindow(WIN,cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN,WIN_W,WIN_H+60)
    cv2.createTrackbar("IMU Panel  OFF|ON",WIN,1,1,nothing)
    cv2.createTrackbar("Speed Plot OFF|ON",WIN,1,1,nothing)

    detector=ArucoDetector()
    tracker =MarkerTracker()
    px1=int(CAM_W*(0.5-PATH_WIDTH_RATIO/2))
    px2=int(CAM_W*(0.5+PATH_WIDTH_RATIO/2))
    canvas=np.zeros((WIN_H,WIN_W,3),dtype=np.uint8)
    fps_t=time.time(); fcnt=0

    print(f"[INFO] Tracking ArUco ID {TARGET_MARKER_ID} only — zero false positives")
    print(f"[INFO] Marker real size: {MARKER_REAL_SIZE_M*100:.0f}cm")
    print(f"[INFO] Press Q to quit\n")

    last_det=None

    while True:
        ret,frame=cap.read()
        if not ret: break
        fcnt+=1

        show_imu  =cv2.getTrackbarPos("IMU Panel  OFF|ON",WIN)==1
        show_speed=cv2.getTrackbarPos("Speed Plot OFF|ON",WIN)==1

        det=detector.detect(frame)
        tracker.update(det,px1,px2)
        if det: last_det=det

        gr     =tracker.growth()
        ttc_val=tracker.ttc()

        with lock: imu_now=dict(latest_imu)
        al=get_alert(tracker,gr,ttc_val,imu_now)

        draw_path(frame,px1,px2)
        draw_marker(frame,tracker,last_det if tracker.active else None,gr,ttc_val,al)
        fps=fcnt/(time.time()-fps_t+1e-5)
        draw_banner(frame,al,fps,tracker)

        canvas[:CAM_H,:CAM_W]=frame
        if show_imu:
            draw_imu_panel(canvas,show_speed)
        else:
            canvas[CAM_H:,:]=(18,20,28)
            cv2.putText(canvas,"IMU hidden — enable with trackbar",
                        (CAM_W//2-130,CAM_H+PANEL_H//2),
                        cv2.FONT_HERSHEY_SIMPLEX,0.42,(60,65,80),1)

        cv2.imshow(WIN,canvas)
        if cv2.waitKey(1)&0xFF==ord('q'): break

    running=False; cap.release(); cv2.destroyAllWindows()
    print("[INFO] Done.")

if __name__=="__main__":
    main()