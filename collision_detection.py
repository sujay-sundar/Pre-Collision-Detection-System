"""
Black Marker Tracker — Single Window Edition
=============================================
• Black HSV object detection + tracking
• Trajectory prediction, growth bar, trail
• IMU accel/gyro/speed plotted INSIDE the OpenCV window
• Trackbars to toggle IMU panel + speed display
• No matplotlib — zero lag
"""

import cv2
import numpy as np
import time
import threading
from collections import deque

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════
CAM_W, CAM_H      = 640, 480
PANEL_H           = 160          # height of IMU panel below camera feed
WIN_W, WIN_H      = CAM_W, CAM_H + PANEL_H

BLACK_V_MAX       = 60
MIN_BLACK_AREA    = 800
MAX_BLACK_AREA    = 80000

GROWTH_WINDOW     = 10
PERSIST_FRAMES    = 18
PATH_WIDTH_RATIO  = 0.40
CROSSING_MIN_FRAMES = 4

GROWTH_APPROACH   = 0.05
GROWTH_FAST       = 0.18
TTC_WARN          = 4.0
TTC_CRITICAL      = 1.8

TRAIL_LEN         = 25
PREDICT_STEPS     = 20
PLOT_BUF          = CAM_W - 20   # samples kept = plot width

IMU_RATE          = 0.033        # ~30 Hz IMU thread

# ══════════════════════════════════════════════════════════════════════════════
# SIMULATED IMU + VELOCITY
# ══════════════════════════════════════════════════════════════════════════════
class SimulatedIMU:
    def __init__(self):
        self.t = 0.0; self.true_speed = 0.0; self.CYCLE = 30.0

    def get_data(self):
        self.t += IMU_RATE; t = self.t; tc = t % self.CYCLE
        true_ax = 0.55 if tc < 5 else (-0.55 if 20 < tc < 25 else 0.0)
        self.true_speed = max(0.0, self.true_speed + true_ax * IMU_RATE)
        ax = true_ax + 0.04*np.sin(t*3.1) + np.random.normal(0, 0.02)
        ay =           0.05*np.sin(t*1.7)  + np.random.normal(0, 0.012)
        az = 1.0     + 0.03*np.sin(t*4.3)  + np.random.normal(0, 0.01)
        if int(t*10) % 200 < 3: az += 1.3
        gx = 0.8*np.sin(t*0.9) + np.random.normal(0, 0.25)
        gy = 0.5*np.sin(t*1.2) + np.random.normal(0, 0.18)
        gz = 0.3*np.sin(t*0.6) + np.random.normal(0, 0.12)
        mag = float(np.sqrt(ax**2 + ay**2 + az**2))
        return dict(ax=ax, ay=ay, az=az, gx=gx, gy=gy, gz=gz,
                    magnitude=mag, true_speed_ms=self.true_speed,
                    moving_forward=(true_ax > 0.05 or self.true_speed > 0.3),
                    sudden_decel=(ax < -0.35), collision=(mag > 2.5))

class VelocityEstimator:
    def __init__(self):
        self.v = 0.0; self.lpf = 0.0; self.still = 0
    def update(self, ax):
        self.lpf = 0.25*ax + 0.75*self.lpf
        if abs(self.lpf) < 0.04: self.still += 1
        else: self.still = 0
        if self.still >= 15: self.v = 0.0; return 0.0
        self.v = max(0.0, (self.v + self.lpf * 9.81 * IMU_RATE) * 0.998)
        return self.v

# ══════════════════════════════════════════════════════════════════════════════
# SHARED STATE
# ══════════════════════════════════════════════════════════════════════════════
lock = threading.Lock()

buf = {k: deque(maxlen=PLOT_BUF) for k in
       ("ax","ay","az","gx","gy","gz","speed_est","speed_true")}

latest_imu   = {}
latest_alert = 0
running      = True

def imu_thread_fn(imu):
    vel = VelocityEstimator()
    while running:
        d = imu.get_data()
        spd = vel.update(d["ax"])
        d["speed_est"]  = spd
        d["speed_kmh"]  = round(spd * 3.6, 1)
        with lock:
            global latest_imu
            latest_imu = d
            for k in ("ax","ay","az","gx","gy","gz"):
                buf[k].append(d[k])
            buf["speed_est"].append(spd)
            buf["speed_true"].append(d["true_speed_ms"])
        time.sleep(IMU_RATE)

# ══════════════════════════════════════════════════════════════════════════════
# BLACK OBJECT DETECTOR
# ══════════════════════════════════════════════════════════════════════════════
_k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
_k7 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))

def detect_black(frame):
    hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([0,0,0]), np.array([180,255,BLACK_V_MAX]))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  _k3)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, _k7)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts: return None
    cnt  = max(cnts, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    if not (MIN_BLACK_AREA <= area <= MAX_BLACK_AREA): return None
    x,y,w,h = cv2.boundingRect(cnt)
    return x+w//2, y+h//2, x, y, w, h, mask, area

# ══════════════════════════════════════════════════════════════════════════════
# TRACKER
# ══════════════════════════════════════════════════════════════════════════════
class Tracker:
    def __init__(self):
        self.cx=self.cy=None; self.bbox=None
        self.areas  = deque(maxlen=GROWTH_WINDOW)
        self.trail  = deque(maxlen=TRAIL_LEN)
        self.vx=self.vy=0.0
        self.missed=0; self.in_path=0
        self.active=False; self.locked=False

    def update(self, det, px1, px2):
        if det:
            cx,cy,x,y,w,h,_,area = det
            if self.cx is not None:
                dvx = cx-self.cx; dvy = cy-self.cy
                # reject teleport (camera shake)
                if abs(dvx)<150 or not self.active:
                    self.vx = 0.3*dvx + 0.7*self.vx
                    self.vy = 0.3*dvy + 0.7*self.vy
            self.cx,self.cy=cx,cy; self.bbox=(x,y,w,h)
            self.areas.append(area); self.trail.append((cx,cy))
            self.missed=0; self.active=True; self.locked=True
            self.in_path = self.in_path+1 if px1<cx<px2 else max(0,self.in_path-1)
        else:
            self.missed += 1
            if self.missed > PERSIST_FRAMES:
                self.cx=self.cy=None; self.bbox=None
                self.areas.clear(); self.trail.clear()
                self.vx=self.vy=0.0; self.missed=0
                self.in_path=0; self.active=False

    def growth(self):
        h=list(self.areas)
        if len(h)<4: return 0.0
        half=len(h)//2
        o=np.mean(h[:half]); r=np.mean(h[half:])
        return (r-o)/o if o>0 else 0.0

    def ttc(self, gr):
        return round(min(1.0/(gr*7.0),99.9),1) if gr>0.005 else 99.9

    def predict(self):
        if self.cx is None or (self.vx==0 and self.vy==0): return []
        return [(int(self.cx+self.vx*i), int(self.cy+self.vy*i))
                for i in range(1, PREDICT_STEPS+1)]

    def crossing(self): return self.active and self.in_path >= CROSSING_MIN_FRAMES
    def parallel(self, gr): return self.crossing() and gr < 0.02

# ══════════════════════════════════════════════════════════════════════════════
# ALERT
# ══════════════════════════════════════════════════════════════════════════════
LABELS = ["CLEAR", "MARKER DETECTED", "PRE-COLLISION", "!! COLLISION !!"]
# BGR
BANNER_BG = [(10,40,10),(20,40,10),(10,40,80),(0,0,60)]
BANNER_FG = [(60,255,60),(0,220,255),(0,160,255),(0,80,255)]

def get_alert(tr, gr, ttc, imu):
    if imu.get("collision"): return 3
    if not tr.active: return 0
    if gr >= GROWTH_FAST: return 3
    if tr.crossing() and ttc < TTC_CRITICAL: return 3
    if tr.crossing() and ttc < TTC_WARN and gr > GROWTH_APPROACH: return 2
    if tr.active: return 1
    return 0

# ══════════════════════════════════════════════════════════════════════════════
# INLINE PLOT RENDERER  — draws directly onto a numpy canvas
# ══════════════════════════════════════════════════════════════════════════════
PLOT_BG   = (18, 20, 28)
PLOT_GRID = (35, 40, 52)

def draw_sparkline(canvas, data, x0, y0, pw, ph, color,
                   vmin=None, vmax=None, thickness=1):
    """Draw a single waveform inside a bounding box (x0,y0,pw,ph)."""
    arr = np.array(data, dtype=np.float32)
    if len(arr) < 2: return
    if vmin is None: vmin = arr.min()
    if vmax is None: vmax = arr.max()
    rng = vmax - vmin
    if rng < 1e-6: rng = 1.0

    xs = np.linspace(x0, x0+pw-1, len(arr)).astype(int)
    ys = (y0+ph-1 - ((arr-vmin)/rng*(ph-2))).astype(int)
    ys = np.clip(ys, y0, y0+ph-1)

    pts = np.stack([xs, ys], axis=1).reshape(-1,1,2)
    cv2.polylines(canvas, [pts], False, color, thickness, cv2.LINE_AA)

def draw_imu_panel(canvas, show_speed):
    """
    Renders a PANEL_H-tall strip below the camera frame.
    Left half  → Accel XYZ + Gyro XYZ
    Right half → Speed (est vs true) if show_speed else big speed number
    """
    y0   = CAM_H
    pw   = CAM_W // 2 - 10
    ph_a = PANEL_H // 2 - 14   # accel plot height
    ph_g = PANEL_H // 2 - 14   # gyro  plot height

    with lock:
        imu  = dict(latest_imu)
        d    = {k: list(buf[k]) for k in buf}

    # ── background ────────────────────────────────────────────────────────────
    canvas[y0:y0+PANEL_H, :] = PLOT_BG

    # horizontal divider
    cv2.line(canvas,(0,y0),(CAM_W,y0),(60,65,80),1)
    # vertical divider
    cv2.line(canvas,(CAM_W//2,y0),(CAM_W//2,y0+PANEL_H),(60,65,80),1)

    # ── grid lines ────────────────────────────────────────────────────────────
    xL = 8
    for row, yb, ph in [(0, y0+12, ph_a), (1, y0+12+ph_a+18, ph_g)]:
        for yi in [yb, yb+ph//2, yb+ph]:
            cv2.line(canvas,(xL,yi),(xL+pw,yi),PLOT_GRID,1)

    # ── accel plot ────────────────────────────────────────────────────────────
    ya = y0+12
    cv2.putText(canvas,"ACCEL (g)",(xL,ya-2),cv2.FONT_HERSHEY_SIMPLEX,0.33,(100,120,160),1)
    draw_sparkline(canvas, d["ax"], xL, ya, pw, ph_a, (80,80,255), -2.5, 2.5)
    draw_sparkline(canvas, d["ay"], xL, ya, pw, ph_a, (80,200,80), -2.5, 2.5)
    draw_sparkline(canvas, d["az"], xL, ya, pw, ph_a, (80,220,220),-2.5, 2.5)
    # live values
    ax=imu.get("ax",0); ay_=imu.get("ay",0); az=imu.get("az",0)
    cv2.putText(canvas,f"X:{ax:+.2f}",(xL,   ya+ph_a+10),cv2.FONT_HERSHEY_SIMPLEX,0.30,(80,80,255),1)
    cv2.putText(canvas,f"Y:{ay_:+.2f}",(xL+44,ya+ph_a+10),cv2.FONT_HERSHEY_SIMPLEX,0.30,(80,200,80),1)
    cv2.putText(canvas,f"Z:{az:+.2f}", (xL+88,ya+ph_a+10),cv2.FONT_HERSHEY_SIMPLEX,0.30,(80,220,220),1)

    # ── gyro plot ─────────────────────────────────────────────────────────────
    yg = ya + ph_a + 18
    cv2.putText(canvas,"GYRO (°/s)",(xL,yg-2),cv2.FONT_HERSHEY_SIMPLEX,0.33,(100,120,160),1)
    draw_sparkline(canvas, d["gx"], xL, yg, pw, ph_g, (200,80,80),  -5, 5)
    draw_sparkline(canvas, d["gy"], xL, yg, pw, ph_g, (200,160,60), -5, 5)
    draw_sparkline(canvas, d["gz"], xL, yg, pw, ph_g, (180,80,200), -5, 5)
    gx=imu.get("gx",0); gy=imu.get("gy",0); gz=imu.get("gz",0)
    cv2.putText(canvas,f"X:{gx:+.1f}",(xL,   yg+ph_g+10),cv2.FONT_HERSHEY_SIMPLEX,0.30,(200,80,80),1)
    cv2.putText(canvas,f"Y:{gy:+.1f}",(xL+44,yg+ph_g+10),cv2.FONT_HERSHEY_SIMPLEX,0.30,(200,160,60),1)
    cv2.putText(canvas,f"Z:{gz:+.1f}",(xL+88,yg+ph_g+10),cv2.FONT_HERSHEY_SIMPLEX,0.30,(180,80,200),1)

    # ── right half: speed ─────────────────────────────────────────────────────
    rx   = CAM_W//2 + 8
    rpw  = CAM_W//2 - 16

    if show_speed:
        # Speed sparkline (est=purple, true=green-dashed)
        cv2.putText(canvas,"SPEED (m/s)",(rx,y0+12),cv2.FONT_HERSHEY_SIMPLEX,0.33,(100,120,160),1)
        sph = PANEL_H - 30
        draw_sparkline(canvas, d["speed_est"],  rx, y0+16, rpw, sph, (180,80,220), 0, 5)
        draw_sparkline(canvas, d["speed_true"], rx, y0+16, rpw, sph, (60,200,100), 0, 5)

        # 15 km/h line (4.17 m/s)
        limit_y = int(y0+16 + sph - (4.17/5.0)*sph)
        cv2.line(canvas,(rx,limit_y),(rx+rpw,limit_y),(40,40,180),1)
        cv2.putText(canvas,"15km/h",(rx+rpw-44,limit_y-2),
                    cv2.FONT_HERSHEY_SIMPLEX,0.28,(80,80,200),1)

        kmh = imu.get("speed_kmh", 0)
        est_ms = imu.get("speed_est",  d["speed_est"][-1]  if d["speed_est"]  else 0)
        tru_ms = imu.get("true_speed_ms", d["speed_true"][-1] if d["speed_true"] else 0)
        drift  = abs(est_ms - tru_ms)

        sc = (60,255,60) if kmh<10 else (0,200,255) if kmh<14 else (0,80,255)
        cv2.putText(canvas,f"{kmh:.1f} km/h",
                    (rx, y0+PANEL_H-22),cv2.FONT_HERSHEY_SIMPLEX,0.55,sc,1)
        dcol=(80,220,80) if drift<0.3 else (0,180,255) if drift<0.8 else (0,80,255)
        cv2.putText(canvas,f"drift:{drift:.3f}m/s",
                    (rx+rpw-90,y0+PANEL_H-8),cv2.FONT_HERSHEY_SIMPLEX,0.28,dcol,1)
        # legend dots
        cv2.circle(canvas,(rx+rpw-10, y0+12),3,(180,80,220),-1)
        cv2.putText(canvas,"est",(rx+rpw-26,y0+14),cv2.FONT_HERSHEY_SIMPLEX,0.27,(180,80,220),1)
        cv2.circle(canvas,(rx+rpw-10, y0+22),3,(60,200,100),-1)
        cv2.putText(canvas,"true",(rx+rpw-28,y0+24),cv2.FONT_HERSHEY_SIMPLEX,0.27,(60,200,100),1)
    else:
        # Speed hidden — show big km/h number only
        kmh = imu.get("speed_kmh", 0)
        sc  = (60,255,60) if kmh<10 else (0,200,255) if kmh<14 else (0,80,255)
        cv2.putText(canvas,f"{kmh:.0f}",(rx+20, y0+90),
                    cv2.FONT_HERSHEY_SIMPLEX,2.8,sc,3,cv2.LINE_AA)
        cv2.putText(canvas,"km/h",(rx+22, y0+118),
                    cv2.FONT_HERSHEY_SIMPLEX,0.55,(120,130,140),1)

    # ── IMU label ─────────────────────────────────────────────────────────────
    moving = "MOVING" if imu.get("moving_forward") else "STILL"
    decel  = " | DECEL!" if imu.get("sudden_decel") else ""
    col_s  = (60,255,60) if imu.get("moving_forward") else (120,130,140)
    cv2.putText(canvas, moving+decel,
                (rx, y0+PANEL_H-8), cv2.FONT_HERSHEY_SIMPLEX, 0.33, col_s, 1)

# ══════════════════════════════════════════════════════════════════════════════
# CV DRAWING — camera region
# ══════════════════════════════════════════════════════════════════════════════
def draw_path(frame, x1, x2):
    ov=frame.copy()
    cv2.rectangle(ov,(x1,0),(x2,CAM_H),(0,255,100),-1)
    cv2.addWeighted(ov,0.07,frame,0.93,0,frame)
    cv2.line(frame,(x1,0),(x1,CAM_H),(0,220,80),1)
    cv2.line(frame,(x2,0),(x2,CAM_H),(0,220,80),1)
    cv2.putText(frame,"PATH",(x1+4,14),cv2.FONT_HERSHEY_SIMPLEX,0.38,(0,180,60),1)

def draw_mask_tint(frame, mask):
    tint=np.zeros_like(frame); tint[mask>0]=(180,220,0)
    cv2.addWeighted(tint,0.22,frame,0.78,0,frame)

def draw_marker(frame, tr, gr, ttc_val, al):
    if not tr.active or tr.bbox is None: return
    x,y,w,h = tr.bbox; cx,cy = tr.cx, tr.cy
    col = BANNER_FG[al]

    # bbox
    cv2.rectangle(frame,(x,y),(x+w,y+h),col,1)
    s=12
    for px,py,dx,dy in [(x,y,1,1),(x+w,y,-1,1),(x,y+h,1,-1),(x+w,y+h,-1,-1)]:
        cv2.line(frame,(px,py),(px+dx*s,py),col,2)
        cv2.line(frame,(px,py),(px,py+dy*s),col,2)

    # crosshair
    cv2.line(frame,(cx-7,cy),(cx+7,cy),col,1)
    cv2.line(frame,(cx,cy-7),(cx,cy+7),col,1)
    cv2.circle(frame,(cx,cy),3,col,-1)

    # trail
    trail=list(tr.trail)
    for i in range(1,len(trail)):
        a=i/len(trail)
        tc=tuple(int(c*a) for c in col)
        cv2.line(frame,trail[i-1],trail[i],tc,1)

    # predicted trajectory
    pred=tr.predict()
    if pred:
        for i,pt in enumerate(pred):
            a=1-i/len(pred)
            cv2.circle(frame,pt,1,(0,int(80*a),int(220*a)),-1)
        cv2.arrowedLine(frame,(cx,cy),pred[-1],(0,60,180),1,tipLength=0.25)

    # growth bar
    bx=x+w+4
    fill=int(np.clip(gr/GROWTH_FAST,0,1)*h)
    cv2.rectangle(frame,(bx,y),(bx+6,y+h),(35,35,35),-1)
    cv2.rectangle(frame,(bx,y+h-fill),(bx+6,y+h),col,-1)

    # label
    area=int(tr.areas[-1]) if tr.areas else 0
    lbl1=f"BLACK  area:{area}px  g:{gr:+.3f}"
    lbl2=(f"TTC:{ttc_val:.1f}s" if ttc_val<99 else "TTC:---")
    if tr.parallel(gr):  lbl2+="  [PARALLEL]"
    elif tr.crossing():  lbl2+="  [IN PATH]"
    cv2.putText(frame,lbl1,(x,max(y-16,12)),cv2.FONT_HERSHEY_SIMPLEX,0.38,col,1)
    cv2.putText(frame,lbl2,(x,max(y-4, 24)),cv2.FONT_HERSHEY_SIMPLEX,0.38,col,1)

def draw_banner(frame, al, fps, tr):
    bg=BANNER_BG[al]; fg=BANNER_FG[al]
    cv2.rectangle(frame,(0,0),(CAM_W,40),bg,-1)
    cv2.putText(frame,LABELS[al],(10,27),cv2.FONT_HERSHEY_SIMPLEX,0.82,fg,2)
    status=f"FPS:{fps:.0f}  {'LOCKED' if tr.locked else 'SEARCHING'}  {'TRACKED' if tr.active else 'LOST'}"
    cv2.putText(frame,status,(CAM_W-200,27),cv2.FONT_HERSHEY_SIMPLEX,0.35,(160,160,160),1)
    if not tr.active and not tr.locked:
        cv2.putText(frame,"Show a BLACK object to begin",
                    (CAM_W//2-120,CAM_H//2),cv2.FONT_HERSHEY_SIMPLEX,0.52,(0,200,255),1)

# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
WIN = "Black Marker Tracker"

def nothing(_): pass   # trackbar callback

def main():
    global running

    # IMU
    imu_obj = SimulatedIMU()
    t = threading.Thread(target=imu_thread_fn, args=(imu_obj,), daemon=True)
    t.start()

    # Camera
    cap = cv2.VideoCapture(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAM_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)        # reduce camera buffer lag
    if not cap.isOpened():
        print("[ERROR] Cannot open camera"); running=False; return

    # Window + trackbars
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, WIN_W, WIN_H + 60)   # +60 for trackbar area

    cv2.createTrackbar("IMU Panel  OFF|ON", WIN, 1, 1, nothing)
    cv2.createTrackbar("Speed Plot OFF|ON", WIN, 1, 1, nothing)

    tracker = Tracker()
    px1 = int(CAM_W*(0.5-PATH_WIDTH_RATIO/2))
    px2 = int(CAM_W*(0.5+PATH_WIDTH_RATIO/2))

    fps_t=time.time(); fcnt=0
    canvas = np.zeros((WIN_H, WIN_W, 3), dtype=np.uint8)

    print("[INFO] Running — press Q to quit")
    print("[INFO] Trackbars: toggle IMU panel and speed plot independently\n")

    while True:
        ret, frame = cap.read()
        if not ret: break
        fcnt += 1

        # read trackbars
        show_imu   = cv2.getTrackbarPos("IMU Panel  OFF|ON", WIN) == 1
        show_speed = cv2.getTrackbarPos("Speed Plot OFF|ON", WIN) == 1

        det = detect_black(frame)
        if det: draw_mask_tint(frame, det[6])

        tracker.update(det, px1, px2)
        gr      = tracker.growth()
        ttc_val = tracker.ttc(gr)

        with lock: imu_now = dict(latest_imu)
        al = get_alert(tracker, gr, ttc_val, imu_now)

        draw_path(frame, px1, px2)
        draw_marker(frame, tracker, gr, ttc_val, al)
        fps = fcnt/(time.time()-fps_t+1e-5)
        draw_banner(frame, al, fps, tracker)

        # compose canvas
        canvas[:CAM_H, :CAM_W] = frame
        if show_imu:
            draw_imu_panel(canvas, show_speed)
        else:
            # blank panel with hint
            canvas[CAM_H:, :] = (18,20,28)
            cv2.putText(canvas,"IMU panel hidden — use trackbar to enable",
                        (CAM_W//2-160, CAM_H+PANEL_H//2),
                        cv2.FONT_HERSHEY_SIMPLEX,0.45,(60,65,80),1)

        cv2.imshow(WIN, canvas)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    running=False; cap.release(); cv2.destroyAllWindows()
    print("[INFO] Done.")

if __name__=="__main__":
    main()