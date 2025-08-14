#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Autonomous Buoy Nav - Single File
- Jetson Orin Nano, Ubuntu 20.04
- OpenCV camera + Ultralytics YOLO .pt model
- Dynamic West/East ↔ PORT/STARBOARD rule decided from initial heading
- FSM + PD control
- MAVLink (pymavlink) RC override to Cube Orange / ArduRover

Install:
  pip install ultralytics opencv-python pymavlink pyyaml

Run example:
  python3 auton_buoy_nav.py --model /path/to/model.pt --cam 0 --fcu udpout:127.0.0.1:14550

Notes:
- West(E)=GREEN, East(E)=RED mapping is enforced regardless of class granularity.
- If your model's class names include 'green'/'red', mapping uses those; else, optional color probe from image patch.
"""

import argparse, math, time, sys, os
import cv2
import numpy as np

from ultralytics import YOLO
from pymavlink import mavutil
import yaml

# -------------------- Utils: geometry --------------------

def wrap_pi(a):
    return (a + math.pi) % (2*math.pi) - math.pi

def pol2xy(r, th):
    return (r*math.cos(th), r*math.sin(th))

def xy2pol(x, y):
    return (math.hypot(x, y), math.atan2(y, x))

# -------------------- Camera & range/angle estimation --------------------

class PolarEstimator:
    """
    Angle θ from horizontal FOV; Range r from bbox height using pinhole (H_real, fx).
    Fallback: r ~ k / bbox_height if H_real not given.
    """
    def __init__(self, width, height, hfov_deg=78.0, buoy_h_real=0.6):
        self.width = width
        self.height = height
        self.hfov = math.radians(hfov_deg)
        self.cx = width/2.0
        self.fx = (width/2.0) / math.tan(self.hfov/2.0)  # pixels
        self.buoy_h_real = buoy_h_real  # [m]; tune to your buoy

    def bbox_to_polar(self, box):
        """
        box: (x1,y1,x2,y2) in pixels
        θ: from bbox center x
        r: from bbox height (pinhole)
        """
        x1,y1,x2,y2 = box
        u = (x1 + x2)/2.0
        h_px = max(1.0, (y2 - y1))
        # Angle:
        # approximate pinhole: tan(theta) ~ (u - cx)/fx
        th = math.atan2((u - self.cx), self.fx)
        # Range:
        if self.buoy_h_real and self.fx:
            r = (self.buoy_h_real * self.fx) / h_px
            r = float(np.clip(r, 0.2, 100.0))
        else:
            # crude fallback
            r = 1000.0 / h_px
        return r, th

# -------------------- Color/side mapping --------------------

def load_names_from_yaml(yaml_path):
    try:
        with open(yaml_path, "r") as f:
            data = yaml.safe_load(f)
        names = data.get("names") or data.get("class_names") or []
        # ensure id->name order if dict
        if isinstance(names, dict):
            names = [names[k] for k in sorted(names.keys(), key=lambda x:int(x))]
        return names
    except Exception:
        return []

def class_to_side_by_name(name: str):
    """
    Robust mapping: if class name mentions 'green' => West, 'red' => East.
    Adjust here if your names are TR/other language keywords.
    """
    lname = name.lower()
    if "green" in lname or "yesil" in lname or "yeşil" in lname:
        return "W"
    if "red" in lname or "kirmizi" in lname or "kırmızı" in lname:
        return "E"
    return None

def color_probe_side(frame, box):
    """
    Optional: probe average hue inside bbox center patch to decide green/red.
    Returns "W" (green) or "E" (red) or None if uncertain.
    """
    x1,y1,x2,y2 = [int(v) for v in box]
    x1 = max(0, x1); y1 = max(0, y1); x2 = min(frame.shape[1]-1, x2); y2 = min(frame.shape[0]-1, y2)
    if x2<=x1 or y2<=y1: return None
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0: return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    h_mean = float(np.mean(h))
    s_mean = float(np.mean(s))
    # Very crude thresholds; tune for your camera/scene
    if s_mean < 40: return None
    # red hue around 0 or 180, green around 60-90
    if h_mean < 15 or h_mean > 170: return "E"   # red
    if 45 < h_mean < 95: return "W"              # green
    return None

# -------------------- MAVLink helpers --------------------

class MavlinkClient:
    def __init__(self, fcu_url):
        self.master = mavutil.mavlink_connection(fcu_url)
        self.master.wait_heartbeat()
        print("[MAV] Heartbeat OK: sys=%d comp=%d" % (self.master.target_system, self.master.target_component))

    def get_initial_heading_deg(self, timeout=5.0):
        """
        Prefer VFR_HUD.heading (deg). Fallback ATTITUDE.yaw (rad).
        """
        t0 = time.time()
        while time.time() - t0 < timeout:
            msg = self.master.recv_match(type=["VFR_HUD","ATTITUDE"], blocking=True, timeout=1.0)
            if not msg: continue
            if msg.get_type() == "VFR_HUD":
                hdg = float(getattr(msg, "heading", None))
                if hdg is not None:
                    return (hdg % 360.0)
            elif msg.get_type() == "ATTITUDE":
                yaw = float(msg.yaw)
                hdg = (math.degrees(yaw) + 360.0) % 360.0
                return hdg
        raise RuntimeError("Heading read timeout")

    # RC override (simple & robust)
    @staticmethod
    def pwm_from_norm(x, center=1500, span=400):
        x = max(-1.0, min(1.0, float(x)))
        return int(center + span*x)

    def send_rc(self, steer_norm, thr_norm, ch_map=(1,3)):
        """
        steer_norm, thr_norm ∈ [-1,1]
        ch_map: (steer_channel_index, throttle_channel_index) 1-based channel numbers (typical CH1 steer, CH3 throttle)
        """
        ch1_idx, ch3_idx = ch_map
        arr = [0]*18
        # map to 0-based indices:
        arr[ch1_idx-1] = self.pwm_from_norm(steer_norm)
        arr[ch3_idx-1] = self.pwm_from_norm(thr_norm)
        self.master.mav.rc_channels_override_send(self.master.target_system, self.master.target_component, *arr)

# -------------------- Main navigation logic --------------------

class Navigator:
    def __init__(self, args):
        self.args = args
        self.cap = cv2.VideoCapture(args.cam)
        if not self.cap.isOpened():
            raise RuntimeError("Camera open failed")
        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError("Camera read failed")
        h, w = frame.shape[:2]
        self.est = PolarEstimator(w, h, hfov_deg=args.hfov, buoy_h_real=args.buoy_h)

        # Load model
        self.model = YOLO(args.model)
        # Names (prefer model.names, else yaml)
        self.names = []
        try:
            self.names = self.model.names if hasattr(self.model, "names") else []
        except Exception:
            self.names = []
        if (not self.names) and args.names_yaml and os.path.exists(args.names_yaml):
            self.names = load_names_from_yaml(args.names_yaml)

        # MAVLink
        self.mav = MavlinkClient(args.fcu)

        # Determine course orientation
        hdg = self.mav.get_initial_heading_deg()
        heading_rad = math.radians(hdg)
        northbound = (math.cos(heading_rad) > 0)  # +N component?
        if northbound:
            self.side_rule = {"W":"PORT", "E":"STARBOARD"}   # S→N
            course_str = "S→N (northbound)"
        else:
            self.side_rule = {"W":"STARBOARD", "E":"PORT"}   # N→S
            course_str = "N→S (southbound)"
        print(f"[RULE] Initial heading={hdg:.1f}°, course={course_str}, mapping: WEST→{self.side_rule['W']}, EAST→{self.side_rule['E']}")

        # FSM & control
        self.expected = args.expected
        self.cooldown = 0.0
        self.prev_e = 0.0
        self.prev_t = time.time()

        # Tunables
        self.Kp = args.Kp; self.Kd = args.Kd; self.Ki = 0.0
        self.v_min = args.v_min; self.v_max = args.v_max
        self.v0 = args.v0; self.kv = args.kv
        self.ahead = args.ahead; self.lat = args.lat
        self.x_pass_gate = args.x_pass_gate; self.x_pass_buoy = args.x_pass_buoy
        self.yaw_rate_max = args.yaw_rate_max
        self.scan_yaw = args.scan_yaw
        self.print_dbg = args.debug

        # Rate
        self.dt_target = 1.0 / args.rate

    # ---- detection helpers ----

    def det_to_side(self, name, frame, box):
        # class name mapping first
        s = class_to_side_by_name(name) if name else None
        if s: return s
        # fallback: color probe
        s = color_probe_side(frame, box)
        return s

    def nearest(self, arr):
        if not arr: return None
        return min(arr, key=lambda b: b[0])  # min r

    # ---- control loop ----

    def loop(self):
        last_send = time.time()
        while True:
            t0 = time.time()
            ok, frame = self.cap.read()
            if not ok: break

            E_list, W_list = [], []

            # YOLO inference
            results = self.model.predict(source=frame, verbose=False, imgsz=self.args.imgsz, conf=self.args.conf)

            for r in results:
                # r.boxes: xyxy, cls, conf
                if not hasattr(r, "boxes"): continue
                for b in r.boxes:
                    xyxy = b.xyxy[0].tolist()
                    cls_id = int(b.cls[0].item()) if b.cls is not None else -1
                    conf = float(b.conf[0].item()) if b.conf is not None else 0.0
                    name = None
                    if self.names and 0 <= cls_id < len(self.names):
                        name = str(self.names[cls_id])
                    side = self.det_to_side(name, frame, xyxy)
                    if side not in ("W","E"):  # unknown color, skip
                        continue
                    r_est, th_est = self.est.bbox_to_polar(xyxy)
                    if side == "W":
                        W_list.append( (r_est, th_est, conf, xyxy) )
                    else:
                        E_list.append( (r_est, th_est, conf, xyxy) )

            # Build target
            have_gate = (len(E_list)>0 and len(W_list)>0)
            target = None
            pass_th = self.x_pass_gate if have_gate else self.x_pass_buoy

            if have_gate:
                e = self.nearest(E_list); w = self.nearest(W_list)
                xe, ye = pol2xy(e[0], e[1])
                xw, yw = pol2xy(w[0], w[1])
                xm, ym = 0.5*(xe+xw), 0.5*(ye+yw)
                r_t, th_t = xy2pol(xm, ym)
                target = (r_t, th_t, True)
            else:
                pool = W_list if self.expected=="W" else E_list
                b = self.nearest(pool)
                if b is None:
                    # search scan: hold still, yaw oscillate
                    yaw_cmd = (self.scan_yaw if (int(time.time())%2==0) else -self.scan_yaw)
                    self.send_cmd(0.0, yaw_cmd)
                    self.sleep_to_rate(t0)
                    continue
                rb, thb = b[0], b[1]
                xb, yb = pol2xy(rb, thb)
                tx, ty = math.cos(thb), math.sin(thb)
                lat_sign = +1.0 if self.side_rule[self.expected]=="PORT" else -1.0
                x_t = xb + self.ahead*tx
                y_t = yb + self.ahead*ty + lat_sign*self.lat
                r_t, th_t = xy2pol(x_t, y_t)
                target = (r_t, th_t, False)

            # Control
            now = time.time()
            dt = max(1e-3, now - self.prev_t)
            self.prev_t = now

            r_t, th_t, is_gate = target
            epsi = wrap_pi(th_t)
            de = (epsi - self.prev_e)/dt
            self.prev_e = epsi

            yaw_rate = self.Kp*epsi + self.Kd*de
            yaw_rate = float(np.clip(yaw_rate, -self.yaw_rate_max, self.yaw_rate_max))

            v = self.v0 - self.kv*abs(epsi)
            v = float(np.clip(v, self.v_min, self.v_max))

            if self.print_dbg:
                info = f"exp={self.expected} gate={is_gate} rt={r_t:.2f} tht={math.degrees(th_t):.1f} epsi={math.degrees(epsi):.1f} v={v:.2f} yaw={yaw_rate:.2f}"
                print(info)

            self.send_cmd(v, yaw_rate)

            # Passed?
            x_forward = r_t * math.cos(th_t)
            if (x_forward < pass_th) and (self.cooldown <= 0.0):
                self.expected = "E" if self.expected=="W" else "W"
                self.cooldown = 0.5  # s
                self.prev_e = 0.0  # small reset

            # cooldown decay
            self.cooldown = max(0.0, self.cooldown - dt)

            self.sleep_to_rate(t0)

    def sleep_to_rate(self, t0):
        dur = time.time() - t0
        extra = self.dt_target - dur
        if extra > 0:
            time.sleep(extra)

    def send_cmd(self, v_ms, yaw_rate_rad_s):
        """
        Map to RC override:
          - steering from yaw_rate normalized to [-1,1] with yaw_rate_max
          - throttle from v normalized to [0,1] with v_max (clip [-1,1] if reverse allowed)
        """
        steer_norm = float(np.clip(yaw_rate_rad_s / self.yaw_rate_max, -1.0, 1.0))
        thr_norm = float(np.clip(v_ms / self.v_max, 0.0, 1.0))  # no reverse
        self.mav.send_rc(steer_norm, thr_norm, ch_map=(self.args.steer_ch, self.args.throttle_ch))

# -------------------- CLI --------------------

def parse_args():
    ap = argparse.ArgumentParser(description="Autonomous Buoy Navigation (single file)")
    ap.add_argument("--model", type=str, required=True, help=".pt model path (Ultralytics YOLO)")
    ap.add_argument("--names-yaml", dest="names_yaml", type=str, default="", help="data.yaml (optional) to read class names")
    ap.add_argument("--cam", type=int, default=0, help="OpenCV camera index")
    ap.add_argument("--fcu", type=str, default="udpout:127.0.0.1:14550", help="pymavlink connection string")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--hfov", type=float, default=78.0, help="camera horizontal FOV in degrees")
    ap.add_argument("--buoy-h", type=float, default=0.6, help="buoy real height estimate (m)")
    ap.add_argument("--expected", type=str, default="W", choices=["W","E"], help="initial expected side")
    ap.add_argument("--Kp", type=float, default=1.2)
    ap.add_argument("--Kd", type=float, default=0.08)
    ap.add_argument("--v0", type=float, default=0.9)
    ap.add_argument("--v_min", type=float, default=0.2)
    ap.add_argument("--v_max", type=float, default=1.2)
    ap.add_argument("--kv", type=float, default=0.8)
    ap.add_argument("--ahead", type=float, default=2.0)
    ap.add_argument("--lat", type=float, default=0.4)
    ap.add_argument("--x-pass-gate", dest="x_pass_gate", type=float, default=0.3)
    ap.add_argument("--x-pass-buoy", dest="x_pass_buoy", type=float, default=0.2)
    ap.add_argument("--yaw-rate-max", type=float, default=0.6)
    ap.add_argument("--rate", type=float, default=15.0, help="control loop Hz")
    ap.add_argument("--steer-ch", type=int, default=1, help="RC steering channel number (1-based)")
    ap.add_argument("--throttle-ch", type=int, default=3, help="RC throttle channel number (1-based)")
    ap.add_argument("--scan-yaw", type=float, default=0.3, help="search yaw rate when no detections")
    ap.add_argument("--debug", action="store_true")
    return ap.parse_args()

def main():
    args = parse_args()
    nav = Navigator(args)
    try:
        nav.loop()
    except KeyboardInterrupt:
        print("\n[INFO] Stopped by user.")
    except Exception as e:
        print("[ERR]", e)
        sys.exit(1)

if __name__ == "__main__":
    main()
