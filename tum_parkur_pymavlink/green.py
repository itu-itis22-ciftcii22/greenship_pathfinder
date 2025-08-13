#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ENTEGRE KOD: Kamera Algılama (YOLO+ArUco) ve MAVLink ile Otonom Navigasyon

Bu betik, kamera verilerini kendi içinde işler ve MAVLink üzerinden otonom
görevleri yürütür. Harici bir veri kaynağına (ROS, UDP vb.) ihtiyaç duymaz.

Görev Akışı:
1.  (Varsa) Dubaların ortasından geçmek için bir global waypoint oluşturur.
2.  10 metre boyunca sağa doğru 1 metre sapma yapan eğrisel bir arama rotası oluşturur.
3.  Bu arama görevini ArduPilot'a yükler ve AUTO modda başlatır.
4.  Arka planda sürekli ArUco ID=2'yi arar.
5.  ArUco ID=2 bulunduğunda, mevcut arama görevini iptal eder ve yeni bir
    park manevrası görevi yükler:
    - 0.45m mesafeye yaklaşma (WP)
    - 2m geri gitme (WP)
    - 90° sola dönüş (YAW komutu)
    - 10m ileri gitme (WP)

Gerekli Kütüphaneler:
  pip install opencv-python ultralytics torch pymavlink

Kullanım:
  python3 <bu_dosyanin_adi.py> --conn udp:127.0.0.1:14550 --weights <yolo_model.pt> --source 0
"""

import argparse
import math
import sys
import threading
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

# Gerekli kütüphaneleri import et
try:
    import cv2
    import numpy as np
    import torch
    from ultralytics import YOLO
    from pymavlink import mavutil
except ImportError as e:
    print(f"Hata: Gerekli bir kütüphane eksik -> {e}")
    print("Lütfen 'pip install opencv-python ultralytics torch pymavlink' komutu ile kurun.")
    sys.exit(1)


# ============================== PARAMETRELER ==============================
TARGET_ARUCO_ID = 2
PARK_DISTANCE_M = 0.45
REVERSE_DISTANCE_M = 2.0
EXIT_DISTANCE_M = 10.0
ACCEPT_RADIUS_M = 0.30
YAW_SPEED_DEG_S = 30.0
CAM_YAW_OFFSET_DEG = 0.0  # Kamera-gövde yaw ofseti (derece)

# ============================== YARDIMCI FONKSİYONLAR ==============================
def deg2rad(d): return d * math.pi / 180.0
def rad2deg(r): return r * 180.0 / math.pi

def enu_to_geodetic(lat_deg: float, lon_deg: float, east_m: float, north_m: float) -> Tuple[float, float]:
    """Lokal ENU (East, North, Up) ofsetlerini global enlem/boylama çevirir."""
    dlat = north_m / 111320.0
    dlon = east_m / (111320.0 * math.cos(math.radians(lat_deg)))
    return lat_deg + dlat, lon_deg + dlon

def body_to_global(east_b: float, north_b: float, heading_deg: float) -> Tuple[float, float]:
    """Aracın gövde koordinat sistemindeki bir ofseti global ENU'ya çevirir."""
    psi = deg2rad(heading_deg)
    # Standart 2D rotasyon matrisi (Kuzey=Y, Doğu=X)
    east_g  =  east_b * math.cos(psi) - north_b * math.sin(psi)
    north_g =  east_b * math.sin(psi) + north_b * math.cos(psi)
    return east_g, north_g

def latlon_to_int(lat: float, lon: float) -> Tuple[int, int]:
    """Enlem/boylamı MAVLink'in beklediği integer formatına çevirir."""
    return int(lat * 1e7), int(lon * 1e7)

# ============================== VERİ SINIFLARI ==============================
@dataclass
class ArucoDet:
    id: int
    distance_m: float
    bearing_deg: float

@dataclass
class BuoyDet:
    color: str
    distance_m: float
    angle_deg: float

# ============================== ENTEGRE KAMERA İŞLEMCİSİ ==============================
class CameraProcessor:
    def __init__(self, weights, source, imgsz, conf, show_video=False):
        print("[CAM] Kamera işlemcisi başlatılıyor...")
        self.lock = threading.Lock()
        self.running = True
        self.show_video = show_video

        # Algılanan son veriler
        self.last_aruco: Optional[ArucoDet] = None
        self.last_buoys: List[BuoyDet] = []

        # Kamera ve YOLO parametreleri
        self.source = int(source) if source.isdigit() else source
        self.imgsz, self.confidence_threshold = imgsz, conf
        self.horizontal_fov = 78.0

        # Bilinen nesne genişlikleri (metre)
        self.CLASS_REAL_WIDTHS_M = {"red_buoy": 0.30, "green_buoy": 0.30}
        self.aruco_marker_size_m = 0.15

        # ArUco
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
        self.aruco_params = cv2.aruco.DetectorParameters()

        self.K, self.dist = None, np.zeros(5)

        # Model Yükleme
        device_str = "CUDA (GPU)" if torch.cuda.is_available() else "CPU"
        print(f"[CAM] Cihaz: {device_str}")
        self.device = 0 if torch.cuda.is_available() else 'cpu'
        
        try:
            self.model = YOLO(weights)
            print(f"[CAM] YOLO modeli '{weights}' yüklendi.")
        except Exception as e:
            raise RuntimeError(f"YOLO modeli yüklenemedi: {e}")

    def _build_default_K(self, frame_shape):
        h, w = frame_shape[:2]
        fx = w / (2.0 * math.tan(math.radians(self.horizontal_fov / 2.0)))
        return np.array([[fx, 0, w/2.0], [0, fx, h/2.0], [0, 0, 1]], dtype=np.float32)

    def run(self):
        """Ana kamera işleme döngüsü (ayrı bir thread'de çalışır)."""
        print("[CAM] Kamera akışı başlatılıyor...")
        stream = self.model.predict(
            source=self.source, stream=True, imgsz=self.imgsz,
            conf=self.confidence_threshold, device=self.device, verbose=False
        )
        
        first_frame = True
        for results in stream:
            if not self.running: break
            im = results.orig_img.copy()
            if first_frame:
                if self.K is None: self.K = self._build_default_K(im.shape)
                first_frame = False

            yolo_dets = self._process_yolo(results)
            aruco_det = self._process_aruco(im)
            
            with self.lock:
                self.last_buoys = yolo_dets
                if aruco_det and aruco_det.id == TARGET_ARUCO_ID:
                    self.last_aruco = aruco_det
                else:
                    self.last_aruco = None
            
            if self.show_video:
                vis_im = self._visualize(im, yolo_dets, self.last_aruco)
                cv2.imshow("Entegre Kamera Algilama", vis_im)
                if cv2.waitKey(1) & 0xFF == 27: self.stop()

        cv2.destroyAllWindows()
        print("[CAM] Kamera işleme durduruldu.")

    def stop(self): self.running = False

    def _process_yolo(self, results):
        detections = []
        if results.boxes is None: return detections
        fx, cx = self.K[0, 0], self.K[0, 2]
        for box in results.boxes:
            class_name = self.model.names[int(box.cls[0])]
            if class_name not in self.CLASS_REAL_WIDTHS_M: continue
            x1, _, x2, _ = map(int, box.xyxy[0])
            w_px = x2 - x1
            if w_px <= 0: continue
            distance = (fx * self.CLASS_REAL_WIDTHS_M[class_name]) / w_px
            bearing = math.degrees(math.atan2(((x1 + x2) / 2.0) - cx, fx))
            detections.append(BuoyDet(class_name.replace("_buoy", ""), distance, bearing))
        return detections

    def _process_aruco(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = cv2.aruco.detectMarkers(gray, self.aruco_dict, parameters=self.aruco_params)
        if ids is None: return None
        for i, marker_id in enumerate(ids.flatten()):
            if marker_id == TARGET_ARUCO_ID:
                rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners[i], self.aruco_marker_size_m, self.K, self.dist)
                tvec = tvecs[0][0]
                distance = np.linalg.norm(tvec)
                bearing = math.degrees(math.atan2(tvec[0], tvec[2]))
                return ArucoDet(marker_id, distance, bearing)
        return None

    def _visualize(self, image, buoys, aruco):
        text_y = 30
        for b in buoys:
            cv2.putText(image, f"{b.color} D={b.distance_m:.1f} B={b.angle_deg:.1f}", (10, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            text_y += 20
        if aruco:
            cv2.putText(image, f"ArUco ID={aruco.id} D={aruco.distance_m:.2f}m, B={aruco.bearing_deg:.1f}deg", (10, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        return image

# ============================== MAVLINK İSTEMCİSİ ==============================
class MavClient:
    def __init__(self, conn_str: str):
        print(f"[MAV] MAVLink bağlanıyor: {conn_str}")
        self.master = mavutil.mavlink_connection(conn_str, autoreconnect=True)
        self.master.wait_heartbeat()
        print(f"[MAV] Heartbeat alındı: sys={self.master.target_system}, comp={self.master.target_component}")

    def get_global_position_heading(self, timeout=3.0) -> Optional[Tuple[float, float, float]]:
        lat = lon = hdg = None
        t0 = time.time()
        while time.time() - t0 < timeout:
            msg = self.master.recv_match(type=["GLOBAL_POSITION_INT", "VFR_HUD"], blocking=True, timeout=1.0)
            if not msg: continue
            if msg.get_type() == "GLOBAL_POSITION_INT": lat, lon = msg.lat / 1e7, msg.lon / 1e7
            elif msg.get_type() == "VFR_HUD": hdg = float(msg.heading)
            if all(v is not None for v in [lat, lon, hdg]): return lat, lon, hdg
        print("[ERR] Konum/başlık bilgisi alınamadı.")
        return None

    def set_mode(self, mode: str):
        if self.master.mode_mapping().get(mode) is None:
            print(f"[ERR] Bilinmeyen mod: {mode}")
            return
        self.master.set_mode(mode)
        print(f"[MAV] Mod → {mode}")

    def arm(self, do_arm: bool):
        action = 1 if do_arm else 0
        self.master.mav.command_long_send(
            self.master.target_system, self.master.target_component,
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            0, action, 0, 0, 0, 0, 0, 0)
        print(f"[MAV] ARM komutu → {do_arm}")

    def upload_mission(self, items: List[dict]):
        if not items:
            print("[WARN] Yüklenecek misyon öğesi yok.")
            return
        count = len(items)
        self.master.waypoint_clear_all_send()
        self.master.mav.mission_count_send(self.master.target_system, self.master.target_component, count, mavutil.mavlink.MAV_MISSION_TYPE_MISSION)
        for i in range(count):
            msg = self.master.recv_match(type=['MISSION_REQUEST'], blocking=True, timeout=3)
            if not msg: raise RuntimeError("Misyon yükleme zaman aşımına uğradı (MISSION_REQUEST).")
            self.master.mav.mission_item_send(
                self.master.target_system, self.master.target_component, msg.seq,
                items[msg.seq]['frame'], items[msg.seq]['command'],
                items[msg.seq]['current'], items[msg.seq]['autocontinue'],
                items[msg.seq]['param1'], items[msg.seq]['param2'], items[msg.seq]['param3'], items[msg.seq]['param4'],
                items[msg.seq]['x'], items[msg.seq]['y'], items[msg.seq]['z']
            )
        ack = self.master.recv_match(type='MISSION_ACK', blocking=True, timeout=3)
        if not ack or ack.type != mavutil.mavlink.MAV_MISSION_ACCEPTED:
            raise RuntimeError(f"Misyon yükleme başarısız, ACK: {ack}")
        print(f"[MAV] Misyon yüklendi: {count} öğe.")

    def wp_nav(self, seq, lat, lon, alt=0.0, accept_radius=ACCEPT_RADIUS_M, yaw_hint=None, current=False):
        return dict(
            seq=seq, frame=mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT,
            command=mavutil.mavlink.MAV_CMD_NAV_WAYPOINT,
            current=1 if current else 0, autocontinue=1,
            param1=0.0, param2=float(accept_radius), param3=0.0,
            param4=float(yaw_hint or 0.0), x=lat, y=lon, z=float(alt)
        )

    def do_yaw(self, seq, angle_deg, speed_deg_s=YAW_SPEED_DEG_S, direction=0.0, relative=False):
        return dict(
            seq=seq, frame=mavutil.mavlink.MAV_FRAME_MISSION,
            command=mavutil.mavlink.MAV_CMD_CONDITION_YAW,
            current=0, autocontinue=1,
            param1=float(angle_deg), param2=float(speed_deg_s),
            param3=float(direction), param4=1.0 if relative else 0.0,
            x=0.0, y=0.0, z=0.0
        )

# ============================== MİSYON OLUŞTURUCULAR ==============================
def build_buoy_center_wp(lat0, lon0, hdg_deg, buoys: List[BuoyDet]) -> Optional[Tuple[float, float]]:
    red = next((b for b in buoys if b.color == "red"), None)
    green = next((b for b in buoys if b.color == "green"), None)
    if not red or not green: return None
    
    avg_dist = 0.5 * (red.distance_m + green.distance_m)
    avg_ang = 0.5 * (red.angle_deg + green.angle_deg) + CAM_YAW_OFFSET_DEG
    east_b, north_b = avg_dist * math.sin(deg2rad(avg_ang)), avg_dist * math.cos(deg2rad(avg_ang))
    east_g, north_g = body_to_global(east_b, north_b, hdg_deg)
    return enu_to_geodetic(lat0, lon0, east_g, north_g)

def build_search_curve_right(lat0, lon0, hdg_deg) -> List[Tuple[float, float]]:
    east_fwd, north_fwd = math.cos(deg2rad(90-hdg_deg)), math.sin(deg2rad(90-hdg_deg))
    east_right, north_right = math.cos(deg2rad(-hdg_deg)), math.sin(deg2rad(-hdg_deg))
    cfg = [(2.5, 0.1), (5.0, 0.3), (7.5, 0.6), (10.0, 1.0)]
    out = []
    for fd, right in cfg:
        e = fd * east_fwd + right * east_right
        n = fd * north_fwd + right * north_right
        out.append(enu_to_geodetic(lat0, lon0, e, n))
    return out

def build_mission_search(mav: MavClient, lat0, lon0, hdg_deg, buoys: List[BuoyDet]) -> List[dict]:
    seq, items = 0, []
    items.append(mav.wp_nav(seq, lat0, lon0, current=True)); seq += 1
    center = build_buoy_center_wp(lat0, lon0, hdg_deg, buoys)
    search_start_lat, search_start_lon, search_start_hdg = (center[0], center[1], hdg_deg) if center else (lat0, lon0, hdg_deg)
    if center: items.append(mav.wp_nav(seq, center[0], center[1])); seq += 1
    
    for lat, lon in build_search_curve_right(search_start_lat, search_start_lon, search_start_hdg):
        items.append(mav.wp_nav(seq, lat, lon)); seq += 1
    return items

def build_mission_aruco_approach(mav: MavClient, lat0, lon0, hdg_deg, det: ArucoDet) -> List[dict]:
    seq, items = 0, []
    items.append(mav.wp_nav(seq, lat0, lon0, current=True)); seq += 1

    total_bearing = det.bearing_deg + CAM_YAW_OFFSET_DEG
    east_b, north_b = det.distance_m * math.sin(deg2rad(total_bearing)), det.distance_m * math.cos(deg2rad(total_bearing))
    east_g, north_g = body_to_global(east_b, north_b, hdg_deg)
    
    ue, un = -east_g / det.distance_m, -north_g / det.distance_m
    e_app, n_app = east_g + PARK_DISTANCE_M * ue, north_g + PARK_DISTANCE_M * un
    lat_app, lon_app = enu_to_geodetic(lat0, lon0, e_app, n_app)
    
    az = math.atan2(east_g - e_app, north_g - n_app)
    yaw_to_marker_deg = (rad2deg(az) + 360.0) % 360.0

    items.append(mav.do_yaw(seq, yaw_to_marker_deg, relative=False)); seq += 1
    items.append(mav.wp_nav(seq, lat_app, lon_app, yaw_hint=yaw_to_marker_deg)); seq += 1
    
    e_rev = e_app - REVERSE_DISTANCE_M * math.cos(deg2rad(90 - yaw_to_marker_deg))
    n_rev = n_app - REVERSE_DISTANCE_M * math.sin(deg2rad(90 - yaw_to_marker_deg))
    lat_rev, lon_rev = enu_to_geodetic(lat0, lon0, e_rev, n_rev)
    items.append(mav.wp_nav(seq, lat_rev, lon_rev, yaw_hint=yaw_to_marker_deg)); seq += 1
    
    items.append(mav.do_yaw(seq, 90.0, direction=-1.0, relative=True)); seq += 1
    
    exit_hdg = (yaw_to_marker_deg - 90.0 + 360.0) % 360.0
    e_exit = e_rev + EXIT_DISTANCE_M * math.cos(deg2rad(90-exit_hdg))
    n_exit = n_rev + EXIT_DISTANCE_M * math.sin(deg2rad(90-exit_hdg))
    lat_exit, lon_exit = enu_to_geodetic(lat0, lon0, e_exit, n_exit)
    items.append(mav.wp_nav(seq, lat_exit, lon_exit, yaw_hint=exit_hdg)); seq += 1
    
    return items

# ============================== ANA AKIŞ ==============================
def main():
    ap = argparse.ArgumentParser(description="Entegre Kamera Algılama ve MAVLink Navigasyon")
    ap.add_argument("--conn", type=str, default="udp:127.0.0.1:14550", help="MAVLink bağlantı adresi")
    ap.add_argument("--weights", required=True, help="YOLO modeli .pt dosyası")
    ap.add_argument("--source", default="0", help="Kamera kaynağı (ID veya video yolu)")
    ap.add_argument("--imgsz", type=int, default=640, help="Görüntü işleme boyutu")
    ap.add_argument("--conf", type=float, default=0.5, help="YOLO güven eşiği")
    ap.add_argument("--no-arm", action="store_true", help="AUTO moda geçmeden önce aracı arm ETME")
    ap.add_argument("--skip-search", action="store_true", help="Arama misyonunu atla, doğrudan ArUco bekle")
    ap.add_argument("--show-video", action="store_true", help="Kamera görüntüsünü ekranda göster")
    args = ap.parse_args()

    camera = CameraProcessor(args.weights, args.source, args.imgsz, args.conf, args.show_video)
    mav = MavClient(args.conn)
    
    cam_thread = threading.Thread(target=camera.run, daemon=True)
    cam_thread.start()

    try:
        print("[MAIN] Başlangıç konumu bekleniyor...")
        pos_data = mav.get_global_position_heading()
        if not pos_data: raise RuntimeError("Başlangıç konumu alınamadı. Çıkılıyor.")
        lat0, lon0, hdg = pos_data
        print(f"[MAIN] Başlangıç: lat={lat0:.7f} lon={lon0:.7f} hdg={hdg:.1f}°")

        if not args.skip_search:
            print("[MAIN] Arama misyonu oluşturuluyor... (Duba tespiti için 3 sn bekle)")
            time.sleep(3)
            with camera.lock: buoys = camera.last_buoys
            search_items = build_mission_search(mav, lat0, lon0, hdg, buoys)
            if len(search_items) > 1:
                mav.upload_mission(search_items)
                mav.set_mode("AUTO")
                if not args.no_arm: mav.arm(True)
            else:
                print("[WARN] Arama için yeterli WP oluşturulamadı; doğrudan ArUco bekleniyor.")

        print(f"[MAIN] ArUco ID={TARGET_ARUCO_ID} bekleniyor...")
        while camera.running:
            with camera.lock: det = camera.last_aruco
            if det:
                print(f"[MAIN] ArUco ID={TARGET_ARUCO_ID} bulundu! Yaklaşma misyonu oluşturuluyor.")
                try:
                    current_pos = mav.get_global_position_heading()
                    if not current_pos:
                        print("[WARN] ArUco bulundu ama güncel konum alınamadı. Tekrar denenecek."); time.sleep(1); continue
                    
                    lat, lon, current_hdg = current_pos
                    approach_items = build_mission_aruco_approach(mav, lat, lon, current_hdg, det)
                    mav.upload_mission(approach_items)
                    mav.set_mode("AUTO")
                    if not args.no_arm: mav.arm(True)
                    print("[MAIN] ArUco yaklaşma misyonu yüklendi. Görev ArduPilot'a devredildi.")
                    break
                except Exception as e:
                    print(f"[ERR] ArUco misyonu oluşturulamadı: {e}")
            time.sleep(0.2)

        print("[MAIN] Program ana döngüsü tamamlandı. Kapatmak için Ctrl+C.")
        while camera.running: time.sleep(1)

    except KeyboardInterrupt:
        print("\n[MAIN] Program kullanıcı tarafından sonlandırıldı.")
    except Exception as e:
        print(f"\n[FATAL] Beklenmedik bir hata oluştu: {e}")
    finally:
        camera.stop()
        if cam_thread.is_alive(): cam_thread.join(timeout=2)
        print("[MAIN] Kapatıldı.")

if __name__ == "__main__":
    main()