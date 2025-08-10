#!/usr/bin/env python3

import rospy
from std_msgs.msg import String
import cv2
from ultralytics import YOLO
import torch
import numpy as np
import os
from datetime import datetime
import time
import math

class SimplifiedCameraDetector:
    def __init__(self, weights, source, imgsz, conf):
        """
        Basitleştirilmiş kamera detector 
        """
        rospy.init_node('simplified_camera_detector', anonymous=True)
        
        # === Tek ROS Publisher (Donma Riskini Azaltır) ===
        self.pub = rospy.Publisher('/camera_detections', String, queue_size=10)
        
        # === Temel Parametreler ===
        self.horizontal_fov = 78.0  # C920 Pro FOV
        self.confidence_threshold = conf
        
        # === GELIŞMIŞ TRACKING SİSTEMİ ===
        self.last_detections = []  # Önceki frame detections
        self.object_tracks = {}    # {track_id: {'positions': [], 'class': '', 'last_seen': time}}
        self.next_track_id = 1
        self.tracking_distance_threshold = 80  # pixels (daha hassas)
        self.tracking_timeout = 2.0  # saniye
        
        # === Performans İzleme ===
        self.frame_count = 0
        self.detection_count = 0
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0
        
        # Log dosyası (opsiyonel - kapatılabilir)
        log_dir = os.path.expanduser("~/simple_camera_logs")
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        try:
            self.log_file = open(os.path.join(log_dir, f"detections_{timestamp}.csv"), "w")
            self.log_file.write("timestamp,label,yaw,confidence,track_id,velocity,is_moving\n")
            self.logging_enabled = True
        except:
            self.logging_enabled = False
            rospy.logwarn("Logging disabled - permission issue")
        
        # === Model Yükleme (Minimal) ===
        rospy.loginfo("=== Gelişmiş Kamera Detector ===")
        rospy.loginfo(f"Model: {weights}")
        rospy.loginfo(f"Source: {source}")
        
        # Device seçimi
        if torch.cuda.is_available():
            self.device = 0
            rospy.loginfo("CUDA kullanılıyor")
        else:
            self.device = 'cpu'
            rospy.loginfo("CPU kullanılıyor")
        
        # Model yükleme
        try:
            self.model = YOLO(weights)
            rospy.loginfo("Model yüklendi")
        except Exception as e:
            rospy.logerr(f"Model hatası: {e}")
            raise
        
        # Kamera parametrelerini sakla (stream'i run'da başlat)
        self.source = source
        self.imgsz = imgsz
        
        rospy.loginfo("Detector hazır")
    
    def pixel_to_angle(self, x_pixel, image_width):
        """
        Pixel'i açıya çevir - Geliştirilmiş hesaplama
        """
        cam_center = image_width / 2
        pixel_offset = x_pixel - cam_center
        angle_per_pixel = self.horizontal_fov / image_width
        return -(pixel_offset * angle_per_pixel)
    
    def process_detections_simple(self, results, image):
        """
        Detection'ları gelişmiş şekilde işle 
        """
        detections = []
        
        if results.boxes is None or len(results.boxes) == 0:
            return detections, image
        
        image_height, image_width = image.shape[:2]
        cam_center = image_width / 2
        
        for box in results.boxes:
            try:
                # Temel detection verisi
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                class_name = self.model.names[class_id]
                
                # Confidence filtreleme
                if confidence < self.confidence_threshold:
                    continue
                
                # Merkez ve açı hesaplama
                x_center = (x1 + x2) / 2
                y_center = (y1 + y2) / 2
                yaw_angle = self.pixel_to_angle(x_center, image_width)
                
                # Bounding box boyutu (mesafe tahmini için)
                box_width = x2 - x1
                box_height = y2 - y1
                box_area = box_width * box_height
                
                # Detection objesi (gelişmiş)
                detection = {
                    'class': class_name,
                    'yaw': yaw_angle,
                    'confidence': confidence,
                    'x_center': x_center,
                    'y_center': y_center,
                    'box_width': box_width,
                    'box_height': box_height,
                    'box_area': box_area,
                    'timestamp': time.time()
                }
                
                detections.append(detection)
                
                # Görsel annotation (gelişmiş)
                # Confidence'e göre renk
                if confidence > 0.8:
                    color = (0, 255, 0)  # Yeşil - yüksek confidence
                elif confidence > 0.6:
                    color = (0, 255, 255)  # Sarı - orta confidence
                else:
                    color = (0, 165, 255)  # Turuncu - düşük confidence
                
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                
                # Label
                label = f"{class_name}: {confidence:.2f} ({yaw_angle:+.1f}°)"
                cv2.putText(image, label, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                # Merkez nokta
                cv2.circle(image, (int(x_center), int(y_center)), 5, (0, 0, 255), -1)
                
                # Açı çizgisi (merkeze doğru)
                center_point = (int(image_width/2), int(image_height/2))
                detection_point = (int(x_center), int(y_center))
                cv2.line(image, center_point, detection_point, (255, 0, 0), 1)
                
            except Exception as e:
                rospy.logwarn(f"Detection processing error: {e}")
                continue
        
        return detections, image
    
    def advanced_tracking(self, current_detections):
        """Gelişmiş tracking sistemi"""
        current_time = time.time()
        
        # Match current detections to existing tracks
        for detection in current_detections:
            best_track_id = None
            best_distance = float('inf')
            
            # Find closest existing track
            for track_id, track in self.object_tracks.items():
                if track['class'] != detection['class']:
                    continue
                
                if len(track['positions']) == 0:
                    continue
                
                last_pos = track['positions'][-1]
                
                # Euclidean distance + size similarity
                position_distance = math.sqrt(
                    (detection['x_center'] - last_pos['x'])**2 + 
                    (detection['y_center'] - last_pos['y'])**2
                )
                
                # Size similarity factor
                size_factor = abs(detection['box_area'] - last_pos.get('area', detection['box_area'])) / max(detection['box_area'], 1)
                combined_distance = position_distance * (1 + size_factor * 0.5)
                
                if combined_distance < self.tracking_distance_threshold and combined_distance < best_distance:
                    best_distance = combined_distance
                    best_track_id = track_id
            
            if best_track_id is not None:
                # Update existing track
                self.object_tracks[best_track_id]['positions'].append({
                    'x': detection['x_center'],
                    'y': detection['y_center'],
                    'area': detection['box_area'],
                    'time': current_time
                })
                self.object_tracks[best_track_id]['last_seen'] = current_time
                
                # Keep only last 10 positions for memory efficiency
                if len(self.object_tracks[best_track_id]['positions']) > 10:
                    self.object_tracks[best_track_id]['positions'] = self.object_tracks[best_track_id]['positions'][-10:]
                
                # Calculate movement
                positions = self.object_tracks[best_track_id]['positions']
                if len(positions) >= 2:
                    dt = positions[-1]['time'] - positions[-2]['time']
                    if dt > 0:
                        dx = positions[-1]['x'] - positions[-2]['x']
                        dy = positions[-1]['y'] - positions[-2]['y']
                        pixel_velocity = math.sqrt(dx*dx + dy*dy) / dt
                        
                        # Calculate average velocity over last 3 measurements
                        if len(positions) >= 3:
                            recent_velocities = []
                            for i in range(len(positions)-1, max(len(positions)-4, 0), -1):
                                if i > 0:
                                    dt_recent = positions[i]['time'] - positions[i-1]['time']
                                    if dt_recent > 0:
                                        dx_recent = positions[i]['x'] - positions[i-1]['x']
                                        dy_recent = positions[i]['y'] - positions[i-1]['y']
                                        vel_recent = math.sqrt(dx_recent*dx_recent + dy_recent*dy_recent) / dt_recent
                                        recent_velocities.append(vel_recent)
                            
                            if recent_velocities:
                                pixel_velocity = sum(recent_velocities) / len(recent_velocities)
                        
                        # Improved velocity estimation
                        detection['velocity'] = pixel_velocity * 0.05  # m/s estimate (calibrated)
                        detection['is_moving'] = pixel_velocity > 3  # pixels/sec threshold (adjusted)
                        detection['track_id'] = best_track_id
                    else:
                        detection['velocity'] = 0.0
                        detection['is_moving'] = False
                        detection['track_id'] = best_track_id
                else:
                    detection['velocity'] = 0.0
                    detection['is_moving'] = False
                    detection['track_id'] = best_track_id
            else:
                # Create new track
                self.object_tracks[self.next_track_id] = {
                    'class': detection['class'],
                    'positions': [{'x': detection['x_center'], 'y': detection['y_center'], 'area': detection['box_area'], 'time': current_time}],
                    'last_seen': current_time
                }
                detection['track_id'] = self.next_track_id
                detection['velocity'] = 0.0
                detection['is_moving'] = False
                self.next_track_id += 1
        
        # Remove old tracks
        tracks_to_remove = []
        for track_id, track in self.object_tracks.items():
            if current_time - track['last_seen'] > self.tracking_timeout:
                tracks_to_remove.append(track_id)
        
        for track_id in tracks_to_remove:
            del self.object_tracks[track_id]
        
        return current_detections
    
    def create_enhanced_message(self, detections):
        """
        Gelişmiş ROS mesajı oluştur 
        """
        if not detections:
            return None
        
        # Multiple detection support with enhanced tracking
        detection_strings = []
        for det in detections:
            det_str = f"{det['class']}:{det['yaw']:.1f}"
            
            # Add tracking info if available
            if 'track_id' in det and 'is_moving' in det:
                if det['is_moving']:
                    det_str += f":MOVING:ID{det['track_id']}:VEL{det['velocity']:.1f}"
                else:
                    det_str += f":STATIC:ID{det['track_id']}"
                
                # Add confidence info for high-confidence detections
                if det['confidence'] > 0.8:
                    det_str += f":CONF{det['confidence']:.2f}"
            
            detection_strings.append(det_str)
        
        return ",".join(detection_strings)
    
    def log_detection(self, detection):
        """
        Gelişmiş logging
        """
        if not self.logging_enabled:
            return
        
        try:
            timestamp = datetime.now().isoformat()
            track_id = detection.get('track_id', -1)
            velocity = detection.get('velocity', 0.0)
            is_moving = detection.get('is_moving', False)
            
            self.log_file.write(f"{timestamp},{detection['class']},{detection['yaw']:.2f},{detection['confidence']:.3f},{track_id},{velocity:.2f},{is_moving}\n")
            self.log_file.flush()
        except:
            pass  # Logging hatası sistemi durdurmasın
    
    def calculate_fps(self):
        """FPS hesaplama"""
        self.fps_counter += 1
        current_time = time.time()
        
        if current_time - self.fps_start_time >= 1.0:  # Her saniyede bir güncelle
            self.current_fps = self.fps_counter / (current_time - self.fps_start_time)
            self.fps_counter = 0
            self.fps_start_time = current_time
    
    def add_enhanced_overlay(self, image):
        """
        Gelişmiş overlay 
        """
        height, width = image.shape[:2]
        
        # Merkez çizgisi
        center_x = width // 2
        cv2.line(image, (center_x, 0), (center_x, height), (0, 255, 255), 1)
        
        # FOV çizgileri (kenar sınırları)
        fov_half = self.horizontal_fov / 2
        left_angle = -fov_half
        right_angle = fov_half
        
        # FOV sınır çizgileri
        cv2.line(image, (0, height//2), (width//4, height//2), (100, 100, 100), 1)
        cv2.line(image, (3*width//4, height//2), (width, height//2), (100, 100, 100), 1)
        
        # Gelişmiş bilgi paneli
        info_text = f"Frame: {self.frame_count} | Det: {self.detection_count} | Tracks: {len(self.object_tracks)} | FPS: {self.current_fps:.1f}"
        cv2.putText(image, info_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Sistem durumu
        status_text = f"FOV: {self.horizontal_fov}° | Conf: {self.confidence_threshold:.2f}"
        cv2.putText(image, status_text, (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Active tracks info
        if self.object_tracks:
            y_offset = 90
            for track_id, track in list(self.object_tracks.items())[:5]:  # Show max 5 tracks
                track_text = f"Track {track_id}: {track['class']}"
                cv2.putText(image, track_text, (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                y_offset += 20
    
    def run(self):
        """
        Ana loop - gelişmiş versiyon
        """
        rospy.loginfo("Gelişmiş detection başlatılıyor...")
        
        try:
            # Stream başlatma (eski kodunuz gibi)
            stream = self.model.predict(
                source=self.source,
                stream=True,
                imgsz=self.imgsz,
                conf=self.confidence_threshold,
                device=self.device,
                verbose=False
            )
            
            rospy.loginfo("Stream başladı")
            
            for results in stream:
                self.frame_count += 1
                
                # FPS hesapla
                self.calculate_fps()
                
                # Görüntüyü al
                im = results.orig_img.copy()
                
                # Detection'ları işle (gelişmiş)
                detections, processed_im = self.process_detections_simple(results, im)
                
                # === GELİŞMİŞ TRACKING EKLE ===
                if detections:
                    detections = self.advanced_tracking(detections)
                
                # Detection varsa publish et
                if detections:
                    self.detection_count += len(detections)
                    
                    # ROS mesajı oluştur (gelişmiş format + tracking)
                    message_str = self.create_enhanced_message(detections)
                    
                    if message_str:
                        # Publish et (tek mesaj - donma riski düşük)
                        msg = String()
                        msg.data = message_str
                        self.pub.publish(msg)
                        
                        # Console log (throttled)
                        if self.frame_count % 10 == 0:  # Her 10 frame'de bir
                            rospy.loginfo(f"DETECTION: {message_str}")
                        
                        # File log (tüm detections)
                        for det in detections:
                            self.log_detection(det)
                
                # Gelişmiş overlay ekle
                self.add_enhanced_overlay(processed_im)
                
                # Görüntüyü göster
                cv2.imshow("Gelişmiş Kamera Detection", processed_im)
                
                # ESC ile çıkış
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC
                    break
                elif key == ord('r'):  # R tuşu ile tracking reset
                    self.object_tracks.clear()
                    self.next_track_id = 1
                    rospy.loginfo("Tracking reset")
                
                # ROS shutdown kontrol
                if rospy.is_shutdown():
                    break
                
                # Her 100 frame'de stats
                if self.frame_count % 100 == 0:
                    rospy.loginfo(f"Frame: {self.frame_count}, Detections: {self.detection_count}, "
                                f"Active tracks: {len(self.object_tracks)}, FPS: {self.current_fps:.1f}")
        
        except Exception as e:
            rospy.logerr(f"Detection loop error: {e}")
            import traceback
            rospy.logerr(traceback.format_exc())
        
        finally:
            cv2.destroyAllWindows()
            if self.logging_enabled:
                self.log_file.close()
            rospy.loginfo("Gelişmiş detector kapatıldı")

def main():
    """
    Ana fonksiyon
    """
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True, help="YOLO model (.pt)")
    parser.add_argument("--source", default="0", help="Camera source")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--conf", type=float, default=0.5, help="Confidence")
    
    args = parser.parse_args()
    
    try:
        detector = SimplifiedCameraDetector(args.weights, args.source, args.imgsz, args.conf)
        detector.run()
        
    except rospy.ROSInterruptException:
        rospy.loginfo("ROS shutdown")
    except KeyboardInterrupt:
        rospy.loginfo("Keyboard interrupt")
    except Exception as e:
        rospy.logerr(f"Error: {e}")

if __name__ == '__main__':
    main()