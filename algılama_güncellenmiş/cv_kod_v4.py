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
        
        # === BASIT TRACKING SİSTEMİ ===
        self.last_detections = []  # Önceki frame detections
        self.object_tracks = {}    # {track_id: {'positions': [], 'class': '', 'last_seen': time}}
        self.next_track_id = 1
        self.tracking_distance_threshold = 100  # pixels
        
        # === Minimal Logging ===
        self.frame_count = 0
        self.detection_count = 0
        
        # Log dosyası (opsiyonel - kapatılabilir)
        log_dir = os.path.expanduser("~/simple_camera_logs")
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        try:
            self.log_file = open(os.path.join(log_dir, f"detections_{timestamp}.csv"), "w")
            self.log_file.write("timestamp,label,yaw,confidence\n")
            self.logging_enabled = True
        except:
            self.logging_enabled = False
            rospy.logwarn("Logging disabled - permission issue")
        
        # === Model Yükleme (Minimal) ===
        rospy.loginfo("=== Basit Kamera Detector ===")
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
        Pixel'i açıya çevir - Eski kodunuzla aynı
        """
        cam_center = image_width / 2
        pixel_offset = x_pixel - cam_center
        angle_per_pixel = self.horizontal_fov / image_width
        return -(pixel_offset * angle_per_pixel)
    def normalize_label(self, name: str) -> str:
        n = str(name).lower().replace(' ', '_')
        if n in ('vessel', 'ship'):
            n = 'boat'
        if 'red' in n and 'buoy' not in n:
            return 'red_buoy'
        if 'green' in n and 'buoy' not in n:
            return 'green_buoy'
        if 'buoy' in n and 'red' in n:
            return 'red_buoy'
        if 'buoy' in n and 'green' in n:
            return 'green_buoy'
        return n

    
    def process_detections_simple(self, results, image):
        """
        Detection'ları işle 
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
                class_name = self.normalize_label(self.model.names[class_id])
                
                # Merkez ve açı hesaplama
                x_center = (x1 + x2) / 2
                yaw_angle = self.pixel_to_angle(x_center, image_width)
                
                # Detection objesi (minimal)
                detection = {
                    'class': class_name,
                    'yaw': yaw_angle,
                    'confidence': confidence,
                    'x_center': x_center,
                    'y_center': (y1 + y2) / 2,
                    'timestamp': time.time()
                }
                
                detections.append(detection)
                
                # Görsel annotation (basit)
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Label
                label = f"{class_name}: {confidence:.2f} ({yaw_angle:+.1f}°)"
                cv2.putText(image, label, (x1, y2 + 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                
                # Merkez nokta
                cv2.circle(image, (int(x_center), int(image_height/2)), 5, (0, 0, 255), -1)
                
            except Exception as e:
                rospy.logwarn(f"Detection processing error: {e}")
                continue
        
        return detections, image
    
    def simple_tracking(self, current_detections):
        """Basit tracking sistemi"""
        import time
        import math
        
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
                distance = math.sqrt(
                    (detection['x_center'] - last_pos['x'])**2 + 
                    (detection['y_center'] - last_pos['y'])**2
                )
                
                if distance < self.tracking_distance_threshold and distance < best_distance:
                    best_distance = distance
                    best_track_id = track_id
            
            if best_track_id is not None:
                # Update existing track
                self.object_tracks[best_track_id]['positions'].append({
                    'x': detection['x_center'],
                    'y': detection['y_center'],
                    'time': current_time
                })
                self.object_tracks[best_track_id]['last_seen'] = current_time
                
                # Keep only last 5 positions
                if len(self.object_tracks[best_track_id]['positions']) > 5:
                    self.object_tracks[best_track_id]['positions'] = self.object_tracks[best_track_id]['positions'][-5:]
                
                # Calculate movement
                positions = self.object_tracks[best_track_id]['positions']
                if len(positions) >= 2:
                    dt = positions[-1]['time'] - positions[-2]['time']
                    if dt > 0:
                        dx = positions[-1]['x'] - positions[-2]['x']
                        dy = positions[-1]['y'] - positions[-2]['y']
                        pixel_velocity = math.sqrt(dx*dx + dy*dy) / dt
                        
                        # Rough conversion to real velocity (very approximate)
                        detection['velocity'] = pixel_velocity * 0.1  # m/s estimate
                        detection['is_moving'] = pixel_velocity > 5  # pixels/sec threshold
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
                    'positions': [{'x': detection['x_center'], 'y': detection['y_center'], 'time': current_time}],
                    'last_seen': current_time
                }
                detection['track_id'] = self.next_track_id
                detection['velocity'] = 0.0
                detection['is_moving'] = False
                self.next_track_id += 1
        
        # Remove old tracks
        tracks_to_remove = []
        for track_id, track in self.object_tracks.items():
            if current_time - track['last_seen'] > 3.0:  # 3 seconds timeout
                tracks_to_remove.append(track_id)
        
        for track_id in tracks_to_remove:
            del self.object_tracks[track_id]
        
        return current_detections
    
    def create_simple_message(self, detections):
        """
        Basit ROS mesajı oluştur 
        """
        if not detections:
            return None
        
        # Multiple detection support with tracking
        detection_strings = []
        for det in detections:
            det_str = f"{det['class']}:{det['yaw']:.1f}"
            
            # Add tracking info if available
            if 'track_id' in det and 'is_moving' in det:
                if det['is_moving']:
                    det_str += f":MOVING:ID{det['track_id']}:VEL{det['velocity']:.1f}"
                else:
                    det_str += f":STATIC:ID{det['track_id']}"
            
            detection_strings.append(det_str)
        
        return ",".join(detection_strings)
    
    def log_detection(self, detection):
        """
        Minimal logging
        """
        if not self.logging_enabled:
            return
        
        try:
            timestamp = datetime.now().isoformat()
            self.log_file.write(f"{timestamp},{detection['class']},{detection['yaw']:.2f},{detection['confidence']:.3f}\n")
            self.log_file.flush()
        except:
            pass  # Logging hatası sistemi durdurmasın
    
    def add_simple_overlay(self, image):
        """
        Basit overlay 
        """
        height, width = image.shape[:2]
        
        # Merkez çizgisi
        center_x = width // 2
        cv2.line(image, (center_x, 0), (center_x, height), (0, 255, 255), 1)
        
        # Basit bilgi
        info_text = f"Frame: {self.frame_count} | Det: {self.detection_count} | Tracks: {len(self.object_tracks)}"
        cv2.putText(image, info_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    def run(self):
        """
        Ana loop 
        """
        rospy.loginfo("Detection başlatılıyor...")
        
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
                
                # Görüntüyü al
                im = results.orig_img.copy()
                
                # Detection'ları işle (minimal)
                detections, processed_im = self.process_detections_simple(results, im)
                
                # === BASIT TRACKING EKLE ===
                if detections:
                    detections = self.simple_tracking(detections)
                
                # Detection varsa publish et
                if detections:
                    self.detection_count += len(detections)
                    
                    # ROS mesajı oluştur (eski format + tracking)
                    message_str = self.create_simple_message(detections)
                    
                    if message_str:
                        # Publish et (tek mesaj - donma riski düşük)
                        msg = String()
                        msg.data = message_str
                        self.pub.publish(msg)
                        
                        # Console log
                        rospy.loginfo(f"DETECTION: {message_str}")
                        
                        # File log (opsiyonel)
                        for det in detections:
                            self.log_detection(det)
                
                # Basit overlay ekle
                self.add_simple_overlay(processed_im)
                
                # Görüntüyü göster
                cv2.imshow("Basit Kamera Detection", processed_im)
                
                # ESC ile çıkış
                if cv2.waitKey(1) & 0xFF == 27:
                    break
                
                # ROS shutdown kontrol
                if rospy.is_shutdown():
                    break
                
                # Her 100 frame'de stats
                if self.frame_count % 100 == 0:
                    rospy.loginfo(f"Frame: {self.frame_count}, Detections: {self.detection_count}, Active tracks: {len(self.object_tracks)}")
        
        except Exception as e:
            rospy.logerr(f"Detection loop error: {e}")
            import traceback
            rospy.logerr(traceback.format_exc())
        
        finally:
            cv2.destroyAllWindows()
            if self.logging_enabled:
                self.log_file.close()
            rospy.loginfo("Detector kapatıldı")

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