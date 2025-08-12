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
    def __init__(self, weights, source, imgsz, conf, show_gui=True):
        """
        Basitleştirilmiş kamera detector (GUI opsiyonel)
        """
        rospy.init_node('simplified_camera_detector', anonymous=True)

        self.pub = rospy.Publisher('/camera_detections', String, queue_size=10)

        self.horizontal_fov = 78.0  # C920 Pro FOV
        self.confidence_threshold = conf

        self.last_detections = []
        self.object_tracks = {}
        self.next_track_id = 1
        self.tracking_distance_threshold = 100  # pixels

        self.frame_count = 0
        self.detection_count = 0

        # GUI kontrolü
        self.show_gui = show_gui

        # Log dosyası (opsiyonel)
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

        rospy.loginfo("=== Basit Kamera Detector ===")
        rospy.loginfo(f"Model: {weights}")
        rospy.loginfo(f"Source: {source}")

        if torch.cuda.is_available():
            self.device = 0
            rospy.loginfo("CUDA kullanılıyor")
        else:
            self.device = 'cpu'
            rospy.loginfo("CPU kullanılıyor")

        try:
            self.model = YOLO(weights)
            rospy.loginfo("Model yüklendi")
        except Exception as e:
            rospy.logerr(f"Model hatası: {e}")
            raise

        self.source = source
        self.imgsz = imgsz

        rospy.loginfo("Detector hazır")

    def pixel_to_angle(self, x_pixel, image_width):
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
        detections = []

        if results.boxes is None or len(results.boxes) == 0:
            return detections, image

        image_height, image_width = image.shape[:2]

        for box in results.boxes:
            try:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                class_name = self.normalize_label(self.model.names[class_id])

                x_center = (x1 + x2) / 2
                yaw_angle = self.pixel_to_angle(x_center, image_width)

                detection = {
                    'class': class_name,
                    'yaw': yaw_angle,
                    'confidence': confidence,
                    'x_center': x_center,
                    'y_center': (y1 + y2) / 2,
                    'timestamp': time.time()
                }

                detections.append(detection)

                # Overlay çizimleri GUI olsun/olmasın yapılabilir (göstermiyoruz)
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{class_name}: {confidence:.2f} ({yaw_angle:+.1f}°)"
                cv2.putText(image, label, (x1, y2 + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                cv2.circle(image, (int(x_center), int(image_height/2)), 5, (0, 0, 255), -1)

            except Exception as e:
                rospy.logwarn(f"Detection processing error: {e}")
                continue

        return detections, image

    def simple_tracking(self, current_detections):
        import time
        import math
        current_time = time.time()

        for detection in current_detections:
            best_track_id = None
            best_distance = float('inf')

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
                self.object_tracks[best_track_id]['positions'].append({
                    'x': detection['x_center'],
                    'y': detection['y_center'],
                    'time': current_time
                })
                self.object_tracks[best_track_id]['last_seen'] = current_time

                if len(self.object_tracks[best_track_id]['positions']) > 5:
                    self.object_tracks[best_track_id]['positions'] = self.object_tracks[best_track_id]['positions'][-5:]

                positions = self.object_tracks[best_track_id]['positions']
                if len(positions) >= 2:
                    dt = positions[-1]['time'] - positions[-2]['time']
                    if dt > 0:
                        dx = positions[-1]['x'] - positions[-2]['x']
                        dy = positions[-1]['y'] - positions[-2]['y']
                        pixel_velocity = math.sqrt(dx*dx + dy*dy) / dt
                        detection['velocity'] = pixel_velocity * 0.1
                        detection['is_moving'] = pixel_velocity > 5
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
                self.object_tracks[self.next_track_id] = {
                    'class': detection['class'],
                    'positions': [{'x': detection['x_center'], 'y': detection['y_center'], 'time': current_time}],
                    'last_seen': current_time
                }
                detection['track_id'] = self.next_track_id
                detection['velocity'] = 0.0
                detection['is_moving'] = False
                self.next_track_id += 1

        tracks_to_remove = []
        for track_id, track in self.object_tracks.items():
            if current_time - track['last_seen'] > 3.0:
                tracks_to_remove.append(track_id)
        for track_id in tracks_to_remove:
            del self.object_tracks[track_id]

        return current_detections

    def create_simple_message(self, detections):
        if not detections:
            return None
        detection_strings = []
        for det in detections:
            det_str = f"{det['class']}:{det['yaw']:.1f}"
            if 'track_id' in det and 'is_moving' in det:
                if det['is_moving']:
                    det_str += f":MOVING:ID{det['track_id']}:VEL{det['velocity']:.1f}"
                else:
                    det_str += f":STATIC:ID{det['track_id']}"
            detection_strings.append(det_str)
        return ",".join(detection_strings)

    def log_detection(self, detection):
        if not self.logging_enabled:
            return
        try:
            timestamp = datetime.now().isoformat()
            self.log_file.write(f"{timestamp},{detection['class']},{detection['yaw']:.2f},{detection['confidence']:.3f}\n")
            self.log_file.flush()
        except:
            pass

    def add_simple_overlay(self, image):
        height, width = image.shape[:2]
        center_x = width // 2
        cv2.line(image, (center_x, 0), (center_x, height), (0, 255, 255), 1)
        info_text = f"Frame: {self.frame_count} | Det: {self.detection_count} | Tracks: {len(self.object_tracks)}"
        cv2.putText(image, info_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    def run(self):
        rospy.loginfo("Detection başlatılıyor...")

        try:
            stream = self.model.predict(
                source=self.source,
                stream=True,
                imgsz=self.imgsz,
                conf=self.confidence_threshold,
                device=self.device,
                verbose=False,
                show=False  # Pencere açmayı zorla kapat
            )

            rospy.loginfo("Stream başladı")

            for results in stream:
                self.frame_count += 1
                im = results.orig_img.copy()

                detections, processed_im = self.process_detections_simple(results, im)

                if detections:
                    detections = self.simple_tracking(detections)

                if detections:
                    self.detection_count += len(detections)
                    message_str = self.create_simple_message(detections)
                    if message_str:
                        msg = String()
                        msg.data = message_str
                        self.pub.publish(msg)
                        rospy.loginfo(f"DETECTION: {message_str}")
                        for det in detections:
                            self.log_detection(det)

                self.add_simple_overlay(processed_im)

                # --- GUI sadece istenirse ---
                if self.show_gui:
                    cv2.imshow("Basit Kamera Detection", processed_im)
                    if cv2.waitKey(1) & 0xFF == 27:
                        break

                if rospy.is_shutdown():
                    break

                if self.frame_count % 100 == 0:
                    rospy.loginfo(f"Frame: {self.frame_count}, Detections: {self.detection_count}, Active tracks: {len(self.object_tracks)}")

        except Exception as e:
            rospy.logerr(f"Detection loop error: {e}")
            import traceback
            rospy.logerr(traceback.format_exc())

        finally:
            # GUI kapatma çağrısı güvenli (pencere açılmamışsa da sorun olmaz)
            try:
                cv2.destroyAllWindows()
            except:
                pass
            if self.logging_enabled:
                self.log_file.close()
            rospy.loginfo("Detector kapatıldı")

def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True, help="YOLO model (.pt)")
    parser.add_argument("--source", default="0", help="Camera source")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--conf", type=float, default=0.5, help="Confidence")
    # --- Yeni bayraklar ---
    parser.add_argument("--no-gui", dest="show_gui", action="store_false",
                        help="OpenCV pencerelerini kapat (headless)")
    parser.add_argument("--headless", dest="show_gui", action="store_false",
                        help="Headless mod (GUI yok)")
    parser.set_defaults(show_gui=True)

    args = parser.parse_args()

    try:
        detector = SimplifiedCameraDetector(args.weights, args.source, args.imgsz, args.conf, show_gui=args.show_gui)
        detector.run()
    except rospy.ROSInterruptException:
        rospy.loginfo("ROS shutdown")
    except KeyboardInterrupt:
        rospy.loginfo("Keyboard interrupt")
    except Exception as e:
        rospy.logerr(f"Error: {e}")

if __name__ == '__main__':
    main()
