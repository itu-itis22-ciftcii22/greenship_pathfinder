#!/usr/bin/env python

import rospy
from std_msgs.msg import String, Float32MultiArray
from sensor_msgs.msg import LaserScan
from message_filters import ApproximateTimeSynchronizer, Subscriber
import numpy as np
from collections import deque
import time
import math


class DebugFusionSystem:
    def __init__(self):
      
        rospy.init_node('debug_fusion_system', anonymous=True)
        
        rospy.loginfo("=== FUSION SYSTEM STARTING ===")
        
        # === ROS Communication ===
        # Outputs
        self.obstacles_pub = rospy.Publisher('/navigation_obstacles', Float32MultiArray, queue_size=10)
        self.fusion_output_pub = rospy.Publisher('/fusion_output', Float32MultiArray, queue_size=10)
        self.fusion_status_pub = rospy.Publisher('/fusion_status', String, queue_size=10)
        self.navigation_alerts_pub = rospy.Publisher('/navigation_alerts', String, queue_size=10)
        
        # === Parameters ===
        self.angle_tolerance = 35.0  
        self.min_distance = 0.3
        self.max_distance = 50.0
        
        # === Data Storage ===
        self.camera_count = 0
        self.lidar_count = 0
        self.fusion_count = 0
        self.last_camera_time = None
        self.last_lidar_time = None
        
        # Son veri cache'i
        self.last_camera_data = None
        self.last_lidar_data = None
        
        # === Individual Subscribers ===
        rospy.Subscriber('/camera_detections', String, self.camera_callback)
        rospy.Subscriber('/lidar_distances', Float32MultiArray, self.lidar_callback)
        
        rospy.loginfo("=== FUSION SYSTEM READY ===")
        rospy.loginfo("Waiting for data on:")
        rospy.loginfo("  - /camera_detections")
        rospy.loginfo("  - /lidar_distances")
        
        # Status timer
        rospy.Timer(rospy.Duration(5.0), self.status_callback)
        
        # Fusion timer - her 200ms'de bir fusion yap
        rospy.Timer(rospy.Duration(0.2), self.fusion_timer_callback)
        
        rospy.spin()
    
    def camera_callback(self, msg):
        """
        Camera data callback with detailed debug
        """
        self.camera_count += 1
        self.last_camera_time = time.time()
        self.last_camera_data = msg
        
        # HER mesajı logla
        rospy.loginfo(f"[CAMERA #{self.camera_count}] Raw data: '{msg.data}'")
        
        # Parse et ve göster
        detections = self.parse_camera_detections(msg)
        rospy.loginfo(f"[CAMERA] Parsed {len(detections)} detections")
        for det in detections:
            rospy.loginfo(f"  - {det['class']} @ {det['angle']:.1f}°")
    
    def lidar_callback(self, msg):
        
        self.lidar_count += 1
        self.last_lidar_time = time.time()
        self.last_lidar_data = msg
        
        # HER 5 mesajda bir detaylı log
        if self.lidar_count % 5 == 1:
            rospy.loginfo(f"[LIDAR #{self.lidar_count}] Data: {[f'{d:.1f}' for d in msg.data[:3]]}")
            if len(msg.data) >= 6:
                rospy.loginfo(f"[LIDAR] Distances: 0°={msg.data[0]:.1f}m, "
                            f"+30°={msg.data[1]:.1f}m, -30°={msg.data[2]:.1f}m")
                rospy.loginfo(f"[LIDAR] Qualities: 0°={msg.data[3]:.2f}, "
                            f"+30°={msg.data[4]:.2f}, -30°={msg.data[5]:.2f}")
    
    def status_callback(self, event):
        """
        Periodic status report
        """
        rospy.loginfo("=== FUSION STATUS ===")
        rospy.loginfo(f"Camera messages: {self.camera_count}")
        rospy.loginfo(f"LiDAR messages: {self.lidar_count}")
        rospy.loginfo(f"Fusion attempts: {self.fusion_count}")
        
        current_time = time.time()
        if self.last_camera_time:
            cam_age = current_time - self.last_camera_time
            rospy.loginfo(f"Last camera: {cam_age:.1f}s ago")
            if self.last_camera_data:
                rospy.loginfo(f"Last camera data: '{self.last_camera_data.data[:50]}...'")
        
        if self.last_lidar_time:
            lidar_age = current_time - self.last_lidar_time
            rospy.loginfo(f"Last LiDAR: {lidar_age:.1f}s ago")
    
    def fusion_timer_callback(self, event):
      
        # Veri kontrolü
        if self.last_camera_data is None:
            rospy.logdebug("No camera data yet")
            return
            
        if self.last_lidar_data is None:
            rospy.logdebug("No LiDAR data yet")
            return
            
        # Veri yaşı kontrolü
        current_time = time.time()
        if self.last_camera_time and (current_time - self.last_camera_time) > 2.0:
            rospy.logwarn("Camera data too old")
            return
            
        if self.last_lidar_time and (current_time - self.last_lidar_time) > 2.0:
            rospy.logwarn("LiDAR data too old")
            return
            
        # Fusion işlemini çağır
        self.perform_fusion()
    
    def perform_fusion(self):
        """
        Main fusion logic
        """
        self.fusion_count += 1
        
        rospy.loginfo(f"\n=== FUSION ATTEMPT #{self.fusion_count} ===")
        
        try:
            # Parse camera data
            camera_detections = self.parse_camera_detections(self.last_camera_data)
            rospy.loginfo(f"[FUSION] Camera detections: {len(camera_detections)}")
            
            if not camera_detections:
                rospy.loginfo("[FUSION] No camera detections to process")
                return
            
            # Parse LiDAR data
            lidar_data = self.parse_lidar_data(self.last_lidar_data)
            if not lidar_data:
                rospy.logwarn("[FUSION] Invalid LiDAR data")
                return
                
            rospy.loginfo("[FUSION] LiDAR data parsed successfully")
            
            # FUSION İŞLEMİ
            obstacles = []
            
            for i, cam_det in enumerate(camera_detections):
                rospy.loginfo(f"\n[FUSION] Processing detection {i+1}: {cam_det['class']} @ {cam_det['angle']:.1f}°")
                
                # En iyi LiDAR eşleşmesini bul
                best_match = None
                best_distance = None
                min_angle_diff = float('inf')
                
                # Her LiDAR sektörünü kontrol et
                for sector_angle, sector_data in lidar_data['sectors'].items():
                    angle_diff = abs(cam_det['angle'] - sector_angle)
                    rospy.loginfo(f"  - Checking LiDAR sector {sector_angle}°: "
                                f"diff={angle_diff:.1f}°, dist={sector_data['distance']:.1f}m")
                    
                    # Geçerli mesafe ve açı toleransı kontrolü
                    if (sector_data['distance'] > 0 and 
                        sector_data['distance'] < self.max_distance and
                        angle_diff < min_angle_diff):
                        
                        min_angle_diff = angle_diff
                        best_distance = sector_data['distance']
                        best_match = sector_angle
                
                # Eşleşme bulunduysa veya bulunamadıysa obstacle oluştur
                if best_match is not None and min_angle_diff <= self.angle_tolerance:
                    obstacle = {
                        'angle': cam_det['angle'],
                        'distance': best_distance,
                        'radius': 1.0,
                        'type': 'DYNAMIC' if 'boat' in cam_det['class'].lower() else 'STATIC',
                        'velocity': cam_det.get('velocity', 0.0),
                        'heading': 0.0
                    }
                    rospy.loginfo(f"  ✓ MATCHED with LiDAR {best_match}°: {best_distance:.1f}m")
                else:
                    # LiDAR eşleşmesi yok, sadece kamera ile devam et
                    default_distance = 15.0  # Default mesafe
                    obstacle = {
                        'angle': cam_det['angle'],
                        'distance': default_distance,
                        'radius': 1.0,
                        'type': 'UNKNOWN',
                        'velocity': 0.0,
                        'heading': 0.0
                    }
                    rospy.logwarn(f"  ✗ NO LIDAR MATCH - using default {default_distance}m")
                
                obstacles.append(obstacle)
                rospy.loginfo(f"  → Obstacle created: {obstacle['type']} @ "
                            f"{obstacle['angle']:.1f}° {obstacle['distance']:.1f}m")
            
            # PUBLISH
            if obstacles:
                rospy.loginfo(f"\n[FUSION] Publishing {len(obstacles)} obstacles")
                
                # Navigation array
                nav_array = self.create_navigation_array(obstacles)
                self.obstacles_pub.publish(nav_array)
                rospy.loginfo(f"[FUSION] Published to /navigation_obstacles")
                
                # Fusion output array
                fusion_array = self.create_fusion_output_array(obstacles)
                self.fusion_output_pub.publish(fusion_array)
                rospy.loginfo(f"[FUSION] Published to /fusion_output: {fusion_array.data}")
                
                # Status
                status = f"FUSION_OK|OBSTACLES:{len(obstacles)}|TIME:{time.time():.1f}"
                self.fusion_status_pub.publish(status)
            else:
                rospy.logwarn("[FUSION] No obstacles created!")
        
        except Exception as e:
            rospy.logerr(f"[FUSION] Error: {e}")
            import traceback
            rospy.logerr(traceback.format_exc())
    
    def parse_camera_detections(self, camera_msg):
        """
        Parse camera detections 
        """
        detections = []
        
        if not camera_msg or not camera_msg.data.strip():
            rospy.logwarn("[PARSE] Empty camera data")
            return detections
        
        rospy.loginfo(f"[PARSE] Parsing camera data: '{camera_msg.data}'")
        
        try:
            
            detection_parts = camera_msg.data.split(',')
            rospy.loginfo(f"[PARSE] Found {len(detection_parts)} detection parts")
            
            for i, detection_str in enumerate(detection_parts):
                detection_str = detection_str.strip()
                if not detection_str:
                    continue
                
                rospy.loginfo(f"[PARSE] Part {i}: '{detection_str}'")
                
                # İki nokta ile ayrılmış parçalar
                parts = detection_str.split(':')
                rospy.loginfo(f"[PARSE] Split into {len(parts)} parts: {parts}")
                
                if len(parts) >= 2:
                    class_name = parts[0].strip()
                    try:
                        angle = float(parts[1].strip())
                        
                        detection = {
                            'class': class_name,
                            'angle': angle,
                            'confidence': 0.8,
                            'is_moving': False,
                            'velocity': 0.0
                        }
                        
                        # Tracking bilgisi varsa
                        if len(parts) > 2:
                            for j, part in enumerate(parts[2:]):
                                if part == 'MOVING':
                                    detection['is_moving'] = True
                                elif part.startswith('VEL'):
                                    try:
                                        detection['velocity'] = float(part[3:])
                                    except:
                                        pass
                        
                        detections.append(detection)
                        rospy.loginfo(f"[PARSE] ✓ Detection: {class_name} @ {angle}°")
                        
                    except ValueError as e:
                        rospy.logerr(f"[PARSE] Failed to parse angle from '{parts[1]}': {e}")
                else:
                    rospy.logwarn(f"[PARSE] Invalid format: '{detection_str}'")
        
        except Exception as e:
            rospy.logerr(f"[PARSE] Camera parse error: {e}")
            import traceback
            rospy.logerr(traceback.format_exc())
        
        rospy.loginfo(f"[PARSE] Total detections parsed: {len(detections)}")
        return detections
    
    def parse_lidar_data(self, lidar_msg):
        """
        Parse LiDAR data with validation
        """
        if not lidar_msg or len(lidar_msg.data) < 6:
            rospy.logwarn(f"[PARSE] Invalid LiDAR data length: {len(lidar_msg.data) if lidar_msg else 0}")
            return None
        
        parsed = {
            'sectors': {
                0: {
                    'angle': 0,
                    'distance': lidar_msg.data[0],
                    'quality': lidar_msg.data[3]
                },
                30: {
                    'angle': 30,
                    'distance': lidar_msg.data[1],
                    'quality': lidar_msg.data[4]
                },
                -30: {
                    'angle': -30,
                    'distance': lidar_msg.data[2],
                    'quality': lidar_msg.data[5]
                }
            }
        }
        
        rospy.loginfo(f"[PARSE] LiDAR parsed: 0°={parsed['sectors'][0]['distance']:.1f}m, "
                     f"30°={parsed['sectors'][30]['distance']:.1f}m, "
                     f"-30°={parsed['sectors'][-30]['distance']:.1f}m")
        
        return parsed
    
    def create_navigation_array(self, obstacles):
        """
        Create navigation array
        """
        max_obstacles = 10
        array_data = [float(len(obstacles))]
        
        for i, obs in enumerate(obstacles[:max_obstacles]):
            type_code = {'UNKNOWN': 0.0, 'STATIC': 1.0, 'DYNAMIC': 2.0}.get(obs['type'], 0.0)
            
            array_data.extend([
                float(obs['angle']),
                float(obs['distance']),
                float(obs.get('radius', 1.0)),
                type_code,
                float(obs.get('velocity', 0.0)),
                float(obs.get('heading', 0.0))
            ])
        
        # Padding
        while len(array_data) < 1 + max_obstacles * 6:
            array_data.extend([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        
        msg = Float32MultiArray()
        msg.data = array_data
        return msg
    
    def create_fusion_output_array(self, obstacles):
        """
        Create fusion output for path planning
        Format: [yaw1, dist1, yaw2, dist2, ...]
        """
        array_data = []
        
        for obs in obstacles:
            array_data.append(float(obs['angle']))     # yaw (derece)
            array_data.append(float(obs['distance']))  # distance (metre)
        
        msg = Float32MultiArray()
        msg.data = array_data
        return msg


if __name__ == '__main__':
    try:
        DebugFusionSystem()
    except rospy.ROSInterruptException:
        rospy.loginfo("Debug Fusion shutdown")
    except Exception as e:
        rospy.logerr(f"Critical error: {e}")
        import traceback
        rospy.logerr(traceback.format_exc())
