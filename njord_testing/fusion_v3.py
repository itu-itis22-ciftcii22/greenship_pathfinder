#!/usr/bin/env python

import rospy
from std_msgs.msg import String, Float32MultiArray
from sensor_msgs.msg import LaserScan
from message_filters import ApproximateTimeSynchronizer, Subscriber
import numpy as np
from collections import deque
import time
import math


class EnhancedFusionSystem:
    def __init__(self):
      
        rospy.init_node('enhanced_fusion_system', anonymous=True)
        
        rospy.loginfo("=== ENHANCED FUSION SYSTEM STARTING ===")
        
        # === ROS Communication ===
        # Outputs
        self.obstacles_pub = rospy.Publisher('/navigation_obstacles', Float32MultiArray, queue_size=10)
        self.fusion_output_pub = rospy.Publisher('/fusion_output', Float32MultiArray, queue_size=10)
        self.fusion_status_pub = rospy.Publisher('/fusion_status', String, queue_size=10)
        self.navigation_alerts_pub = rospy.Publisher('/navigation_alerts', String, queue_size=10)
        
        # === Enhanced Parameters ===
        self.angle_tolerance = 35.0  
        self.min_distance = 0.3
        self.max_distance = 50.0
        self.fusion_timeout = 2.0  # Data freshness timeout
        self.confidence_threshold = 0.6  # Minimum confidence for camera detections
        
        # === Data Storage with Quality Tracking ===
        self.camera_count = 0
        self.lidar_count = 0
        self.fusion_count = 0
        self.successful_fusions = 0
        self.last_camera_time = None
        self.last_lidar_time = None
        
        # Data cache with quality metrics
        self.last_camera_data = None
        self.last_lidar_data = None
        self.camera_quality = 0.0
        self.lidar_quality = 0.0
        
        # Historical data for trend analysis
        self.camera_history = deque(maxlen=10)
        self.lidar_history = deque(maxlen=10)
        
        # === Individual Subscribers ===
        rospy.Subscriber('/camera_detections', String, self.camera_callback)
        rospy.Subscriber('/lidar_distances', Float32MultiArray, self.lidar_callback)
        
        rospy.loginfo("=== ENHANCED FUSION SYSTEM READY ===")
        rospy.loginfo("Waiting for data on:")
        rospy.loginfo("  - /camera_detections")
        rospy.loginfo("  - /lidar_distances")
        
        # Status timer
        rospy.Timer(rospy.Duration(5.0), self.status_callback)
        
        # Fusion timer - optimized frequency
        rospy.Timer(rospy.Duration(0.1), self.fusion_timer_callback)  # 10Hz
        
        rospy.spin()
    
    def camera_callback(self, msg):
        """
        Enhanced camera data callback
        """
        self.camera_count += 1
        self.last_camera_time = time.time()
        self.last_camera_data = msg
        
        # Store in history
        self.camera_history.append({
            'timestamp': self.last_camera_time,
            'data': msg.data,
            'detection_count': len(msg.data.split(',')) if msg.data else 0
        })
        
        # Calculate camera data quality
        self.camera_quality = self.assess_camera_quality(msg)
        
        # Log every camera message with quality
        rospy.loginfo(f"[CAMERA #{self.camera_count}] Quality: {self.camera_quality:.2f} | Data: '{msg.data[:50]}...'")
        
        # Parse and show detections
        detections = self.parse_camera_detections(msg)
        if detections:
            rospy.loginfo(f"[CAMERA] Parsed {len(detections)} detections:")
            for i, det in enumerate(detections):
                status_info = f"conf:{det['confidence']:.2f}"
                if 'track_id' in det:
                    status_info += f" track:{det['track_id']}"
                if det.get('is_moving', False):
                    status_info += f" vel:{det.get('velocity', 0):.1f}"
                
                rospy.loginfo(f"  {i+1}. {det['class']} @ {det['angle']:+.1f}° ({status_info})")
    
    def lidar_callback(self, msg):
        """
        Enhanced LiDAR data callback
        """
        self.lidar_count += 1
        self.last_lidar_time = time.time()
        self.last_lidar_data = msg
        
        # Store in history
        self.lidar_history.append({
            'timestamp': self.last_lidar_time,
            'data': list(msg.data),
            'valid_measurements': sum(1 for d in msg.data[:3] if d > 0)
        })
        
        # Calculate LiDAR data quality
        self.lidar_quality = self.assess_lidar_quality(msg)
        
        # Log every 5th message with enhanced info
        if self.lidar_count % 5 == 1:
            rospy.loginfo(f"[LIDAR #{self.lidar_count}] Quality: {self.lidar_quality:.2f}")
            if len(msg.data) >= 6:
                rospy.loginfo(f"[LIDAR] Distances: 0°={msg.data[0]:.1f}m, "
                            f"+30°={msg.data[1]:.1f}m, -30°={msg.data[2]:.1f}m")
                rospy.loginfo(f"[LIDAR] Qualities: 0°={msg.data[3]:.2f}, "
                            f"+30°={msg.data[4]:.2f}, -30°={msg.data[5]:.2f}")
    
    def assess_camera_quality(self, msg):
        """Assess camera data quality"""
        if not msg or not msg.data:
            return 0.0
        
        try:
            detections = self.parse_camera_detections(msg)
            if not detections:
                return 0.1
            
            # Quality factors
            detection_count = len(detections)
            avg_confidence = sum(det['confidence'] for det in detections) / detection_count
            tracking_ratio = sum(1 for det in detections if 'track_id' in det) / detection_count
            
            # Combined quality score
            quality = (avg_confidence * 0.6 + 
                      min(detection_count / 5.0, 1.0) * 0.3 + 
                      tracking_ratio * 0.1)
            
            return min(quality, 1.0)
            
        except:
            return 0.2
    
    def assess_lidar_quality(self, msg):
        """Assess LiDAR data quality"""
        if not msg or len(msg.data) < 6:
            return 0.0
        
        try:
            # Distance validity
            valid_distances = sum(1 for d in msg.data[:3] if 0.3 < d < 50.0)
            distance_quality = valid_distances / 3.0
            
            # Quality scores
            avg_quality = sum(msg.data[3:6]) / 3.0
            
            # Combined quality
            quality = distance_quality * 0.7 + avg_quality * 0.3
            
            return min(quality, 1.0)
            
        except:
            return 0.0
    
    def status_callback(self, event):
        """
        Enhanced status report
        """
        rospy.loginfo("=== ENHANCED FUSION STATUS ===")
        rospy.loginfo(f"Camera messages: {self.camera_count} (Quality: {self.camera_quality:.2f})")
        rospy.loginfo(f"LiDAR messages: {self.lidar_count} (Quality: {self.lidar_quality:.2f})")
        rospy.loginfo(f"Fusion attempts: {self.fusion_count} (Success: {self.successful_fusions})")
        
        current_time = time.time()
        
        # Data freshness
        if self.last_camera_time:
            cam_age = current_time - self.last_camera_time
            status = "FRESH" if cam_age < 1.0 else "STALE" if cam_age < 3.0 else "OLD"
            rospy.loginfo(f"Last camera: {cam_age:.1f}s ago [{status}]")
        
        if self.last_lidar_time:
            lidar_age = current_time - self.last_lidar_time
            status = "FRESH" if lidar_age < 1.0 else "STALE" if lidar_age < 3.0 else "OLD"
            rospy.loginfo(f"Last LiDAR: {lidar_age:.1f}s ago [{status}]")
        
        # Success rate
        if self.fusion_count > 0:
            success_rate = (self.successful_fusions / self.fusion_count) * 100
            rospy.loginfo(f"Fusion success rate: {success_rate:.1f}%")
    
    def fusion_timer_callback(self, event):
        """
        Enhanced fusion timer with quality checks
        """
        # Data availability check
        if self.last_camera_data is None:
            rospy.logdebug("No camera data yet")
            return
            
        if self.last_lidar_data is None:
            rospy.logdebug("No LiDAR data yet")
            return
            
        # Data freshness check
        current_time = time.time()
        camera_age = current_time - self.last_camera_time if self.last_camera_time else float('inf')
        lidar_age = current_time - self.last_lidar_time if self.last_lidar_time else float('inf')
        
        if camera_age > self.fusion_timeout:
            rospy.logwarn_throttle(5.0, f"Camera data too old: {camera_age:.1f}s")
            return
            
        if lidar_age > self.fusion_timeout:
            rospy.logwarn_throttle(5.0, f"LiDAR data too old: {lidar_age:.1f}s")
            return
        
        # Quality check
        if self.camera_quality < 0.3:
            rospy.logwarn_throttle(5.0, f"Camera quality too low: {self.camera_quality:.2f}")
            return
            
        if self.lidar_quality < 0.3:
            rospy.logwarn_throttle(5.0, f"LiDAR quality too low: {self.lidar_quality:.2f}")
            return
            
        # Perform fusion
        self.perform_enhanced_fusion()
    
    def perform_enhanced_fusion(self):
        """
        Enhanced fusion logic with quality assessment
        """
        self.fusion_count += 1
        
        rospy.loginfo(f"\n=== ENHANCED FUSION #{self.fusion_count} ===")
        
        try:
            # Parse camera data with quality filtering
            camera_detections = self.parse_camera_detections(self.last_camera_data)
            
            # Filter detections by confidence
            quality_detections = [
                det for det in camera_detections 
                if det['confidence'] >= self.confidence_threshold
            ]
            
            rospy.loginfo(f"[FUSION] Camera detections: {len(camera_detections)} total, {len(quality_detections)} high-quality")
            
            if not quality_detections:
                rospy.loginfo("[FUSION] No high-quality camera detections")
                # Still publish empty result for system awareness
                self.publish_empty_result()
                return
            
            # Parse LiDAR data
            lidar_data = self.parse_lidar_data(self.last_lidar_data)
            if not lidar_data:
                rospy.logwarn("[FUSION] Invalid LiDAR data")
                return
                
            rospy.loginfo("[FUSION] LiDAR data parsed successfully")
            
            # ENHANCED FUSION PROCESS
            obstacles = []
            fusion_successes = 0
            
            for i, cam_det in enumerate(quality_detections):
                rospy.loginfo(f"\n[FUSION] Processing detection {i+1}: {cam_det['class']} @ {cam_det['angle']:.1f}° (conf: {cam_det['confidence']:.2f})")
                
                # Find best LiDAR match
                best_match = None
                best_distance = None
                min_angle_diff = float('inf')
                match_quality = 0.0
                
                # Check each LiDAR sector
                for sector_angle, sector_data in lidar_data['sectors'].items():
                    angle_diff = abs(cam_det['angle'] - sector_angle)
                    
                    # Enhanced matching criteria
                    if (sector_data['distance'] > 0 and 
                        sector_data['distance'] < self.max_distance and
                        sector_data['quality'] > 0.3 and  # Minimum LiDAR quality
                        angle_diff < self.angle_tolerance):
                        
                        # Calculate match quality
                        angle_factor = 1.0 - (angle_diff / self.angle_tolerance)
                        distance_factor = 1.0 - (sector_data['distance'] / self.max_distance)
                        quality_factor = sector_data['quality']
                        
                        current_match_quality = (angle_factor * 0.4 + 
                                               distance_factor * 0.3 + 
                                               quality_factor * 0.3)
                        
                        if current_match_quality > match_quality:
                            match_quality = current_match_quality
                            min_angle_diff = angle_diff
                            best_distance = sector_data['distance']
                            best_match = sector_angle
                    
                    rospy.loginfo(f"  - LiDAR sector {sector_angle}°: "
                                f"diff={angle_diff:.1f}°, dist={sector_data['distance']:.1f}m, "
                                f"qual={sector_data['quality']:.2f}")
                
                # Create obstacle based on match quality
                if best_match is not None and match_quality > 0.5:
                    # High-quality match
                    obstacle_type = 'DYNAMIC' if cam_det.get('is_moving', False) else 'STATIC'
                    
                    obstacle = {
                        'angle': cam_det['angle'],
                        'distance': best_distance,
                        'radius': max(1.0, 3.0 - match_quality * 2.0),  # Smaller radius for better matches
                        'type': obstacle_type,
                        'velocity': cam_det.get('velocity', 0.0),
                        'heading': 0.0,
                        'confidence': cam_det['confidence'],
                        'match_quality': match_quality
                    }
                    
                    fusion_successes += 1
                    rospy.loginfo(f"  ✓ HIGH-QUALITY MATCH with LiDAR {best_match}°: "
                                f"{best_distance:.1f}m (quality: {match_quality:.2f})")
                    
                elif best_match is not None:
                    # Lower quality match
                    obstacle = {
                        'angle': cam_det['angle'],
                        'distance': best_distance,
                        'radius': 2.0,
                        'type': 'UNCERTAIN',
                        'velocity': cam_det.get('velocity', 0.0),
                        'heading': 0.0,
                        'confidence': cam_det['confidence'],
                        'match_quality': match_quality
                    }
                    
                    rospy.loginfo(f"  ≈ LOW-QUALITY MATCH with LiDAR {best_match}°: "
                                f"{best_distance:.1f}m (quality: {match_quality:.2f})")
                else:
                    # No LiDAR match - camera only
                    default_distance = min(15.0, 10.0 + cam_det['confidence'] * 10.0)  # Confidence-based distance
                    
                    obstacle = {
                        'angle': cam_det['angle'],
                        'distance': default_distance,
                        'radius': 3.0,  # Larger uncertainty
                        'type': 'VISUAL_ONLY',
                        'velocity': cam_det.get('velocity', 0.0),
                        'heading': 0.0,
                        'confidence': cam_det['confidence'],
                        'match_quality': 0.0
                    }
                    
                    rospy.logwarn(f"  ✗ NO LIDAR MATCH - using camera-only {default_distance:.1f}m")
                
                obstacles.append(obstacle)
                rospy.loginfo(f"  → Obstacle: {obstacle['type']} @ "
                            f"{obstacle['angle']:.1f}° {obstacle['distance']:.1f}m")
            
            # PUBLISH RESULTS
            if obstacles:
                self.successful_fusions += 1
                
                rospy.loginfo(f"\n[FUSION] Publishing {len(obstacles)} obstacles "
                            f"({fusion_successes} with LiDAR match)")
                
                # Navigation array
                nav_array = self.create_navigation_array(obstacles)
                self.obstacles_pub.publish(nav_array)
                
                # Fusion output array
                fusion_array = self.create_fusion_output_array(obstacles)
                self.fusion_output_pub.publish(fusion_array)
                rospy.loginfo(f"[FUSION] Published fusion output: {len(fusion_array.data)} elements")
                
                # Enhanced status
                status = (f"FUSION_OK|OBSTACLES:{len(obstacles)}|"
                         f"MATCHED:{fusion_successes}|"
                         f"CAM_QUAL:{self.camera_quality:.2f}|"
                         f"LIDAR_QUAL:{self.lidar_quality:.2f}|"
                         f"TIME:{time.time():.1f}")
                
                self.fusion_status_pub.publish(status)
                
                # Navigation alerts for critical obstacles
                critical_obstacles = [obs for obs in obstacles if obs['distance'] < 5.0]
                if critical_obstacles:
                    alert = f"CRITICAL_PROXIMITY|COUNT:{len(critical_obstacles)}|MIN_DIST:{min(obs['distance'] for obs in critical_obstacles):.1f}"
                    self.navigation_alerts_pub.publish(alert)
                    
            else:
                rospy.logwarn("[FUSION] No obstacles created!")
                self.publish_empty_result()
        
        except Exception as e:
            rospy.logerr(f"[FUSION] Error: {e}")
            import traceback
            rospy.logerr(traceback.format_exc())
            self.publish_empty_result()
    
    def publish_empty_result(self):
        """Publish empty results for system awareness"""
        empty_nav = Float32MultiArray()
        empty_nav.data = [0.0]  # Zero obstacles
        self.obstacles_pub.publish(empty_nav)
        
        empty_fusion = Float32MultiArray()
        empty_fusion.data = []
        self.fusion_output_pub.publish(empty_fusion)
        
        status = f"FUSION_EMPTY|TIME:{time.time():.1f}"
        self.fusion_status_pub.publish(status)
    
    def parse_camera_detections(self, camera_msg):
        """
        Enhanced camera detection parsing
        """
        detections = []
        
        if not camera_msg or not camera_msg.data.strip():
            return detections
        
        try:
            detection_parts = camera_msg.data.split(',')
            
            for detection_str in detection_parts:
                detection_str = detection_str.strip()
                if not detection_str:
                    continue
                
                parts = detection_str.split(':')
                
                if len(parts) >= 2:
                    class_name = parts[0].strip()
                    try:
                        angle = float(parts[1].strip())
                        
                        detection = {
                            'class': class_name,
                            'angle': angle,
                            'confidence': 0.8,  # Default confidence
                            'is_moving': False,
                            'velocity': 0.0
                        }
                        
                        # Enhanced parsing for additional data
                        for part in parts[2:]:
                            part = part.strip()
                            if part == 'MOVING':
                                detection['is_moving'] = True
                            elif part == 'STATIC':
                                detection['is_moving'] = False
                            elif part.startswith('VEL'):
                                try:
                                    detection['velocity'] = float(part[3:])
                                except:
                                    pass
                            elif part.startswith('ID'):
                                try:
                                    detection['track_id'] = int(part[2:])
                                except:
                                    pass
                            elif part.startswith('CONF'):
                                try:
                                    detection['confidence'] = float(part[4:])
                                except:
                                    pass
                        
                        detections.append(detection)
                        
                    except ValueError as e:
                        rospy.logwarn(f"[PARSE] Failed to parse angle from '{parts[1]}': {e}")
                else:
                    rospy.logwarn(f"[PARSE] Invalid format: '{detection_str}'")
        
        except Exception as e:
            rospy.logerr(f"[PARSE] Camera parse error: {e}")
        
        return detections
    
    def parse_lidar_data(self, lidar_msg):
        """
        Enhanced LiDAR data parsing with validation
        """
        if not lidar_msg or len(lidar_msg.data) < 6:
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
        
        return parsed
    
    def create_navigation_array(self, obstacles):
        """
        Create enhanced navigation array
        """
        max_obstacles = 10
        array_data = [float(len(obstacles))]
        
        # Sort obstacles by distance (closest first)
        sorted_obstacles = sorted(obstacles, key=lambda x: x['distance'])
        
        for i, obs in enumerate(sorted_obstacles[:max_obstacles]):
            type_code = {
                'UNKNOWN': 0.0, 
                'STATIC': 1.0, 
                'DYNAMIC': 2.0,
                'UNCERTAIN': 3.0,
                'VISUAL_ONLY': 4.0
            }.get(obs['type'], 0.0)
            
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
        Create enhanced fusion output for path planning
        """
        array_data = []
        
        # Sort by criticality (distance and type)
        def criticality_score(obs):
            distance_factor = 1.0 / max(obs['distance'], 0.1)
            type_factor = {'DYNAMIC': 2.0, 'STATIC': 1.5, 'UNCERTAIN': 1.0, 'VISUAL_ONLY': 0.8}.get(obs['type'], 1.0)
            return distance_factor * type_factor
        
        sorted_obstacles = sorted(obstacles, key=criticality_score, reverse=True)
        
        for obs in sorted_obstacles:
            array_data.append(float(obs['angle']))     # yaw (degrees)
            array_data.append(float(obs['distance']))  # distance (meters)
        
        msg = Float32MultiArray()
        msg.data = array_data
        return msg


if __name__ == '__main__':
    try:
        EnhancedFusionSystem()
    except rospy.ROSInterruptException:
        rospy.loginfo("Enhanced Fusion shutdown")
    except Exception as e:
        rospy.logerr(f"Critical error: {e}")
        import traceback
        rospy.logerr(traceback.format_exc())