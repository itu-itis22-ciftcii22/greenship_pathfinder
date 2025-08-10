#!/usr/bin/env python
"""
ENHANCED PATHPLANNING LIDAR PUBLISHER
====================================

Output Format: Float32MultiArray
[front_0°, right_30°, left_-30°, quality_front, quality_right, quality_left]

Enhanced Features:
- Adaptive filtering for better data quality
- Performance monitoring and optimization
- Enhanced error handling and recovery
- Dynamic quality assessment
"""

import rospy
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32MultiArray
import numpy as np
import math
import time
from collections import deque


class EnhancedPathPlanningLidarPublisher:
    def __init__(self):
       
        rospy.init_node('enhanced_pathplanning_lidar_publisher', anonymous=True)

        # === ROS Communication ===
        self.pub = rospy.Publisher('/lidar_distances', Float32MultiArray, queue_size=10)
        rospy.Subscriber('/scan', LaserScan, self.scan_callback)

        # === Enhanced LiDAR Parameters ===
        self.expected_points = 1947
        self.min_valid_distance = 0.3  # 30cm minimum
        self.max_distance_reliable = 8.0  # Black objects (reliable)
        self.max_distance_extended = 25.0  # White objects (extended)

        # Adaptive sector definitions
        self.target_sectors = [
            {'name': 'FRONT', 'angle': 0, 'tolerance': 8.0},
            {'name': 'RIGHT', 'angle': 30, 'tolerance': 8.0},
            {'name': 'LEFT', 'angle': -30, 'tolerance': 8.0}
        ]

        # === Performance Monitoring ===
        self.scan_count = 0
        self.error_count = 0
        self.last_stats_time = rospy.Time.now()
        self.processing_times = deque(maxlen=100)
        self.data_quality_history = deque(maxlen=50)
        
        # === Adaptive Filtering ===
        self.distance_filters = {
            'FRONT': deque(maxlen=5),
            'RIGHT': deque(maxlen=5),
            'LEFT': deque(maxlen=5)
        }
        
        # === Quality Metrics ===
        self.sector_reliability = {'FRONT': 1.0, 'RIGHT': 1.0, 'LEFT': 1.0}
        self.consecutive_errors = 0
        self.max_consecutive_errors = 10

        rospy.loginfo("=== Enhanced Path Planning LiDAR Publisher ===")
        rospy.loginfo(f"Output format: [dist_0°, dist_+30°, dist_-30°, qual_0°, qual_+30°, qual_-30°]")
        rospy.loginfo(f"Reliable range: {self.max_distance_reliable}m")
        rospy.loginfo(f"Extended range: {self.max_distance_extended}m")
        rospy.loginfo("Enhanced features: Adaptive filtering, Quality assessment, Performance monitoring")
        
        rospy.spin()

    def angle_to_index(self, target_angle, total_points):
        """
        Enhanced angle to index conversion with bounds checking
        """
        normalized_angle = (target_angle + 360) % 360
        index = int(round(normalized_angle * total_points / 360.0))
        return max(0, min(index, total_points - 1))

    def extract_sector_distance_enhanced(self, ranges, target_angle, sector_name, total_points):
        """
        Enhanced sector distance extraction with adaptive filtering
        """
        center_index = self.angle_to_index(target_angle, total_points)
        
        # Dynamic tolerance based on sector reliability
        base_tolerance = 8.0
        tolerance_points = int(base_tolerance * total_points / 360.0 * self.sector_reliability[sector_name])
        tolerance_points = max(5, tolerance_points)  # Minimum tolerance
        
        start_idx = max(0, center_index - tolerance_points)
        end_idx = min(total_points, center_index + tolerance_points + 1)

        # Collect all valid measurements
        valid_measurements = []
        reliable_measurements = []
        
        for i in range(start_idx, end_idx):
            distance = ranges[i]

            if (self.min_valid_distance <= distance <= self.max_distance_extended and
                    not math.isinf(distance) and not math.isnan(distance)):

                # Calculate angular weight (closer to center = higher weight)
                angular_distance = abs(i - center_index)
                weight = 1.0 / (1.0 + angular_distance * 0.1)
                
                valid_measurements.append({
                    'distance': distance,
                    'weight': weight,
                    'index': i
                })

                if distance <= self.max_distance_reliable:
                    reliable_measurements.append({
                        'distance': distance,
                        'weight': weight,
                        'index': i
                    })

        # Enhanced distance selection strategy
        if not valid_measurements:
            return -1.0, 0.0, {}

        # Prefer reliable measurements
        target_measurements = reliable_measurements if reliable_measurements else valid_measurements
        
        # Weighted average for smoother results, but biased towards minimum for safety
        if len(target_measurements) == 1:
            selected_distance = target_measurements[0]['distance']
        else:
            # Combine minimum distance (safety) with weighted average (accuracy)
            min_distance = min(m['distance'] for m in target_measurements)
            
            weighted_sum = sum(m['distance'] * m['weight'] for m in target_measurements)
            weight_sum = sum(m['weight'] for m in target_measurements)
            avg_distance = weighted_sum / weight_sum if weight_sum > 0 else min_distance
            
            # Blend minimum and average (safety-first approach)
            selected_distance = min_distance * 0.7 + avg_distance * 0.3

        # Apply temporal filtering
        self.distance_filters[sector_name].append(selected_distance)
        if len(self.distance_filters[sector_name]) > 1:
            # Simple moving average filter
            filtered_distance = sum(self.distance_filters[sector_name]) / len(self.distance_filters[sector_name])
            
            # Don't filter too aggressively for rapid changes
            if abs(filtered_distance - selected_distance) > 2.0:
                selected_distance = selected_distance * 0.8 + filtered_distance * 0.2
            else:
                selected_distance = filtered_distance

        # Enhanced quality calculation
        data_density = len(valid_measurements) / (2 * tolerance_points)
        reliable_ratio = len(reliable_measurements) / max(len(valid_measurements), 1)
        
        # Measurement consistency
        if len(target_measurements) > 1:
            distances = [m['distance'] for m in target_measurements]
            std_dev = np.std(distances)
            consistency = 1.0 / (1.0 + std_dev)
        else:
            consistency = 0.5
        
        # Angular accuracy (how close to target angle)
        if target_measurements:
            best_measurement = min(target_measurements, key=lambda x: abs(x['index'] - center_index))
            angular_accuracy = 1.0 - abs(best_measurement['index'] - center_index) / tolerance_points
        else:
            angular_accuracy = 0.0

        # Combined quality score
        base_quality = 0.8 if reliable_measurements else 0.5
        quality_score = base_quality * (
            0.3 * data_density + 
            0.3 * reliable_ratio + 
            0.2 * consistency + 
            0.2 * angular_accuracy
        )
        
        # Update sector reliability
        self.sector_reliability[sector_name] = (
            self.sector_reliability[sector_name] * 0.9 + 
            quality_score * 0.1
        )
        
        quality_score = min(quality_score, 1.0)
        
        # Additional metrics for debugging
        metrics = {
            'valid_count': len(valid_measurements),
            'reliable_count': len(reliable_measurements),
            'data_density': data_density,
            'consistency': consistency,
            'angular_accuracy': angular_accuracy,
            'sector_reliability': self.sector_reliability[sector_name]
        }

        return selected_distance, quality_score, metrics

    def validate_scan_data_enhanced(self, msg):
        """
        Enhanced scan data validation with adaptive thresholds
        """
        if not msg.ranges:
            return False, ["Empty ranges array"], 0.0

        total_points = len(msg.ranges)

        # Adaptive point count validation
        expected_range = (self.expected_points - 300, self.expected_points + 300)
        if not (expected_range[0] <= total_points <= expected_range[1]):
            return False, [f"Point count outside adaptive range: {total_points}"], 0.0

        # Enhanced data quality analysis
        valid_count = 0
        inf_count = 0
        zero_count = 0
        max_count = 0
        
        for distance in msg.ranges:
            if math.isinf(distance):
                inf_count += 1
            elif distance == 0.0:
                zero_count += 1
            elif distance >= msg.range_max:
                max_count += 1
            elif (self.min_valid_distance <= distance <= self.max_distance_extended and
                  not math.isnan(distance)):
                valid_count += 1

        valid_ratio = valid_count / total_points
        inf_ratio = inf_count / total_points
        zero_ratio = zero_count / total_points
        max_ratio = max_count / total_points

        # Adaptive quality threshold based on recent performance
        recent_quality = np.mean(self.data_quality_history) if self.data_quality_history else 0.5
        adaptive_threshold = max(0.05, recent_quality * 0.5)

        # Quality assessment
        overall_quality = valid_ratio
        
        # Store quality for adaptive thresholds
        self.data_quality_history.append(overall_quality)

        # Validation with context
        error_messages = []
        
        if valid_ratio < adaptive_threshold:
            return False, [f"Valid data below adaptive threshold: {valid_ratio:.1%} < {adaptive_threshold:.1%}"], overall_quality

        # Warnings for suboptimal conditions
        if inf_ratio > 0.8:
            error_messages.append(f"High infinity ratio: {inf_ratio:.1%}")
        if zero_ratio > 0.5:
            error_messages.append(f"High zero ratio: {zero_ratio:.1%}")

        return True, error_messages, overall_quality

    def create_output_array_enhanced(self, distances, qualities, metrics):
        """
        Create enhanced output array with metadata
        """
        output_data = distances + qualities
        
        array_msg = Float32MultiArray()
        array_msg.data = output_data
        
        return array_msg

    def scan_callback(self, msg):
        """
        Enhanced scan callback with performance monitoring
        """
        start_time = time.time()
        
        try:
            self.scan_count += 1
            current_time = rospy.Time.now()

            # === ENHANCED DATA VALIDATION ===
            is_valid, validation_messages, data_quality = self.validate_scan_data_enhanced(msg)

            if not is_valid:
                self.error_count += 1
                self.consecutive_errors += 1
                
                rospy.logwarn(f"Invalid LiDAR data (#{self.consecutive_errors}): {validation_messages}")
                
                # Publish safe default values
                error_distances = [-1.0, -1.0, -1.0]
                error_qualities = [0.0, 0.0, 0.0]
                error_array = self.create_output_array_enhanced(error_distances, error_qualities, {})
                self.pub.publish(error_array)
                
                # Reset system if too many consecutive errors
                if self.consecutive_errors >= self.max_consecutive_errors:
                    rospy.logerr("Too many consecutive LiDAR errors! Resetting filters...")
                    self.reset_filters()
                    self.consecutive_errors = 0
                
                return
            else:
                self.consecutive_errors = 0  # Reset error counter on success

            # Log warnings if any
            if validation_messages:
                rospy.logdebug(f"LiDAR warnings: {validation_messages}")

            # === ENHANCED SECTOR PROCESSING ===
            total_points = len(msg.ranges)
            sector_distances = []
            sector_qualities = []
            all_metrics = {}

            for sector in self.target_sectors:
                distance, quality, metrics = self.extract_sector_distance_enhanced(
                    msg.ranges, sector['angle'], sector['name'], total_points
                )

                sector_distances.append(distance)
                sector_qualities.append(quality)
                all_metrics[sector['name']] = metrics

            # === PUBLISH ENHANCED OUTPUT ===
            output_array = self.create_output_array_enhanced(sector_distances, sector_qualities, all_metrics)
            self.pub.publish(output_array)

            # === PERFORMANCE MONITORING ===
            processing_time = time.time() - start_time
            self.processing_times.append(processing_time)
            
            # === ENHANCED LOGGING ===
            # Detailed log every 25 scans (optimized frequency)
            if self.scan_count % 25 == 0:
                avg_processing_time = np.mean(self.processing_times) if self.processing_times else 0
                
                rospy.loginfo(f"=== Enhanced LiDAR Scan #{self.scan_count} ===")
                rospy.loginfo(f"Points: {total_points} | Quality: {data_quality:.1%} | Process: {processing_time*1000:.1f}ms (avg: {avg_processing_time*1000:.1f}ms)")

                for i, sector in enumerate(self.target_sectors):
                    if sector_distances[i] > 0:
                        metrics = all_metrics[sector['name']]
                        status = "EXCELLENT" if sector_qualities[i] > 0.8 else "GOOD" if sector_qualities[i] > 0.6 else "POOR"
                        
                        rospy.loginfo(f"{sector['name']:>5} {sector['angle']:+3.0f}°: "
                                      f"{sector_distances[i]:5.1f}m (Q:{sector_qualities[i]:.2f}) [{status}]")
                        rospy.logdebug(f"  Metrics: valid={metrics['valid_count']}, reliable={metrics['reliable_count']}, "
                                      f"density={metrics['data_density']:.2f}, consistency={metrics['consistency']:.2f}")
                    else:
                        rospy.loginfo(f"{sector['name']:>5} {sector['angle']:+3.0f}°: NO_DATA")

                # Performance stats
                time_elapsed = (current_time - self.last_stats_time).to_sec()
                if time_elapsed > 0:
                    scan_rate = 25 / time_elapsed
                    rospy.loginfo(f"Scan rate: {scan_rate:.1f} Hz | Error rate: {self.error_count/self.scan_count*100:.1f}%")
                    
                    # Adaptive performance warnings
                    if scan_rate < 8.0:
                        rospy.logwarn(f"Low scan rate detected: {scan_rate:.1f} Hz")
                    if avg_processing_time > 0.05:
                        rospy.logwarn(f"High processing time: {avg_processing_time*1000:.1f}ms")

                self.last_stats_time = current_time

            # Critical proximity alert (every scan)
            min_distance = min([d for d in sector_distances if d > 0], default=999)
            if min_distance < 2.0:
                rospy.logwarn_throttle(1.0, f"CRITICAL PROXIMITY: {min_distance:.1f}m!")
            elif min_distance < 5.0:
                rospy.loginfo_throttle(5.0, f"Close proximity: {min_distance:.1f}m")

        except Exception as e:
            self.error_count += 1
            rospy.logerr(f"Enhanced LiDAR callback error: {str(e)}")
            
            # Publish safe defaults on critical error
            error_distances = [-1.0, -1.0, -1.0]
            error_qualities = [0.0, 0.0, 0.0]
            error_array = self.create_output_array_enhanced(error_distances, error_qualities, {})
            self.pub.publish(error_array)

    def reset_filters(self):
        """Reset all adaptive filters and reliability scores"""
        for key in self.distance_filters:
            self.distance_filters[key].clear()
        
        for key in self.sector_reliability:
            self.sector_reliability[key] = 1.0
        
        self.data_quality_history.clear()
        self.processing_times.clear()
        
        rospy.loginfo("LiDAR filters and reliability scores reset")


if __name__ == '__main__':
    try:
        EnhancedPathPlanningLidarPublisher()
    except rospy.ROSInterruptException:
        rospy.loginfo("Enhanced LiDAR Publisher normal shutdown")
    except Exception as e:
        rospy.logerr(f"Enhanced LiDAR Publisher critical error: {e}")