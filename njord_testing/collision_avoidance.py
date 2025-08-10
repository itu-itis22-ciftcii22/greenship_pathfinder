import sys
import signal
import rospy
import numpy as np
import os
import math
from std_msgs.msg import Float32MultiArray, String
from autopath_core.pathfinder import NavDomain
from autopath_core.vehicle_interface import Vehicle
from autopath_core.core import Commander, ned_to_global

class COLREGCollisionAvoidance(Commander):
    def __init__(self, rate, domain, vehicle):
        super().__init__(rate, domain, vehicle)
        
        # Fusion data integration
        rospy.Subscriber('/fusion_output', Float32MultiArray, self.fusion_callback)
        rospy.Subscriber('/navigation_alerts', String, self.navigation_alerts_callback)
        
        # COLREG-specific data
        self.fusion_obstacles = []
        self.last_fusion_time = None
        self.detected_vessels = []
        self.colreg_maneuver_active = False
        self.maneuver_start_time = None
        
        # COLREG Parameters
        self.STARBOARD_GIVE_WAY_ANGLE_MIN = 5.0   # 5° sağ taraftan
        self.STARBOARD_GIVE_WAY_ANGLE_MAX = 112.5 # 112.5° sağ taraftan (COLREG crossing situation)
        self.SAFE_PASSING_DISTANCE = 50.0        # Güvenli geçiş mesafesi (metre)
        self.DETECTION_SIGNAL_DISTANCE = 100.0   # Tespit sinyali mesafesi
        self.STERN_PASSING_OFFSET = 30.0          # Arkadan geçiş için offset (metre)
        
        # Detection signaling
        self.detection_pub = rospy.Publisher('/vessel_detection_signal', String, queue_size=10)
        self.last_detection_signal = None
        
        rospy.loginfo("COLREG Collision Avoidance Commander initialized")
        rospy.loginfo(f"Starboard give-way angle range: {self.STARBOARD_GIVE_WAY_ANGLE_MIN}° to {self.STARBOARD_GIVE_WAY_ANGLE_MAX}°")
    
    def fusion_callback(self, msg):
        """Process fusion obstacle data with COLREG analysis"""
        self.last_fusion_time = rospy.Time.now()
        self.fusion_obstacles = []
        self.detected_vessels = []
        
        # Parse [angle1, dist1, angle2, dist2, ...] format
        for i in range(0, len(msg.data), 2):
            if i + 1 < len(msg.data):
                angle = msg.data[i]
                distance = msg.data[i + 1]
                
                # Convert to NED coordinates
                angle_rad = np.radians(angle)
                obs_north = self.vehicle_ned[0] + distance * np.cos(angle_rad)
                obs_east = self.vehicle_ned[1] + distance * np.sin(angle_rad)
                
                self.fusion_obstacles.append([obs_north, obs_east])
                
                # Analyze for COLREG compliance (assume moving obstacles are vessels)
                if distance < self.DETECTION_SIGNAL_DISTANCE:
                    vessel_info = self.analyze_vessel_for_colreg(angle, distance)
                    if vessel_info:
                        self.detected_vessels.append(vessel_info)
        
        rospy.logdebug(f"Fusion callback: {len(self.fusion_obstacles)} obstacles, {len(self.detected_vessels)} vessels")
    
    def navigation_alerts_callback(self, msg):
        """Handle navigation alerts from fusion system"""
        if "CRITICAL_PROXIMITY" in msg.data:
            rospy.logwarn(f"Navigation Alert: {msg.data}")
    
    def analyze_vessel_for_colreg(self, relative_angle, distance):
        """
        Analyze detected vessel for COLREG compliance
        relative_angle: angle relative to our heading (+ = starboard, - = port)
        distance: distance to vessel
        """
        vessel_info = {
            'angle': relative_angle,
            'distance': distance,
            'colreg_rule': None,
            'action_required': None,
            'is_starboard': relative_angle > 0
        }
        
        # COLREG Rule 15: Crossing Situations
        if (vessel_info['is_starboard'] and 
            self.STARBOARD_GIVE_WAY_ANGLE_MIN <= relative_angle <= self.STARBOARD_GIVE_WAY_ANGLE_MAX):
            
            vessel_info['colreg_rule'] = "Rule 15 - Crossing (Give Way)"
            vessel_info['action_required'] = "GIVE_WAY_STARBOARD"
            
            # Signal detection
            self.signal_vessel_detection(vessel_info)
            
            rospy.loginfo(f"COLREG: Starboard vessel detected at {relative_angle:.1f}°, {distance:.1f}m - GIVE WAY required")
        
        elif relative_angle < 0 and abs(relative_angle) <= self.STARBOARD_GIVE_WAY_ANGLE_MAX:
            vessel_info['colreg_rule'] = "Rule 15 - Crossing (Stand On)"
            vessel_info['action_required'] = "STAND_ON"
            rospy.loginfo(f"COLREG: Port vessel detected at {relative_angle:.1f}°, {distance:.1f}m - STAND ON")
        
        elif abs(relative_angle) <= 5.0:  # Head-on situation
            vessel_info['colreg_rule'] = "Rule 14 - Head-on"
            vessel_info['action_required'] = "ALTER_COURSE_STARBOARD"
            rospy.loginfo(f"COLREG: Head-on vessel detected at {relative_angle:.1f}°, {distance:.1f}m - Alter course to starboard")
        
        return vessel_info
    
    def signal_vessel_detection(self, vessel_info):
        """Signal vessel detection as required by competition rules"""
        current_time = rospy.Time.now()
        
        # Avoid spamming signals
        if (self.last_detection_signal is None or 
            (current_time - self.last_detection_signal).to_sec() > 2.0):
            
            signal_msg = f"VESSEL_DETECTED|ANGLE:{vessel_info['angle']:.1f}|DIST:{vessel_info['distance']:.1f}|RULE:{vessel_info['colreg_rule']}|ACTION:{vessel_info['action_required']}"
            
            self.detection_pub.publish(signal_msg)
            self.last_detection_signal = current_time
            
            rospy.loginfo(f"🚢 VESSEL DETECTION SIGNAL: {signal_msg}")
    
    def plan_colreg_maneuver(self, current_coord, target_coord):
        """
        Plan path according to COLREG rules
        """
        # Check for starboard give-way vessels
        starboard_vessels = [v for v in self.detected_vessels if v['action_required'] == 'GIVE_WAY_STARBOARD']
        
        if not starboard_vessels:
            # No COLREG maneuver needed, use normal path planning
            return self.domain.a_star_search(
                current_coord,
                target_coord,
                corridor=self._find_corridor(self._find_buoy_pairs()),
                moving_obstacles=[
                    self.domain.Coordinate(*map(int, np.round(obs))) 
                    for obs in self.fusion_obstacles
                ]
            )
        
        # COLREG Maneuver: Give way to starboard vessel
        closest_starboard = min(starboard_vessels, key=lambda x: x['distance'])
        
        rospy.loginfo(f"🚢 COLREG MANEUVER: Giving way to starboard vessel at {closest_starboard['angle']:.1f}°, {closest_starboard['distance']:.1f}m")
        
        # Strategy: Pass behind (stern passing) the starboard vessel
        return self.plan_stern_passing_maneuver(current_coord, target_coord, closest_starboard)
    
    def plan_stern_passing_maneuver(self, current_coord, target_coord, starboard_vessel):
        """
        Plan a stern passing maneuver for COLREG compliance
        """
        self.colreg_maneuver_active = True
        if self.maneuver_start_time is None:
            self.maneuver_start_time = rospy.Time.now()
        
        # Calculate vessel position in NED
        vessel_angle_rad = np.radians(starboard_vessel['angle'])
        vessel_north = self.vehicle_ned[0] + starboard_vessel['distance'] * np.cos(vessel_angle_rad)
        vessel_east = self.vehicle_ned[1] + starboard_vessel['distance'] * np.sin(vessel_angle_rad)
        
        # Create stern passing waypoint
        # Assuming vessel is moving roughly perpendicular to our path
        stern_offset_north = -self.STERN_PASSING_OFFSET  # Go behind
        stern_offset_east = 0
        
        stern_waypoint_north = vessel_north + stern_offset_north
        stern_waypoint_east = vessel_east + stern_offset_east
        
        # Create intermediate coordinate for stern passing
        stern_coord = self.domain.Coordinate(
            int(np.round(stern_waypoint_north)), 
            int(np.round(stern_waypoint_east))
        )
        
        rospy.loginfo(f"🚢 STERN PASSING: Waypoint at N={stern_waypoint_north:.1f}, E={stern_waypoint_east:.1f}")
        
        # Plan path to stern waypoint first, then to target
        # For simplicity, we'll create an expanded obstacle for the vessel's predicted path
        expanded_obstacles = []
        
        for obs in self.fusion_obstacles:
            expanded_obstacles.append(self.domain.Coordinate(*map(int, np.round(obs))))
        
        # Add safety zone around the starboard vessel
        safety_radius = 3  # Grid units
        for dr in range(-safety_radius, safety_radius + 1):
            for dc in range(-safety_radius, safety_radius + 1):
                if dr*dr + dc*dc <= safety_radius*safety_radius:
                    safe_coord = self.domain.Coordinate(
                        int(vessel_north) + dr,
                        int(vessel_east) + dc
                    )
                    if (0 <= safe_coord.row < self.domain.rows and 
                        0 <= safe_coord.col < self.domain.cols):
                        expanded_obstacles.append(safe_coord)
        
        # Try to find path directly to target, avoiding vessel
        path = self.domain.a_star_search(
            current_coord,
            target_coord,
            corridor=self._find_corridor(self._find_buoy_pairs()),
            moving_obstacles=expanded_obstacles
        )
        
        return path
    
    def execute_task_code_collision_avoidance(self):
        """Execute COLREG-compliant collision avoidance mission"""
        rospy.loginfo_once("🚢 COLREG Collision Avoidance Mission Started")
        
        # Give MAVROS topics time to fully establish
        rospy.loginfo("Waiting for MAVROS connections to stabilize...")
        rospy.sleep(2.0)
        
        MISSION_RADIUS = 2.0  # meters
        TARGET_SPEED = 2.0    # 2 knots as per competition rules
        
        # Wait for AUTO mode and home position
        if not self._wait_for(
            lambda: (self.vehicle.state.mode == "AUTO" and 
                    self.vehicle.home_position_global.latitude != 0),
            "AUTO mode and home position"
        ):
            rospy.logerr("Failed to initialize mission")
            return False
        
        # Store home position
        self.home_global = np.array([
            self.vehicle.home_position_global.latitude,
            self.vehicle.home_position_global.longitude
        ])
        rospy.loginfo(f"🏠 Home position: lat={self.home_global[0]:.6f}, lon={self.home_global[1]:.6f}")
        
        # Wait for mission waypoints
        if not self._wait_for(
            lambda: len(self.vehicle.current_waypoints) > 0,
            "mission waypoints"
        ):
            rospy.logerr("No mission waypoints available")
            return False
        
        # Convert waypoints to NED
        self._convert_waypoints()
        rospy.loginfo(f"📍 Mission loaded: {len(self.mission_ned_list)} waypoints")
        
        # Switch to GUIDED mode and arm
        if not self.vehicle.setMode("GUIDED"):
            rospy.logerr("Failed to enter GUIDED mode")
            return False
        
        if not self.vehicle.arm():
            rospy.logerr("Failed to arm vehicle")
            return False
        
        # Set speed to 2 knots (competition requirement)
        rospy.loginfo(f"🚤 Setting speed to {TARGET_SPEED} knots")
        # Note: Speed setting implementation depends on your vehicle interface
        
        rospy.loginfo("🚢 === COLREG MISSION EXECUTION STARTED ===")
        
        # Navigate through waypoints with COLREG compliance
        for i, target_ned in enumerate(self.mission_ned_list):
            rospy.loginfo(f"🎯 Navigating to waypoint {i+1}/{len(self.mission_ned_list)}")
            
            target_coord = self.domain.Coordinate(*map(int, np.round(target_ned)))
            
            if not self.vehicle.state.guided:
                rospy.logerr("Lost GUIDED mode! Aborting mission")
                return False
            
            # Navigate to waypoint with COLREG compliance
            reached = False
            start_time = rospy.Time.now()
            timeout_duration = rospy.Duration(120.0)  # 2 minutes
            
            while not reached and not rospy.is_shutdown():
                if (rospy.Time.now() - start_time) > timeout_duration:
                    rospy.logwarn(f"Timeout reaching waypoint {i+1}")
                    break
                
                # Check if reached
                current_distance = np.linalg.norm(target_ned - self.vehicle_ned)
                if current_distance <= MISSION_RADIUS:
                    rospy.loginfo(f"✅ Reached waypoint {i+1}")
                    reached = True
                    break
                
                # COLREG-compliant path planning
                try:
                    current_coord = self.domain.Coordinate(*map(int, np.round(self.vehicle_ned)))
                    
                    # Use fusion data if available
                    if (self.last_fusion_time and 
                        (rospy.Time.now() - self.last_fusion_time).to_sec() < 2.0):
                        
                        # Plan with COLREG compliance
                        path = self.plan_colreg_maneuver(current_coord, target_coord)
                        
                        rospy.logdebug(f"Using COLREG-compliant path planning with {len(self.detected_vessels)} vessels")
                    else:
                        # Fallback to basic path planning
                        path = self.domain.a_star_search(
                            current_coord,
                            target_coord,
                            corridor=self._find_corridor(self._find_buoy_pairs())
                        )
                        rospy.logwarn_throttle(5.0, "No fresh sensor data, using basic navigation")
                    
                    if not path:
                        rospy.logwarn_throttle(5.0, "No valid path found, retrying...")
                        self.rate.sleep()
                        continue
                    
                    # Execute path
                    look_ahead = min(5, len(path) - 1) if self.colreg_maneuver_active else min(3, len(path) - 1)
                    next_coord = path[look_ahead]
                    next_ned = np.array([next_coord.row, next_coord.col])
                    
                    # Convert to global and send
                    next_global = ned_to_global(
                        self.home_ned[0], self.home_ned[1],
                        next_ned[0], next_ned[1],
                        self.home_global[0], self.home_global[1]
                    )
                    
                    if not self.vehicle.setTargetPositionGlobal(*next_global):
                        rospy.logwarn_throttle(2.0, "Failed to set target position")
                    else:
                        rospy.logdebug(f"🧭 Moving to: N={next_ned[0]:.1f}, E={next_ned[1]:.1f}")
                
                except Exception as e:
                    rospy.logerr(f"COLREG path planning error: {e}")
                
                self.rate.sleep()
            
            # Reset COLREG maneuver state after reaching waypoint
            self.colreg_maneuver_active = False
            self.maneuver_start_time = None
        
        rospy.loginfo("🏁 === COLREG MISSION COMPLETED ===")
        return True

def clean_exit(signum=None, frame=None):
    rospy.loginfo("🛑 Exiting COLREG collision avoidance...")
    global vehicle
    try:
        if vehicle:
            vehicle.setMode("MANUAL")
            vehicle.disarm()
    except Exception as e:
        rospy.logerr(f"Error during cleanup: {str(e)}")
    sys.exit(0)

signal.signal(signal.SIGINT, clean_exit)
signal.signal(signal.SIGTERM, clean_exit)

if __name__ == "__main__":
    try:
        rospy.init_node("colreg_collision_avoidance")
        
        # Set debug level
        log_level = rospy.get_param('~log_level', os.getenv('ROS_LOG_LEVEL', 'info')).upper()
        rospy.loginfo(f"Setting log level to {log_level}")
        import logging
        logging.getLogger('rosout').setLevel(getattr(logging, log_level))
        
        rate = 10.0
        domain_size = 300
        domain = NavDomain(domain_size, domain_size)
        vehicle = Vehicle(rate)
        command = COLREGCollisionAvoidance(rate, domain, vehicle)
        
        rospy.loginfo("🚢 Starting COLREG-compliant collision avoidance system")
        
        while not rospy.is_shutdown():
            rospy.loginfo("🚀 Starting new COLREG mission execution")
            success = command.execute_task_code_collision_avoidance()
            
            if rospy.is_shutdown():
                break
            
            if success:
                rospy.loginfo("✅ COLREG mission completed successfully")
                rospy.sleep(5.0)
            else:
                rospy.logwarn("❌ COLREG mission failed, restarting...")
                rospy.sleep(10.0)
    
    except rospy.ROSInterruptException:
        pass