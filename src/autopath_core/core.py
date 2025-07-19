import numpy as np
import rospy
import math
from threading import Lock
from autopath_core.pathfinder import NavDomain
from autopath_core.vehicle_interface import Vehicle
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import PoseStamped

class Commander:
    def __init__(self, rate, domain : NavDomain, vehicle: Vehicle):
        """Initialize the Commander node
        
        Args:
            rate: ROS rate for control loop
            domain: Navigation domain for path planning
            vehicle: Vehicle interface for control
        """
        # Configuration parameters
        self.rate = rospy.Rate(rate)
        self.rate_value = rate
        self.domain = domain
        self.vehicle = vehicle
        
        # Initialize home position at domain center
        self.home_ned = np.array([self.domain.nrow/2, self.domain.ncol/2])
        
        # Thread-safe state variables
        self._lock = Lock()
        self._vehicle_ned = np.zeros(2)
        self._home_global = np.zeros(2)
        self._mission_ned_list = []
        self._dynamic_obstacle_ned_list = []
        
        # Last update timestamps for monitoring 
        self._last_obstacle_update = rospy.Time.now()
        self._last_position_update = rospy.Time.now()
        
        # Configuration
        self.obstacle_timeout = rospy.Duration(2.0)  # Consider obstacles stale after 2s
        self.position_timeout = rospy.Duration(5.0)  # Position data timeout
        
        # Publishers and subscribers
        rospy.Subscriber('/fusion/dynamic_obstacles', Float32MultiArray, 
                        self._dynamic_obstacle_callback, queue_size=2)
        self.vehicle_pos_pub = rospy.Publisher('autopath_core/vehicle_position', 
                                             PoseStamped, queue_size=2)
        self.vehicle_pos_pub_seq = 0
                        
        # Start monitoring and publishing loops
        rospy.Timer(rospy.Duration(10.0/self.rate_value), self._monitor_timeouts)
        rospy.Timer(rospy.Duration(1.0/self.rate_value), self._publish_vehicle_position)  # Use rate from constructor

    def _dynamic_obstacle_callback(self, msg):
        """Convert polar coordinate obstacles to NED coordinates
        
        The incoming message contains obstacles in polar coordinates relative to vehicle.
        We convert these to NED coordinates accounting for vehicle position.
        """
        with self._lock:  # Thread safety for vehicle position access
            self._dynamic_obstacle_ned_list = []  # Access the underlying list directly
            data = msg.data
            
            if len(data) % 2 != 0:
                rospy.logwarn("Invalid obstacle data: odd number of elements")
                return
                
            try:
                for i in range(0, len(data), 2):
                    r = float(data[i])  # Range in meters
                    theta_deg = float(data[i + 1])  # Angle in degrees
                    
                    # Convert polar to cartesian coordinates
                    x = self._vehicle_ned[0] + r * math.cos(math.radians(theta_deg))
                    y = self._vehicle_ned[1] + r * math.sin(math.radians(theta_deg))
                    
                    # Store as numpy array for efficient operations
                    self._dynamic_obstacle_ned_list.append(np.array((x, y)))
                    
                """if self._dynamic_obstacle_ned_list:
                    rospy.logdebug(f"Updated {len(self._dynamic_obstacle_ned_list)} dynamic obstacles")"""
                self._last_obstacle_update = rospy.Time.now()
            except (ValueError, IndexError) as e:
                rospy.logerr(f"Error processing obstacle data: {e}")
                
    def _monitor_timeouts(self, event):
        """Monitor data freshness and log warnings if data is stale"""
        now = rospy.Time.now()
        
        # Check obstacle data freshness
        if (now - self._last_obstacle_update) > self.obstacle_timeout:
            rospy.logwarn_throttle(
                10.0,  # Warn every 10 seconds
                "No obstacle updates received for {:.1f} seconds".format(
                    (now - self._last_obstacle_update).to_sec()
                )
            )
            
        # Check position data freshness
        if (now - self._last_position_update) > self.position_timeout:
            rospy.logwarn_throttle(
                10.0,
                "No position updates received for {:.1f} seconds".format(
                    (now - self._last_position_update).to_sec()
                )
            )
            
    @property
    def vehicle_ned(self):
        """Thread-safe access to vehicle NED position"""
        with self._lock:
            return self._vehicle_ned.copy()
            
    @vehicle_ned.setter
    def vehicle_ned(self, value):
        """Thread-safe update of vehicle NED position"""
        with self._lock:
            self._vehicle_ned = np.array(value)
            self._last_position_update = rospy.Time.now()
            
    @property
    def dynamic_obstacle_ned_list(self):
        """Thread-safe access to obstacle list"""
        with self._lock:
            return self._dynamic_obstacle_ned_list.copy()
            
    def _publish_vehicle_position(self, event):
        """Publish vehicle NED position as PoseStamped"""
        # Convert current global position to NED
        current_ned = global_to_ned(
            self.vehicle.home_position_global.latitude, self.vehicle.home_position_global.longitude,
            self.vehicle.vehicle_position_global.latitude, self.vehicle.vehicle_position_global.longitude,
            self.home_ned[0], self.home_ned[1]
        )
        self.vehicle_ned = current_ned
            
        # Create and publish message
        msg = PoseStamped()
        msg.header.seq = self.vehicle_pos_pub_seq
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = 'ned'
        
        with self._lock:
            msg.pose.position.x = self._vehicle_ned[0]  # North
            msg.pose.position.y = self._vehicle_ned[1]  # East
            msg.pose.position.z = 0.0  # Down (not used)
            
        self.vehicle_pos_pub.publish(msg)
        self.vehicle_pos_pub_seq += 1

    def _wait_for(self, condition_func, message, timeout=None):
        """Helper function to wait for a condition with proper shutdown handling
        
        Args:
            condition_func: Function that returns True when condition is met
            message: Message to display while waiting
            timeout: Optional timeout in seconds
            
        Returns:
            bool: True if condition was met, False if shutdown or timeout occurred
        """
        start_time = rospy.Time.now()
        rospy.logdebug(f"Starting to wait for: {message}")  # Debug level for start
        
        while not rospy.is_shutdown():
            if condition_func():
                rospy.loginfo(f"Condition met: {message}")  # Info level for success
                return True
                
            if timeout and (rospy.Time.now() - start_time).to_sec() > timeout:
                rospy.logerr(f"Timeout waiting for: {message}")  # Error level for timeout
                return False
                
            rospy.loginfo_throttle(5, f"Waiting for: {message}")  # Throttled status updates
            self.rate.sleep()
            
        rospy.logdebug("Wait interrupted by shutdown")  # Debug level for shutdown
        return False

    def _convert_waypoints(self):
        """Convert global waypoints to NED coordinates"""
        self.mission_ned_list = []
        for i, waypoint in enumerate(self.vehicle.current_waypoints):
            rospy.loginfo(f"Converting waypoint {i+1}:")
            rospy.loginfo(f"  Waypoint global: lat={waypoint.x_lat:.8f}, lon={waypoint.y_long:.8f}")
            rospy.loginfo(f"  Home reference: lat={self.home_global[0]:.8f}, lon={self.home_global[1]:.8f}")
            
            ned_pos = global_to_ned(
                self.home_global[0], self.home_global[1],
                waypoint.x_lat, waypoint.y_long,
                self.home_ned[0], self.home_ned[1]
            )
            rospy.loginfo(f"  Converted to NED: N={ned_pos[0]:.1f}, E={ned_pos[1]:.1f}")
            self.mission_ned_list.append(np.array(ned_pos))
            
    def execute_mission(self):
        """Execute the mission sequence"""
        rospy.loginfo_once("Mission executor node started")
        
        # Give MAVROS topics time to fully establish
        rospy.loginfo("Waiting for MAVROS connections to stabilize...")
        rospy.sleep(2.0)  # Wait for subscribers to establish
        
        MISSION_RADIUS = 2.0  # meters, distance to consider waypoint reached
        
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
        rospy.loginfo(f"Home position set to: lat={self.home_global[0]:.6f}, lon={self.home_global[1]:.6f}")
        rospy.loginfo(f"Vehicle current global position: lat={self.vehicle.vehicle_position_global.latitude:.6f}, lon={self.vehicle.vehicle_position_global.longitude:.6f}")
        rospy.loginfo(f"NED origin at grid: N={self.home_ned[0]:.1f}, E={self.home_ned[1]:.1f}")
        
        # Wait for mission waypoints
        if not self._wait_for(
            lambda: len(self.vehicle.current_waypoints) > 0,
            "mission waypoints"
        ):
            rospy.logerr("No mission waypoints available")
            return False
        
        # Convert waypoints to NED
        self._convert_waypoints()
        rospy.loginfo(f"Loaded mission with {len(self.mission_ned_list)} waypoints")
        rospy.logdebug("Mission waypoints (NED):")
        for i, wp in enumerate(self.mission_ned_list):
            rospy.logdebug(f"  {i+1}: N={wp[0]:.1f}, E={wp[1]:.1f}")
        
        # Switch to GUIDED mode and arm
        if not self.vehicle.setMode("GUIDED"):
            rospy.logerr("Failed to enter GUIDED mode")
            return False
            
        if not self.vehicle.arm():
            rospy.logerr("Failed to arm vehicle")
            return False
        
        rospy.loginfo("=== Starting mission execution ===")
        
        # Navigate through waypoints
        for i, target_ned in enumerate(self.mission_ned_list):
            rospy.loginfo(f"Navigating to waypoint {i+1}/{len(self.mission_ned_list)}")
            rospy.logdebug(f"Target position (NED): N={target_ned[0]:.1f}, E={target_ned[1]:.1f}")
            
            target_coord = self.domain.Coordinate(*map(int, np.round(target_ned)))
            
            if not self.vehicle.state.guided:
                rospy.logerr("Lost GUIDED mode! Aborting mission")
                return False
                
            # Keep trying to reach the waypoint until timeout
            reached = False
            start_time = rospy.Time.now()
            timeout_duration = rospy.Duration(60.0)  # 60 seconds timeout
            
            while not reached and not rospy.is_shutdown():
                if (rospy.Time.now() - start_time) > timeout_duration:
                    rospy.logwarn(f"Timeout reaching waypoint {i+1}, skipping to next")
                    break
                    
                # Check if we've reached the waypoint
                current_distance = np.linalg.norm(target_ned - self.vehicle_ned)
                if current_distance <= MISSION_RADIUS:
                    rospy.loginfo(f"Reached waypoint {i+1}")
                    reached = True
                    break
                
                # If not reached, update path planning and movement
                try:
                    # Try to find path avoiding obstacles
                    current_coord = self.domain.Coordinate(*map(int, np.round(self.vehicle_ned)))
                    obstacle_coords = [
                        self.domain.Coordinate(*map(int, np.round(obs))) 
                        for obs in self.dynamic_obstacle_ned_list
                    ]
                    
                    path = self.domain.a_star_search(
                        current_coord, 
                        target_coord,
                        moving_obstacles=obstacle_coords
                    )
                    
                    if not path:
                        rospy.logwarn_throttle(5.0, "No valid path found, retrying...")
                        self.rate.sleep()
                        continue
                        
                    path_len = len(path)
                    rospy.logdebug(f"Found path with {path_len} waypoints")
                    
                    # Take next waypoint with a reasonable look-ahead
                    look_ahead = min(3, path_len - 1)
                    next_coord = path[look_ahead]
                    next_ned = np.array([next_coord.row, next_coord.col])
                    
                    # Convert to global coordinates and send
                    next_global = ned_to_global(
                        self.home_ned[0], self.home_ned[1],
                        next_ned[0], next_ned[1],
                        self.home_global[0], self.home_global[1]
                    )
                    if not self.vehicle.setTargetPositionGlobal(*next_global):
                        rospy.logwarn_throttle(2.0, "Failed to set next waypoint")
                    else:
                        rospy.logdebug(f"Moving to intermediate point at NED: {next_ned}")
                        
                except Exception as e:
                    rospy.logerr(f"Error in path planning: {e}")
                    
                self.rate.sleep()
        
        # Mission complete - switch back to MANUAL mode
        if not rospy.is_shutdown():
            rospy.loginfo("Mission complete - switching to MANUAL mode")
            self.vehicle.setMode("MANUAL")
            return True
        return False


# Coordinate conversion functions (could be moved to a utilities module)
def ned_to_global(origin_n, origin_e, position_n, position_e, origin_lat, origin_lon):
    offset_n = position_n - origin_n
    offset_e = position_e - origin_e
    R = 6378137.0  # Earth radius in meters
    position_lat = origin_lat + math.degrees(offset_n / R)
    position_lon = origin_lon + math.degrees(offset_e / (R * math.cos(math.radians(origin_lat))))
    return position_lat, position_lon

def global_to_ned(origin_lat, origin_lon, position_lat, position_lon, origin_n, origin_e):
    offset_lat = math.radians(position_lat - origin_lat)
    offset_lon = math.radians(position_lon - origin_lon)
    R = 6378137.0
    offset_n = R * offset_lat
    offset_e = R * offset_lon * math.cos(math.radians(origin_lat))
    position_n = origin_n + offset_n
    position_e = origin_e + offset_e
    return position_n, position_e