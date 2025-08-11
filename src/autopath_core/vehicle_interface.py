import rospy
from mavros_msgs.srv import CommandBool, SetMode
from mavros_msgs.msg import State, HomePosition, WaypointList, VFR_HUD
from sensor_msgs.msg import NavSatFix
from geographic_msgs.msg import GeoPoint, GeoPoseStamped
# Lock for thread-safe access to shared variables
from threading import Lock

class Vehicle:
    def __init__(self, rate):
        self.rate = rospy.Rate(rate)
        self.rate_value = rate
        self.state = State()
        self.home_position_global = GeoPoint() # geographic_msgs/GeoPoint
        self.vehicle_position_global = GeoPoint() # geographic_msgs/GeoPoint
        self.vehicle_heading_compass = int()
        self.current_waypoints = [] # list of mavros_msgs/Waypoint.msg
        self._lock = Lock()

        rospy.wait_for_service('/mavros/cmd/arming')
        rospy.wait_for_service('/mavros/set_mode')
        self.arm_service = rospy.ServiceProxy('/mavros/cmd/arming', CommandBool)
        self.mode_service = rospy.ServiceProxy('/mavros/set_mode', SetMode)

        rospy.Subscriber('/mavros/state', State, self._state_cb)
        rospy.Subscriber('/mavros/home_position/home', HomePosition, self._home_cb)
        rospy.Subscriber('/mavros/global_position/global', NavSatFix, self._position_cb)
        rospy.Subscriber('/mavros/vfr_hud', VFR_HUD, self._orientation_cb)
        rospy.Subscriber('/mavros/mission/waypoints', WaypointList, self._wp_list_cb)

        self.target_global_pub = rospy.Publisher('/mavros/setpoint_position/global', GeoPoseStamped, queue_size=2)
        self.target_global_pub_seq = 0
        self.target_global_pub_flag = False
        self.target_global = GeoPoseStamped()
        rospy.Timer(rospy.Duration(1.0/self.rate_value), self._publish_target_global)

    def _state_cb(self, msg: State):
        """Update vehicle state from MAVROS state message"""
        old_state = self.state
        self.state = msg
        
        # Log important state changes
        if old_state.mode != msg.mode:
            rospy.loginfo(f"Vehicle mode changed: {old_state.mode} -> {msg.mode}")
        if old_state.armed != msg.armed:
            rospy.loginfo(f"Vehicle armed state changed: {old_state.armed} -> {msg.armed}")
        if old_state.connected != msg.connected:
            rospy.loginfo(f"Vehicle connection state changed: {old_state.connected} -> {msg.connected}")

    def _home_cb(self, msg: HomePosition):
        """Handle home position updates"""
        self.home_position_global = msg.geo
        rospy.loginfo_once("Home position received")
        rospy.logdebug(f"Home position set to: lat={msg.geo.latitude:.6f}, lon={msg.geo.longitude:.6f}")

    def _position_cb(self, msg: NavSatFix):
        """Handle GPS position updates"""
        self.vehicle_position_global.latitude = msg.latitude
        self.vehicle_position_global.longitude = msg.longitude
        self.vehicle_position_global.altitude = msg.altitude
        rospy.logdebug_throttle(5, f"Position: lat={msg.latitude:.6f}, lon={msg.longitude:.6f}, alt={msg.altitude:.1f}")

    def _orientation_cb(self, msg: VFR_HUD):
        """Handle heading updates"""
        self.vehicle_heading_compass = msg.heading
        rospy.logdebug_throttle(5, f"Heading: {msg.heading}")

    def _wp_list_cb(self, msg: WaypointList):
        """Handle mission waypoint updates"""
        old_wp_count = len(self.current_waypoints)
        self.current_waypoints = msg.waypoints
        if old_wp_count != len(msg.waypoints):
            rospy.loginfo(f"Mission waypoints updated: {len(msg.waypoints)} points")

    def _publish_target_global(self, event=None):
        """Publish target position if in GUIDED mode and have a target"""
        with self._lock:
            if not self.state.connected:
                rospy.logwarn_throttle(10, "Vehicle not connected, cannot publish target")
                self.target_global_pub_flag = False
                return
                
            if not self.state.guided:
                rospy.logwarn_throttle(5, "Not in GUIDED mode, skipping target position publish")
                self.target_global_pub_flag = False
                return
                
            if not self.target_global_pub_flag:
                self.target_global = GeoPoseStamped()
                return
                
            try:
                self.target_global.header.seq = self.target_global_pub_seq
                self.target_global.header.stamp = rospy.Time.now()
                self.target_global_pub.publish(self.target_global)
                self.target_global_pub_seq += 1
                rospy.logdebug_throttle(
                    5, 
                    f"Publishing target: lat={self.target_global.pose.position.latitude:.6f}, "
                    f"lon={self.target_global.pose.position.longitude:.6f}"
                )
            except Exception as e:
                rospy.logerr(f"Error publishing target position: {str(e)}")
                self.target_global_pub_flag = False

    def _wait_for_condition(self, condition_func, message, timeout=10.0):
        """Generic wait helper with timeout"""
        start_time = rospy.Time.now()
        while not rospy.is_shutdown():
            if condition_func():
                return True
            if (rospy.Time.now() - start_time).to_sec() > timeout:
                rospy.logerr(f"Timeout waiting for {message}")
                return False
            self.rate.sleep()
        return False

    def arm(self, timeout=10.0):
        """Arm the vehicle with timeout and error handling"""
        rospy.loginfo("Attempting to arm vehicle...")
        try:
            self.arm_service(True)
            success = self._wait_for_condition(
                lambda: self.state.armed,
                "arming",
                timeout
            )
            if success:
                rospy.loginfo("Vehicle armed successfully")
            return success
        except rospy.ServiceException as e:
            rospy.logerr(f"Arming service call failed: {str(e)}")
            return False

    def disarm(self, timeout=10.0):
        """Disarm the vehicle with timeout and error handling"""
        rospy.loginfo("Attempting to disarm vehicle...")
        try:
            self.arm_service(False)
            success = self._wait_for_condition(
                lambda: not self.state.armed,
                "disarming",
                timeout
            )
            if success:
                rospy.loginfo("Vehicle disarmed successfully")
            return success
        except rospy.ServiceException as e:
            rospy.logerr(f"Disarming service call failed: {str(e)}")
            return False

    def setMode(self, mode, timeout=10.0):
        """Set vehicle mode with timeout and error handling"""
        rospy.loginfo(f"Attempting to set mode to {mode}...")
        try:
            self.mode_service(base_mode=0, custom_mode=mode)
            success = self._wait_for_condition(
                lambda: self.state.mode == mode,
                f"mode change to {mode}",
                timeout
            )
            if success:
                rospy.loginfo(f"Vehicle mode changed to {mode}")
            return success
        except rospy.ServiceException as e:
            rospy.logerr(f"Mode change service call failed: {str(e)}")
            return False
            
    def setTargetPositionGlobal(self, latitude, longitude):
        """Set global position target"""
        rospy.logdebug(f"Setting target position: lat={latitude:.6f}, lon={longitude:.6f}")
        
        if not self.state.guided:
            rospy.logwarn("Cannot set target: vehicle not in GUIDED mode")
            return False
            
        msg = GeoPoseStamped()
        msg.pose.position.latitude = latitude
        msg.pose.position.longitude = longitude
        msg.pose.position.altitude = 0
        
        with self._lock:
            self.target_global = msg
            self.target_global_pub_flag = True
            
        rospy.logdebug("Target position set successfully")
        return True