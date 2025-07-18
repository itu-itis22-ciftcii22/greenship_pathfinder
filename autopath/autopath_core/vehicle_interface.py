import rospy
import threading
from mavros_msgs.srv import CommandBool, SetMode
from mavros_msgs.msg import State, HomePosition, WaypointList
from sensor_msgs.msg import NavSatFix
from geographic_msgs.msg import GeoPoint, GeoPoseStamped
class Vehicle:
    def __init__(self, rate : rospy.Rate):
        self.state = State()
        self.home_position_global = GeoPoint() # geographic_msgs/GeoPoint
        self.vehicle_position_global = GeoPoint() # geographic_msgs/GeoPoint
        self.current_waypoints = [] # list of mavros_msgs/Waypoint.msg
        self._state_lock = threading.Lock()
        rospy.wait_for_service('/mavros/cmd/arming')
        rospy.wait_for_service('/mavros/set_mode')
        self.arm_service = rospy.ServiceProxy('/mavros/cmd/arming', CommandBool)
        self.mode_service = rospy.ServiceProxy('/mavros/set_mode', SetMode)
        rospy.Subscriber('/mavros/state', State, self._state_cb)
        rospy.Subscriber('/mavros/home_position/home', HomePosition, self._home_cb)
        rospy.Subscriber('/mavros/global_position/global', NavSatFix, self._gps_cb)
        rospy.Subscriber('/mavros/mission/waypoints', WaypointList, self._wp_list_cb)
        self.target_global_pub = rospy.Publisher('/mavros/setpoint_position/global', GeoPoseStamped, queue_size=2)
        self.target_global = GeoPoseStamped()
        self.publish_target_global = False
        rospy.Timer(rospy.Duration(secs=1/rate), self._publishers)
    def _state_cb(self, msg: State):
        with self._state_lock:
            self.state = msg
        """
        std_msgs/Header header
        bool connected
        bool armed
        bool guided
        bool manual_input
        string mode
        uint8 system_status
        """
    def _home_cb(self, msg: HomePosition):
        self.home_position_global = msg.geo
    def _gps_cb(self, msg: NavSatFix):
        self.vehicle_position_global.latitude = msg.latitude
        self.vehicle_position_global.longitude = msg.longitude
        self.vehicle_position_global.altitude = msg.altitude
    def _wp_list_cb(self, msg: WaypointList):
        self.current_waypoints = msg.waypoints
    def _publishers(self):
        if self.state and self.state.guided and self.publish_target_global:
            print(f"Publishing target global position: {self.target_global.pose.position.latitude}, {self.target_global.pose.position.longitude}")
            self.target_global.header.stamp = rospy.Time.now()
            self.target_global_pub.publish(self.target_global)
        else:
            print("Not in GUIDED mode or publish_target_global is False, skipping publish.")
            self.publish_target_global = False
            self.target_global = GeoPoseStamped()
    def arm(self):
        self.arm_service(True)
    def disarm(self):
        self.arm_service(False)
    def getMode(self):
        with self._state_lock:
            return self.state.mode if self.state else None
    def setMode(self, mode):
        self.mode_service(base_mode=0, custom_mode=mode)
    def setTargetPositionGlobal(self, latitude, longitude):
        msg = GeoPoseStamped()
        msg.pose.position.latitude = latitude
        msg.pose.position.longitude = longitude
        msg.pose.position.altitude = 0
        self.target_global = msg
        self.publish_target_global = True