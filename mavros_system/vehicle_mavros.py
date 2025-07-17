import rospy
import threading
import time
from mavros_msgs.srv import CommandBool, SetMode
from mavros_msgs.msg import State, HomePosition, Waypoint, WaypointList
from sensor_msgs.msg import NavSatFix
from geographic_msgs import GeoPoint, GeoPoseStamped
from geometry_msgs.msg import Point, PoseStamped

class Vehicle_Mavros:
    def __init__(self, rate=20):
        rospy.init_node('vehicle', anonymous=True)

        self.state = None
        self.home_position_global = GeoPoint() # geographic_msgs/GeoPoint
        self.home_position_local = Point() # geometry_msgs/Point
        self.vehicle_position_global = GeoPoint() # geographic_msgs/GeoPoint
        self.vehicle_position_local = Point() # geometry_msgs/Point
        self.current_waypoints = [] # list of mavros_msgs/Waypoint.msg
        self._state_lock = threading.Lock()

        rospy.wait_for_service('/mavros/cmd/arming')
        rospy.wait_for_service('/mavros/set_mode')

        self.arm_service = rospy.ServiceProxy('/mavros/cmd/arming', CommandBool)
        self.mode_service = rospy.ServiceProxy('/mavros/set_mode', SetMode)

        rospy.Subscriber('/mavros/state', State, self._state_cb)
        rospy.Subscriber('/mavros/home_position/home', HomePosition, self._home_cb)
        rospy.Subscriber('/mavros/global_position/global', NavSatFix, self._gps_cb)
        rospy.Subscriber('/mavros/local_position/pose', PoseStamped, self._ned_cb)
        rospy.Subscriber('/mavros/mission/waypoints', WaypointList, self._wp_list_cb)

        self.target_global_pub = rospy.Publisher('/mavros/setpoint_position/global', GeoPoseStamped, queue_size=1)
        self.target_global = GeoPoseStamped()
        self.publish_target_global = False
        self.target_local_pub = rospy.Publisher('/mavros/setpoint_position/local', PoseStamped, queue_size=1)
        self.target_local = PoseStamped()
        self.publish_target_local = False

        self.rate = rospy.Rate(rate)
        threading.Thread(target=self._publish_loop).start()


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
        self.home_position_local = msg.position

    def _gps_cb(self, msg: NavSatFix):
        self.vehicle_position_global.latitude = msg.latitude
        self.vehicle_position_global.longitude = msg.longitude
        self.vehicle_position_global.altitude = msg.altitude

    def _ned_cb(self, msg: PoseStamped):
        self.vehicle_position_local = msg.pose.position

    def _wp_list_cb(self, msg: WaypointList):
        self.current_waypoints = msg.waypoints

    def _publish_loop(self):
        while not rospy.is_shutdown() and self.state and self.state.connected:

            if self.state.guided:
                if self.publish_target_global and self.target_global:
                    self.target_global_pub.publish(self.target_global)
                elif self.publish_target_local and self.target_local:
                    self.target_local_pub.publish(self.target_local)
            else:
                self.target_global = GeoPoseStamped()
                self.target_local = PoseStamped()

            self.rate.sleep()
    
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
        msg.header.stamp = rospy.Time.now()
        msg.pose.position.latitude = latitude
        msg.pose.position.longitude = longitude
        msg.pose.position.altitude = 0
        self.publish_target_local = False
        self.target_global = msg
        self.publish_target_global = True

    def setTargetPositionLocal(self, x, y):
        msg = PoseStamped()
        msg.header.stamp = rospy.Time.now()
        msg.pose.position.x = x
        msg.pose.position.y = y
        msg.pose.position.z = 0
        self.target_local = msg
        self.publish_target_global = False
        self.target_local = msg
        self.publish_target_local = True