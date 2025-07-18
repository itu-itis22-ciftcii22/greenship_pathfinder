import numpy as np
import rospy
from autopath_core.pathfinder import NavDomain
from autopath_core.vehicle_interface import Vehicle
from std_msgs.msg import Header, Float32MultiArray
from geometry_msgs.msg import PoseArray, PoseStamped, Pose, Point, Quaternion

class RosIO:
    def __init__(self, rate : rospy.Rate, domain : NavDomain, vehicle: Vehicle):
        
        # Configuration parameters
        self.rate = rate
        self.domain = domain
        self.vehicle = vehicle
        self.home_ned = np.array([self.domain.nrow/2, self.domain.ncol/2])
        
        # State variables
        self.vehicle_ned = np.zeros(2)
        self.home_global = np.zeros(2)
        self.mission_ned_list = []
        self.obstacle_ned_list = []
        
        # ROS setup
        rospy.Subscriber('/obstacles', Float32MultiArray, self._obstacle_callback)
        self.position_pub = rospy.Publisher('/autopath_core/position', Point, queue_size=2)
        self.position_pub_seq = 0
        rospy.Timer(rospy.Duration(1/self.rate), self._publish_position)

        self.missions_pub = rospy.Publisher('/autopath_core/missions', Point, queue_size=2)
        self.missions_pub_seq = 0
        rospy.Timer(rospy.Duration(1/self.rate), self._publish_missions)

        rospy.Subscriber("/commandeer/set_target/flag", Bool, self._target_handler)
        rospy.Subscriber("/commandeer/set_target/point", PoseStamped, self._target_handler)
        self.target_flag = False
        self.target_ned = np.zeros(2)


    def _obstacle_callback(self, msg):
        self.obstacle_list = []
        data = msg.data
        
        for i in range(0, len(data), 2):
            r = data[i]
            theta_deg = data[i + 1]
            x = self.current_position[0] + r * math.cos(math.radians(theta_deg))
            y = self.current_position[1] + r * math.sin(math.radians(theta_deg))
            self.obstacle_list.append(np.array((x, y)))

    def _publish_position(self, event=None):
        position_global = np.array((
            self.vehicle.vehicle_position_global.latitude,
            self.vehicle.vehicle_position_global.longitude
        ))
        position_ned = np.array(global_to_ned(
            self.home_global[0], self.home_global[1],
            position_global[0], position_global[1],
            self.home_ned[0], self.home_ned[1] 
        ))
        self.current_position = position_ned
        msg = PoseStamped()
        msg.header = Header()
        msg.header.seq = self.position_pub_seq
        self.position_pub_seq += 1
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = "map"
        msg.pose = Pose()
        msg.pose.position = Point(self.current_position[0], self.current_position[1], 0.0)
        pose1.orientation = Quaternion(0.0, 0.0, 0.0, 1.0)
        self.position_pub.publish(msg)

    def _publish_missions(self):
        self.missions_ned = []
        for mission_global in self.vehicle.current_waypoints:
            mission_ned = np.array(global_to_ned(
                self.home_global[0], self.home_global[1],
                mission_global.x_lat, mission_global.y_long,
                self.home_ned[0], self.home_ned[1] 
            ))
            self.missions_ned.append(mission_ned)
        msg = PoseArray()
        msg.header = Header()
        msg.header.seq = seq
        self.missions_pub_seq += 1
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = "map"
        for mission_ned in self.missions_ned:
            pose = Pose()
            pose.position = Point(mission_ned[0], mission_ned[1], 0.0)
            pose.orientation = Quaternion(0.0, 0.0, 0.0, 1.0)      
            msg.poses.append(pose)
        pub.publish(msg)

    def _target_handler(self, msg):
        if isinstance(msg, Bool):
            if msg is True:
                self.target_flag = True
            else:
                self.target_ned = np.zeros(2)
        elif isinstance(msg, PoseStamped):
            self.target_ned = np.array((msg.pose.position.x, msg.pose.position.y))
        if self.target_flag and self.target_ned != np.zeros(2):
            target_global = np.array(ned_to_global(self.home_ned[0], self.home_ned[1],
                self.target_ned[0], self.target_ned[1],
                self.home_global[0], self.home_global[1]))
            self.vehicle.setTargetPositionGlobal(target_global[0], target_global[1])

    def execute_mission(self):
        """Main mission execution loop"""
        rospy.loginfo("Starting mission execution")
        
        # Wait for AUTO mode
        while self.vehicle.getMode() != "AUTO" and not rospy.is_shutdown():
            rospy.loginfo_throttle(5, f"Current mode: {self.vehicle.getMode()} - Waiting for AUTO")
            rospy.sleep(0.5)
        
        # Get home position
        while not hasattr(self.vehicle, 'home_position_global') or \
              self.vehicle.home_position_global.latitude == 0:
            rospy.loginfo_throttle(5, "Waiting for home position...")
            rospy.sleep(0.5)
        
        self.home_global = np.array([
            self.vehicle.home_position_global.latitude,
            self.vehicle.home_position_global.longitude
        ])
        
        # Get mission waypoints
        while len(self.vehicle.current_waypoints) == 0:
            rospy.loginfo_throttle(5, "Waiting for mission waypoints...")
            rospy.sleep(0.5)
        
        self.convert_waypoints()
        rospy.loginfo(f"Converted {len(self.missions_ned)} waypoints to NED")
        
        # Switch to GUIDED mode
        self.vehicle.setMode("GUIDED")
        rospy.loginfo("Switched to GUIDED mode")
        
        # Arm vehicle
        while not self.vehicle.state.armed:
            self.vehicle.arm()
            rospy.loginfo_throttle(1, "Arming vehicle...")
            rospy.sleep(0.5)
        
        rospy.loginfo("Vehicle armed - Starting navigation")
        
        # Navigate through waypoints
        for i, target_ned in enumerate(self.missions_ned):
            if not self.vehicle.state.guided:
                rospy.logerr("Vehicle not in GUIDED mode! Aborting mission")
                break
            
            rospy.loginfo(f"Navigating to waypoint {i+1}/{len(self.missions_ned)}: {target_ned}")
            target_coord = self.domain.Coordinate(*map(int, np.round(target_ned)))
            
            while not rospy.is_shutdown():
                # Update current position
                self.current_position = self.get_vehicle_position()
                
                # Check arrival
                if np.linalg.norm(target_ned - self.current_position) <= self.mission_radius:
                    rospy.loginfo(f"Reached waypoint {i+1}")
                    break
                
                # Path planning
                current_coord = self.domain.Coordinate(*map(int, np.round(self.current_position)))
                obstacle_coords = [
                    self.domain.Coordinate(*map(int, np.round(obs))) 
                    for obs in self.obstacle_list
                ]
                
                path = self.domain.a_star_search(
                    current_coord, 
                    target_coord,
                    moving_obstacles=obstacle_coords
                )
                
                if path:
                    next_idx = min(self.step_size, len(path)-1)
                    next_coord = path[next_idx]
                    next_ned = np.array([next_coord.row, next_coord.col])
                    self.send_position_setpoint(next_ned)
                else:
                    rospy.logwarn("No valid path found! Waiting...")
                
                rospy.sleep(1/self.control_rate)



# Coordinate conversion functions (could be moved to a utilities module)
def ned_to_global(origin_n, origin_e, position_n, position_e, origin_lat, origin_lon):
    offset_n = position_n - origin_n
    offset_e = position_e - origin_e
    R = 6378137.0  # Earth radius in meters
    position_lat = origin_lat + math.degrees(offset_n / R)
    position_lon = origin_lon + math.degrees(offset_e / (R * math.cos(math.radians(origin_lat))))
    return position_lat, position_lon

def global_to_ned(origin_lat, origin_lon, position_lat, position_lon, origin_n, origin_e):
    offset_lat = math.radians(lat - origin_lat)
    offset_lon = math.radians(lon - origin_lon)
    R = 6378137.0
    offset_n = R * offset_lat
    offset_e = R * offset_lon * math.cos(math.radians(origin_lat))
    position_n = origin_n + offset_n
    position_e = origin_e + offset_e
    return position_n, position_e