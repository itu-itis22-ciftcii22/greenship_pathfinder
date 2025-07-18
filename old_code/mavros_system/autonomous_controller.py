import math
import numpy as np
import rospy
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import Point
import signal
import sys
import time
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

class APFPlannerNode:
    def __init__(self):
        # Initialize ROS node
        rospy.init_node('apf_planner')
        
        # Configuration parameters
        self.domain_size = (300, 300)
        self.home_ned = np.array([150.0, 150.0])
        self.step_size = 3.0
        self.mission_radius = 3.0
        self.control_rate = 0.5  # Hz
        
        # State variables
        self.obstacle_list = []
        self.current_position = np.zeros(2)
        self.missions_ned = []
        self.home_global = None
        
        # ROS setup
        self.obstacle_sub = rospy.Subscriber('/obstacles', Float32MultiArray, self.obstacle_callback)
        self.position_pub = rospy.Publisher('/apf_planner/current_position', Point, queue_size=2)
        self.control_timer = rospy.Timer(rospy.Duration(1/self.control_rate), self.publish_position)
        
        # Visualization setup
        self.setup_visualization()
        
        # External interfaces (would be implemented separately)
        self.domain = Domain(*self.domain_size)
        self.vehicle = Vehicle_Mavros()
        
        # Signal handling
        signal.signal(signal.SIGINT, self.clean_exit)
        signal.signal(signal.SIGTERM, self.clean_exit)

    def setup_visualization(self):
        """Initialize matplotlib visualization"""
        plt.ion()
        self.fig, self.ax = plt.subplots()
        self.ax.set_aspect('equal')
        self.ax.set_xlim(0, self.domain_size[0])
        self.ax.set_ylim(0, self.domain_size[1])
        self.vehicle_dot, = self.ax.plot([], [], 'bo', markersize=10)
        self.obstacle_dots = [self.ax.plot([], [], 'ro')[0] for _ in range(4)]

    def obstacle_callback(self, msg):
        """Process obstacle detection messages"""
        self.obstacle_list = []
        data = msg.data
        
        for i in range(0, len(data), 2):
            r = data[i]
            theta_deg = data[i + 1]
            x = self.current_position[0] + r * math.cos(math.radians(theta_deg))
            y = self.current_position[1] + r * math.sin(math.radians(theta_deg))
            self.obstacle_list.append(np.array([x, y]))
        
        self.update_visualization()

    def publish_position(self, event=None):
        """Publish current position"""
        msg = Point()
        msg.x = self.current_position[0]
        msg.y = self.current_position[1]
        msg.z = 0
        self.position_pub.publish(msg)

    def update_visualization(self):
        """Update visualization elements"""
        self.vehicle_dot.set_data([self.current_position[0]], [self.current_position[1]])
        
        # Update obstacle markers
        for i, dot in enumerate(self.obstacle_dots):
            if i < len(self.obstacle_list):
                dot.set_data([self.obstacle_list[i][0]], [self.obstacle_list[i][1]])
            else:
                dot.set_data([], [])
        
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def get_vehicle_position(self):
        """Get current vehicle position in NED coordinates"""
        position_global = np.array([
            self.vehicle.vehicle_position_global.latitude,
            self.vehicle.vehicle_position_global.longitude
        ])
        offset_n, offset_e = global_to_ned(
            self.home_global[0], self.home_global[1],
            position_global[0], position_global[1]
        )
        return np.array([offset_n, offset_e]) + self.home_ned

    def convert_waypoints(self):
        """Convert mission waypoints to NED coordinates"""
        self.missions_ned = []
        for wp in self.vehicle.current_waypoints:
            offset_n, offset_e = global_to_ned(
                self.home_global[0], self.home_global[1],
                wp.x_lat, wp.y_long
            )
            self.missions_ned.append(np.array([offset_n, offset_e]) + self.home_ned)

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

    def send_position_setpoint(self, ned_position):
        """Send position setpoint to vehicle"""
        # Convert to global coordinates
        local_offset = ned_position - self.home_ned
        lat, lon = ned_to_global(
            self.home_global[0], self.home_global[1],
            local_offset[0], local_offset[1]
        )
        
        # Send to vehicle
        self.vehicle.setTargetPositionGlobal(lat, lon)
        rospy.logdebug(f"Sent setpoint: LAT={lat:.6f}, LON={lon:.6f}")

    def clean_exit(self, signum=None, frame=None):
        """Clean up resources on exit"""
        rospy.loginfo("Exiting and cleaning up...")
        try:
            if self.vehicle:
                self.vehicle.setMode("MANUAL")
                self.vehicle.disarm()
        except Exception as e:
            rospy.logerr(f"Error during cleanup: {str(e)}")
        sys.exit(0)

# Coordinate conversion functions (could be moved to a utilities module)
def ned_to_global(origin_lat, origin_lon, offset_n, offset_e):
    R = 6378137.0  # Earth radius in meters
    lat = origin_lat + math.degrees(offset_n / R)
    lon = origin_lon + math.degrees(offset_e / (R * math.cos(math.radians(origin_lat))))
    return lat, lon

def global_to_ned(origin_lat, origin_lon, lat, lon):
    R = 6378137.0
    dlat = math.radians(lat - origin_lat)
    dlon = math.radians(lon - origin_lon)
    offset_n = R * dlat
    offset_e = R * dlon * math.cos(math.radians(origin_lat))
    return offset_n, offset_e

if __name__ == '__main__':
    try:
        node = APFPlannerNode()
        node.execute_mission()
    except rospy.ROSInterruptException:
        pass