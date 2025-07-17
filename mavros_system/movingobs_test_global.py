#!/usr/bin/env python

#-------------------------------------------------------------------
import numpy as np
import heapq
class Domain:
    class Cell:
        def __init__(self):
            self.contains = "empty"
            self.blocked = False
            self.attractval = 0
            self.repelval = 0
            self.parent_i = 0
            self.parent_j = 0
    def __init__(self, nrow, ncol, containables=None):
        if not isinstance(nrow, int) or not isinstance(ncol, int):
            raise TypeError("Row and column must be integers.")
        if nrow <= 0 or ncol <= 0:
            raise ValueError("Row and column  must be positive numbers.")
        self.map = [[self.Cell() for _ in range(ncol)] for _ in range(nrow)]
        self.nrow = nrow
        self.ncol = ncol
        if containables is None:
            self.containables = ("pon", "target", "waypoint")
        else:
            self.containables = containables
    class Coordinate:
        def __init__(self, row, col):
            if not isinstance(row, int) or not isinstance(col, int):
                raise TypeError("Row and column must be integers.")
            self.row = row
            self.col = col
    class Cost:
        def __init__(self):
            self.f = float('inf')
            self.g = float('inf')
            self.h = 0
    def isValid(self, coord: Coordinate):
        return (coord.row >= 0) and (coord.row < self.nrow) and (coord.col >= 0) and (coord.col < self.ncol)
    def updateCell(self, coord: Coordinate, contains=None, blocked=None, value=None, is_attractor=False, is_repellor=False):
        if self.isValid(coord):
            if value is not None:
                if is_attractor:
                    self.map[coord.row][coord.col].attractval = value
                elif is_repellor:
                    self.map[coord.row][coord.col].repelval = value
            if contains is not None:
                self.map[coord.row][coord.col].contains = contains
            if blocked is not None:
                self.map[coord.row][coord.col].blocked = blocked
    def updateMap(self, coord: Coordinate, contains=None, blocked=None, value=None, radius=None, is_attractor=False, is_repellor=False):
        self.updateCell(coord, contains, blocked, value, is_attractor, is_repellor)
        if (value is not None) and (is_attractor or is_repellor):
            if radius is None:
                radius = 1
            i = coord.row
            j = coord.col
            for x in range(max(0, i - radius), min(self.nrow, i + radius + 1)):
                for y in range(max(0, j - radius), min(self.ncol, j + radius + 1)):
                    distance = ((i - x) ** 2 + (j - y) ** 2) ** 0.5
                    if distance <= radius:
                        new_value = np.exp(np.log(value)*(radius-distance)/(radius-1))
                        if (new_value > self.map[x][y].attractval and is_attractor) or (new_value > self.map[x][y].repelval and is_repellor):
                            self.updateCell(self.Coordinate(x, y), value=new_value, is_attractor=is_attractor, is_repellor=is_repellor)
    def trace_path(self, dest: Coordinate):
        path = []
        row = self.map[dest.row][dest.col].parent_i
        col = self.map[dest.row][dest.col].parent_j
        while not (self.map[row][col].parent_i == row and self.map[row][col].parent_j == col):
            path.append(self.Coordinate(row, col))
            temp_row = self.map[row][col].parent_i
            temp_col = self.map[row][col].parent_j
            row = temp_row
            col = temp_col
        # Do not add the source cell to the path
        # path.append(self.Coordinate(row, col))
        path.reverse()
        return path
    def a_star_search(self, src: Coordinate, dest: Coordinate, corridor=None, moving_obstacles=None):
        if not self.isValid(src) or not self.isValid(dest):
            print("Source or destination is invalid")
            print("Source:")
            print(src.row, src.col)
            print("Destination:")
            print(dest.row, dest.col)
            return None
        if src.row == dest.row and src.col == dest.col:
            print("We are already at the destination")
            return None
        closed_list = [[False for _ in range(self.ncol)] for _ in range(self.nrow)]
        cell_costs = [[self.Cost() for _ in range(self.ncol)] for _ in range(self.nrow)]
        i = src.row
        j = src.col
        cell_costs[i][j].f = 0
        cell_costs[i][j].g = 0
        cell_costs[i][j].h = 0
        self.map[i][j].parent_i = i
        self.map[i][j].parent_j = j
        open_list = []
        heapq.heappush(open_list, (0.0, i, j))
        while len(open_list) > 0:
            p = heapq.heappop(open_list)
            i = p[1]
            j = p[2]
            closed_list[i][j] = True
            directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
            for direction in directions:
                new_i = i + direction[0]
                new_j = j + direction[1]
                if self.isValid(self.Coordinate(new_i, new_j)) and not self.map[new_i][new_j].blocked and not closed_list[new_i][new_j]:
                    if new_i == dest.row and new_j == dest.col:
                        self.map[new_i][new_j].parent_i = i
                        self.map[new_i][new_j].parent_j = j
                        path = self.trace_path(self.Coordinate(new_i, new_j))
                        return path
                    else:
                        distance = (direction[0] ** 2 + direction[1] ** 2) ** 0.5
                        g_new = cell_costs[i][j].g + distance
                        h_new = ((new_i - dest.row) ** 2 + (new_j - dest.col) ** 2) ** 0.5
                        f_new = g_new + h_new
                        f_new += -self.map[new_i][new_j].attractval + self.map[new_i][new_j].repelval
                        if corridor is not None and len(corridor) > 0:
                            query_point = np.array([new_i, new_j])
                            lcf_penalty = float('inf')
                            for idx in range(len(corridor)):
                                d = np.linalg.norm(query_point - np.array([corridor[idx][0], corridor[idx][1]]))
                                if d < lcf_penalty:
                                    lcf_penalty = d
                            f_new += lcf_penalty ** 2
                        for moving_obstacle in moving_obstacles:
                            distance = ((new_i - moving_obstacle.row) ** 2 + (new_j - moving_obstacle.col) ** 2) ** 0.5
                            if distance <= 4:
                                penalty = np.exp(np.log(50) * (4 - distance) / (4 - 1))
                                f_new += penalty
                        if (cell_costs[new_i][new_j].f > f_new):
                            heapq.heappush(open_list, (f_new, new_i, new_j))
                            cell_costs[new_i][new_j].f = f_new
                            cell_costs[new_i][new_j].g = g_new
                            cell_costs[new_i][new_j].h = h_new
                            self.map[new_i][new_j].parent_i = i
                            self.map[new_i][new_j].parent_j = j
        path = self.trace_path(self.Coordinate(i, j))
        return path
#-------------------------------------------------------------------
#-------------------------------------------------------------------
import rospy
import threading
from mavros_msgs.srv import CommandBool, SetMode
from mavros_msgs.msg import State, HomePosition, WaypointList
from sensor_msgs.msg import NavSatFix
from geographic_msgs.msg import GeoPoint, GeoPoseStamped
class Vehicle_Mavros:
    def __init__(self, rate : rospy.Rate = rospy.Rate(10)):
        self.state = None
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
#-------------------------------------------------------------------

import math
import numpy as np
import rospy
from std_msgs.msg import Float32MultiArray
import signal
import sys
import time
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

def ned_to_global(origin_lat, origin_lon, offset_n, offset_e):
    R = 6378137.0  # Radius of Earth in meters
    latitude = origin_lat + (offset_n / R) * (180 / math.pi)
    longitude = origin_lon + (offset_e / (R * math.cos(math.radians(origin_lat))) * (180 / math.pi))
    return latitude, longitude

def global_to_ned(origin_lat, origin_lon, latitude, longitude):
    R = 6378137.0  # Radius of Earth in meters
    delta_lat = latitude - origin_lat
    delta_lon = longitude - origin_lon
    offset_n = (delta_lat * math.pi / 180) * R
    offset_e = (delta_lon * math.pi / 180) * R * math.cos(math.radians(origin_lat))
    return offset_n, offset_e

def clean_exit(signum, frame):
    global vehicle
    print("\nExiting and cleaning up...")
    if vehicle:
        try:
            vehicle.setMode("MANUAL")
            vehicle.disarm()
        except:
            pass
    sys.exit(0)

signal.signal(signal.SIGINT, clean_exit)
signal.signal(signal.SIGTERM, clean_exit)

# Global obstacle list
obstacle_list = []
current_position = np.zeros(2)

def obstacle_callback(msg):
    global obstacle_list, current_position
    obstacle_list = []
    data = msg.data  # [r1, θ1, r2, θ2, ...]
    
    for i in range(0, len(data), 2):
        r = data[i]
        theta_deg = data[i + 1]
        theta_rad = np.deg2rad(theta_deg)
        
        # Convert obstacle to global coordinates
        x = current_position[0] + r * np.cos(theta_rad)
        y = current_position[1] + r * np.sin(theta_rad)
        obstacle_list.append(np.array([x, y]))

def start_listener():
    rospy.Subscriber('/obstacles', Float32MultiArray, obstacle_callback)

if __name__ == '__main__':
    plt.ion()
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.set_xlim(0, 300)
    ax.set_ylim(0, 300)
    vehicle_dot, = ax.plot([], [], 'bo')
    obstacles_dot = []
    for _ in range(4):
        dot, = ax.plot([], [], 'ro')
        obstacles_dot.append(dot)

    rospy.init_node('apf_planner')
    start_listener()
    rate = rospy.Rate(0.5)
    domain = Domain(300, 300)
    print("Started")

    # rospy.Timer(rospy.Duration(secs=1/rate), self._publishers)

    # Initialize MAVROS vehicle
    vehicle = Vehicle_Mavros()

while not rospy.is_shutdown():
    try:
        print("[STATUS] Waiting for AUTO mode to be engaged...")
        while vehicle.getMode() != "AUTO":
            print(f"[WAITING] Current mode: {vehicle.getMode()} - Waiting for AUTO")
            rate.sleep()
        
        print("[SUCCESS] AUTO mode engaged")
        
        # Get home position
        print("[STATUS] Waiting for valid home position...")
        while not hasattr(vehicle, 'home_position_global') or \
              vehicle.home_position_global.latitude == 0:
            print("[WAITING] Home position not available yet...")
            rate.sleep()
            
        home_global = np.array([
            vehicle.home_position_global.latitude,
            vehicle.home_position_global.longitude
        ])
        print(f"[SUCCESS] Home position received: lat={home_global[0]}, lon={home_global[1]}")
        home_ned = np.array([150.0, 150.0])
        print(f"Home NED offset: {home_ned}")
        
        # Get mission waypoints
        print("[STATUS] Waiting for mission waypoints...")
        while len(vehicle.current_waypoints) == 0:
            print(f"[WAITING] Waypoints not available yet... (current count: {len(vehicle.current_waypoints)})")
            rate.sleep()
            
        missions_global = []
        for i, wp in enumerate(vehicle.current_waypoints):
            missions_global.append((wp.x_lat, wp.y_long))
            print(f"Waypoint {i+1}: lat={wp.x_lat}, lon={wp.y_long}")
            
        missions_global = np.array(missions_global)
        print(f"[SUCCESS] Received {len(missions_global)} waypoints")
        
        # Convert to NED coordinates relative to home
        missions_ned = np.array([
            global_to_ned(
                home_global[0], home_global[1],
                mission_global[0], mission_global[1]
            ) for mission_global in missions_global
        ]) + home_ned
        print(f"[CONVERSION] Converted waypoints to NED coordinates:\n{missions_ned}")

        # Switch to GUIDED mode for setpoint control
        print("[STATUS] Switching to GUIDED mode...")
        vehicle.setMode("GUIDED")
        print("[STATUS] Arming vehicle...")
        while not vehicle.state.armed:
            vehicle.arm()
            print("[WAITING] Waiting for vehicle to arm...")
            rate.sleep()
        print("[SUCCESS] GUIDED mode armed and ready")
        
        step_size = 3
        mission_radius = 3
        print(f"[PARAMS] Step size: {step_size}m, Mission radius: {mission_radius}m")
        
        for i, mission_ned in enumerate(missions_ned):
            if not vehicle.state.guided:
                print("[ERROR] Vehicle is not in GUIDED mode, aborting mission")
                break
            print(f"\n[WAYPOINT {i+1}/{len(missions_ned)}] Target NED: {mission_ned}")
            mission_coord = domain.Coordinate(
                int(round(mission_ned[0])),
                int(round(mission_ned[1]))
            )
            
            waypoint_start_time = time.time()
            attempt_count = 0
            
            while vehicle.state.guided and not rospy.is_shutdown():
                attempt_count += 1
                if attempt_count % 10 == 0:  # Print every 10 iterations
                    print(f"[WP {i+1} ATTEMPT {attempt_count}] Still trying to reach waypoint...")
                
                # Get current position
                position_global = np.array([
                    vehicle.vehicle_position_global.latitude,
                    vehicle.vehicle_position_global.longitude
                ])
                position_ned = np.array(global_to_ned(
                    home_global[0], home_global[1],
                    position_global[0], position_global[1]
                )) + home_ned
                current_position = position_ned

                vehicle_dot.set_data([position_ned[0]], [position_ned[1]])
                for i, obstacle_ned in enumerate(obstacle_list):
                    obstacles_dot[i].set_data([obstacle_ned[0]], [obstacle_ned[1]])
                fig.canvas.draw()
                fig.canvas.flush_events()
                
                position_coord = domain.Coordinate(
                    int(round(position_ned[0])),
                    int(round(position_ned[1]))
                )
                
                # Check if mission point reached
                distance = np.linalg.norm(mission_ned - position_ned)
                print(f"[POSITION] Current NED: {position_ned}, Distance to target: {distance:.2f}m")
                
                if distance <= mission_radius:
                    print(f"[SUCCESS] Reached waypoint {i+1} in {time.time()-waypoint_start_time:.1f}s")
                    break
                
                # Convert obstacles to domain coordinates
                obstacle_coords = []
                for obs in obstacle_list:
                    obstacle_coords.append(domain.Coordinate(
                        int(round(obs[0])),
                        int(round(obs[1]))
                    ))
                print(f"[OBSTACLES] {len(obstacle_coords)} obstacles detected")
                
                # Path planning with A*
                print(f"[PATH PLANNING] Calculating path from {position_coord.row, position_coord.col} to {mission_coord.row, mission_coord.col}")
                wps_coord = domain.a_star_search(
                    position_coord, 
                    mission_coord,
                    moving_obstacles=obstacle_coords
                )
                
                if wps_coord:
                    print(f"[PATH FOUND] {len(wps_coord)} waypoints in path")
                    # Select next waypoint in path
                    next_idx = min(step_size, len(wps_coord)-1)
                    next_coord = wps_coord[next_idx]
                    
                    # Convert to local NED coordinates (relative to home)
                    next_ned = np.array([
                        next_coord.row - home_ned[0],
                        next_coord.col - home_ned[1]
                    ])
                    offset_ned = next_ned - home_ned
                    next_global = np.array(ned_to_global(
                            home_global[0], home_global[1],
                            offset_ned[0], offset_ned[1]
                        ))
                    
                    # Send setpoint in GUIDED mode
                    print(f"[SETPOINT] Sending new target: {next_global} from: {position_global}")
                    vehicle.setTargetPositionGlobal(
                        next_global[0], 
                        next_global[1]
                    )
                else:
                    print("[WARNING] No valid path found! Waiting...")
                
                rate.sleep()

    except Exception as e:
        print(f"[ERROR] Exception occurred: {str(e)}")
        import traceback
        traceback.print_exc()
