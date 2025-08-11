#!/usr/bin/env python
import rospy
import numpy as np
from std_msgs.msg import Float32MultiArray, Float32
from geometry_msgs.msg import PoseStamped
import threading
import time
import random
import os
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

class ObstacleSimulator:
    def __init__(self, path_func, update_interval=0.01):
        self.path_func = path_func
        self.dt = update_interval
        self.start_t = time.time()
        self.position = np.zeros(2)
        self._running = True
        self._lock = threading.Lock()
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def run(self):
        while self._running:
            t = time.time() - self.start_t
            new_pos = np.array(self.path_func(t))
            with self._lock:
                self.position = new_pos
            time.sleep(self.dt)

    def get_position(self):
        with self._lock:
            return self.position.copy()

    def stop(self):
        self._running = False

def linear_oscillation(t, center, amplitude=15.0, velocity=2.0):
    """
    Linear oscillation perpendicular to East-West axis (oscillates North-South)
    In NED frame: oscillation along North axis
    velocity: 2 m/s oscillation speed
    """
    # Calculate oscillation distance based on time and velocity
    distance = amplitude * np.sin(velocity * t / amplitude)
    # Oscillate North-South (perpendicular to East-West gate line)
    return center + np.array([distance, 0])

class Gate:
    def __init__(self, center_position, gate_id):
        self.center = center_position
        self.gate_id = gate_id
        self.width = 5.0  # 5m gate width
        
        # NED Frame: Red buoy on port (left/west), Green buoy on starboard (right/east) when heading north
        # Gate perpendicular to North-South axis (running East-West)
        self.red_buoy = center_position + np.array([0, -self.width/2])   # West side (port)
        self.green_buoy = center_position + np.array([0, self.width/2])  # East side (starboard)

class EnvironmentPublisher:
    def __init__(self):
        self.pub_dynamic_obstacles = rospy.Publisher('/fusion_output/dynamic_obstacles', Float32MultiArray, queue_size=10)
        self.pub_red_buoys = rospy.Publisher('/fusion_output/red_buoy', Float32MultiArray, queue_size=10)
        self.pub_green_buoys = rospy.Publisher('/fusion_output/green_buoy', Float32MultiArray, queue_size=10)
        self.dynamic_obstacles = []
        self.gates = []
        
        # Vehicle state (NED frame)
        self.home_position = np.array([0.0, 0.0])  # Start at origin (NED)
        self.vehicle_position = self.home_position.copy()
        self.vehicle_heading = 0.0  # compass degrees
        self._vehicle_lock = threading.Lock()
        
        # Setup visualization
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(10, 12))
        self.ax.set_aspect('equal')
        self.ax.set_xlabel('East (m)')
        self.ax.set_ylabel('North (m)')
        self.ax.set_xlim(-50, 50)
        self.ax.set_ylim(0, 300)
        self.ax.grid(True)
        
        # Initialize plot elements
        self.vehicle_dot, = self.ax.plot([], [], 'bo', markersize=12, label='Vehicle')
        self.home_dot, = self.ax.plot(self.home_position[1], self.home_position[0], 'gs', 
                                    markersize=10, label='Home')
        
        # Create gates along North-South axis (4 gates total)
        self.create_gates()
        
        # Create ONE oscillating obstacle between the median gates (between gates 2 and 3)
        self.create_dynamic_obstacle()
        
        # Create plot elements for gates and obstacles
        self.red_buoy_dots = []
        self.green_buoy_dots = []
        self.obstacle_dots = []
        
        for i, gate in enumerate(self.gates):
            # Red buoys
            red_dot, = self.ax.plot(gate.red_buoy[1], gate.red_buoy[0], 'ro', 
                                   markersize=10, label='Red Buoy' if i == 0 else "")
            self.red_buoy_dots.append(red_dot)
            
            # Green buoys  
            green_dot, = self.ax.plot(gate.green_buoy[1], gate.green_buoy[0], 'go', 
                                     markersize=10, label='Green Buoy' if i == 0 else "")
            self.green_buoy_dots.append(green_dot)
            
            # Gate number annotation
            self.ax.text(gate.center[1], gate.center[0] + 3, f'{gate.gate_id}', 
                        ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        # Obstacle markers (only one)
        self.obstacle_dot, = self.ax.plot([], [], 'rs', markersize=10, label='Dynamic Obstacle')
        
        self.ax.legend(loc='upper right')
        plt.title('Gate Environment with Dynamic Obstacles (NED Frame)')
        
        # ROS subscribers
        rospy.Subscriber('autopath_core/vehicle_relative_pose', PoseStamped, self.vehicle_pose_cb)
        
        # Publisher timer
        rospy.Timer(rospy.Duration(0.1), self.publish_obstacles)
        rospy.on_shutdown(self.cleanup)

    def create_gates(self):
        """Create 4 gates along East-West axis with random spacing between 20-80m"""
        current_east = 50 # Start 50m east of home
        gate_center_north = 0.0  # Center line (North=0 in NED)
        
        for gate_id in range(1, 5):  # Gates 1, 2, 3, 4
            gate_center = np.array([current_east, gate_center_north])
            gate = Gate(gate_center, gate_id)
            self.gates.append(gate)
            
            # Random distance to next gate (20-80m)
            if gate_id < 4:  # Don't add distance after last gate
                distance_to_next = random.uniform(20, 80)
                current_east += distance_to_next
                
        rospy.loginfo(f"Created {len(self.gates)} gates along North-South axis")

    def create_dynamic_obstacle(self):
        """Create one oscillating obstacle between gates 2 and 3 (median gates)"""
        if len(self.gates) >= 3:
            # Position obstacle between gates 2 and 3 (index 1 and 2)
            gate2_pos = self.gates[1].center
            gate3_pos = self.gates[2].center
            obstacle_center = (gate2_pos + gate3_pos) / 2.0
            
            # Add slight random offset
            obstacle_center += np.array([random.uniform(-5, 5), random.uniform(-3, 3)])
            
            obstacle = ObstacleSimulator(
                lambda t, center=obstacle_center: linear_oscillation(t, center)
            )
            self.dynamic_obstacles.append(obstacle)
            
        rospy.loginfo(f"Created 1 dynamic obstacle between median gates")

    def vehicle_pose_cb(self, msg):
        with self._vehicle_lock:
            self.vehicle_position = np.array([msg.pose.position.x, msg.pose.position.y]) + self.home_position
            rotation = Rotation.from_quat(msg.pose.orientation)  # Input [x, y, z, w]
            self.vehicle_heading = rotation.as_euler('z')


    def publish_obstacles(self, event):
        dynamic_obstacle_data = []
        red_buoy_data = []
        green_buoy_data = []
        
        with self._vehicle_lock:
            vehicle_pos = self.vehicle_position.copy()
            vehicle_heading = self.vehicle_heading
        
        # Update vehicle position in visualization
        self.vehicle_dot.set_data(vehicle_pos[1], vehicle_pos[0])
        
        # Calculate and publish buoy positions
        for gate in self.gates:
            # Red buoy data
            red_pos = gate.red_buoy
            dx_red = red_pos[0] - vehicle_pos[0]  # North difference
            dy_red = red_pos[1] - vehicle_pos[1]  # East difference
            r_red = np.sqrt(dx_red**2 + dy_red**2)
            theta_red_global = np.arctan2(dy_red, dx_red)
            theta_red_global_deg = np.rad2deg(theta_red_global)
            relative_bearing_red = theta_red_global_deg - vehicle_heading
            
            # Normalize red buoy bearing to [-180, 180]
            while relative_bearing_red > 180:
                relative_bearing_red -= 360
            while relative_bearing_red < -180:
                relative_bearing_red += 360
                
            red_buoy_data.append(r_red)
            red_buoy_data.append(relative_bearing_red)
            
            # Green buoy data
            green_pos = gate.green_buoy
            dx_green = green_pos[0] - vehicle_pos[0]  # North difference
            dy_green = green_pos[1] - vehicle_pos[1]  # East difference
            r_green = np.sqrt(dx_green**2 + dy_green**2)
            theta_green_global = np.arctan2(dy_green, dx_green)
            theta_green_global_deg = np.rad2deg(theta_green_global)
            relative_bearing_green = theta_green_global_deg - vehicle_heading
            
            # Normalize green buoy bearing to [-180, 180]
            while relative_bearing_green > 180:
                relative_bearing_green -= 360
            while relative_bearing_green < -180:
                relative_bearing_green += 360
                
            green_buoy_data.append(r_green)
            green_buoy_data.append(relative_bearing_green)
        
        # Update obstacles (only one obstacle)
        if len(self.dynamic_obstacles) > 0:
            obs = self.dynamic_obstacles[0]
            pos = obs.get_position()
            
            # Calculate relative position (NED frame)
            dx = pos[0] - vehicle_pos[0]  # North difference
            dy = pos[1] - vehicle_pos[1]  # East difference
            
            # Distance
            r = np.sqrt(dx**2 + dy**2)
            
            # Angle relative to north, then adjust for vehicle heading
            theta_global = np.arctan2(dy, dx)  # Angle from north in global frame
            theta_global_deg = np.rad2deg(theta_global)
            
            # Convert to relative bearing (vehicle heading = 0 degrees)
            relative_bearing = theta_global_deg - vehicle_heading
            
            # Normalize to [-180, 180]
            while relative_bearing > 180:
                relative_bearing -= 360
            while relative_bearing < -180:
                relative_bearing += 360
            
            dynamic_obstacle_data.append(r)
            dynamic_obstacle_data.append(relative_bearing)
            
            # Update obstacle marker
            self.obstacle_dot.set_data(pos[1], pos[0])
        
        # Update plot
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        
        # Publish all data
        obstacle_msg = Float32MultiArray(data=dynamic_obstacle_data)
        red_buoy_msg = Float32MultiArray(data=red_buoy_data)
        green_buoy_msg = Float32MultiArray(data=green_buoy_data)
        
        self.pub_dynamic_obstacles.publish(obstacle_msg)
        self.pub_red_buoys.publish(red_buoy_msg)
        self.pub_green_buoys.publish(green_buoy_msg)

    def cleanup(self):
        for obs in self.dynamic_obstacles:
            obs.stop()
        plt.close('all')

if __name__ == '__main__':
    try:
        rospy.init_node('EnvironmentAndFusionSimulation')
        log_level = rospy.get_param('~log_level', os.getenv('ROS_LOG_LEVEL', 'info')).upper()
        rospy.loginfo(f"Setting log level to {log_level}")
        import logging
        logging.getLogger('rosout').setLevel(getattr(logging, log_level))
            
        publisher = EnvironmentPublisher()
        rospy.loginfo("Gate environment simulation started")
        
        while not rospy.is_shutdown():
            plt.pause(0.1)
            
    except rospy.ROSInterruptException:
        pass