#!/usr/bin/env python
import rospy
import numpy as np
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import PoseStamped
import threading
import time
import random
import os
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

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

def linear_oscillation(t, base, target, period=10):
    omega_val = 2 * np.pi / period 
    frac = 0.5 * (1 + np.sin(omega_val * t))
    return base + frac * (target - base)

def circular_motion(t, center, radius=5.0, period=10):
    omega_val = 2 * np.pi / period 
    return center + radius * np.array([np.cos(omega_val * t), np.sin(omega_val * t)])

class ObstaclePublisher:
    def __init__(self):
        self.pub = rospy.Publisher('/fusion/dynamic_obstacles', Float32MultiArray, queue_size=10)
        self.obstacles = []
        self.home_position = np.array([150.0, 150.0])
        self.vehicle_position = self.home_position.copy()  # Initialize with home position
        self._vehicle_lock = threading.Lock()  # Lock for vehicle position
        
        # Setup visualization
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        self.ax.set_aspect('equal')
        self.ax.set_xlabel('East (m)')
        self.ax.set_ylabel('North (m)')
        self.ax.set_xlim(0, 300)
        self.ax.set_ylim(0, 300)
        self.ax.grid(True)
        
        # Initialize plot elements
        self.vehicle_dot, = self.ax.plot([], [], 'bo', markersize=10, label='Vehicle')
        self.home_dot, = self.ax.plot(self.home_position[1], self.home_position[0], 'gs', 
                                    markersize=10, label='Home')
        
        # Create obstacles
        for _ in range(10):
            pattern = random.choice([0, 1])
            with self._vehicle_lock:
                hp = self.home_position.copy()
                
            if pattern == 0:  # Linear motion
                base = self.random_point_in_radius(hp, 20, 60)
                target = self.random_point_in_radius(hp, 20, 60)
                obstacle = ObstacleSimulator(
                    lambda t, b=base, tgt=target: linear_oscillation(t, b, tgt, 100)
                )
            else:  # Circular motion
                center = self.random_point_in_radius(hp, 20, 60)
                radius = random.uniform(5, 10)
                obstacle = ObstacleSimulator(
                    lambda t, c=center, r=radius: circular_motion(t, c, r, 50)
                )
            self.obstacles.append(obstacle)
        
        # Create obstacle markers AFTER obstacle generation
        self.obstacle_dots = []
        for i in range(len(self.obstacles)):
            # Only label first obstacle for legend
            dot, = self.ax.plot([], [], 'ro', markersize=8, label='Obstacle')
            self.obstacle_dots.append(dot)
        
        self.ax.legend()
        plt.title('Dynamic Obstacles Visualization (NED Frame)')
        
        rospy.Subscriber('autopath_core/vehicle_position', PoseStamped, self.vehicle_pos_cb)
        rospy.Timer(rospy.Duration(0.1), self.publish_obstacles)
        rospy.on_shutdown(self.cleanup)

    def vehicle_pos_cb(self, msg):
        with self._vehicle_lock:
            self.vehicle_position = np.array([msg.pose.position.x, msg.pose.position.y])

    def random_point_in_radius(self, ref_point, min_r, max_r):
        angle = random.uniform(0, 2*np.pi)
        distance = random.uniform(min_r, max_r)
        return ref_point + distance * np.array([np.cos(angle), np.sin(angle)])

    def publish_obstacles(self, event):
        data = []
        with self._vehicle_lock:
            vehicle_pos = self.vehicle_position.copy()
        
        # Update vehicle position in visualization
        self.vehicle_dot.set_data(vehicle_pos[1], vehicle_pos[0])
        
        # Update obstacles
        for i, obs in enumerate(self.obstacles):
            pos = obs.get_position()
            dx = pos[0] - vehicle_pos[0]
            dy = pos[1] - vehicle_pos[1]
            r = np.sqrt(dx**2 + dy**2)
            theta = np.arctan2(dy, dx)
            data.append(r)
            data.append(np.rad2deg(theta))
            
            # Update obstacle marker
            self.obstacle_dots[i].set_data(pos[1], pos[0])
        
        # Update plot
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        
        # Publish obstacle data
        msg = Float32MultiArray(data=data)
        self.pub.publish(msg)

    def cleanup(self):
        for obs in self.obstacles:
            obs.stop()
        plt.close('all')

if __name__ == '__main__':
    try:
        rospy.init_node('dynamic_obstacle_simulation')
        log_level = rospy.get_param('~log_level', os.getenv('ROS_LOG_LEVEL', 'info')).upper()
        rospy.loginfo(f"Setting log level to {log_level}")
        import logging
        logging.getLogger('rosout').setLevel(getattr(logging, log_level))
            
        publisher = ObstaclePublisher()
        while not rospy.is_shutdown():
            plt.pause(0.1)
            
    except rospy.ROSInterruptException:
        pass