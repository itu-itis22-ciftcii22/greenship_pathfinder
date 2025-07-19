#!/usr/bin/env python
import rospy
import numpy as np
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import Point
import threading
import time
import random

class ObstacleSimulator:
    def __init__(self, path_func, base, target, update_interval=0.01):
        self.path_func = path_func
        self.base = base
        self.target = target
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
            new_pos = np.array(self.path_func(t, self.base, self.target))
            with self._lock:
                self.position = new_pos
            time.sleep(self.dt)

    def get_position(self):
        with self._lock:
            return self.position.copy()

    def stop(self):
        self._running = False

def linear_oscillation(t, base, target, period=0.01):
    frac = 0.5 * (1 + np.sin(2 * np.pi * t / period))
    return base + frac * (target - base)

def circular_motion(t, base, center, radius=5.0, omega=100):
    return center + radius * np.array([np.cos(omega * t), np.sin(omega * t)])

class ObstaclePublisher:
    def __init__(self):
        rospy.init_node('obstacle_publisher')
        self.pub = rospy.Publisher('/obstacles', Float32MultiArray, queue_size=10)
        self.obstacles = []
        self.home_position = np.array([150.0, 150.0])
        self.vehicle_position = np.zeros(2)
        rospy.Subscriber('/mavros/global_position/global', Point, self.vehicle_pos_cb)
        
        for _ in range(10):
            pattern = random.choice([0, 1])  # 0: linear, 1: circular
            
            if pattern == 0:  # Linear oscillation
                base = self.random_point_in_radius(50, 60)
                target = self.random_point_in_radius(60, 70)
                obstacle = ObstacleSimulator(
                    lambda t, b, tg: linear_oscillation(t, b, tg, 1),
                    base,
                    target
                )
            else:  # Circular motion
                center = self.random_point_in_radius(50, 70)
                radius = random.uniform(5, 10)
                obstacle = ObstacleSimulator(
                    lambda t, b, tg: circular_motion(t, b, center, radius, 1),
                    np.zeros(2),
                    np.zeros(2)
                )
            self.obstacles.append(obstacle)
        
        rospy.Timer(rospy.Duration(0.1), self.publish_obstacles)
        rospy.on_shutdown(self.cleanup)

    def vehicle_pos_cb(self, msg):
        vehicle_position_global = np.array([msg.latitude, msg.longitude])
        self.vehicle_position = np.array([msg.latitude, msg.longitude])

    def random_point_in_radius(self, min_radius, max_radius):
        angle = random.uniform(0, 2*np.pi)
        distance = random.uniform(min_radius, max_radius)
        return self.vehicle_position + distance * np.array([np.cos(angle), np.sin(angle)])

    def publish_obstacles(self, event):
        data = []
        for obs in self.obstacles:
            pos = obs.get_position()
            dx = pos[0] - self.vehicle_position[0]
            dy = pos[1] - self.vehicle_position[1]
            r = np.sqrt(dx**2 + dy**2)
            theta = np.arctan2(dy, dx)
            data.append(r)
            data.append(np.rad2deg(theta))
        
        msg = Float32MultiArray(data=data)
        self.pub.publish(msg)

    def cleanup(self):
        for obs in self.obstacles:
            obs.stop()

if __name__ == '__main__':
    publisher = ObstaclePublisher()
    rospy.spin()