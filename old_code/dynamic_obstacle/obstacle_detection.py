# obstacle_detection.py
import rospy
import numpy as np
from std_msgs.msg import Float32MultiArray

# Global obstacle listesi
obstacle_list = []

def obstacle_callback(msg):
    global obstacle_list
    obstacle_list = []

    data = msg.data  # [r1, θ1, r2, θ2, ...]
    for i in range(0, len(data), 2):
        r = data[i]
        theta_deg = data[i + 1]
        theta_rad = np.deg2rad(theta_deg)

        # Mevcut konum ayrıdan main_loop.py'den gelecek
        from main_loop import current_position

        x = current_position[0] + r * np.cos(theta_rad)
        y = current_position[1] + r * np.sin(theta_rad)
        obstacle_list.append(np.array([x, y]))

def start_listener():
    rospy.Subscriber('/obstacles', Float32MultiArray, obstacle_callback)
