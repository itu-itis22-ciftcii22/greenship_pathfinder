# main_loop.py
import rospy
import numpy as np
from apf_core import attractive_force, repulsive_force
from obstacle_detection import obstacle_list, start_listener

current_position = np.array([0.0, 0.0]) #burası mavlinkten gelecek koordinata göre güncellenecek
goal = np.array([10.0, 10.0]) # bu bilgi mavlinkten gelecek
step_size = 0.1
goal_thresh = 0.5

def apf_navigation():
    global current_position

    rate = rospy.Rate(10)  # 10 Hz
    path = []

    while not rospy.is_shutdown():
        f_att = attractive_force(current_position, goal)
        f_rep = repulsive_force(current_position, obstacle_list)
        total_force = f_att + f_rep

        move = step_size * total_force / np.linalg.norm(total_force)
        current_position += move
        path.append(current_position.copy())

        rospy.loginfo(f"Current position: {current_position}")
        if np.linalg.norm(goal - current_position) < goal_thresh:
            rospy.loginfo("Reached goal!")
            break

        rate.sleep()

if __name__ == '__main__':
    rospy.init_node('apf_planner')
    start_listener()
    apf_navigation()
