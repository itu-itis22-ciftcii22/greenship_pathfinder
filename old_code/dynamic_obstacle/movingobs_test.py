#!/usr/bin/env python
import rospy
import numpy as np
from std_msgs.msg import Float32MultiArray
from domain import Domain
from utils import ned_to_global_scaled, global_scaled_to_ned
from vehicle import Vehicle
import signal
import sys

def clean_exit(signum, frame):
    global vehicle
    print("\nExiting and cleaning up...")
    if vehicle.connection:
        vehicle.connection.set_mode_manual()
        vehicle.disarm()
        vehicle.connection.close()
    sys.exit(0)


signal.signal(signal.SIGINT, clean_exit)
signal.signal(signal.SIGTERM, clean_exit)


# Global obstacle listesi
obstacle_list = []
current_position = np.zeros(2)

def obstacle_callback(msg):
    global obstacle_list
    global current_position
    obstacle_list = []

    data = msg.data  # [r1, θ1, r2, θ2, ...]
    for i in range(0, len(data), 2):
        r = data[i]
        theta_deg = data[i + 1]
        theta_rad = np.deg2rad(theta_deg)

        x = current_position[0] + r * np.cos(theta_rad)
        y = current_position[1] + r * np.sin(theta_rad)
        obstacle_list.append(np.array([x, y]))

def start_listener():
    rospy.Subscriber('/obstacles', Float32MultiArray, obstacle_callback)
    

if __name__ == '__main__':
    rospy.init_node('apf_planner')
    start_listener()
    rate = rospy.Rate(10)  # 10 Hz
    domain = Domain(150, 150)
    print("Started")

    while not rospy.is_shutdown():
        vehicle = Vehicle("/dev/ttyACM0", baud=57600)
        while True:
            try:
                print("Waiting for AUTO…")
                while True:
                    mode = vehicle.getMode()
                    if mode == "AUTO":
                        break
                    else:
                        print(mode)
                        continue
                print("AUTO engaged, starting")

                msg = None
                while msg is None:
                    msg = vehicle.getHome()
                home_global = np.array([msg.latitude, msg.longitude])
                home_ned = np.array([75.0, 75.0])
                home_coord = domain.Coordinate(75, 75)

                position_global = home_global
    
                missions_global = []
                for wp in vehicle.getWPList():
                    missions_global.append((wp.x, wp.y))
                missions_global = np.array(missions_global)
    
                missions_ned = np.array([
                    global_scaled_to_ned(home_global[0], home_global[1], mission_global[0], mission_global[1])
                    for mission_global in missions_global
                ]) + home_ned
    
                obstacles = obstacle_list
    
                step_size = 2
                mission_radius = 1
                first_move_sent = False
    
                for mission_ned in missions_ned:
                    if vehicle.getMode() != "AUTO":
                        break
                    mission_coord = domain.Coordinate(int(round(mission_ned[0].item())), int(round(mission_ned[1].item())))
                    while True:
                        if vehicle.getMode() != "AUTO":
                            break
                        msg = vehicle.getLocationRaw()
                        if msg is not None:
                            position_global = np.array([msg.lat, msg.lon])
                        position_ned = global_scaled_to_ned(home_global[0], home_global[1], position_global[0], position_global[1]) + home_ned
                        current_position = position_ned
                        position_coord = domain.Coordinate(int(round(position_ned[0].item())), int(round(position_ned[1].item())))
                        obstacle_ned = obstacles[0]
                        obstacle_coord = domain.Coordinate(int(round(obstacle_ned[0].item())), int(round(obstacle_ned[1].item())))
                        if np.linalg.norm(mission_ned - position_ned) > mission_radius:
                            wps_coord = domain.a_star_search(position_coord, mission_coord, moving_obstacles=[obstacle_coord])
                            for wp_coord in wps_coord:
                                print(wp_coord.row, wp_coord.col)
                            if wps_coord is not None:
                                wps_len = len(wps_coord)
                                if wps_len > step_size:
                                    first_lat, first_lon = ned_to_global_scaled(home_global[0], home_global[1],
                                                                              wps_coord[step_size].row-75, wps_coord[step_size].col-75)
                                    if wps_len > step_size*2:
                                        second_lat, second_lon = ned_to_global_scaled(home_global[0], home_global[1],
                                                                                      wps_coord[step_size*2].row-75, wps_coord[step_size*2].col-75)
                                    else:
                                        second_lat, second_lon = ned_to_global_scaled(home_global[0], home_global[1],
                                                                                    wps_coord[wps_len-1].row-75, wps_coord[wps_len-1].col-75)
                                    vehicle.assignWPs([(first_lat, first_lon), (second_lat, second_lon)])
                                else:
                                    first_lat, first_lon = ned_to_global_scaled(home_global[0], home_global[1],
                                                                              wps_coord[wps_len-1].row-75, wps_coord[wps_len-1].col-75)
                                    vehicle.assignWPs([(first_lat, first_lon)])
    
    
                            if not first_move_sent:
                                vehicle.arm()
                                first_move_sent = True
                                
                            rate.sleep()
                        else:
                            break
            except Exception as e:
                print(e)
