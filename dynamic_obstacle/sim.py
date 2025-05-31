import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from domain import Domain
from utils import ned_to_global_scaled, global_scaled_to_ned
from vehicle import Vehicle
from obstacle import ObstacleSimulator, circular_between

import signal
import sys

def clean_exit(signum, frame):
    global vehicle
    print("\nExiting and cleaning up...")
    if vehicle.connection:
        vehicle.disarm()
        vehicle.connection.close()
    sys.exit(0)


signal.signal(signal.SIGINT, clean_exit)
signal.signal(signal.SIGTERM, clean_exit)


if __name__ == '__main__':

    plt.ion()
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.set_xlim(0, 150)
    ax.set_ylim(0, 150)
    vehicle_dot, = ax.plot([], [], 'bo')
    obstacle_dot, = ax.plot([], [], 'ro')

    domain = Domain(150, 150)

    vehicle = Vehicle("udpin:localhost:14550")
    while True:
        print("Waiting for AUTO…")
        while True:
            mode = vehicle.getMode()
            if mode == "AUTO":
                break
            else:
                print(mode)
                continue
        print("AUTO engaged, starting")

        msg = vehicle.getHome()
        home_global = np.array([msg.latitude, msg.longitude])
        home_ned = np.array([75.0, 75.0])
        home_coord = domain.Coordinate(75, 75)

        missions_global = []
        for wp in vehicle.getWPList():
            missions_global.append((wp.x, wp.y))
        missions_global = np.array(missions_global)

        missions_ned = np.array([
            global_scaled_to_ned(home_global[0], home_global[1], mission_global[0], mission_global[1])
            for mission_global in missions_global
        ]) + home_ned

        obstacles = []
        for mission_ned in missions_ned:
            obstacle = ObstacleSimulator(circular_between, home_ned, mission_ned)
            obstacle.start()
            obstacles.append(obstacle)

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
                msg = vehicle.getLocationGlobal()
                position_global = np.array([msg.lat, msg.lon])
                position_ned = global_scaled_to_ned(home_global[0], home_global[1], position_global[0], position_global[1]) + home_ned
                position_coord = domain.Coordinate(int(round(position_ned[0].item())), int(round(position_ned[1].item())))
                obstacle_ned = obstacles[0].get_position()
                obstacle_coord = domain.Coordinate(int(round(obstacle_ned[0].item())), int(round(obstacle_ned[1].item())))
                vehicle_dot.set_data([position_ned[0]], [position_ned[1]])
                obstacle_dot.set_data([obstacle_ned[0]], [obstacle_ned[1]])
                fig.canvas.draw()
                fig.canvas.flush_events()
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
                else:
                    break

        for obstacle in obstacles:
            obstacle.stop()
            obstacle.join()
        vehicle.connection.set_mode_manual()
        vehicle.disarm()
