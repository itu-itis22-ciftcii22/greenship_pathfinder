import numpy as np
from apf_core import attractive_force, repulsive_force
from utils import ned_to_global_scaled, global_scaled_to_ned
from vehicle import Vehicle
from obstacle import ObstacleSimulator, circular_path
import time


if __name__ == '__main__':

    obs_sim = ObstacleSimulator(lambda t: circular_path(t, 10, 0.2))
    obs_sim.start()

    vehicle = Vehicle("udpin:localhost:14550")
    print("Waiting for AUTO…")
    vehicle.waitAuto()
    print("AUTO engaged, starting")

    msg = vehicle.getHome()
    home_global = np.array([msg.latitude, msg.longitude])
    home_ned = np.array([0.0, 0.0])

    missions_global = []
    for wp in vehicle.getWPList():
        missions_global.append((wp.x, wp.y))
    missions_global = np.array(missions_global)

    missions_ned = np.array([
        global_scaled_to_ned(home_global[0], home_global[1], mission_global[0], mission_global[1])
        for mission_global in missions_global
    ]) + home_ned


    step_size = 3
    mission_radius = 0.5
    first_move_sent = False

    for mission_ned in missions_ned:
        msg = vehicle.getLocationGlobal()
        position_global = np.array([msg.lat, msg.lon])
        position_ned = global_scaled_to_ned(home_global[0], home_global[1], position_global[0], position_global[1]) + home_ned
        obstacle_ned = obs_sim.position
        if np.linalg.norm(mission_ned - position_ned) > mission_radius:
            f_att = attractive_force(position_ned, mission_ned)
            f_rep = repulsive_force(position_ned, obstacle_ned)
            total_force = f_att + f_rep

            move = step_size * total_force / np.linalg.norm(total_force)

            next_position_ned = position_ned + move

            next_lat, next_lon = ned_to_global_scaled(home_global[0], home_global[1], next_position_ned[0], next_position_ned[1])

            vehicle.assignWPs([(next_lat, next_lon)])

            if not first_move_sent:
                vehicle.arm()
                first_move_sent = True
                time.sleep(0.5)

            time.sleep(0.2)
        else:
            continue

    obs_sim.stop()
    obs_sim.join()
    vehicle.connection.close()
