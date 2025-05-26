import random
import math
import copy
from domain import Domain

def camera(territory, base, heading, obstacles):
    # Convert heading and field-of-view angles to radians
    heading_rad = math.radians(heading)
    left_rad = math.radians((heading - 45) % 360)
    right_rad = math.radians((heading + 45) % 360)

    # Get base coordinates
    x0, y0 = base

    obstacles_identified = []
    for obs in obstacles:
        x, y = obs
        dx = x - x0
        dy = y - y0

        distance = math.sqrt(dx**2+dy**2)

        # Calculate angle from base to obstacle
        angle = math.atan2(dy, dx) % (2 * math.pi)
        heading_rad_mod = heading_rad % (2 * math.pi)

        # Calculate angular differences
        left_diff = (angle - left_rad) % (2 * math.pi)
        right_diff = (right_rad - angle) % (2 * math.pi)

        # Check if angle is within the 120-degree FOV
        if distance <= 30 and (left_diff <= math.radians(90)) and (right_diff <= math.radians(90)):
            obstacles_identified.append((x, y, territory[x][y]))

    return obstacles_identified

import multiprocessing as mp

def lidar_ray(args):
    angle, area, nrow, ncol, base = args
    x = base[0]
    while True:
        if x > nrow - 1 or x < 0:
            break
        y = math.tan(math.radians(angle)) * (x - base[0]) + base[1]
        if 0 <= y <= ncol - 1:
            if area[int(round(x))][int(round(y))] >= 1:
                return (round(x), round(y))  # Obstacle found
        if 270 < angle <= 360 or 0 <= angle <= 90:
            x += 0.01
        elif 90 < angle <= 270:
            x -= 0.01
        else:
            break
    return None  # No obstacle detected


def lidar(area, nrow, ncol, base, interval):
    start, end = interval
    if start <= end:
        angles = list(range(start, end + 1))
    else:
        angles = list(range(start, 360)) + list(range(0, end + 1))

    with mp.Pool() as pool:
        results = pool.map(lidar_ray, [(i, area, nrow, ncol, base) for i in angles])

    obs_list = list({pt for pt in results if pt is not None})
    return obs_list

if __name__ == '__main__':
    size = (180, 30)

    custom_contains = ("pon", "pairable_pon",  "target", "waypoint")
    custom_colors = {
        "pon": [0, 0, 0],
        "pairable_pon": [1, 1, 0],
        "target": [0, 0, 1],
        "waypoint": [0, 1, 0]
    }

    domain = Domain(size[0], size[1], custom_contains)

    base = (0,0)
    heading = 0

    territory = [[0 for _ in range(size[1])] for _ in range(size[0])]

    targets = []

    territory[15][5] = 2
    territory[15][15] = 2
    targets.append((15, 10))

    territory[30][15] = 2
    territory[30][25] = 2
    targets.append((30, 20))

    territory[45][8] = 2
    territory[45][18] = 2
    targets.append((45, 13))

    territory[60][12] = 2
    territory[60][22] = 2
    targets.append((60, 17))

    territory[75][11] = 2
    territory[75][21] = 2

    territory[90][10] = 2
    territory[90][20] = 2
    territory[110][10] = 2
    territory[110][20] = 2
    territory[130][10] = 2
    territory[130][20] = 2
    territory[150][10] = 2
    territory[150][20] = 2
    targets.append((150, 15))

    territory[80][17] = 1
    territory[85][16] = 1
    territory[96][17] = 1
    territory[108][18] = 1
    territory[110][13] = 1
    territory[122][15] = 1
    territory[131][12] = 1
    territory[140][14] = 1

    territory[165][15] = 3
    territory[172][18] = 3
    territory[172][12] = 3

    for target in targets:
        domain.updateCell(domain.Coordinate(target[0], target[1]), contains=domain.containables[2])

    object_list = []
    targets_old = []

    max_value = 50
    parkour = 1

    while True:
        new_tars = False
        objects = lidar(territory.copy(), size[0], size[1], base, (heading+1, heading))
        for object in camera(territory, base, heading, objects):
            if object not in object_list:
                object_list.append(object)
                if object[2] == 1:
                    domain.updateMap(domain.Coordinate(object[0], object[1]), contains=domain.containables[0], blocked=True,
                                 value=max_value, radius=5, is_repellor=True)
                    if parkour == 1:
                        parkour = 2
                if object[2] == 2:
                    domain.updateMap(domain.Coordinate(object[0], object[1]), contains=domain.containables[1], blocked=True,
                                 value=max_value, radius=10, is_repellor=True)
                if object[2] == 3:
                    if parkour == 2:
                        domain.updateCell(domain.Coordinate(object[0], object[1]), contains=domain.containables[2])
                        targets.append((object[0], object[1]))
                        new_tars = True
                        parkour = 3

        corr_list = []
        if parkour == 2:
            for x in range(len(object_list)):
                if object_list[x][2] != 2:
                    continue
                for y in range(x, len(object_list)):
                    if object_list[y][2] != 2:
                        continue
                    distancex = abs(object_list[x][0]-object_list[y][0])
                    distancey = abs(object_list[x][1]-object_list[y][1])
                    distance = round(math.sqrt(distancex**2+distancey**2))
                    if 8 <= distance <= 12:
                        corr_row = round((object_list[x][0]+object_list[y][0])/2)
                        corr_col = round((object_list[x][1]+object_list[y][1])/2)
                        corr = (corr_row, corr_col)
                        if corr not in corr_list:
                            corr_list.append(corr)
                            if parkour == 1 and corr not in targets and corr not in targets_old:
                                targets.append(corr)
                                new_tars = True

        if new_tars:
            def calculate_distance(target):
                return math.sqrt((target[0] - base[0]) ** 2 + (target[1] - base[1]) ** 2)
            targets = sorted(targets, key=calculate_distance)

        pos = domain.Coordinate(int(base[0]), int(base[1]))
        wp_list = []
        for target in targets:
            next_pos = domain.Coordinate(int(target[0]), int(target[1]))
            wp_segment = domain.a_star_search(pos, next_pos, corr_list)
            if len(wp_segment) > 0:
                for i in range(5, len(wp_segment), 5):
                    wp_list.append(domain.Coordinate(wp_segment[i].row, wp_segment[i].col))
            wp_list.append(next_pos)
            pos = next_pos

        domain_copy = copy.deepcopy(domain)

        if wp_list is not None:
            for x in wp_list:
                if domain.isValid(x) and domain.map[x.row][x.col].contains == "empty":
                    domain_copy.updateCell(x, contains=domain.containables[3])

        domain_copy.plotMap(max_value, custom_colors)

        if len(wp_list) > 0:
            base = wp_list.pop(0)
            base = (base.row, base.col)
            if base == targets[0]:
                target_old = targets.pop(0)
                targets_old.append(target_old)
                domain.updateCell(domain.Coordinate(target_old[0], target_old[1]), contains="empty")
            if len(wp_list) > 0:
                next = wp_list.pop(0)
                next = (next.row, next.col)
            else:
                next = base
            heading = int(math.degrees(math.atan2(next[1] - base[1], next[0] - base[0])))
        else:
            break