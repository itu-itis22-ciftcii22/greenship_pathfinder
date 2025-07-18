import copy

from domain import Domain

import random
import math

def pairedpon(base, heading, distance, width):
    c = math.cos(math.radians(heading))
    s = math.sin(math.radians(heading))
    xd = distance*c
    yd = distance*s
    c2 = -s
    s2 = c
    xd1 = (width/2)*c2
    yd1 = (width/2)*s2
    xd2 = -xd1
    yd2 = -yd1
    x1 = round(base[0]+xd+xd1)
    y1 = round(base[1]+yd+yd1)
    x2 = round(base[0]+xd+xd2)
    y2 = round(base[1]+yd+yd2)
    x = round(base[0]+xd)
    y = round(base[1]+yd)
    return (x,y), (x1,y1), (x2,y2)

def generate_paired_pons(base, heading, num_pairs):
    paired_pons = []
    for _ in range(num_pairs):
        # Generate random distances and a heading offset
        distance = 10
        new_base, pon1, pon2 = pairedpon(base, heading, distance, 5)

        paired_pons.append((pon1, pon2, new_base))

        # Move base forward (simulate path progression)
        base = new_base
        heading += random.randint(-45, 45)  # Random new heading

    return paired_pons


def generate_obstacles(center_point, radius=3, num_obstacles=5):
    """
    Generate random obstacle positions around a center point within a specified radius.

    Args:
        center_point: tuple (x, y) representing the center point coordinates
        radius: radius in meters around the center point (default 3)
        num_obstacles: number of obstacles to generate (default 5)

    Returns:
        List of tuples containing (x, y) coordinates of the obstacles
    """

    obstacles = []
    attempts = 0
    max_attempts = num_obstacles * 3  # Prevent infinite loops

    while len(obstacles) < num_obstacles and attempts < max_attempts:
        # Generate random angle and distance
        angle = random.uniform(0, 2 * math.pi)
        # Use sqrt for uniform distribution within circle
        distance = random.uniform(0, radius)

        # Calculate coordinates
        x = int(center_point[0] + distance * math.cos(angle))
        y = int(center_point[1] + distance * math.sin(angle))

        # Check if the position is valid (within grid and not at center)
        if ((x, y) != center_point and (x, y) not in obstacles):
            obstacles.append((x, y))

        attempts += 1

    return obstacles


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

        # Calculate angle from base to obstacle
        angle = math.atan2(dy, dx) % (2 * math.pi)
        heading_rad_mod = heading_rad % (2 * math.pi)

        # Calculate angular differences
        left_diff = (angle - left_rad) % (2 * math.pi)
        right_diff = (right_rad - angle) % (2 * math.pi)

        # Check if angle is within the 90-degree FOV
        if (left_diff <= math.radians(90)) and (right_diff <= math.radians(90)):
            obstacles_identified.append((x, y, territory[x][y]))

    return obstacles_identified

import multiprocessing as mp

def lidar_ray(args):
    angle, area, base = args
    x = base[0]
    while True:
        if x > len(area) - 1 or x < 0:
            break
        y = math.tan(math.radians(angle)) * (x - base[0]) + base[1]
        if 0 <= y <= len(area) - 1:
            if area[int(round(x))][int(round(y))] >= 1:
                return (round(x), round(y))  # Obstacle found
        if 270 < angle <= 360 or 0 <= angle <= 90:
            x += 0.01
        elif 90 < angle <= 270:
            x -= 0.01
        else:
            break
    return None  # No obstacle detected


def lidar(area, base, interval):
    start, end = interval
    if start <= end:
        angles = list(range(start, end + 1))
    else:
        angles = list(range(start, 360)) + list(range(0, end + 1))

    with mp.Pool() as pool:
        results = pool.map(lidar_ray, [(i, area, base) for i in angles])

    obs_list = list({pt for pt in results if pt is not None})
    return obs_list

if __name__ == '__main__':

    size = 200

    custom_contains = ("pon", "pairable_pon",  "target", "waypoint")
    custom_colors = {
        "pon": [0, 0, 1],
        "pairable_pon": [0, 0, 0],
        "target": [1, 0, 0],
        "waypoint": [0, 1, 0]
    }

    domain = Domain(size, size, custom_contains)

    base = (round(size/2), round(size/2))
    heading =  random.randint(0, 359)

    paired_pons = generate_paired_pons(base, heading, num_pairs=9)

    territory = [[0 for _ in range(size)] for _ in range(size)]
    for pon1, pon2, target in paired_pons:
        territory[pon1[0]][pon1[1]] = 2
        territory[pon2[0]][pon2[1]] = 2
        for obstacle in generate_obstacles(target, 3, 2):
            territory[obstacle[0]][obstacle[1]] = 1

    object_list = []

    targets = [(paired_pons[8][2][0], paired_pons[8][2][1])]
    for target in targets:
        domain.updateCell(domain.Coordinate(target[0], target[1]), contains=domain.containables[2])
    for obstacle in generate_obstacles(targets[0], 3, 2):
        territory[obstacle[0]][obstacle[1]] = 3

    while True:
        new_tars = False
        objects = lidar(territory.copy(), base, (heading+1, heading))
        for object in camera(territory, base, heading, objects):
            if object not in object_list:
                object_list.append(object)
                if object[2] == 1:
                    domain.updateMap(domain.Coordinate(object[0], object[1]), contains=domain.containables[0], blocked=True,
                                 value=25, radius=10, is_repellor=True)
                if object[2] == 2:
                    domain.updateMap(domain.Coordinate(object[0], object[1]), contains=domain.containables[1], blocked=True,
                                 value=10, radius=5, is_repellor=True)
                if object[2] == 3:
                    domain.updateCell(domain.Coordinate(object[0], object[1]), contains=domain.containables[2])
                    targets.append((object[0], object[1]))
                    new_tars = True

        corr_list = []
        for x in range(len(object_list)):
            if object_list[x][2] != 2:
                continue
            for y in range(x, len(object_list)):
                if object_list[y][2] != 2:
                    continue
                distancex = abs(object_list[x][0]-object_list[y][0])
                distancey = abs(object_list[x][1]-object_list[y][1])
                distance = round(math.sqrt(distancex**2+distancey**2))
                if 4 <= distance <= 6:
                    corr_row = round((object_list[x][0]+object_list[y][0])/2)
                    corr_col = round((object_list[x][1]+object_list[y][1])/2)
                    corr = (corr_row, corr_col)
                    if corr not in corr_list:
                        corr_list.append(corr)

        if new_tars:
            def calculate_distance(target):
                return math.sqrt((target[0] - base[0]) ** 2 + (target[1] - base[1]) ** 2)
            tars_current = sorted(targets, key=calculate_distance)

        pos = domain.Coordinate(int(base[0]), int(base[1]))
        wp_list = []
        for target in targets:
            next_pos = domain.Coordinate(int(target[0]), int(target[1]))
            wp_segment = domain.a_star_search(pos, next_pos, corr_list)
            for i in range(5, len(wp_segment), 5):
                wp_list.append(domain.Coordinate(wp_segment[i].row, wp_segment[i].col))
            wp_list.append(next_pos)
            pos = next_pos

        domain_copy = copy.deepcopy(domain)

        if wp_list is not None:
            for x in wp_list:
                if domain.map[x.row][x.col].contains == "empty":
                    domain_copy.updateCell(x, contains=domain.containables[3])

        domain_copy.plotMap(custom_colors)

        if len(wp_list) > 0:
            base = wp_list.pop(0)
            base = (base.row, base.col)
            if base == targets[0]:
                target_old = targets.pop(0)
                domain.updateCell(domain.Coordinate(target_old[0], target_old[1]), contains="empty")
            if len(wp_list) > 0:
                next = wp_list.pop(0)
                next = (next.row, next.col)
            else:
                next = base
            heading = int(math.degrees(math.atan2(next[1] - base[1], next[0] - base[0])))
        else:
            break