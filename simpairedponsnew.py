from domain import Domain

import random
import matplotlib.pyplot as plt
import math
import numpy as np



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
        distance = random.randint(15, 20)
        new_base, pon1, pon2 = pairedpon(base, heading, distance, 10)

        paired_pons.append((pon1, pon2, new_base))

        # Move base forward (simulate path progression)
        base = new_base
        heading += random.randint(-45, 45)  # Random new heading

    return paired_pons

def lidar(area, base, interval):
    obs_list = []

    # Normalize the interval to handle wrapping cases
    start, end = interval
    if start <= end:
        angles = range(start, end + 1)  # Regular case
    else:
        angles = list(range(start, 360)) + list(range(0, end + 1))  # Wrapping case
    for i in angles:
        x = base[0]
        while True:
            if x > len(area)-1 or x < 0:
                break
            y = math.tan(math.radians(i)) * (x - base[0]) + base[1]
            if 0 <= y <= len(area)-1:
                if area[int(round(x))][int(round(y))] == 1:  # Obstacle found
                    if (round(x), round(y)) in obs_list:
                        pass
                    else:
                        obs_list.append((round(x), round(y)))
                    break
                else:
                    area[int(round(x))][int(round(y))] = -1  # Mark as scanned
            if 270 < i <= 360 or 0 <= i <= 90:
                x += 10 ** -2
            elif 90 < i <= 270:
                x -= 10 ** -2
            else:
                pass
    return obs_list

# Hermite spline function
def hermite_spline_function(p0, p1, m0, m1, t):
    h00 = 2 * t ** 3 - 3 * t ** 2 + 1  # Basis polynomial 1
    h10 = t ** 3 - 2 * t ** 2 + t  # Basis polynomial 2
    h01 = -2 * t ** 3 + 3 * t ** 2  # Basis polynomial 3
    h11 = t ** 3 - t ** 2  # Basis polynomial 4
    return h00 * p0 + h10 * m0 + h01 * p1 + h11 * m1

def hermite_spline(points, heading):
    # Robot's initial angle (in radians) and magnitude for starting tangent
    theta_0 = heading * np.pi/180  # 45 degrees
    magnitude = 6  # Tangent magnitude factor

    # Compute tangents
    tangents = []

    # Initial tangent (based on starting angle)
    t0 = magnitude * np.array([np.cos(theta_0), np.sin(theta_0)])
    tangents.append(t0)

    # Intermediate tangents (smooth transitions between checkpoints)
    for i in range(1, len(points) - 1):
        xdiff = points[i + 1][0] - points[i - 1][0]
        ydiff = points[i + 1][1] - points[i - 1][1]
        distance = np.sqrt(xdiff ** 2 + ydiff ** 2)
        new_tangent = magnitude * np.array([xdiff / distance, ydiff / distance])  # Incoming tangent
        tangents.append(new_tangent)

    # Final tangent (based on home vector)
    xdiff = points[0][0] - points[-1][0]
    ydiff = points[0][1] - points[-1][1]
    distance = np.sqrt(xdiff ** 2 + ydiff ** 2)
    tf = magnitude * np.array([xdiff / distance, ydiff / distance])
    tangents.append(tf)

    # Generate and plot spline
    t_values = np.linspace(0, 1, 100)
    spline_points = []

    for i in range(len(points) - 1):
        p0 = points[i]
        p1 = points[i + 1]
        m0 = tangents[i]
        m1 = tangents[i + 1]
        spline_points.extend([hermite_spline_function(p0, p1, m0, m1, t) for t in t_values])


    spline_points = np.array(spline_points)

    # Plot the splines and tangents
    if spline_points is not None:
        plt.plot(spline_points[:, 0], spline_points[:, 1], label="Hermite Spline Path")
    plt.scatter(*zip(*points), color="red", label="Checkpoints")
    for i, p in enumerate(points):
        plt.quiver(p[0], p[1], tangents[i][0], tangents[i][1], angles="xy", scale_units="xy", label=f"Tangent {i}")
    plt.legend()
    plt.title("Hermite Spline Path with Tangents")
    plt.show()

    return spline_points

if __name__ == '__main__':

    size = 300

    base = (round(size/2), round(size/2))
    heading =  random.randint(0, 359)
    paired_pons = generate_paired_pons(base, heading, num_pairs=8)


    obs_list = []
    targets_current = []
    targets_distances = {}
    targets_old = []

    while True:
        area = [[0 for _ in range(size)] for _ in range(size)]
        for pon1, pon2, target in paired_pons:
            area[pon1[0]][pon1[1]] = 1
            area[pon2[0]][pon2[1]] = 1

        for i in range(4):
            obstacles = lidar(area, base, ((heading-45)%360, (heading+45)%360))
            for obs in obstacles:
                if obs not in obs_list:
                    obs_list.append(obs)
            heading += 90

        for x in range(len(obs_list)):
            for y in range(x, len(obs_list)):
                distancex = abs(obs_list[x][0]-obs_list[y][0])
                distancey = abs(obs_list[x][1]-obs_list[y][1])
                distance = round(math.sqrt(distancex**2+distancey**2))
                if 9 <= distance <= 11:
                    targetrow = round((obs_list[x][0]+obs_list[y][0])/2)
                    targetcol = round((obs_list[x][1]+obs_list[y][1])/2)
                    target_distancex = abs(targetrow-base[0])
                    target_distancey = abs(targetcol-base[1])
                    target_distance = round(math.sqrt(target_distancex**2+target_distancey**2))
                    target = (targetrow, targetcol)
                    if target not in targets_current and target not in targets_old:
                        targets_current.append(target)
                        targets_distances.update({target: target_distance})

        current = base
        targets_current_copy = targets_current.copy()
        targets_current_sorted = []
        for _ in range(len(targets_current)):
            index_closest = 0
            distance_closest = float('inf')
            for x in range(len(targets_current_copy)):
                distancex = targets_current_copy[x][0] - current[0]
                distancey = targets_current_copy[x][1] - current[1]
                distance = round(math.sqrt(distancex ** 2 + distancey ** 2))
                if distance < distance_closest:
                    index_closest = x
                    distance_closest = distance
            current = targets_current_copy.pop(index_closest)
            targets_current_sorted.append(current)

        targets_current = targets_current_sorted

        sorted_targets = sorted(targets_current, key=lambda x: targets_distances[x])
        domain = Domain(size, size)

        for obs in obs_list:
            domain.updateMap(domain.Coordinate(obs[0], obs[1]), contains=domain.containables[0], blocked=True, value=25, radius=10, is_repellor=True)

        for target in targets_current:
            domain.updateMap(domain.Coordinate(int(target[0]), int(target[1])), contains=domain.containables[1],  value=25, radius=10, is_attractor=True)

        spline_points = hermite_spline(np.array([base] + targets_current), heading)
        spline_points_rounded = []
        for point in spline_points:
            point = (round(point[0]), round(point[1]))
            if point not in spline_points_rounded:
                domain.updateMap(domain.Coordinate(point[0], point[1]), value=50, radius=2, is_attractor=True)
                spline_points_rounded.append(point)

        # Prepare visualization grid
        area_np = np.array(area).transpose()

        # Plot the grid
        plt.figure(figsize=(10, 10))
        plt.imshow(area_np, cmap='gray', vmin=-1, vmax=1, origin='lower')

        # Highlight scanned cells
        scanned_cells = np.argwhere(area_np == -1)
        plt.scatter(scanned_cells[:, 1], scanned_cells[:, 0], c='blue', s=10, label='Scanned Cells')

        # Highlight obstacles
        obstacle_cells = np.argwhere(area_np == 1)
        plt.scatter(obstacle_cells[:, 1], obstacle_cells[:, 0], c='red', s=50, label='Obstacles')

        # Highlight lidar base position
        plt.scatter(base[0], base[1], c='green', s=100, label='Lidar Base')

        # Add gridlines
        plt.grid(visible=True, which='both', color='black', linewidth=0.5, linestyle='--')

        # Labels and legend
        plt.title("LiDAR Scan Visualization", fontsize=16)
        plt.xlabel("X Coordinate")
        plt.ylabel("Y Coordinate")
        plt.legend()

        # Display the plot
        plt.show()

        pos = domain.Coordinate(int(base[0]), int(base[1]))
        wp_list = []
        for target in targets_current:
            next_pos = domain.Coordinate(int(target[0]), int(target[1]))
            wp_segment = domain.a_star_search(pos, next_pos)
            if wp_segment is not None:
                for waypoint in wp_segment:
                    if waypoint not in wp_list:
                        wp_list.append(waypoint)
            pos = next_pos

        if wp_list is not None:
            for x in wp_list:
                if domain.map[x.row][x.col].contains == "empty":
                    domain.updateCell(x, contains=domain.containables[2])

        domain.plotMap()

        cond = input("Press Enter to continue, 'end' to terminate...")

        if cond == "":  # Enter key
            base = targets_current.pop(0)
            targets_old.append(base)
        elif cond.lower() == "end":  # User types 'end' to terminate
            break
