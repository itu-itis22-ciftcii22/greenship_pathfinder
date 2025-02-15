import numpy as np

from domain import Domain
#from vehicle import Vehicle

import signal
import sys
import random
import matplotlib.pyplot as plt

"""def clean_exit(signum, frame):
    global vehicle
    print("\nExiting and cleaning up...")
    if vehicle.connection:
        vehicle.connection.close()
    sys.exit(0)


signal.signal(signal.SIGINT, clean_exit)
signal.signal(signal.SIGTERM, clean_exit)"""


def generate_paired_pons(base, heading, num_pairs):
    """
    Generates paired left and right pons defining a path corridor and calculates subwaypoints.

    :param base: Tuple (x, y) as the starting base point.
    :param heading: Initial heading angle in degrees.
    :param num_pairs: Number of left/right pon pairs to generate.
    :param min_distance: Minimum distance between base and pons (in grid units).
    :param max_distance: Maximum distance for pon generation.
    :return: List of paired pons [(left_pon, right_pon, subwaypoint), ...].
    """
    paired_pons = []
    for _ in range(num_pairs):
        # Generate random distances and a heading offset
        distance = random.randint(10, 15)
        angle = random.randint(-45, 45)
        left_angle = heading + angle - 15  # Angle for left pon
        right_angle = heading + angle + 15  # Angle for right pon

        # Generate left and right pons relative to the base
        left_pon = Domain.findRelativePoint(distance, left_angle, heading, base[0], base[1])
        right_pon = Domain.findRelativePoint(distance, right_angle, heading, base[0], base[1])

        # Calculate subwaypoint (midpoint of the pon pair)
        subtarget = ((left_pon[0] + right_pon[0]) // 2, (left_pon[1] + right_pon[1]) // 2)
        paired_pons.append((left_pon, right_pon, subtarget))

        # Move base forward (simulate path progression)
        base = subtarget
        heading += random.randint(-15, 15)  # Random new heading

    return paired_pons

import math


def ned_to_global_scaled(origin_lat, origin_lon, offset_n, offset_e):
    """
    Converts NED frame (North-East-Down) coordinates to global latitude, longitude, and altitude.

    :param origin_lat: Origin latitude in degrees
    :param origin_lon: Origin longitude in degrees
    :param origin_alt: Origin altitude in meters
    :param offset_n: Offset in the North direction (meters)
    :param offset_e: Offset in the East direction (meters)
    :param offset_d: Offset in the Down direction (meters)
    :return: (latitude_scaled, longitude_scaled, altitude) where lat/lon are in degrees * 10^7
    """
    R = 6378137.0  # Radius of Earth in meters
    new_lat = origin_lat + int(((offset_n / R) * (180 / math.pi)) * 1e7)
    new_lon = origin_lon + int(((offset_e / (R * math.cos(math.pi * origin_lat / 180))) * (180 / math.pi)) * 1e7)

    # Scale latitude and longitude by 10^7
    latitude_scaled = int(new_lat)
    longitude_scaled = int(new_lon)

    return latitude_scaled, longitude_scaled

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
    plt.plot(spline_points[:, 0], spline_points[:, 1], label="Hermite Spline Path")
    plt.scatter(*zip(*points), color="red", label="Checkpoints")
    for i, p in enumerate(points):
        plt.quiver(p[0], p[1], tangents[i][0], tangents[i][1], angles="xy", scale_units="xy", label=f"Tangent {i}")
    plt.legend()
    plt.title("Hermite Spline Path with Tangents")
    plt.show()

    return spline_points

if __name__ == '__main__':
    """vehicle = Vehicle("udpin:localhost:14550")  # Aracın bağlantı noktası
    vehicle.arm()
    vehicle.connection.set_mode_loiter()"""

    domain = Domain(300, 300)

    base = (150, 150)

    heading =  random.randint(0, 359) #vehicle.getCompass().heading
    if heading is None:
        sys.exit(1)

    pon_pairs = generate_paired_pons(base, heading, num_pairs=10)
    left_pons = []
    right_pons = []
    targets = []
    for ponleft, ponright, subtarget in pon_pairs:
        left_pons.append(ponleft)
        right_pons.append(ponright)
        targets.append(subtarget)

    for lpon, rpon in zip(left_pons, right_pons):
        domain.updateMap(lpon, contains=domain.containables[0], blocked=True, value=25, radius=15, is_repellor=True)
        domain.updateMap(rpon, contains=domain.containables[0], blocked=True, value=25, radius=15, is_repellor=True)

    targetx = -1
    targety = -1
    while not domain.isValid(targetx, targety) or domain.map[targetx][targety].blocked:
        random.seed()
        targetx = int(random.random() * 300)
        random.seed()
        targety = int(random.random() * 300)
    target = (targetx, targety)
    targets.append(target)

    for target in targets:
        domain.updateMap(target, value=25, radius=15, contains=domain.containables[1], is_attractor=True)

    spline_points = hermite_spline(np.array([base] + targets), heading)

    for point in spline_points:
        locat = (int(point[0]), int(point[1]))
        domain.updateMap(locat, value=50, radius=3, is_attractor=True)

    WPList = []
    for target in targets:
        wp = domain.a_star_search(base, target)
        if wp is not None:
            WPList.append(wp)
        base = target

    if WPList is None:
        sys.exit(1)

    flat_wp_list = []

    flat_wp_list = []
    for segment in WPList:
        for waypoint in segment:
            if waypoint not in flat_wp_list:
                flat_wp_list.append(waypoint)

    for x in flat_wp_list:
        if domain.map[x[0]][x[1]].contains == "empty":
            domain.updateCell(x, contains=domain.containables[2])
    # domain.map[WPList[0][0]][WPList[0][1]] = -100

    domain.writeMapToFile("map.txt")

    """pos = vehicle.getLocationGlobal()
    for i in range(len(WPList)):
        WPList[i] = [WPList[i][0] - base[0], WPList[i][1] - base[1]]
        WPList[i] = ned_to_global_scaled(pos.lat, pos.lon, WPList[i][0], WPList[i][1])

    vehicle.assignWP(WPList)

    vehicle.connection.close()"""

    domain.plotMap()
