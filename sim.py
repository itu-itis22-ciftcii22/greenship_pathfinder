from domain import Domain
from vehicle import Vehicle

import signal
import sys
import random

def clean_exit(signum, frame):
    global vehicle
    print("\nExiting and cleaning up...")
    if vehicle.connection:
        vehicle.connection.close()
    sys.exit(0)
    
signal.signal(signal.SIGINT, clean_exit)
signal.signal(signal.SIGTERM, clean_exit)

def findRelativePoint(distance, angle, heading, x_base=0, y_base=0):
    # Calculate the total angle in degrees (account for clockwise direction)
    total_angle = -(angle + heading)  # Negative to convert clockwise to counterclockwise

    # Convert angle to radians
    total_angle_rad = math.radians(total_angle)

    # Calculate displacements
    x_displacement = distance * math.cos(total_angle_rad)
    y_displacement = distance * math.sin(total_angle_rad)

    # Add displacements to the base-point coordinates
    x = x_base + x_displacement
    y = y_base + y_displacement

    return int(x), int(y)

def generatePons(degree2, local, number):
    ponlist = []
    random.seed()
    for _ in range(int(random.random()*number)):
        random.seed()
        distance = int(random.random()*100)
        random.seed()
        degree1 = int(random.random()*360)
        pon = findRelativePoint(distance, degree1, degree2, local.row, local.col)
        if not pon[0] == local.row and not pon[1] == local.col:
            ponlist.append(pon)
    return ponlist

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
    new_lat = origin_lat + int(((offset_n / R) * (180 / math.pi))*1e7)
    new_lon = origin_lon + int(((offset_e / (R * math.cos(math.pi * origin_lat / 180))) * (180 / math.pi))*1e7)

    # Scale latitude and longitude by 10^7
    latitude_scaled = int(new_lat)
    longitude_scaled = int(new_lon)

    return latitude_scaled, longitude_scaled

def global_scaled_to_ned(origin_lat, origin_lon, target_lat_scaled, target_lon_scaled):
    """
    Converts global coordinates (scaled by 1e7) to NED frame offsets in meters.

    :param origin_lat: Origin latitude in degrees
    :param origin_lon: Origin longitude in degrees
    :param target_lat_scaled: Target latitude in degrees * 1e7
    :param target_lon_scaled: Target longitude in degrees * 1e7
    :return: (offset_n, offset_e) in meters
    """
    R = 6378137.0  # Earth radius in meters

    # Convert scaled lat/lon to degrees
    target_lat = target_lat_scaled * 1e-7
    target_lon = target_lon_scaled * 1e-7

    # Compute offsets
    delta_lat = target_lat - origin_lat
    delta_lon = target_lon - origin_lon

    offset_n = (delta_lat * math.pi / 180) * R
    offset_e = (delta_lon * math.pi / 180) * R * math.cos(math.radians(origin_lat))

    return offset_n, offset_e



if __name__ == '__main__':
    vehicle = Vehicle("udpin:localhost:14550")
    vehicle.waitAuto()
    missions = vehicle.getWPList()

    domain = Domain(100, 100)

    base = (50, 50)

    heading = vehicle.getCompass().heading
    if heading is None:
        sys.exit(1)

    ponlist = generatePons(heading, base, 100)
    for pon in ponlist:
            domain.updateMap(domain.Coordinate(pon[0], pon[1]), contains=domain.containables[0], blocked=True,
                             value=50, radius=5, is_repellor=True)

    locat = vehicle.getLocationGlobal()
    missions_ned = []
    for mission in missions:
        mission_ned = global_scaled_to_ned(locat.lat, locat.lon, mission.x, mission.y)
        missions_ned.append((mission_ned[0] + base[0], mission_ned[1] + base[1]))

    wps = []
    pos = base
    for mission_ned in missions_ned:
        wps_segment= domain.a_star_search(domain.Coordinate(pos[0], pos[1]), domain.Coordinate(mission_ned[0], mission_ned[1]))
        if wps_segment is None:
            sys.exit(1)
        for wp in wps_segment:
            wps.append(wp)
        pos = missions_ned

    """domain_copy = copy.deepcopy(domain)

    for x in wp_list:
        if domain.isValid(x) and domain.map[x.row][x.col].contains == "empty":
            domain_copy.updateCell(x, contains=domain.containables[2])


    domain_copy.plotMap(50)"""
    
    pos = vehicle.getLocationGlobal()
    for i in range(len(wps)):
        wps[i] = [wps[i].row - base[0], wps[i].col - base[1]]
        wps[i] = ned_to_global_scaled(pos.lat, pos.lon, wps[i][0], wps[i][1])


    vehicle.assignWPs(wps)

    vehicle.arm()

    vehicle.connection.close()