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
        pon = findRelativePoint(distance, degree1, degree2, local[0], local[1])
        if not pon[0] == local[0]and not pon[1] == local[1]:
            ponlist.append(pon)
    return ponlist

import math

def ned_to_global_scaled(origin_lat_scaled, origin_lon_scaled, offset_n, offset_e):
    R = 6378137.0  # Radius of Earth in meters
    latitude_scaled = int(origin_lat_scaled + ((offset_n / R) * (180 / math.pi))*1e7)
    longitude_scaled = int(origin_lon_scaled + ((offset_e / (R * math.cos(math.pi * origin_lat_scaled / 180))) * (180 / math.pi))*1e7)

    return latitude_scaled, longitude_scaled

def global_scaled_to_ned(origin_lat_scaled, origin_lon_scaled, latitude_scaled, longitude_scaled):
    R = 6378137.0  # Earth radius in meters

    # Compute offsets
    delta_lat = latitude_scaled - origin_lat_scaled
    delta_lon = longitude_scaled - origin_lon_scaled


    offset_n = int((delta_lat * math.pi / 180) * R * 1e-7)
    offset_e = int((delta_lon * math.pi / 180) * R * math.cos(math.radians(origin_lat_scaled))  * 1e-7)


    return offset_n, offset_e



if __name__ == '__main__':
    vehicle = Vehicle("udpin:localhost:14550")
    print("Waiting")
    vehicle.waitAuto()
    print("Started")
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
    print(missions_ned)

    wps = []
    pos = base
    for mission_ned in missions_ned:
        wps_segment= domain.a_star_search(domain.Coordinate(pos[0], pos[1]), domain.Coordinate(mission_ned[0], mission_ned[1]))
        if wps_segment is None:
            sys.exit(1)
        for i in range(0, len(wps_segment), 3):
            wps.append(wps_segment[i])
        pos = mission_ned

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