from domain import Domain
from vehicle import Vehicle
from path_finder import PathFinder

import signal
import sys
import random
import matplotlib.pyplot as plt

def clean_exit(signum, frame):
    global vehicle
    print("\nExiting and cleaning up...")
    if vehicle.connection:
        vehicle.connection.close()
    sys.exit(0)
    
signal.signal(signal.SIGINT, clean_exit)
signal.signal(signal.SIGTERM, clean_exit)

def generatePons(degree2, local, number):
    ponlist = []
    random.seed()
    for _ in range(int(random.random()*number)):
        random.seed()
        distance = int(random.random()*100)
        random.seed()
        degree1 = int(random.random()*360)
        pon = Domain.findRelativePoint(distance, degree1, degree2, local[0], local[1])
        if not pon[0] == local[0] and not pon[1] == local[1]:
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
    hf
    return latitude_scaled, longitude_scaled


if __name__ == '__main__':
    vehicle = Vehicle("udpin:localhost:14550")  # Aracın bağlantı noktası
    vehicle.arm()
    vehicle.connection.set_mode_loiter()

    domain = Domain(100, 100, float("inf"))

    base = (50, 50)

    heading = vehicle.getCompass().heading
    if heading is None:
        sys.exit(1)

    ponlist = generatePons(heading, base, 100)
    for pon in ponlist:
        domain.updateMap(pon, 15)
    
    endx = -1
    endy = -1
    while not domain.is_valid(endx, endy) or not domain.is_unblocked(endx, endy):
        random.seed()
        endx = int(random.random()*50)
        random.seed()
        endy = int(random.random()*50)
    endpoint = (endx, endy)

    pathfind = PathFinder(domain.nrow, domain.ncol, domain.blockval, domain.map)
    WPList = pathfind.a_star_search(base, endpoint)
    
    if WPList is None:
        sys.exit(1)

    for x in WPList:
        domain.map[x[0]][x[1]] = -100
    #domain.map[WPList[0][0]][WPList[0][1]] = -100

    domain.writeMapToFile("map.txt")
    
    pos = vehicle.getLocationGlobal()
    for i in range(len(WPList)):
        WPList[i] = [WPList[i][0] - base[0], WPList[i][1] - base[1]]
        WPList[i] = ned_to_global_scaled(pos.lat, pos.lon, WPList[i][0], WPList[i][1])


    vehicle.assignWP(WPList)

    vehicle.connection.close()

    plt.figure(figsize=(10, 10))
    plt.imshow(domain.map, cmap='RdBu', interpolation='nearest', vmin=-100, vmax=100, origin='lower')  # Choose a colormap like 'viridis'
    plt.gca().set_aspect('auto')
    #plt.colorbar(label='Value')  # Add a color bar for reference
    plt.title("Weighted Grid with Path (Obstacles marked as blue, Path marked as red)")
    plt.savefig("path_visualization.jpg", bbox_inches='tight')
    plt.show()