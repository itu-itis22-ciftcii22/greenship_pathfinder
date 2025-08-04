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


    offset_n = (delta_lat * math.pi / 180) * R * 1e-7
    offset_e = (delta_lon * math.pi / 180) * R * math.cos(math.radians(origin_lat_scaled))  * 1e-7


    return offset_n, offset_e