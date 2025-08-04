#!/usr/bin/env python
import rospy
import numpy as np
from std_msgs.msg import Float32MultiArray
from domain import Domain
from utils import ned_to_global_scaled, global_scaled_to_ned
from vehicle import Vehicle
import signal
import sys
from scipy.spatial import KDTree

def clean_exit(signum, frame):
    global vehicle
    print("\nExiting and cleaning up...")
    if vehicle and hasattr(vehicle, 'connection') and vehicle.connection:
        try:
            vehicle.connection.set_mode_manual()
            vehicle.disarm()
            vehicle.connection.close()
        except:
            pass
    sys.exit(0)


signal.signal(signal.SIGINT, clean_exit)
signal.signal(signal.SIGTERM, clean_exit)


# Global obstacle storage using KDTrees
red_buoy_positions = []
green_buoy_positions = []
red_buoy_kdtree = None
green_buoy_kdtree = None

# Temporary lists for new detections
new_red_buoy_list = []
new_green_buoy_list = []
current_position = np.zeros(2)

MIN_DISTANCE_THRESHOLD = 2.0

def rebuild_kdtrees():
    """
    KDTree'leri yeniden oluştur
    """
    global red_buoy_kdtree, green_buoy_kdtree
    global red_buoy_positions, green_buoy_positions
    
    # Red buoy KDTree
    if red_buoy_positions:
        red_buoy_kdtree = KDTree(red_buoy_positions)
    else:
        red_buoy_kdtree = None
        
    # Green buoy KDTree
    if green_buoy_positions:
        green_buoy_kdtree = KDTree(green_buoy_positions)
    else:
        green_buoy_kdtree = None

def red_buoy_callback(msg):
    """
    Kırmızı duba tespiti - KDTree kullanarak filtreleme
    """
    global red_buoy_kdtree, new_red_buoy_list, current_position

    new_red_buoy_list = []
    data = msg.data

    for i in range(0, len(data), 2):
        if i+1 < len(data):
            yaw_deg = data[i]
            dist = data[i+1]
            
            if dist > 0:
                theta_rad = np.deg2rad(yaw_deg)
                x = current_position[0] + dist * np.cos(theta_rad)
                y = current_position[1] + dist * np.sin(theta_rad)
                
                new_position = np.array([x, y])
                
                # KDTree ile en yakın komşu ara
                is_new_buoy = True
                if red_buoy_kdtree is not None:
                    distance, _ = red_buoy_kdtree.query(new_position)
                    if distance < MIN_DISTANCE_THRESHOLD:
                        is_new_buoy = False
                        rospy.logdebug(f"Mevcut KIRMIZI duba tespit edildi: Açı={yaw_deg:.1f}°, "
                                     f"Mesafe={dist:.1f}m, En yakın dubaya uzaklık={distance:.1f}m")
                
                if is_new_buoy:
                    new_red_buoy_list.append(new_position)
                    rospy.logdebug(f"YENİ kırmızı duba eklendi: Açı={yaw_deg:.1f}°, "
                                 f"Mesafe={dist:.1f}m, Pozisyon=({x:.1f}, {y:.1f})")

def green_buoy_callback(msg):
    """
    Yeşil duba tespiti - KDTree kullanarak filtreleme
    """
    global green_buoy_kdtree, new_green_buoy_list, current_position

    new_green_buoy_list = []
    data = msg.data

    for i in range(0, len(data), 2):
        if i+1 < len(data):
            yaw_deg = data[i]
            dist = data[i+1]
            
            if dist > 0:
                theta_rad = np.deg2rad(yaw_deg)
                x = current_position[0] + dist * np.cos(theta_rad)
                y = current_position[1] + dist * np.sin(theta_rad)
                
                new_position = np.array([x, y])
                
                # KDTree ile en yakın komşu ara
                is_new_buoy = True
                if green_buoy_kdtree is not None:
                    distance, _ = green_buoy_kdtree.query(new_position)
                    if distance < MIN_DISTANCE_THRESHOLD:
                        is_new_buoy = False
                        rospy.logdebug(f"Mevcut YEŞİL duba tespit edildi: Açı={yaw_deg:.1f}°, "
                                     f"Mesafe={dist:.1f}m, En yakın dubaya uzaklık={distance:.1f}m")
                
                if is_new_buoy:
                    new_green_buoy_list.append(new_position)
                    rospy.logdebug(f"YENİ yeşil duba eklendi: Açı={yaw_deg:.1f}°, "
                                 f"Mesafe={dist:.1f}m, Pozisyon=({x:.1f}, {y:.1f})")

def update_buoy_kdtrees():
    """
    Yeni tespit edilen dubaları KDTree'lere ekle ve yeniden oluştur
    """
    global red_buoy_positions, green_buoy_positions
    global new_red_buoy_list, new_green_buoy_list
    
    # Yeni kırmızı dubaları ekle
    if new_red_buoy_list:
        red_buoy_positions.extend(new_red_buoy_list)
        rospy.loginfo(f"Toplam {len(new_red_buoy_list)} yeni kırmızı duba eklendi. "
                     f"Toplam kırmızı duba sayısı: {len(red_buoy_positions)}")
        new_red_buoy_list = []
    
    # Yeni yeşil dubaları ekle
    if new_green_buoy_list:
        green_buoy_positions.extend(new_green_buoy_list)
        rospy.loginfo(f"Toplam {len(new_green_buoy_list)} yeni yeşil duba eklendi. "
                     f"Toplam yeşil duba sayısı: {len(green_buoy_positions)}")
        new_green_buoy_list = []
    
    # KDTree'leri yeniden oluştur
    rebuild_kdtrees()

def get_buoy_counts():
    """
    Mevcut duba sayılarını döndür
    """
    return len(red_buoy_positions), len(green_buoy_positions)

def find_nearest_buoys(position, color='both', k=1):
    """
    Belirtilen pozisyona en yakın dubaları bul
    
    Args:
        position: np.array([x, y]) - Arama pozisyonu
        color: 'red', 'green', 'both' - Hangi renk dubaları ara
        k: int - Kaç tane en yakın duba döndürülecek
        
    Returns:
        dict: {'red': [(distance, index), ...], 'green': [(distance, index), ...]}
    """
    results = {}
    
    if color in ['red', 'both'] and red_buoy_kdtree is not None:
        distances, indices = red_buoy_kdtree.query(position, k=min(k, len(red_buoy_positions)))
        if k == 1:
            distances = [distances]
            indices = [indices]
        results['red'] = list(zip(distances, indices))
    
    if color in ['green', 'both'] and green_buoy_kdtree is not None:
        distances, indices = green_buoy_kdtree.query(position, k=min(k, len(green_buoy_positions)))
        if k == 1:
            distances = [distances]
            indices = [indices]
        results['green'] = list(zip(distances, indices))
    
    return results

def find_buoy_pairs(max_pair_distance=7.0):
    """
    Kırmızı ve yeşil duba çiftlerini bul
    
    Args:
        max_pair_distance: float - Maksimum çift mesafesi
        
    Returns:
        list: [(red_idx, green_idx, distance), ...] - Duba çiftleri
    """
    pairs = []
    
    if red_buoy_kdtree is None or green_buoy_kdtree is None:
        return pairs
    
    for red_idx, red_pos in enumerate(red_buoy_positions):
        # Her kırmızı duba için en yakın yeşil dubayı bul
        distance, green_idx = green_buoy_kdtree.query(red_pos)
        
        if distance <= max_pair_distance:
            pairs.append((red_idx, green_idx))
    
    
    return pairs

def find_corridor(pairs):
    """
    Kırmızı ve yeşil duba çiftlerinin orta noktalarını hesaplar
    
    Args:
        pairs: list - (red_idx, green_idx) çiftlerini içeren liste
        
    Returns:
        list: [(mid_x, mid_y), ...] - Çiftlerin orta noktaları
    """
    midpoints = []
    for red_idx, green_idx in pairs:
        red_pos = red_buoy_positions[red_idx]
        green_pos = green_buoy_positions[green_idx]
        mid_x = (red_pos[0] + green_pos[0]) / 2.0
        mid_y = (red_pos[1] + green_pos[1]) / 2.0
        midpoints.append((mid_x, mid_y))
    return midpoints

def clear_all_buoys():
    """
    Tüm duba verilerini temizle
    """
    global red_buoy_positions, green_buoy_positions
    global red_buoy_kdtree, green_buoy_kdtree
    global new_red_buoy_list, new_green_buoy_list
    
    red_buoy_positions = []
    green_buoy_positions = []
    red_buoy_kdtree = None
    green_buoy_kdtree = None
    new_red_buoy_list = []
    new_green_buoy_list = []
    
    rospy.loginfo("Tüm duba verileri temizlendi")

def start_listener():
    rospy.Subscriber('/fusion_output/red_buoy', Float32MultiArray, red_buoy_callback)
    rospy.Subscriber('/fusion_output/green_buoy', Float32MultiArray, green_buoy_callback)
    rospy.loginfo("Fusion output kırmızı duba dinleniyor: /fusion_output/red_buoy")
    rospy.loginfo("Fusion output yeşil duba dinleniyor: /fusion_output/green_buoy")

if __name__ == '__main__':
    rospy.init_node('apf_planner')
    start_listener()
    rate = rospy.Rate(10)  # 10 Hz
    domain = Domain(150, 150)
    vehicle = None  # Global tanımlama
    
    rospy.loginfo("=== Otonom Yol Planlama Başlatıldı ===")

    while not rospy.is_shutdown():
        try:
         
            vehicle = Vehicle("/dev/ttyACM0", baud=57600)
            
            while not rospy.is_shutdown():
                try:
                    rospy.loginfo("AUTO modu bekleniyor...")
                    
                    # AUTO modunu bekle
                    while not rospy.is_shutdown():
                        mode = vehicle.getMode()
                        if mode == "AUTO":
                            break
                        else:
                            rospy.loginfo(f"Mevcut mod: {mode}")
                            rospy.sleep(1.0)
                            
                    if rospy.is_shutdown():
                        break
                        
                    rospy.loginfo("AUTO modu aktif, planlama başlıyor")

                    # Home pozisyonunu al
                    msg = None
                    retry_count = 0
                    while msg is None and retry_count < 5:
                        msg = vehicle.getHome()
                        if msg is None:
                            rospy.logwarn(f"Home pozisyonu alınamadı, yeniden deneniyor... ({retry_count+1}/5)")
                            rospy.sleep(1.0)
                            retry_count += 1
                            
                    if msg is None:
                        rospy.logerr("Home pozisyonu alınamadı!")
                        continue
                        
                    home_global = np.array([msg.latitude, msg.longitude])
                    home_ned = np.array([75.0, 75.0])
                    home_coord = domain.Coordinate(75, 75)
                    
                    rospy.loginfo(f"Home pozisyonu: {home_global}")

                    # Waypoint listesini al
                    wp_list = vehicle.getWPList()
                    if not wp_list:
                        rospy.logwarn("Waypoint listesi boş!")
                        continue
                        
                    missions_global = []
                    for wp in wp_list:
                        missions_global.append((wp.x, wp.y))
                    missions_global = np.array(missions_global)
                    
                    rospy.loginfo(f"{len(missions_global)} waypoint yüklendi")

                    # Waypoint'leri NED koordinatlarına çevir
                    missions_ned = np.array([
                        global_scaled_to_ned(home_global[0], home_global[1], mission_global[0], mission_global[1])
                        for mission_global in missions_global
                    ]) + home_ned

                    # Parametreler
                    step_size = 2
                    mission_radius = 1
                    first_move_sent = False

                    # Her waypoint için yol planla
                    for mission_idx, mission_ned in enumerate(missions_ned):
                        if vehicle.getMode() != "AUTO" or rospy.is_shutdown():
                            break
                            
                        mission_coord = domain.Coordinate(
                            int(round(mission_ned[0].item())), 
                            int(round(mission_ned[1].item()))
                        )
                        
                        rospy.loginfo(f"Waypoint {mission_idx+1}/{len(missions_ned)} hedefleniyor")
                        
                        while not rospy.is_shutdown():
                            if vehicle.getMode() != "AUTO":
                                break
                                
                            # Mevcut pozisyonu al
                            msg = vehicle.getLocationRaw()
                            if msg is not None:
                                position_global = np.array([msg.lat, msg.lon])
                                position_ned = global_scaled_to_ned(
                                    home_global[0], home_global[1], 
                                    position_global[0], position_global[1]
                                ) + home_ned
                                current_position = position_ned
                                
                                position_coord = domain.Coordinate(
                                    int(round(position_ned[0].item())), 
                                    int(round(position_ned[1].item()))
                                )
                                
                                # Hedefe olan mesafeyi kontrol et
                                distance_to_mission = np.linalg.norm(mission_ned - position_ned)
                                
                                if distance_to_mission > mission_radius:
                                    # DÜZELTİLDİ: Engel koordinatlarını düzgün şekilde oluştur
                                    update_buoy_kdtrees()
                                    corridor = find_corridor(find_buoy_pairs)
                                    
                                    # A* ile yol planla
                                    wps_coord = domain.a_star_search(
                                        position_coord, 
                                        mission_coord, 
                                        corridor=corridor
                                    )
                                    
                                    if wps_coord is not None and len(wps_coord) > 0:
                                        wps_len = len(wps_coord)
                                        rospy.loginfo(f"Yol planlandı: {wps_len} nokta")
                                        
                                        # Waypoint'leri belirle
                                        waypoints_to_send = []
                                        
                                        if wps_len > step_size:
                                            # İlk waypoint
                                            first_lat, first_lon = ned_to_global_scaled(
                                                home_global[0], home_global[1],
                                                wps_coord[step_size].row-75, 
                                                wps_coord[step_size].col-75
                                            )
                                            waypoints_to_send.append((first_lat, first_lon))
                                            
                                            # İkinci waypoint
                                            if wps_len > step_size*2:
                                                second_lat, second_lon = ned_to_global_scaled(
                                                    home_global[0], home_global[1],
                                                    wps_coord[step_size*2].row-75, 
                                                    wps_coord[step_size*2].col-75
                                                )
                                            else:
                                                second_lat, second_lon = ned_to_global_scaled(
                                                    home_global[0], home_global[1],
                                                    wps_coord[wps_len-1].row-75, 
                                                    wps_coord[wps_len-1].col-75
                                                )
                                            waypoints_to_send.append((second_lat, second_lon))
                                        else:
                                            # Sadece son noktayı gönder
                                            first_lat, first_lon = ned_to_global_scaled(
                                                home_global[0], home_global[1],
                                                wps_coord[wps_len-1].row-75, 
                                                wps_coord[wps_len-1].col-75
                                            )
                                            waypoints_to_send.append((first_lat, first_lon))
                                        
                                        # Waypoint'leri araca gönder
                                        vehicle.assignWPs(waypoints_to_send)
                                        rospy.loginfo(f"{len(waypoints_to_send)} waypoint gönderildi")
                                    else:
                                        rospy.logwarn("Yol planlanamadı!")
                                    
                                    # İlk harekette arm et
                                    if not first_move_sent:
                                        vehicle.arm()
                                        first_move_sent = True
                                        rospy.loginfo("Araç arm edildi")
                                    
                                    rate.sleep()
                                else:
                                    rospy.loginfo(f"Waypoint {mission_idx+1} ulaşıldı")
                                    break
                            else:
                                rospy.logwarn("Pozisyon alınamadı")
                                rate.sleep()
                                
                except Exception as e:
                    rospy.logerr(f"Ana döngü hatası: {e}")
                    import traceback
                    traceback.print_exc()
                    rospy.sleep(1.0)
                    
        except Exception as e:
            rospy.logerr(f"Bağlantı hatası: {e}")
            rospy.sleep(5.0)
            
        finally:
            # Güvenli kapanış
            if vehicle and hasattr(vehicle, 'connection'):
                try:
                    vehicle.connection.set_mode_manual()
                    vehicle.disarm()
                    rospy.loginfo("Araç güvenli modda")
                except:
                    pass
