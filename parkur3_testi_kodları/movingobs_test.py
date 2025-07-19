#!/usr/bin/env python
import rospy
import numpy as np
from std_msgs.msg import Float32MultiArray
from domain import Domain
from utils import ned_to_global_scaled, global_scaled_to_ned
from vehicle import Vehicle
import signal
import sys

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


# Global obstacle listesi
obstacle_list = []
current_position = np.zeros(2)

def obstacle_callback(msg):
    """
    fusion_v3.py'den gelen engel verileri
    Format: [yaw1, dist1, yaw2, dist2, ...]
    """
    global obstacle_list
    global current_position

    obstacle_list = []
    data = msg.data  # [yaw1, dist1, yaw2, dist2, ...]

    # Her iki değer bir engeli temsil ediyor
    for i in range(0, len(data), 2):
        if i+1 < len(data):  # Güvenlik kontrolü
            yaw_deg = data[i]      # fusion.py'den gelen açı (derece)
            dist = data[i+1]       # fusion.py'den gelen mesafe (metre)
            
            # Geçerli engel kontrolü
            if dist > 0:  # -1 değeri geçersiz engel demek
                # derece → radyan
                theta_rad = np.deg2rad(yaw_deg)
                
                # Relatif pozisyondan mutlak pozisyona çevir
                x = current_position[0] + dist * np.cos(theta_rad)
                y = current_position[1] + dist * np.sin(theta_rad)
                
                obstacle_list.append(np.array([x, y]))
                rospy.logdebug(f"Engel eklendi: Açı={yaw_deg:.1f}°, Mesafe={dist:.1f}m, Pozisyon=({x:.1f}, {y:.1f})")

def start_listener():
    
    rospy.Subscriber('/fusion_output', Float32MultiArray, obstacle_callback)
    rospy.loginfo("Fusion output dinleniyor: /fusion_output")
    

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
                                    moving_obstacles = []
                                    for obs in obstacle_list:
                                        obs_coord = domain.Coordinate(
                                            int(round(obs[0])), 
                                            int(round(obs[1]))
                                        )
                                        if domain.isValid(obs_coord):
                                            moving_obstacles.append(obs_coord)
                                    
                                    if moving_obstacles:
                                        rospy.loginfo(f"{len(moving_obstacles)} engel algılandı")
                                    
                                    # A* ile yol planla
                                    wps_coord = domain.a_star_search(
                                        position_coord, 
                                        mission_coord, 
                                        moving_obstacles=moving_obstacles
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
