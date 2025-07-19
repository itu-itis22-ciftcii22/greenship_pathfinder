#!/usr/bin/env python
"""

Output Format: Float32MultiArray
[front_0°, right_30°, left_-30°, quality_front, quality_right, quality_left]

"""

import rospy
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32MultiArray
import numpy as np
import math


class PathPlanningLidarPublisher:
    def __init__(self):
       
        rospy.init_node('pathplanning_lidar_publisher', anonymous=True)

        # === ROS Communication ===
        # Output: [dist_0, dist_30, dist_-30, qual_0, qual_30, qual_-30]
        self.pub = rospy.Publisher('/lidar_distances', Float32MultiArray, queue_size=10)
        # Input: Ham LiDAR tarama
        rospy.Subscriber('/scan', LaserScan, self.scan_callback)

        # === LiDAR Parametreleri ===
        # RPLiDAR A3 spesifikasyonları
        self.expected_points = 1947
        self.min_valid_distance = 0.3  # 30cm minimum
        self.max_distance_reliable = 8.0  # Siyah objeler (güvenilir)
        self.max_distance_extended = 25.0  # Beyaz objeler (genişletilmiş)

        # Sektör tanımları
        self.target_sectors = [
            {'name': 'FRONT', 'angle': 0, 'index_offset': 0},
            {'name': 'RIGHT', 'angle': 30, 'index_offset': None},  # Hesaplanacak
            {'name': 'LEFT', 'angle': -30, 'index_offset': None}  # Hesaplanacak
        ]

        # Açı toleransı (±derece)
        self.sector_tolerance = 8.0  # ±8° her sektör için

        # === Performans İzleme ===
        self.scan_count = 0
        self.error_count = 0
        self.last_stats_time = rospy.Time.now()

        rospy.loginfo("=== Path Planning LiDAR Publisher ===")
        rospy.loginfo(f"Çıktı formatı: [dist_0°, dist_+30°, dist_-30°, qual_0°, qual_+30°, qual_-30°]")
        rospy.loginfo(f"Güvenilir menzil: {self.max_distance_reliable}m")
        rospy.loginfo(f"Genişletilmiş menzil: {self.max_distance_extended}m")
        rospy.spin()

    def angle_to_index(self, target_angle, total_points):
        """
        Açıyı LiDAR array index'ine çevir

        Input:
        - target_angle: Hedef açı (-180 to +180)
        - total_points: Toplam LiDAR noktası

        Output:
        - index: Array index
        """
        # Açıyı 0-360 normalize et
        normalized_angle = (target_angle + 360) % 360
        # Index hesapla
        index = int(round(normalized_angle * total_points / 360.0))
        return min(max(index, 0), total_points - 1)

    def extract_sector_distance(self, ranges, target_angle, total_points):
        """
        Belirli sektörden en güvenilir mesafeyi çıkar

        Input:
        - ranges: LiDAR mesafe array'i
        - target_angle: Hedef açı
        - total_points: Toplam nokta sayısı

        Output:
        - distance: En güvenilir mesafe (metre, -1 = geçersiz)
        - quality: Kalite skoru (0.0-1.0)
        """
        center_index = self.angle_to_index(target_angle, total_points)

        # Sektör aralığı hesapla
        tolerance_points = int(self.sector_tolerance * total_points / 360.0)
        start_idx = max(0, center_index - tolerance_points)
        end_idx = min(total_points, center_index + tolerance_points + 1)

        # Sektördeki tüm geçerli mesafeleri topla
        valid_distances = []
        reliable_distances = []

        for i in range(start_idx, end_idx):
            distance = ranges[i]

            # Temel geçerlilik kontrolü
            if (self.min_valid_distance <= distance <= self.max_distance_extended and
                    not math.isinf(distance) and not math.isnan(distance)):

                valid_distances.append(distance)

                # Güvenilir aralık kontrolü
                if distance <= self.max_distance_reliable:
                    reliable_distances.append(distance)

        # Mesafe seçimi stratejisi
        if not valid_distances:
            return -1.0, 0.0

        # Güvenilir mesafeler varsa onları öncelikle
        if reliable_distances:
            selected_distance = min(reliable_distances)  # En yakın güvenilir
            quality_base = 0.8
        else:
            selected_distance = min(valid_distances)  # En yakın geçerli
            quality_base = 0.4

        # Kalite skoru hesaplama
        data_density = len(valid_distances) / (2 * tolerance_points)
        reliable_ratio = len(reliable_distances) / max(len(valid_distances), 1)

        quality_score = quality_base * (0.5 + 0.3 * data_density + 0.2 * reliable_ratio)
        quality_score = min(quality_score, 1.0)

        return selected_distance, quality_score

    def validate_scan_data(self, msg):
        """
        LiDAR tarama verisini doğrula

        Input: LaserScan msg
        Output: (is_valid, error_messages)
        """
        if not msg.ranges:
            return False, ["Boş ranges array"]

        total_points = len(msg.ranges)

        # Nokta sayısı kontrolü (±200 tolerance)
        if abs(total_points - self.expected_points) > 200:
            return False, [f"Beklenmedik nokta sayısı: {total_points}"]

        # Veri kalitesi analizi
        valid_count = 0
        inf_count = 0

        for distance in msg.ranges:
            if math.isinf(distance):
                inf_count += 1
            elif (self.min_valid_distance <= distance <= self.max_distance_extended and
                  not math.isnan(distance)):
                valid_count += 1

        valid_ratio = valid_count / total_points
        inf_ratio = inf_count / total_points

        # Minimum veri kalitesi kontrolü
        if valid_ratio < 0.05:  # %5'ten az geçerli veri
            return False, [f"Çok az geçerli veri: %{valid_ratio * 100:.1f}"]

        # Çok fazla infinity uyarısı
        error_messages = []
        if inf_ratio > 0.9:  # %90'dan fazla infinity
            error_messages.append(f"Yüksek infinity oranı: %{inf_ratio * 100:.1f}")

        return True, error_messages

    def create_output_array(self, distances, qualities):
        """
        Çıktı array'ini oluştur

        Input:
        - distances: [front, right, left] mesafeler
        - qualities: [front, right, left] kalite skorları

        Output:
        - Float32MultiArray: [dist_f, dist_r, dist_l, qual_f, qual_r, qual_l]
        """
        # Array formatı: [mesafeler, kaliteler]
        output_data = distances + qualities

        # ROS mesajı oluştur
        array_msg = Float32MultiArray()
        array_msg.data = output_data

        return array_msg

    def scan_callback(self, msg):
        """
        Ana LiDAR callback fonksiyonu

        Input: LaserScan mesajı
        Output: Float32MultiArray publish
        """
        try:
            self.scan_count += 1
            current_time = rospy.Time.now()

            # === VERİ DOĞRULAMA ===
            is_valid, validation_messages = self.validate_scan_data(msg)

            if not is_valid:
                self.error_count += 1
                rospy.logwarn(f"Geçersiz LiDAR verisi: {validation_messages}")

                # Hata durumunda güvenli default değerler
                error_distances = [-1.0, -1.0, -1.0]
                error_qualities = [0.0, 0.0, 0.0]
                error_array = self.create_output_array(error_distances, error_qualities)
                self.pub.publish(error_array)
                return

            # Uyarı mesajları varsa logla
            if validation_messages:
                rospy.logdebug(f"LiDAR uyarıları: {validation_messages}")

            # === SEKTÖR MESAFELERİNİ ÇIKAR ===
            total_points = len(msg.ranges)
            sector_distances = []
            sector_qualities = []
            sector_info = []

            # Her sektör için mesafe ve kalite hesapla
            for sector in self.target_sectors:
                distance, quality = self.extract_sector_distance(
                    msg.ranges, sector['angle'], total_points
                )

                sector_distances.append(distance)
                sector_qualities.append(quality)

                sector_info.append({
                    'name': sector['name'],
                    'angle': sector['angle'],
                    'distance': distance,
                    'quality': quality
                })

            # === ÇIKTI ARRAY'İNİ OLUŞTUR VE PUBLISH ET ===
            output_array = self.create_output_array(sector_distances, sector_qualities)
            self.pub.publish(output_array)

            # === LOGGING ===
            # Her 20 scan'de bir detaylı log
            if self.scan_count % 20 == 0:
                rospy.loginfo(f"=== LiDAR Scan #{self.scan_count} ===")
                rospy.loginfo(f"Nokta sayısı: {total_points}")

                for info in sector_info:
                    if info['distance'] > 0:
                        status = "OK" if info['quality'] > 0.6 else "LOW_QUAL"
                        rospy.loginfo(f"{info['name']:>5} {info['angle']:+3.0f}°: "
                                      f"{info['distance']:5.1f}m (Q:{info['quality']:.2f}) [{status}]")
                    else:
                        rospy.loginfo(f"{info['name']:>5} {info['angle']:+3.0f}°: NO_DATA")

                # Performans istatistikleri
                time_elapsed = (current_time - self.last_stats_time).to_sec()
                if time_elapsed > 0:
                    scan_rate = 20 / time_elapsed
                    rospy.loginfo(f"Scan rate: {scan_rate:.1f} Hz")

                    if self.error_count > 0:
                        error_rate = self.error_count / self.scan_count * 100
                        rospy.loginfo(f"Error rate: {error_rate:.1f}%")

                self.last_stats_time = current_time

            # Kritik yakınlık uyarısı
            min_distance = min([d for d in sector_distances if d > 0], default=999)
            if min_distance < 3.0:  # 3m'den yakın
                rospy.logwarn(f"KRİTİK YAKINLIK: {min_distance:.1f}m!")

        except Exception as e:
            self.error_count += 1
            rospy.logerr(f"LiDAR callback hatası: {str(e)}")

            # Kritik hata durumunda güvenli değerler
            error_distances = [-1.0, -1.0, -1.0]
            error_qualities = [0.0, 0.0, 0.0]
            error_array = self.create_output_array(error_distances, error_qualities)
            self.pub.publish(error_array)


if __name__ == '__main__':
    try:
        PathPlanningLidarPublisher()
    except rospy.ROSInterruptException:
        rospy.loginfo("LiDAR Publisher normal shutdown")
    except Exception as e:
        rospy.logerr(f"LiDAR Publisher kritik hata: {e}")
