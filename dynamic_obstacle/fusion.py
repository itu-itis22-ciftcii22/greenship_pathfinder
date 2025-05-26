#!/usr/bin/env python
import rospy
from std_msgs.msg import String, Float32MultiArray
from message_filters import ApproximateTimeSynchronizer, Subscriber

class CameraLidarFusion:
    def __init__(self):
        rospy.init_node('camera_lidar_fusion_node', anonymous=True)

        self.cam_sub = Subscriber('/camera_detections', String)
        self.lidar_sub = Subscriber('/filtered_scan_points', Float32MultiArray)

        self.pub = rospy.Publisher('/fusion_output', String, queue_size=10)

        ats = ApproximateTimeSynchronizer(
            [self.cam_sub, self.lidar_sub], queue_size=10, slop=0.1, allow_headerless=True)
        ats.registerCallback(self.fusion_callback)

        rospy.loginfo("Camera-Lidar fusion node started.")
        rospy.spin()

    def fusion_callback(self, cam_msg, lidar_msg):
        try:
            lidar_values = lidar_msg.data  # [front, right, left, rear]
            label, yaw = cam_msg.data.split(":")
            yaw = float(yaw)

            output = f"Object: {label}, Yaw: {yaw:.1f} deg, LIDAR: {lidar_values}"
            self.pub.publish(output)
            rospy.loginfo(f"Fusion: {output}")
        except Exception as e:
            rospy.logwarn(f"Fusion error: {e}")

if __name__ == '__main__':
    try:
        CameraLidarFusion()
    except rospy.ROSInterruptException:
        pass