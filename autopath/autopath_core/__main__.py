import rospy
from autopath_core.ros_interface import RosIO
from autopath_core.vehicle_interface import Vehicle
from autopath_core.pathfinder import NavDomain

if __name__ == "__main__":
    try:
        rospy.init_node("autopath_core")
        rate = Rospy.Rate(10)
        domain_size = 300
        domain = NavDomain(300, 300)
        vehicle = Vehicle(rate)
        rosIO = RosIO(domain, vehicle)
    except rospy.ROSInterruptException:
        pass