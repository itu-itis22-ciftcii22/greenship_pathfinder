import sys
import signal
import rospy
from autopath_core.pathfinder import NavDomain
from autopath_core.vehicle_interface import Vehicle
from autopath_core.ros_interface import RosIO

def clean_exit(self, signum=None, frame=None):
    rospy.loginfo("Exiting and cleaning up...")
    global vehicle
    try:
        if vehicle:
            self.vehicle.setMode("MANUAL")
            self.vehicle.disarm()
    except Exception as e:
        rospy.logerr(f"Error during cleanup: {str(e)}")
    sys.exit(0)

signal.signal(signal.SIGINT, clean_exit)
signal.signal(signal.SIGTERM, clean_exit)

if __name__ == "__main__":
    try:
        rospy.init_node("autopath_core")
        rate = rospy.Rate(10)
        domain_size = 300
        domain = NavDomain(300, 300)
        vehicle = Vehicle(rate)
        rosIO = RosIO(domain, vehicle)
    except rospy.ROSInterruptException:
        pass