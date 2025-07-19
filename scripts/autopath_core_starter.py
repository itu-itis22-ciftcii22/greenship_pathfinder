import sys
import signal
import rospy
from autopath_core.pathfinder import NavDomain
from autopath_core.vehicle_interface import Vehicle
from autopath_core.core import Commander

def clean_exit(signum=None, frame=None):
    rospy.loginfo("Exiting and cleaning up...")
    global vehicle
    try:
        if vehicle:
            vehicle.setMode("MANUAL")
            vehicle.disarm()
    except Exception as e:
        rospy.logerr(f"Error during cleanup: {str(e)}")
    sys.exit(0)

signal.signal(signal.SIGINT, clean_exit)
signal.signal(signal.SIGTERM, clean_exit)

if __name__ == "__main__":
    try:
        rospy.init_node("autopath_core")
        rate = 10
        domain_size = 300
        domain = NavDomain(domain_size, domain_size)
        vehicle = Vehicle(rospy.Rate(rate))
        command = Commander(rospy.Rate(rate/2), domain, vehicle)
        command.execute_mission()
    except rospy.ROSInterruptException:
        pass