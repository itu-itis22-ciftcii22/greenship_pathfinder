import sys
import signal
import rospy
import os
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
        
        # Set debug level from parameter or environment variable
        log_level = rospy.get_param('~log_level', os.getenv('ROS_LOG_LEVEL', 'info')).upper()
        rospy.loginfo(f"Setting log level to {log_level}")
        import logging
        logging.getLogger('rosout').setLevel(getattr(logging, log_level))
        
        rate = 10.0
        domain_size = 300
        domain = NavDomain(domain_size, domain_size)
        vehicle = Vehicle(rate)
        command = Commander(rate, domain, vehicle)
        
        rospy.logdebug("Starting autopath_core with debug logging enabled")
        
        while not rospy.is_shutdown():
            rospy.loginfo("Starting new mission execution")
            success = command.execute_mission()
            
            if rospy.is_shutdown():
                break
                
            if success:
                rospy.loginfo("Mission completed successfully, restarting in 5 seconds...")
                rospy.sleep(5.0)  # Wait 5 seconds before starting next mission
            else:
                rospy.logwarn("Mission failed, restarting in 10 seconds...")
                rospy.sleep(10.0)  # Wait longer if there was a failure
                
    except rospy.ROSInterruptException:
        pass