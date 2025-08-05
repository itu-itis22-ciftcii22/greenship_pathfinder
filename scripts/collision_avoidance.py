import sys
import signal
import rospy
import numpy as np
import os
from autopath_core.pathfinder import NavDomain
from autopath_core.vehicle_interface import Vehicle
from autopath_core.core import Commander, ned_to_global

class CommanderCollisionAvoidance(Commander):
    def execute_task_code_collision_avoidance(self):
        """Execute the mission sequence"""
        rospy.loginfo_once("Mission executor node started")
        
        # Give MAVROS topics time to fully establish
        rospy.loginfo("Waiting for MAVROS connections to stabilize...")
        rospy.sleep(2.0)  # Wait for subscribers to establish
        
        MISSION_RADIUS = 2.0  # meters, distance to consider waypoint reached
        
        # Wait for AUTO mode and home position
        if not self._wait_for(
            lambda: (self.vehicle.state.mode == "AUTO" and 
                    self.vehicle.home_position_global.latitude != 0),
            "AUTO mode and home position"
        ):
            rospy.logerr("Failed to initialize mission")
            return False
            
        # Store home position
        self.home_global = np.array([
            self.vehicle.home_position_global.latitude,
            self.vehicle.home_position_global.longitude
        ])
        rospy.loginfo(f"Home position set to: lat={self.home_global[0]:.6f}, lon={self.home_global[1]:.6f}")
        rospy.loginfo(f"Vehicle current global position: lat={self.vehicle.vehicle_position_global.latitude:.6f}, lon={self.vehicle.vehicle_position_global.longitude:.6f}")
        rospy.loginfo(f"NED origin at grid: N={self.home_ned[0]:.1f}, E={self.home_ned[1]:.1f}")
        
        # Wait for mission waypoints
        if not self._wait_for(
            lambda: len(self.vehicle.current_waypoints) > 0,
            "mission waypoints"
        ):
            rospy.logerr("No mission waypoints available")
            return False
        
        # Convert waypoints to NED
        self._convert_waypoints()
        rospy.loginfo(f"Loaded mission with {len(self.mission_ned_list)} waypoints")
        rospy.logdebug("Mission waypoints (NED):")
        for i, wp in enumerate(self.mission_ned_list):
            rospy.logdebug(f"  {i+1}: N={wp[0]:.1f}, E={wp[1]:.1f}")
        
        # Switch to GUIDED mode and arm
        if not self.vehicle.setMode("GUIDED"):
            rospy.logerr("Failed to enter GUIDED mode")
            return False
            
        if not self.vehicle.arm():
            rospy.logerr("Failed to arm vehicle")
            return False
        
        rospy.loginfo("=== Starting mission execution ===")
        
        # Navigate through waypoints
        for i, target_ned in enumerate(self.mission_ned_list):
            rospy.loginfo(f"Navigating to waypoint {i+1}/{len(self.mission_ned_list)}")
            rospy.logdebug(f"Target position (NED): N={target_ned[0]:.1f}, E={target_ned[1]:.1f}")
            
            target_coord = self.domain.Coordinate(*map(int, np.round(target_ned)))
            
            if not self.vehicle.state.guided:
                rospy.logerr("Lost GUIDED mode! Aborting mission")
                return False
                
            # Keep trying to reach the waypoint until timeout
            reached = False
            start_time = rospy.Time.now()
            timeout_duration = rospy.Duration(60.0)  # 60 seconds timeout
            
            while not reached and not rospy.is_shutdown():
                if (rospy.Time.now() - start_time) > timeout_duration:
                    rospy.logwarn(f"Timeout reaching waypoint {i+1}, skipping to next")
                    break
                    
                # Check if we've reached the waypoint
                current_distance = np.linalg.norm(target_ned - self.vehicle_ned)
                if current_distance <= MISSION_RADIUS:
                    rospy.loginfo(f"Reached waypoint {i+1}")
                    reached = True
                    break
                
                # If not reached, update path planning and movement
                try:
                    # Try to find path avoiding obstacles
                    current_coord = self.domain.Coordinate(*map(int, np.round(self.vehicle_ned)))

                    dynamic_obstacle_coords = [
                        self.domain.Coordinate(*map(int, np.round(obs))) 
                        for obs in self.dynamic_obstacle_ned_list
                    ]

                    path = self.domain.a_star_search(
                        current_coord, 
                        target_coord,
                        corridor=self._find_corridor(self._find_buoy_pairs()),
                        moving_obstacles=dynamic_obstacle_coords
                    )
                    
                    if not path:
                        rospy.logwarn_throttle(5.0, "No valid path found, retrying...")
                        self.rate.sleep()
                        continue
                        
                    path_len = len(path)
                    rospy.logdebug(f"Found path with {path_len} waypoints")
                    
                    # Take next waypoint with a reasonable look-ahead
                    look_ahead = min(3, path_len - 1)
                    next_coord = path[look_ahead]
                    next_ned = np.array([next_coord.row, next_coord.col])
                    
                    # Convert to global coordinates and send
                    next_global = ned_to_global(
                        self.home_ned[0], self.home_ned[1],
                        next_ned[0], next_ned[1],
                        self.home_global[0], self.home_global[1]
                    )
                    if not self.vehicle.setTargetPositionGlobal(*next_global):
                        rospy.logwarn_throttle(2.0, "Failed to set next waypoint")
                    else:
                        rospy.logdebug(f"Moving to intermediate point at NED: {next_ned}")
                        
                except Exception as e:
                    rospy.logerr(f"Error in path planning: {e}")
                    
                self.rate.sleep()

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
        command = CommanderCollisionAvoidance(rate, domain, vehicle)
        
        rospy.logdebug("Starting autopath_core with debug logging enabled")
        
        while not rospy.is_shutdown():
            rospy.loginfo("Starting new mission execution")
            success = command.execute_task_code_collision_avoidance()
            
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