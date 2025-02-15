from pymavlink import mavutil

class Vehicle():

    def __init__(self, port):
        self.connection = mavutil.mavlink_connection(port)
        self.connection.wait_heartbeat()
        print("Heartbeat from system (system %u component %u)" 
              % (self.connection.target_system, self.connection.target_component))

    def getLocationGlobal(self):
        self.connection.mav.heartbeat_send(mavutil.mavlink.MAV_TYPE_ONBOARD_CONTROLLER,
                                mavutil.mavlink.MAV_AUTOPILOT_INVALID, 0, 0, 0)
        reval = self.connection.recv_match(type='GLOBAL_POSITION_INT', blocking=True, timeout=10)
        if reval is None:
            print("timeout\n")
        return reval
        
    def getLocationLocal(self):
        self.connection.mav.heartbeat_send(mavutil.mavlink.MAV_TYPE_ONBOARD_CONTROLLER,
                                mavutil.mavlink.MAV_AUTOPILOT_INVALID, 0, 0, 0)
        reval = self.connection.recv_match(type='LOCAL_POSITION_NED', blocking=True, timeout=10)
        if reval is None:
            print("timeout\n")
        return reval

    def arm(self):
        self.connection.mav.command_long_send(
            self.connection.target_system,
            self.connection.target_component,
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            0,
            1, 0, 0, 0, 0, 0, 0)

        print("Waiting for the vehicle to arm")
        self.connection.motors_armed_wait()
        print('Armed!')

    def disarm(self):
        self.connection.mav.command_long_send(
            self.connection.target_system,
            self.connection.target_component,
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            0,
            0, 0, 0, 0, 0, 0, 0)
        self.connection.motors_disarmed_wait()
        print('Disarmed!')

    def getCompass(self):
        self.connection.mav.heartbeat_send(mavutil.mavlink.MAV_TYPE_ONBOARD_CONTROLLER,
                                mavutil.mavlink.MAV_AUTOPILOT_INVALID, 0, 0, 0)
        reval = self.connection.recv_match(type='VFR_HUD', blocking=True, timeout=10)
        if reval is None:
            print("timeout\n")
        return reval
    
    def assignWP(self, waypoints):
        self.connection.mav.heartbeat_send(mavutil.mavlink.MAV_TYPE_ONBOARD_CONTROLLER,
                                                    mavutil.mavlink.MAV_AUTOPILOT_INVALID, 0, 0, 0)
        
        self.connection.waypoint_clear_all_send()
        self.connection.waypoint_count_send(len(waypoints))

        for i in range(len(waypoints)):
            self.connection.mav.mission_item_int_send(
                target_system=self.connection.target_system,
                target_component=self.connection.target_component,
                seq=i,
                frame=mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT,
                command=mavutil.mavlink.MAV_CMD_NAV_WAYPOINT,
                current=0,
                autocontinue=1,
                param1=0,
                param2=0,
                param3=0,
                param4=0,
                x=waypoints[i][0],
                y=waypoints[i][1],
                z=0)
