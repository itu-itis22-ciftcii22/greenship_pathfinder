from pymavlink import mavutil

class Vehicle():

    def __init__(self, port):
        self.connection = mavutil.mavlink_connection(port)
        self.connection.wait_heartbeat()
        print("Heartbeat from system (system %u component %u)" 
              % (self.connection.target_system, self.connection.target_component))

    def getLocationGlobal(self):
        reval = self.connection.recv_match(type='GLOBAL_POSITION_INT', blocking=True, timeout=10)
        if reval is None:
            raise RuntimeError("Timed out waiting for GLOBAL_POSITION_INT")
        return reval
        
    def getLocationLocal(self):
        reval = self.connection.recv_match(type='LOCAL_POSITION_NED', blocking=True, timeout=10)
        if reval is None:
            raise RuntimeError("Timed out waiting for LOCAL_POSITION_NED")
        return reval

    def arm(self):
        self.connection.mav.command_long_send(
            self.connection.target_system,
            self.connection.target_component,
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            0,
            1, 0, 0, 0, 0, 0, 0)

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
        reval = self.connection.recv_match(type='VFR_HUD', blocking=True, timeout=10)
        if reval is None:
            raise RuntimeError("Timed out waiting for VFR_HUD")
        return reval

    def waitAuto(self):
        while True:
            msg = self.connection.recv_match(type='HEARTBEAT', blocking=True, timeout=1)
            if msg is None:
                continue
            mode_flags = msg.base_mode
            if mode_flags & mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED:
                if self.connection.flightmode.upper() == 'AUTO':
                    break

    def getWPList(self):
        self.connection.waypoint_request_list_send()
        reval = self.connection.recv_match(type="MISSION_COUNT", blocking=True, timeout=10)
        if reval is None:
            raise RuntimeError("Timed out waiting for MISSION_COUNT")
        count = reval.count
        waypoints = []
        for seq in range(count):
            self.connection.mav.mission_request_int_send(
                target_system=self.connection.target_system,
                target_component=self.connection.target_component,
                seq=seq)
            reval = self.connection.recv_match(type="MISSION_ITEM_INT", blocking=True, timeout=10)
            if reval is None:
                raise RuntimeError("Timed out waiting for MISSION_ITEM_INT")
            waypoints.append(reval)

        return waypoints
    
    def assignWPs(self, waypoints):
        
        self.connection.waypoint_clear_all_send()
        self.connection.waypoint_count_send(len(waypoints))

        while True:
            reval = self.connection.recv_match(type="MISSION_REQUEST_INT", blocking=True, timeout=10)
            if reval is None:
                break
            seq = reval.seq

            self.connection.mav.mission_item_int_send(
                target_system=self.connection.target_system,
                target_component=self.connection.target_component,
                seq=seq,
                frame=mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT,
                command=mavutil.mavlink.MAV_CMD_NAV_WAYPOINT,
                current=0,
                autocontinue=1,
                param1=0,
                param2=0,
                param3=0,
                param4=0,
                x=waypoints[seq][0],
                y=waypoints[seq][1],
                z=0)

        ack = self.connection.recv_match(type="MISSION_ACK", blocking=True, timeout=10)
        if ack is None or ack.type != 0:
            raise RuntimeError("Mission upload failed")

        self.connection.mission_set_current_send(0)

