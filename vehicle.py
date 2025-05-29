from pymavlink import mavutil
import time

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

    def getCompass(self):
        reval = self.connection.recv_match(type='VFR_HUD', blocking=True, timeout=10)
        if reval is None:
            raise RuntimeError("Timed out waiting for VFR_HUD")
        return reval

    def getHome(self):
        self.connection.mav.command_long_send(
            self.connection.target_system,
            self.connection.target_component,
            mavutil.mavlink.MAV_CMD_GET_HOME_POSITION,
            0,
            1, 0, 0, 0, 0, 0, 0)
        reval = self.connection.recv_match(type='HOME_POSITION', blocking=True, timeout=10)
        if reval is None:
            raise RuntimeError("Timed out waiting for HOME_POSITION")
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
            print("No mission uploaded on vehicle.")
            return
        count = reval.count
        waypoints = []
        for seq in range(1, count):
            self.connection.mav.mission_request_int_send(
                target_system=self.connection.target_system,
                target_component=self.connection.target_component,
                seq=seq)
            reval = self.connection.recv_match(type="MISSION_ITEM_INT", blocking=True, timeout=10)
            if reval is None:
                raise RuntimeError("Timed out waiting for MISSION_ITEM_INT")
            waypoints.append(reval)

        self.connection.mav.mission_ack_send(target_system=self.connection.target_system,
                target_component=self.connection.target_component,
                type=0)

        return waypoints

    def assignWPs(self, waypoints):
        self.connection.waypoint_clear_all_send()
        self.connection.waypoint_count_send(len(waypoints))

        sent = 0
        while sent < len(waypoints):
            reval = self.connection.recv_match(type=["MISSION_REQUEST_INT", "MISSION_REQUEST"], blocking=True, timeout=10)
            if reval is None:
                raise RuntimeError(f"Timeout waiting for MISSION_REQUEST_INT at seq {sent}")
            seq = reval.seq
            lat, lon = waypoints[seq][0], waypoints[seq][1]
            alt = waypoints[seq][2] if len(waypoints[seq]) > 2 else 0

            self.connection.mav.mission_item_int_send(
                self.connection.target_system,
                self.connection.target_component,
                seq,
                mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT,
                mavutil.mavlink.MAV_CMD_NAV_WAYPOINT,
                0,  # current
                1,  # autocontinue
                0, 0, 0, 0,
                lat,
                lon,
                alt
            )

            sent += 1

        ack = self.connection.recv_match(type="MISSION_ACK", blocking=True, timeout=10)
        if ack is None or ack.type != 0:
            raise RuntimeError("Mission upload failed or was rejected")

        self.connection.mav.command_long_send(
            self.connection.target_system,
            self.connection.target_component,
            mavutil.mavlink.MAV_CMD_DO_SET_MISSION_CURRENT,
            0, 1, 1,
            0, 0, 0, 0, 0
        )
        reval = self.connection.recv_match(type="MISSION_CURRENT", blocking=True, timeout=10)
        if reval is None:
            raise RuntimeError("Timeout waiting for MISSION_CURRENT")
        """else:
            print(reval.mission_state, reval.mission_mode, reval.seq)"""




