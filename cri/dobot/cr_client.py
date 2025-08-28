"""Python client interface for Dobot CR using sockets.
Firmware version 3.5.2.8
Uses https://github.com/Dobot-Arm/TCP-IP-CR-Python-CMD
"""

import threading
import numpy as np
from time import sleep

from cri.dobot.cr.dobot_api import DobotApiDashboard, DobotApi, DobotApiMove, MyType
from cri.transforms import euler2quat, quat2euler


class CRClient:
    """Python client interface for Dobot CR
    """

    class CommandFailed(RuntimeError):
        pass

    class InvalidZone(ValueError):
        pass

    def __init__(self, ip="192.168.5.1"):
        self.set_units('millimeters', 'degrees')     
        self.connect(ip)

    def __repr__(self):
        return "{} ({})".format(self.__class__.__name__, self.get_info())

    def __str__(self):
        return self.__repr__()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()        
        
    def set_units(self, linear, angular):
        """Sets linear and angular units.
        """
        units_l = {'millimeters' : 1.0,
               'meters' : 1000.0,
               'inches' : 25.4,
               }
        units_a = {'degrees' : 1.0,
               'radians' : 57.2957795,
               }
        self._scale_linear = units_l[linear]
        self._scale_angle  = units_a[angular]

    def connect(self, ip):
        """Connects to Dobot CR.
        """
        self._dashboard = DobotApiDashboard(ip, 29999)
        self._move = DobotApiMove(ip, 30003)
        self._feed = DobotApi(ip, 30005)

        self._dashboard.ClearError()
        self._dashboard.EnableRobot()
        self._dashboard.User(0)
        self._dashboard.Tool(0)

        feed_thread = threading.Thread(target=self.get_feed, args=(self._feed,))
        feed_thread.setDaemon(True)
        feed_thread.start()

    def get_feed(self, feed: DobotApi):
        global feedback
        hasRead = 0
        while True:
            data = bytes()
            while hasRead < 1440:
                temp = feed.socket_dobot.recv(1440 - hasRead)
                if len(temp) > 0:
                    hasRead += len(temp)
                    data += temp
            hasRead = 0

            a = np.frombuffer(data, dtype=MyType)
            if hex((a['test_value'][0])) == '0x123456789abcdef':

                # Refresh Properties
                feedback = a[0]

            sleep(0.001)

    def get_info(self):
        """Returns a unique robot identifier string.
        """
        pass

    def set_servo_mode(self, servo_mode):
        """Sets whether move commands will use servo. 
        """
        if type(servo_mode) is not bool: 
            raise Exception("Servo mode must be [True, False]")
        self._servo_mode = servo_mode

    def get_servo_mode(self):
        """Returns whether move commands will use servo.
        """
        return self._servo_mode

    def set_servo_delay(self, servo_delay):
        """Sets time delay for servo move commands.
        """
        self._servo_delay = servo_delay

    def get_servo_delay(self):
        """Gets time delay for servo move commands. 
        """
        return self._servo_delay

    def move_joints(self, joint_angles):
        """Executes an immediate move to the specified joint angles.

        joint_angles = (j0, j1, j2, rx, ry, rz)
        j0, j1, j2, rx, ry, rz are numbered from base to end effector and are
        measured in degrees (default)
        """       
        joint_angles = np.array(joint_angles, dtype=np.float32).ravel()
        joint_angles *= self._scale_angle 
        if self._servo_mode:
            self._move.ServoJ(*joint_angles)
            sleep(self._servo_delay)
        else:
            self._move.JointMovJ(*joint_angles)
            self._move.Sync()

    def move_linear(self, pose):
        """Executes a linear/cartesian move from the current base frame pose to
        the specified pose.

        pose = (x, y, z, rx, ry, rz)
        x, y, z specify a Euclidean position (default mm)
        """
        pose = np.array(pose, dtype=np.float32).ravel()
        pose = quat2euler(pose, 'sxyz')
        pose[:3] *= self._scale_linear
        pose[3:] *= self._scale_angle
        if self._servo_mode:
            self._move.ServoP(*pose)
            sleep(self._servo_delay)
        else:    
            self._move.MovL(*pose)
            self._move.Sync()

    def set_linear_speed(self, linear_speed):
        """Sets the linear speed (% of maximum)
        """
        if linear_speed < 1 or linear_speed > 100: 
            raise Exception("Linear speed value outside range of 1-100%")
        self._dashboard.SpeedL(round(linear_speed))
        self._move.Sync()

    def set_angular_speed(self, angular_speed):
        """Sets the angular speed (% of maximum)
        """
        if angular_speed < 1 or angular_speed > 100: 
            raise Exception("Angular speed value outside range of 1-100%")
        self._dashboard.SpeedJ(round(angular_speed))
        self._move.Sync()

    def set_speed(self, speed):
        """Sets the angular speed (% of maximum)
        """
        if speed < 1 or speed > 100: 
            raise Exception("Speed value outside range of 1-100%")
        self._dashboard.SpeedFactor(round(speed))
        self._move.Sync()

    def get_speed(self):
        """Gets the speed (% of maximum)
        """
        return feedback["speed_scaling"] 
    
    def get_joint_angles(self):
        """returns the robot joint angles.

        joint_angles = (j0, j1, j2, rx, ry, rz)
        j0, j1, j2, rx, ry, rz are numbered from base to end effector and are
        measured in degrees (default)
        """
        return feedback["q_actual"] / self._scale_angle

    def get_pose(self):
        """retvalsurns the TCP pose in the reference coordinate frame.

        pose = (x, y, z, rx, ry, rz)
        x, y, z specify a Euclidean position (default mm)
        rz rotation of end effector
        """
        pose_actual = feedback["tool_vector_actual"]
        pose = np.array(pose_actual, dtype=np.float32)
        pose[:3] /= self._scale_linear        
        pose[3:] /= self._scale_angle        
        pose = euler2quat(pose, 'sxyz')
        return pose

    def move_circular(self, via_pose, end_pose):
        """Executes a movement in a circular path from the current base frame
        pose, through via_pose, to end_pose.
        
        via_pose, end_pose = (x, y, z, qw, qx, qy, qz)
        x, y, z specify a Cartesian position (default mm)
        qw, qx, qy, qz specify a quaternion rotation
        """     
        via_pose = np.array(via_pose, dtype=np.float32).ravel()
        via_pose = quat2euler(via_pose, 'sxyz')
        via_pose[:3] *= self._scale_linear
        via_pose[3:] *= self._scale_angle
        end_pose = np.array(end_pose, dtype=np.float32).ravel()
        end_pose = quat2euler(end_pose, 'sxyz')
        end_pose[:3] *= self._scale_linear   
        end_pose[3:] *= self._scale_angle   
        pose = self.get_pose()
        pose = quat2euler(pose, 'sxyz')
        pose[:3] *= self._scale_linear   
        pose[3:] *= self._scale_angle   
        self._move.MovL(*pose)
        self._move.Circle(1, *via_pose, *end_pose) # does nothing
        self._move.Sync()

    def close(self):
        """Releases any resources held by the controller (e.g., sockets). And disconnects from Dobot magician
        """
        self._move.Sync()
        self._dashboard.ClearError()
        self._dashboard.close()
        print("Disconnecting Dobot...")
