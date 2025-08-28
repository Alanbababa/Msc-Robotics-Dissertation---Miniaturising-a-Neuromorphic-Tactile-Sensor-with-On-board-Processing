"""Robot controller interface/implementations provide a common, low-level
interface to various robot arms.
"""

import warnings

from cri.dobot.mg400_client import MG400Client
from cri.controller import RobotController


class MG400Controller(RobotController):
    """Dobot MG400 controller class implements common interface robot arms.   
    
    Poses and coordinate frames are specified using 3D Cartesian positions
    and quaternion rotations.  This format makes it easy to perform
    coordinate transformations.
    """

    def __init__(self, ip="192.168.1.6"):
        self._ip = ip
        self._client = MG400Client(ip)
        try:
            # self.tcp = (0, 0, 0, 1, 0, 0, 0)     # base frame (quaternion)
            # self.linear_speed = 10               # mm/s

            self._commanded_joint_angles = None
            self._commanded_pose = None
        except:
            self._client.close()
            raise

    @property
    def info(self):
        """Returns a unique robot identifier string.
        """
        return "ip: {}, {}".format(
                self._ip,
                self._client.get_info(),
                )

    @property
    def tcp(self):
        """Returns the tool center point (TCP) of the robot.
        """
        warnings.warn("TCP property not implemented in MG400 API.")
        return None

    @tcp.setter
    def tcp(self, tcp):
        """Sets the tool center point (TCP) of the robot.
        """
        warnings.warn("TCP property not implemented in MG400 API.")

    @property
    def linear_speed(self):
        """Returns the linear speed of the robot TCP (% of max).
        """
        warnings.warn("MG400 API only reads a single global speed.")
        return self._client.get_speed()
    
    @linear_speed.setter
    def linear_speed(self, speed):
        """Sets the linear speed of the robot TCP (% of max).
        """
        warnings.warn("Linear speed not working in MG400 API.")
        self._client.set_linear_speed(speed)

    @property
    def angular_speed(self):
        """Returns the angular speed of the robot TCP (% of max).
        """
        warnings.warn("MG400 API only reads a single global speed.")
        return self._client.get_speed()

    @angular_speed.setter
    def angular_speed(self, speed):
        """Sets the angular speed of the robot TCP (% of max).
        """
        warnings.warn("Angular speed not working in MG400 API.")
        self._client.set_angular_speed(speed)

    @property
    def speed(self):
        """Returns the speed of the robot TCP (% of max).
        """
        return self._client.get_speed()

    @speed.setter
    def speed(self, speed):
        """Sets the speed of the robot TCP (% of max).
        """
        self._client.set_speed(speed)

    @property
    def blend_radius(self):
        """Returns the robot blend radius (mm).
        """
        warnings.warn("blend_radius property not implemented in Dobot MG400 Controller")
        pass
    
    @blend_radius.setter
    def blend_radius(self, blend_radius):
        """Sets the robot blend radius (mm).
        """
        warnings.warn("blend_radius property not implemented in Dobot MG400 Controller")
        pass

    @property
    def joint_angles(self):
        """Returns the current joint angles.
        """
        return self._client.get_joint_angles()

    @property
    def commanded_joint_angles(self):
        """ Returns the commanded joint angles.
        """
        return self._commanded_joint_angles

    @property
    def pose(self):
        """Returns the current TCP pose.
        """
        return self._client.get_pose()

    @property
    def commanded_pose(self):
        """Returns the commanded TCP pose.
        """
        return self._commanded_pose

    @property
    def elbow(self):
        """Returns the current elbow angle.
        """
        warnings.warn("elbow property not implemented in Dobot MG400 Controller")
        return None

    def commanded_elbow(self):
        """Returns the commanded elbow angle.
        """
        warnings.warn("elbow property not implemented in Dobot MG400 controller")
        return None

    def move_joints(self, joint_angles):
        """Executes an immediate move to the specified joint angles.
        """
        self._commanded_joint_angles = joint_angles
        self._client.move_joints(joint_angles)

    def move_linear(self, pose, elbow=None):
        """Executes linear/Cartesian move from base frame pose to specified pose.
        """
        if elbow is not None:
            warnings.warn("elbow property not implemented in Dobot MG400 Controller")
        self._commanded_pose = pose
        self._client.move_linear(pose)

    def move_circular(self, via_pose, end_pose, elbow=None):
        """Executes a movement in a circular path from the current base frame
        pose, through via_pose, to end_pose.
        """
        if elbow is not None:
            warnings.warn("elbow property not implemented in Dobot MG400 Controller")
        warnings.warn("move_circular not working in Dobot API")
        self._client.move_circular(via_pose, end_pose)
    
    def close(self):
        """Releases any resources held by the controller (e.g., sockets).
        """
        self._client.close()

