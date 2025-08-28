"""Robot controller interface/implementations provide a common, low-level
interface to various robot arms.
"""

import warnings

from cri.dobot.magician_client import MagicianClient
from cri.controller import RobotController


class MagicianController(RobotController):
    """Dobot Magician controller class implements common interface robot arms.   
    
    Poses and coordinate frames are specified using 3D Cartesian positions
    and quaternion rotations.  This format makes it easy to perform
    coordinate transformations.
    """

    def __init__(self, port=""):
        self._port = port
        self._client = MagicianClient(port)
        try:
            self.tcp = (0, 0, 0, 1, 0, 0, 0)     # base frame (quaternion)
            self.linear_speed = 100              # mm/s
            self.angular_speed = 100             # deg/s

            self._commanded_joint_angles = None
            self._commanded_pose = None
        except:
            self._client.close()
            raise

    @property
    def info(self):
        """Returns a unique robot identifier string.
        """
        return "port: {}, {}".format(
                self._port,
                self._client.get_info(),
                )

    @property    
    def tcp(self):
        """Returns the tool center point (TCP) of the robot.
        """
        return self._tcp

    @tcp.setter    
    def tcp(self, tcp):
        """Sets the tool center point (TCP) of the robot.
        """
        self._client.set_tcp(tcp)
        self._tcp = tcp

    @property
    def linear_speed(self):
        """Returns the linear speed of the robot TCP (mm/s).
        """
        return self._client.get_speed()[0]

    @linear_speed.setter
    def linear_speed(self, speed):
        """Sets the linear speed of the robot TCP (mm/s).
        """
        try:
            self._angular_speed
        except AttributeError:
            self._client.set_speed(linear_speed=speed, 
                                   angular_speed=100)
        else:
            self._client.set_speed(linear_speed=speed, 
                                   angular_speed=self._angular_speed)
        self._linear_speed = speed

    @property
    def angular_speed(self):
        """Returns the angular speed of the robot TCP (deg/s).
        """
        return self._client.get_speed()[1]
    
    @angular_speed.setter
    def angular_speed(self, speed):
        """Sets the angular speed of the robot TCP (deg/s).
        """
        try:
            self._linear_speed
        except AttributeError:
            self._client.set_speed(linear_speed=100, 
                                   angular_speed=speed)
        else:
            self._client.set_speed(linear_speed=self._linear_speed, 
                                   angular_speed=speed)
        self._angular_speed = speed

    @property
    def blend_radius(self):
        """Returns the robot blend radius (mm).
        """
        warnings.warn("blend_radius property not implemented in Dobot Magician Controller")
        pass
    
    @blend_radius.setter
    def blend_radius(self, blend_radius):
        """Sets the robot blend radius (mm).
        """
        warnings.warn("blend_radius property not implemented in Dobot Magician Controller")
        pass

    @property
    def joint_angles(self):
        """Returns the robot joint angles.
        """
        return self._client.get_joint_angles()

    @property
    def commanded_joint_angles(self):
        """ Returns the commanded joint angles.
        """
        return self._commanded_joint_angles

    @property
    def pose(self):
        """Returns the TCP pose in the reference coordinate frame.
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
        warnings.warn("elbow property not implemented in Dobot Magician Controller")
        return None

    def commanded_elbow(self):
        """Returns the commanded elbow angle.
        """
        warnings.warn("elbow property not implemented in Dobot Magician controller")
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
            warnings.warn("elbow property not implemented in Dobot Magician Controller")
        self._commanded_pose = pose
        self._client.move_linear(pose)

    def move_circular(self, via_pose, end_pose, elbow=None):
        """Executes a movement in a circular path from the current base frame
        pose, through via_pose, to end_pose.
        """
        if elbow is not None:
            warnings.warn("elbow property not implemented in Dobot Magician Controller")
        warnings.warn("move_circular method not implemented in Dobot Magician Controller")
        pass
    
    def close(self):
        """Releases any resources held by the controller (e.g., sockets).
        """
        self._client.close()
        
