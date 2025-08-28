"""Robot controller interface/implementations provide a common, low-level
interface to various robot arms.
"""

from cri.controller import RobotController


class DummyController(RobotController):
    """Dummy controller class does nothing for testing purposes
    """

    def __init__(self, ip='127.0.0.1', port=5000):
        return None

    @property
    def info(self):
        """Returns a unique robot identifier string.
        """
        return "Dummy controller: does nothing"

    @property    
    def tcp(self):
        """Returns the tool center point (TCP) of the robot.
        """
        return None

    @tcp.setter    
    def tcp(self, tcp):
        """Sets the tool center point (TCP) of the robot.
        """
        pass

    @property
    def linear_speed(self):
        """Returns the linear speed of the robot TCP (mm/s).
        """
        return None

    @linear_speed.setter
    def linear_speed(self, speed):
        """Sets the linear speed of the robot TCP (mm/s).
        """
        pass

    @property
    def angular_speed(self):
        """Returns the angular speed of the robot TCP (deg/s).
        """
        return None

    @angular_speed.setter
    def angular_speed(self, speed):
        """Sets the angular speed of the robot TCP (deg/s).
        """
        pass

    @property
    def blend_radius(self):
        """Returns the robot blend radius (mm).
        """
        pass
    
    @blend_radius.setter
    def blend_radius(self, blend_radius):
        """Sets the robot blend radius (mm).
        """
        pass

    @property
    def joint_angles(self):
        """Returns the robot joint angles.
        """
        return [0,0,0,0,0,0]

    @property
    def commanded_joint_angles(self):
        """ Returns the commanded joint angles.
        """
        return [0,0,0,0,0,0]
        
    @property
    def pose(self):
        """Returns the TCP pose in the reference coordinate frame.
        """
        return [0,0,0,0,0,0,1]

    @property
    def commanded_pose(self):
        """Returns the commanded TCP pose.
        """
        return [0,0,0,0,0,0,1]

    @property
    def elbow(self):
        """Returns the current elbow angle.
        """
        return None

    def commanded_elbow(self):
        """Returns the commanded elbow angle.
        """
        return None

    def move_joints(self, joint_angles):
        """Executes an immediate move to the specified joint angles.
        """
        pass

    def move_linear(self, pose, elbow=None):
        """Executes linear/Cartesian move from base frame pose to specified pose.
        """
        pass

    def move_circular(self, via_pose, end_pose, elbow=None):
        """Executes a movement in a circular path from the current base frame
        pose, through via_pose, to end_pose.
        """
        pass
    
    def close(self):
        """Releases any resources held by the controller (e.g., sockets).
        """
        pass