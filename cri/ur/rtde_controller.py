"""Robot controller interface/implementations provide a common, low-level
interface to various robot arms.
"""

import warnings
import numpy as np

from cri.transforms import quat2axangle, axangle2quat
from cri.ur.rtde_client import RTDEClient
from cri.controller import RobotController


class RTDEController(RobotController):
    """UR RTDE controller class implements common interface to robot arms.
    
    Poses and coordinate frames are specified using 3D Cartesian positions
    and quaternion rotations.  This format makes it easy to perform
    coordinate transformations.
    """
    def __init__(self, ip='192.168.125.1'):
        self._ip = ip
        self._client = RTDEClient(ip)        
        try:   
            self.tcp = (0, 0, 0, 1, 0, 0, 0)    # base frame (quaternion)
            self.linear_accel = 500             # mm/s/s
            self.linear_speed = 20              # mm/s
            self.angular_accel = 50             # deg/s/s
            self.angular_speed = 20             # deg/s
            self.blend_radius = 0               # mm
        except:
            self._client.close()
            raise

    @property    
    def info(self):
        """Returns a unique robot identifier string.
        """
        return "ip: {}, info: {}".format(self._ip, self._client.get_info())

    @property    
    def tcp(self):
        """Returns the tool center point (TCP) of the robot.
        """
        return self._tcp

    @tcp.setter    
    def tcp(self, tcp):
        """Sets the tool center point (TCP) of the robot.
        """
        self._client.set_tcp(quat2axangle(tcp))
        self._tcp = tcp

    @property
    def linear_accel(self):
        """Returns the linear acceleration of the robot TCP (mm/s/s).
        """
        return self._linear_accel
    
    @linear_accel.setter    
    def linear_accel(self, accel):
        """Sets the linear acceleration of the robot TCP (mm/s/s).
        """
        self._client.set_linear_accel(accel)
        self._linear_accel = accel

    @property   
    def linear_speed(self):
        """Returns the linear speed of the robot TCP (mm/s).
        """
        return self._linear_speed

    @linear_speed.setter    
    def linear_speed(self, speed):
        """Sets the linear speed of the robot TCP (mm/s).
        """
        self._client.set_linear_speed(speed)
        self._linear_speed = speed

    @property
    def angular_accel(self):
        """Returns the angular acceleration of the robot TCP (deg/s/s).
        """
        return self._angular_accel

    @angular_accel.setter
    def angular_accel(self, accel):
        """Sets the angular acceleration of the robot TCP (deg/s/s).
        """
        self._client.set_angular_accel(accel)
        self._angular_accel = accel

    @property
    def angular_speed(self):
        """Returns the angular speed of the robot TCP (deg/s).
        """
        return self._angular_speed

    @angular_speed.setter
    def angular_speed(self, speed):
        """Sets the angular speed of the robot TCP (deg/s).
        """
        self._client.set_angular_speed(speed)
        self._angular_speed = speed

    @property
    def blend_radius(self):
        """Returns the robot blend radius (mm).
        """
        return self._blend_radius

    @blend_radius.setter    
    def blend_radius(self, blend_radius):
        """Sets the robot blend radius (mm).
        """
        self._client.set_blend_radius(blend_radius)
        self._blend_radius = blend_radius

    @property
    def joint_angles(self):
        """Returns the current joint angles.
        """
        return self._client.get_joint_angles()

    @property
    def commanded_joint_angles(self):
        """Returns the commanded joint angles.
        """
        return self._client.get_target_joint_angles()

    @property
    def joint_velocities(self):
        """Returns the current joint velocities.
        """
        joint_velocities = self._client.get_joint_speeds()
        return joint_velocities

    @property
    def commanded_joint_velocities(self):
        """Returns the commanded joint velocities.
        """
        joint_velocities = self._client.get_target_joint_speeds()
        return joint_velocities

    @property
    def pose(self):
        """Returns the current base frame pose.
        """
        return axangle2quat(self._client.get_pose())

    @property
    def commanded_pose(self):
        """Returns the commanded base frame pose.
        """
        return axangle2quat(self._client.get_target_pose())

    @property
    def linear_velocity(self):
        """Returns the linear velocity.
        """
        return self._client.get_linear_speed()

    @property
    def commanded_linear_velocity(self):
        """Returns the commanded linear velocity.
        """
        return self._client.get_target_linear_speed()

    @property
    def elbow(self):
        """Returns the current elbow angle.
        """
        warnings.warn("elbow property not implemented in RTDE controller")
        return None

    @property
    def commanded_elbow(self):
        """Returns the commanded elbow angle.
        """
        warnings.warn("elbow property not implemented in RTDE controller")
        return None

    def move_joints(self, joint_angles):
        """Executes an immediate move to the specified joint angles.
        """
        self._client.move_joints(joint_angles)

    def move_linear(self, pose, elbow=None):
        """Executes a linear/cartesian move from the current base frame pose to
        the specified pose.
        """
        if elbow is not None:
            warnings.warn("elbow property not implemented in RTDE controller")
        self._client.move_linear(quat2axangle(pose))

    def move_joints_velocity(self, joints_velocity, joints_accel, return_time=None):
        """Executes an immediate move to the specified joint velocities.

        Accelerate linearly in joint space and continue with constant joint
        speed. The return time is optional; if provided the function will
        return after that time, regardless of the target speed has been reached.
        If the return time is not provided, the function will return when the
        target speed is reached.

        joint_speeds = (jd0, jd1, jd2, jd3, jd4, jd5)
        jd0, jd1, jd2, jd3, jd4, jd5 are numbered from base to end effector and are
        measured in deg/s
        joint_accel is measured in deg/s/s (of leading axis)
        return_time is measured in secs before the function returns (optional)
        """
        joints_velocity = np.array(joints_velocity, dtype=np.float64).ravel()
        self._client.move_joint_speed(joints_velocity, joints_accel, return_time)

    def move_linear_velocity(self, linear_velocity, linear_accel, return_time=None):
        """Executes an immediate move to the specified linear velocity.

        Accelerate linearly in Cartesian space and continue with constant tool
        speed. The return time is optional; if provided the function will return after
        that time, regardless of the target speed has been reached. If the return time
        is not provided, the function will return when the target speed is reached

        linear speed = (xd, yd, zd, axd, ayd, azd)
        xd, yd, zd specify a translational velocity (mm/s)
        axd, ayd, azd specify an axis-angle rotational velocity (deg/s)
        linear_accel is measured in mm/s/s
        return_time is measured in secs before the function returns (optional)
        """
        linear_velocity = np.array(linear_velocity, dtype=np.float64).ravel()
        self._client.move_linear_speed(linear_velocity, linear_accel, return_time)

    def stop_joints_velocity(self, joints_accel):
        """Decelerate joints velocity to zero.

        joints_accel is measured in deg/s/s (of leading axis)
        """
        self._client.stop_joints(joints_accel)

    def stop_linear_velocity(self, linear_accel):
        """Decelerate linear speed to zero.

        linear_accel is measured in mm/s/s
        """
        self._client.stop_linear(linear_accel)

    def move_circular(self, via_pose, end_pose, elbow=None):
        """Executes a movement in a circular path from the current base frame
        pose, through via_pose, to end_pose.
        """
        if elbow is not None:
            warnings.warn("elbow property not implemented in RTDE controller")
        self._client.move_circular(quat2axangle(via_pose),
                                   quat2axangle(end_pose))
      
    def close(self):
        """Releases any resources held by the controller (e.g., sockets).
        """
        return self._client.close()
