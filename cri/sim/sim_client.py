"""Python client interface for Pybullet sim.
"""
import numpy as np

from cri.transforms import euler2quat, quat2euler


class SimClient:
    """Python client interface for Pybullet sim.
    """

    def __init__(self, robot_arm):
        self.set_units('inv_meters', 'inv_radians')
        self.connect(robot_arm)

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
        units_l = {'millimeters': 1.0,
                   'inv_meters': 0.001,  # for pybullet
                   'meters': 1000.0,
                   'inches': 25.4,
                   }
        units_a = {'degrees': 1.0,
                   'inv_radians': np.pi/180,  # for pybullet
                   'radians': 180/np.pi,
                   }
        self._scale_linear = units_l[linear]
        self._scale_angle = units_a[angular]

    def connect(self, robot_arm):
        """Connects to simulation
        """
        self._sim_robot_arm = robot_arm
        print("Client connected ...")

    def get_info(self):
        """returns a unique robot identifier string.
        """
        return self._sim_robot_arm.name

    def move_joints(self, joint_angles):
        """Executes an immediate move to the specified joint angles.
        joint_angles = (j0, j1, j2, j3, j4, j5)
        j0, j1, j2, j3, j4, j5 are numbered from base to end effector and are
        measured in degrees (default)
        """
        joint_angles = np.array(joint_angles, dtype=np.float32).ravel()
        joint_angles *= self._scale_angle
        self._sim_robot_arm.move_joints(joint_angles, quick_mode=False)

    def move_linear(self, pose):
        """Executes a linear/cartesian move from the current base frame pose to
        the specified pose.
        pose = (x, y, z, qw, qx, qy, qz)
        x, y, z specify a Cartesian position (default mm)
        qw, qx, qy, qz specify a quaternion rotation
        """
        pose = np.array(pose, dtype=np.float32).ravel()
        pose = quat2euler(pose, 'sxyz')
        pose[:3] *= self._scale_linear
        pose[3:] *= self._scale_angle
        self._sim_robot_arm.move_linear(pose, quick_mode=False)

    def get_joint_angles(self):
        """returns the robot joint angles.

        joint_angles = (j0, j1, j2, j3, j4, j5)
        j0, j1, j2, j3, j4, j5 are numbered from base to end effector and are
        measured in degrees (default)
        """
        joint_angles = self._sim_robot_arm.get_joint_angles()
        joint_angles /= self._scale_angle
        return joint_angles

    def get_pose(self):
        """returns the TCP pose in the reference coordinate frame.

        pose = (x, y, z, qw, qx, qy, qz)
        x, y, z specify a Cartesian position (default mm)
        qw, qx, qy, qz specify a quaternion rotation
        """
        pose = self._sim_robot_arm.get_tcp_pose()
        pose[:3] /= self._scale_linear
        pose[3:] /= self._scale_angle
        pose = euler2quat(pose, 'sxyz')
        return pose

    def close(self):
        """Releases any resources held by the controller (e.g., sockets).
        """
        print("Shutting down client ...")
        self._sim_robot_arm.close()
