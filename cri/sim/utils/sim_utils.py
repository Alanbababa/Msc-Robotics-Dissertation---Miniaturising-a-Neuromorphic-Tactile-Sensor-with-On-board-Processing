import numpy as np

from tactile_sim.utils.pybullet_utils import connect_pybullet
from tactile_sim.utils.pybullet_utils import load_standard_environment
from tactile_sim.utils.pybullet_utils import set_debug_camera
from tactile_sim.embodiments import create_embodiment
from tactile_sim.assets.default_rest_poses import rest_poses_dict


def setup_pybullet_env():
    timestep = 1/240.0
    show_gui = True
    
    embodiment_type = 'arm'        #  ['arm', 'tactile_arm', 'visual_arm', 'visuotactile_arm']
    arm_type = 'ur5'               # ['ur5', 'franka_panda', 'kuka_iiwa', 'cr3', 'mg400']
    sensor_type ='standard_tactip' #  ['standard_tactip', 'standard_digit', 'standard_digitac', 'mini_tactip', 'flat_tactip', 'right_angle_tactip', 'right_angle_digit', 'right_angle_digitac']

    # define sensor parameters
    robot_arm_params = {
        "type": arm_type,
        "rest_poses": rest_poses_dict[arm_type],
        "tcp_lims": np.column_stack([-np.inf * np.ones(6), np.inf * np.ones(6)]),
    }

    tactile_sensor_params = {
        "type": sensor_type,
        "core": "no_core",
        "dynamics": {},  # {'stiffness': 50, 'damping': 100, 'friction': 10.0},
        "image_size": (128, 128),
        "turn_off_border": False,
        "show_tactile": False,
    }

    # set debug camera position
    visual_sensor_params = {
        'image_size': [128, 128],
        'dist': 1.0,
        'yaw': 90.0,
        'pitch': -25.0,
        'pos': [0.6, 0.0, 0.0525],
        'fov': 75.0,
        'near_val': 0.1,
        'far_val': 100.0,
        'show_vision': False
    }

    pb = connect_pybullet(timestep, show_gui)
    load_standard_environment(pb)
    embodiment = create_embodiment(
        pb,
        embodiment_type,
        robot_arm_params,
        tactile_sensor_params,
        visual_sensor_params
    )
    set_debug_camera(pb, visual_sensor_params)
    return embodiment


if __name__ == '__main__':
    pass
