from cri.robot import SyncRobot, AsyncRobot
from cri.controller import MG400Controller as Controller

from tactile_sensor_speck import SpeckTac
import time

import numpy as np

global_speed = 10


def main():
    base_frame = (0, 0, 0, 0, 0, 0)  # base frame在机械臂底座
    work_frame = (300, 0, 0, -180, 0, 0)  # 工作空间设为这个，在x轴上300位置，并且将坐标系绕x旋转180.导致y轴左右颠倒，z轴上下颠倒


    with AsyncRobot(SyncRobot(Controller())) as robot, SpeckTac(duration=3) as sensor:
        # Set TCP, linear speed,  angular speed and coordinate frame
        robot.coord_frame = work_frame  # 参考坐标系设为工作空间
        robot.tcp = (0, 0, 20, 0, 0, 0)  # 机械臂末端工具距离法兰盘的距离，设置这个会使运动命令的目标位置是末端工具的位置，而不是法兰盘的位置
        robot.speed = global_speed

        for angle in range(15):  # 15个角度
            robot.speed = global_speed  # 除了按压抬起的时候慢点，其他时候大幅移动尽量快点
            # 以工作空间为参考坐标系，将物体移动到传感器尖端中心点上方并旋转到特定角度
            robot.coord_frame = work_frame
            robot.async_move_linear((-14.5, -14.5, 122.5, 0, 0, angle*12))
            robot.async_result()  # 对于异步移动来说，这两句话是绑定的，中间能插运动无关的命令，边运动边执行
            # 将上一个运动的目标位置设为新的参考坐标系
            robot.coord_frame = base_frame  # 直接切换到新的位置的坐标系是危险的，因此需要过渡一下
            robot.coord_frame = robot.target_pose
            for y_offset in np.linspace(-3, 3, 61):  # 左右偏移
                robot.async_move_linear((0, y_offset, 0, 0, 0, 0))
                robot.async_result()
                for press_depth in np.linspace(1,2,11):
                    for repeat_times in range(1):  # 每个角度重复次数
                        print(f"第{angle+1}个角度，第{y_offset}个偏移，第{press_depth}个深度，第{repeat_times+1}次：开始...")
                        robot.speed = 3  # 下压抬起的时候速度慢一些
                        # 下压
                        robot.async_move_linear((0, y_offset, press_depth+3, 0, 0, 0))
                        thread = sensor.threaded_record_once(f"angle{angle*12}_yOffset{y_offset}_pressDepth{press_depth+1}_sample{repeat_times+1:02d}.bin")  # 保存到/home/Speck_DVS_data/xxx.bin
                        robot.async_result()
                        # 抬起
                        robot.async_move_linear((0, y_offset, 0, 0, 0, 0))
                        robot.async_result()
                        thread.join()

        # 回到工作空间原点
        robot.speed = global_speed
        robot.coord_frame = work_frame
        robot.async_move_linear((0, 0, 0, 0, 0, 0))
        robot.async_result()


if __name__ == '__main__':
    main()