import os
import re
import random
import shutil

def split_dataset(
    data_dir: str,
    output_dir: str,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
):
    """
    按角度将 data_dir 下的 .bin 文件按 train/val/test 比例拆分到 output_dir 中，
    并在每个子集里为每个角度创建一个以标签（0,1,2,...）命名的文件夹。
    假定文件名格式为：
      angle<angle>_yOffset<offset>_pressDepth<depth>_sample<sample>.bin
    例如：angle12_yOffset-3.0_pressDepth2_sample01.bin
    """

    # 1. 检查比例是否合理
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-8:
        raise ValueError(f"train+val+test ratio must sum to 1.0 (got {total})")

    # 2. 正则提取 angle 字段
    pattern = re.compile(r'^angle(?P<angle>\d+)_yOffset.+\.bin$')

    # 3. 列出所有文件并按 angle 分组
    angle_groups = {}
    for fname in os.listdir(data_dir):
        if not fname.endswith('.bin'):
            continue
        m = pattern.match(fname)
        if not m:
            continue
        angle = int(m.group('angle'))
        angle_groups.setdefault(angle, []).append(fname)

    if not angle_groups:
        raise RuntimeError("在 data_dir 中未找到任何符合命名格式的 .bin 文件。")

    # 4. 为每个 angle 分配一个从 0 开始的标签
    sorted_angles = sorted(angle_groups.keys())
    label_map = {angle: idx for idx, angle in enumerate(sorted_angles)}

    # 5. 创建输出目录结构：output_dir/{train,val,test}/{label}/
    subsets = ['train', 'val', 'test']
    for subset in subsets:
        for label in label_map.values():
            path = os.path.join(output_dir, subset, str(label))
            os.makedirs(path, exist_ok=True)

    # 6. 固定随机种子
    random.seed(seed)

    # 7. 对每个角度组进行随机拆分并复制
    for angle in sorted_angles:
        files = angle_groups[angle]
        random.shuffle(files)
        n = len(files)

        n_train = int(n * train_ratio)
        n_val   = int(n * val_ratio)
        # 测试集取剩余
        n_test  = n - n_train - n_val

        train_files = files[:n_train]
        val_files   = files[n_train:n_train + n_val]
        test_files  = files[n_train + n_val:]

        label = label_map[angle]
        for fname in train_files:
            shutil.copy2(
                os.path.join(data_dir, fname),
                os.path.join(output_dir, 'train', str(label), fname)
            )
        for fname in val_files:
            shutil.copy2(
                os.path.join(data_dir, fname),
                os.path.join(output_dir, 'val',   str(label), fname)
            )
        for fname in test_files:
            shutil.copy2(
                os.path.join(data_dir, fname),
                os.path.join(output_dir, 'test',  str(label), fname)
            )

        # 打印当前角度/标签的拆分情况
        print(f"Angle {angle:3d} → label {label:2d} | "
              f"total: {n:4d} | "
              f"train: {len(train_files):3d} | "
              f"val: {len(val_files):3d} | "
              f"test: {len(test_files):3d}")

    print("==== 全部数据集拆分完成 ====")


if __name__ == "__main__":
    # —— 用户需修改这两行路径 —— #
    data_directory = 'data/Speck_DVS_data'
    output_directory = 'data_all_surface_split_planE'
    # —————————————————————— #

    split_dataset(
        data_dir=data_directory,
        output_dir=output_directory,
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=42
    )



# 改错误文件名的 例如：angle96_yOffset-0.8999999999999999_pressDepth2.7_sample01.bin → angle96_yOffset-0.9_pressDepth2.7_sample01.bin
# import os
# import re
#
# # TODO: 修改为你的数据文件所在目录
# data_dir = r'data/Speck_DVS_data'
#
# # 匹配原始文件名
# pattern = re.compile(
#     r'^angle(?P<angle>[-\d\.]+)_'
#     r'yOffset(?P<yOffset>[-\d\.]+)_'
#     r'pressDepth(?P<pressDepth>[-\d\.]+)_'
#     r'sample(?P<sample>\d+)\.bin$'
# )
#
# for fname in os.listdir(data_dir):
#     m = pattern.match(fname)
#     if not m:
#         # 跳过不符合命名格式的文件
#         print(f"Skipping: {fname}")
#         continue
#
#     # 解析并转换数值
#     angle = float(m.group('angle'))
#     y_offset = float(m.group('yOffset'))
#     press_depth = float(m.group('pressDepth'))
#     sample = int(m.group('sample'))
#
#     # 四舍五入到合理精度
#     angle_i = int(round(angle))                  # 整数度数
#     y_off_r = round(y_offset, 1)                 # 保留 0.1 mm
#     pdep_r = round(press_depth, 1)               # 保留 0.1 mm
#     sample_i = sample                            # 已经是整数
#
#     # 避免出现 "-0.0"
#     if abs(y_off_r) < 1e-6:
#         y_off_r = 0.0
#     if abs(pdep_r) < 1e-6:
#         pdep_r = 0.0
#
#     # 重新格式化文件名
#     new_name = (
#         f"angle{angle_i}"
#         f"_yOffset{y_off_r:.1f}"
#         f"_pressDepth{pdep_r:.1f}"
#         f"_sample{sample_i:02d}.bin"
#     )
#
#     src = os.path.join(data_dir, fname)
#     dst = os.path.join(data_dir, new_name)
#     if src != dst:
#         print(f"Renaming:\n  {fname}\n→ {new_name}")
#         os.rename(src, dst)
