import os
import numpy as np
import torch
from torch.utils.data import Dataset

class SpeckTacDataset(Dataset):  # GPU上训练、验证、测试用的
    def __init__(self, root_dir, n_time_bins=30, resolution=(128, 128), dual_polarity=False, max_duration_us=None):
        self.samples = []  # list of (bin_path, label)
        self.n_time_bins = n_time_bins
        self.resolution = resolution
        self.dual_polarity = dual_polarity
        self.max_duration_us = max_duration_us  # <-- 新增参数

        for label_str in sorted(os.listdir(root_dir)):
            label_path = os.path.join(root_dir, label_str)
            if not os.path.isdir(label_path):
                continue
            label = int(label_str)
            for fname in os.listdir(label_path):
                if fname.endswith(".bin"):
                    self.samples.append((os.path.join(label_path, fname), label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        bin_path, label = self.samples[idx]
        events = self._bin_to_events(bin_path)
        frames = self._events_to_frames(events)
        return torch.from_numpy(frames).float(), label

    def _bin_to_events(self, path):
        data = np.fromfile(path, dtype='<u4').reshape(-1, 4)
        x, y, p, t = data[:, 0], data[:, 1], data[:, 2], data[:, 3]
        t_min = t.min()

        # 根据 max_duration_us 截断事件
        if self.max_duration_us is not None:
            mask = (t - t_min) <= self.max_duration_us
            x, y, p, t = x[mask], y[mask], p[mask], t[mask]

        return np.stack([t, x, y, p], axis=1)  # shape: [N, 4], order: [t, x, y, p]

    def _events_to_frames(self, events):
        T = self.n_time_bins
        H, W = self.resolution
        polarity_channels = 2 if self.dual_polarity else 1
        frames = np.zeros((T, polarity_channels, H, W), dtype=np.float32)

        t_min, t_max = events[:, 0].min(), events[:, 0].max()
        duration = t_max - t_min + 1
        bin_width = duration / T

        for t, x, y, p in events:
            bin_idx = int((t - t_min) / bin_width)
            bin_idx = min(bin_idx, T - 1)

            if self.dual_polarity:
                frames[bin_idx, int(p), y, x] += 1.0
            else:
                frames[bin_idx, 0, y, x] += 1.0

        return frames  # shape: [T, C, H, W]




class SpeckTacWindowDataset(Dataset):
    """
    与 SpeckTacDataset 类似，但只使用 [start_us, end_us] 时间窗内的事件（相对每个样本的 t_min）。
    缺省为 [300ms, 800ms]。若窗口内无事件，返回全零帧以避免报错。
    """
    def __init__(self,
                 root_dir,
                 n_time_bins=30,
                 resolution=(128, 128),
                 dual_polarity=False,
                 window_us=(300_000, 800_000)):
        self.samples = []  # list of (bin_path, label)
        self.n_time_bins = n_time_bins
        self.resolution = resolution
        self.dual_polarity = dual_polarity
        self.window_us = (int(window_us[0]), int(window_us[1]))

        for label_str in sorted(os.listdir(root_dir)):
            label_path = os.path.join(root_dir, label_str)
            if not os.path.isdir(label_path):
                continue
            label = int(label_str)
            for fname in os.listdir(label_path):
                if fname.endswith(".bin"):
                    self.samples.append((os.path.join(label_path, fname), label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        bin_path, label = self.samples[idx]
        events = self._bin_to_events(bin_path)

        # 窗口内无事件 -> 返回全零帧，保持下游稳定
        if events.size == 0:
            T = self.n_time_bins
            H, W = self.resolution
            C = 2 if self.dual_polarity else 1
            frames = np.zeros((T, C, H, W), dtype=np.float32)
            return torch.from_numpy(frames).float(), label

        frames = self._events_to_frames(events)
        return torch.from_numpy(frames).float(), label

    def _bin_to_events(self, path):
        # 与原类一致的读取方式：<u4，并按 [x, y, p, t] 读取后重排为 [t, x, y, p]
        data = np.fromfile(path, dtype='<u4').reshape(-1, 4)
        x, y, p, t = data[:, 0], data[:, 1], data[:, 2], data[:, 3]

        # 相对每个样本的最早时间 t_min 截取 [start, end] 微秒窗口（含端点）
        t_min = t.min()
        rel_t = t - t_min
        start_us, end_us = self.window_us
        mask = (rel_t >= start_us) & (rel_t <= end_us)

        if not np.any(mask):
            return np.empty((0, 4), dtype=np.uint32)

        x, y, p, t = x[mask], y[mask], p[mask], t[mask]
        return np.stack([t, x, y, p], axis=1)  # shape: [N, 4] -> [t, x, y, p]

    def _events_to_frames(self, events):
        # 完全模仿你的分桶逻辑：按事件内 t 的最小/最大确定 bin 宽度并累加到 [T, C, H, W]
        T = self.n_time_bins
        H, W = self.resolution
        polarity_channels = 2 if self.dual_polarity else 1
        frames = np.zeros((T, polarity_channels, H, W), dtype=np.float32)

        t_min, t_max = events[:, 0].min(), events[:, 0].max()
        duration = t_max - t_min + 1
        bin_width = duration / T

        for t, x, y, p in events:
            bin_idx = int((t - t_min) / bin_width)
            bin_idx = min(bin_idx, T - 1)

            if self.dual_polarity:
                frames[bin_idx, int(p), y, x] += 1.0
            else:
                frames[bin_idx, 0, y, x] += 1.0

        return frames  # [T, C, H, W]




import os
import numpy as np
import torch
from torch.utils.data import Dataset

class SpeckTacEventDataset(Dataset):  # speck devkit上用的
    def __init__(self, root_dir, max_duration_us=None):
        self.samples = []  # list of (bin_path, label)
        self.max_duration_us = max_duration_us

        for label_str in sorted(os.listdir(root_dir)):
            label_path = os.path.join(root_dir, label_str)
            if not os.path.isdir(label_path):
                continue
            label = int(label_str)
            for fname in os.listdir(label_path):
                if fname.endswith(".bin"):
                    self.samples.append((os.path.join(label_path, fname), label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        bin_path, label = self.samples[idx]
        events = self._bin_to_events(bin_path)
        # 返回每条事件为 dict 格式，方便你在 samna 中构建 Spike 对象
        event_list = [{'t': t, 'x': x, 'y': y, 'p': p} for t, x, y, p in events]
        return event_list, label

    def _bin_to_events(self, path):
        data = np.fromfile(path, dtype='<u4').reshape(-1, 4)
        x, y, p, t = data[:, 0], data[:, 1], data[:, 2], data[:, 3]
        t_min = t.min()

        # 根据 max_duration_us 截断事件
        if self.max_duration_us is not None:
            mask = (t - t_min) <= self.max_duration_us
            x, y, p, t = x[mask], y[mask], p[mask], t[mask]

        return np.stack([t, x, y, p], axis=1)  # shape: [N, 4]


import os
import numpy as np
from torch.utils.data import Dataset

class SpeckTacEventWindowDataset(Dataset):
    """
    返回 [start_us, end_us] 时间窗内的事件流（list of dict），窗口相对于每个样本的 t_min。
    默认窗口：300ms~800ms。
    """
    def __init__(self, root_dir, window_us=(300_000, 800_000)):
        self.samples = []  # list of (bin_path, label)
        self.start_us = int(window_us[0])
        self.end_us = int(window_us[1])
        if self.end_us <= self.start_us:
            raise ValueError("window_us 无效：end_us 必须大于 start_us")

        for label_str in sorted(os.listdir(root_dir)):
            label_path = os.path.join(root_dir, label_str)
            if not os.path.isdir(label_path):
                continue
            label = int(label_str)
            for fname in os.listdir(label_path):
                if fname.endswith(".bin"):
                    self.samples.append((os.path.join(label_path, fname), label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        bin_path, label = self.samples[idx]
        events = self._bin_to_events(bin_path)  # [N,4]: [t,x,y,p] (uint32)

        # 转为 list[dict]，与 SpeckTacEventDataset 一致
        event_list = [{'t': int(t), 'x': int(x), 'y': int(y), 'p': int(p)}
                      for t, x, y, p in events]
        return event_list, label

    def _bin_to_events(self, path):
        # 与你现有代码相同的读取 & 字段顺序
        data = np.fromfile(path, dtype='<u4').reshape(-1, 4)
        x, y, p, t = data[:, 0], data[:, 1], data[:, 2], data[:, 3]

        # 以样本最早时间为零点，截取 [start_us, end_us]（含端点）
        t_min = t.min()
        rel_t = t - t_min
        mask = (rel_t >= self.start_us) & (rel_t <= self.end_us)

        if not np.any(mask):
            return np.empty((0, 4), dtype=np.uint32)

        x, y, p, t = x[mask], y[mask], p[mask], t[mask]

        # 保险起见，按时间排序
        order = np.argsort(t)
        x, y, p, t = x[order], y[order], p[order], t[order]

        return np.stack([t, x, y, p], axis=1)  # [N,4] -> [t,x,y,p]
