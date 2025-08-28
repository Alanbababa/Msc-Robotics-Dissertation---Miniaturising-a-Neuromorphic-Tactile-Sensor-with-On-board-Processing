#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm

import sinabs
import sinabs.layers as sl
from sinabs.activation.surrogate_gradient_fn import PeriodicExponential
from MyDataset import SpeckTacDataset, SpeckTacWindowDataset, SpeckTacEventWindowDataset

# -----------------------------------------------------------------------------
# 超参
# -----------------------------------------------------------------------------
BATCH_SIZE   = 4
N_TIME_BINS  = 50
NUM_WORKERS  = 16
EPOCHES      = 4
LR           = 1e-4
PATIENCE     = 5
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BIT_WIDTH    = 8         # 量化位宽（权重/偏置：int8）
STATE_BITS   = 16        # 新增：阈值/状态位宽（int16）——用于共同scale约束

torch.cuda.empty_cache()

# reproducibility
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark     = False

# -----------------------------------------------------------------------------
# FakeQuant / QAT 基础组件
# -----------------------------------------------------------------------------
class FakeQuantizeSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale, zero_point, qmin, qmax):
        q = torch.clamp(torch.round(x / scale) + zero_point, qmin, qmax)
        return (q - zero_point) * scale

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None, None, None


class FakeQuantizer(nn.Module):
    def __init__(self, bitwidth=8, symmetric=True, per_channel=False, ch_axis=0):
        super().__init__()
        self.bitwidth    = bitwidth
        self.symmetric   = symmetric
        self.per_channel = per_channel
        self.ch_axis     = ch_axis

        # 量化上下界
        if symmetric:
            self.qmin = -(2**(bitwidth-1) - 1)
            self.qmax =  2**(bitwidth-1) - 1
        else:
            self.qmin = 0
            self.qmax = 2**bitwidth - 1

        # per-tensor 模式下，直接注册一个 scalar buffer
        if not per_channel:
            self.register_buffer('scale',      torch.tensor(1.0))
            self.register_buffer('zero_point', torch.tensor(0, dtype=torch.int64))
        # per-channel 下，不事先创建 scale / zero_point buffer
        # （注册留到第一次 update_params 时做）

    @torch.no_grad()
    def update_params(self, x: torch.Tensor):
        # 统计 min/max
        if self.per_channel:
            dims  = [i for i in range(x.dim()) if i != self.ch_axis]
            x_min = x.amin( dim=dims, keepdim=True )
            x_max = x.amax( dim=dims, keepdim=True )
        else:
            x_min = x.min()
            x_max = x.max()

        # 计算 scale / zero_point
        if self.symmetric:
            max_val    = torch.max(x_min.abs(), x_max.abs())
            scale      = max_val / self.qmax
            zero_point = torch.zeros_like(scale, dtype=torch.int64)
        else:
            scale      = (x_max - x_min) / float(self.qmax - self.qmin)
            zero_point = (torch.round(-x_min/scale)
                          .to(torch.int64) + self.qmin)

        # 防止 scale 为 0
        scale = torch.where(scale>0, scale, torch.ones_like(scale))

        # per-channel 首次调用时动态注册 buffer，后续直接 copy_
        if self.per_channel:
            if not hasattr(self, 'scale'):
                # buffer 名不存在，第一次注册
                self.register_buffer('scale',      scale)
                self.register_buffer('zero_point', zero_point)
            else:
                # 已有 buffer，直接更新数据
                self.scale.data.copy_(scale)
                self.zero_point.data.copy_(zero_point)
        else:
            # per-tensor 模式直接更新
            self.scale.data.copy_(scale)
            self.zero_point.data.copy_(zero_point)

    def forward(self, x: torch.Tensor):
        # 保存输入用于 calibrate
        self.last_input = x.detach()

        # 如果 per-channel 且还没校准，直接直通
        if self.per_channel and not hasattr(self, 'scale'):
            return x

        # reshape scale/zero_point
        if self.per_channel:
            view_shape = [
                x.size(self.ch_axis) if i == self.ch_axis else 1
                for i in range(x.dim())
            ]
            sp = self.scale.view(view_shape)
            zp = self.zero_point.view(view_shape)
        else:
            sp, zp = self.scale, self.zero_point

        # 调用 STE 版本的 fake-quant，保留梯度
        return FakeQuantizeSTE.apply(x, sp, zp, self.qmin, self.qmax)


class QuantConv2d(nn.Module):
    def __init__(self, conv: nn.Conv2d, w_q: FakeQuantizer, a_q: FakeQuantizer):
        super().__init__()
        self.stride, self.padding, self.dilation, self.groups = \
            conv.stride, conv.padding, conv.dilation, conv.groups
        self.weight = conv.weight
        self.bias   = conv.bias
        self.w_q    = w_q
        self.a_q    = a_q

    def forward(self, x):
        x_q = self.a_q(x)
        w_q = self.w_q(self.weight)
        return F.conv2d(x_q, w_q, self.bias,
                        self.stride, self.padding,
                        self.dilation, self.groups)


class QuantLinear(nn.Module):
    def __init__(self, lin: nn.Linear, w_q: FakeQuantizer, a_q: FakeQuantizer):
        super().__init__()
        self.in_features  = lin.in_features
        self.out_features = lin.out_features
        self.weight       = lin.weight
        self.bias         = lin.bias
        self.w_q          = w_q
        self.a_q          = a_q

    def forward(self, x):
        x_q = self.a_q(x)
        w_q = self.w_q(self.weight)
        return F.linear(x_q, w_q, self.bias)


def convert_qat_module(orig: nn.Sequential, bitwidth=8):
    """
    【改动点】把 per_channel=True 改为 False → per-tensor 权重量化
    """
    modules = []
    for m in orig:
        if isinstance(m, nn.Conv2d):
            wq = FakeQuantizer(
                bitwidth   = bitwidth,
                symmetric  = True,
                per_channel= False,   # <<< 改为 per-tensor
                ch_axis    = 0,
            )
            aq = nn.Identity()     # 激活直通
            modules.append(QuantConv2d(m, wq, aq))

        elif isinstance(m, nn.Linear):
            wq = FakeQuantizer(
                bitwidth   = bitwidth,
                symmetric  = True,
                per_channel= False,   # <<< 改为 per-tensor
                ch_axis    = 0,
            )
            aq = nn.Identity()
            modules.append(QuantLinear(m, wq, aq))

        else:
            modules.append(m)

    return nn.Sequential(*modules)


# -----------------------------------------------------------------------------
# 数据加载
# -----------------------------------------------------------------------------
train_dataset = SpeckTacWindowDataset("data_all_surface_split_planE/train",
                                n_time_bins=N_TIME_BINS,
                                resolution=(128, 128),
                                dual_polarity=True,
                                window_us=(300_000, 800_000))
val_dataset   = SpeckTacWindowDataset("data_all_surface_split_planE/val",
                                n_time_bins=N_TIME_BINS,
                                resolution=(128, 128),
                                dual_polarity=True,
                                window_us=(300_000, 800_000))
test_dataset  = SpeckTacWindowDataset("data_all_surface_split_planE/test",
                                n_time_bins=N_TIME_BINS,
                                resolution=(128, 128),
                                dual_polarity=True,
                                window_us=(300_000, 800_000))

train_loader = DataLoader(train_dataset,
                          batch_size=BATCH_SIZE,
                          shuffle=True,
                          num_workers=NUM_WORKERS,
                          drop_last=True)
val_loader   = DataLoader(val_dataset,
                          batch_size=BATCH_SIZE,
                          shuffle=False,
                          num_workers=NUM_WORKERS,
                          drop_last=True)
test_loader  = DataLoader(test_dataset,
                          batch_size=BATCH_SIZE,
                          shuffle=False,
                          num_workers=NUM_WORKERS,
                          drop_last=True)

# -----------------------------------------------------------------------------
# 原始 SNN 浮点模型
# -----------------------------------------------------------------------------
float_best_model = torch.load('best_model_all_surface_planE.pth', map_location=DEVICE, weights_only=False)
float_best_model.eval()

# 将模型转换为 QAT 版（per-tensor）
snn_qat = convert_qat_module(float_best_model, bitwidth=BIT_WIDTH).to(DEVICE).train()

# -----------------------------------------------------------------------------
# 阈值工具（拿阈值；某些版本字段名不同）
# -----------------------------------------------------------------------------
def get_iaf_threshold_tensor(iaf: nn.Module) -> torch.Tensor:
    for name in ["threshold", "v_threshold", "v_thr", "v_th"]:
        if hasattr(iaf, name):
            t = getattr(iaf, name)
            if torch.is_tensor(t):
                return t
    # 兜底：给一个不会报错的标量
    return torch.tensor(1.0, device=next(iaf.parameters(), torch.tensor(0.)).device)

# -----------------------------------------------------------------------------
# 校准：把“权重(8bit) + 紧随的 IAF 阈值(16bit)”纳入共同 scale 约束
# -----------------------------------------------------------------------------
def calibrate_fake_quant(model: nn.Module, calib_loader: torch.utils.data.DataLoader):
    """
    【改动点】
    1) per-tensor：用权重自身更新 w_q.scale
    2) Conv/Linear + 紧随 IAF：计算共同 scale = max( max|W|/qmax8, max|thr|/qmax16 )
       并把该共同 scale 写回这一层的 w_q.scale（零点保持 0）
    """
    print("开始校准(含共同scale约束)...")
    time.sleep(0.1)
    model.eval()

    # 先基于权重做一次 per-tensor 的 scale 更新（不需要跑数据）
    for m in model.modules():
        if isinstance(m, QuantConv2d) and hasattr(m, "w_q"):
            m.w_q.update_params(m.weight.detach())
        if isinstance(m, QuantLinear) and hasattr(m, "w_q"):
            m.w_q.update_params(m.weight.detach())

    # 共同 scale 约束
    qmax8  = 2**(BIT_WIDTH-1)  - 1
    qmax16 = 2**(STATE_BITS-1) - 1

    prev_layer = None  # 记录上一层是否为 QuantConv2d / QuantLinear
    for mod in model.modules():
        if isinstance(mod, (QuantConv2d, QuantLinear)):
            prev_layer = mod
        elif isinstance(mod, sl.IAFSqueeze) and prev_layer is not None:
            w_absmax   = prev_layer.weight.detach().abs().max()
            thr_tensor = get_iaf_threshold_tensor(mod).detach()
            thr_absmax = thr_tensor.abs().max()

            scale_w   = (w_absmax / qmax8).clamp(min=1e-8)
            scale_thr = (thr_absmax / qmax16).clamp(min=1e-8)
            scale_c   = torch.max(scale_w, scale_thr)

            prev_layer.w_q.scale.data.copy_(scale_c.to(prev_layer.w_q.scale.device))
            # 共同 scale 下，零点保持 0（对称量化）
            if hasattr(prev_layer.w_q, "zero_point"):
                prev_layer.w_q.zero_point.data.zero_()

            prev_layer = None

    model.train()
    print("校准结束。")
    time.sleep(0.1)

# -----------------------------------------------------------------------------
# 量化感知训练 (QAT)
# -----------------------------------------------------------------------------
optimizer = torch.optim.Adam(snn_qat.parameters(), lr=LR)
criterion = CrossEntropyLoss()
best_val_loss = float('inf')
counter = 0

# 训练前先做一次校准（得到 per-tensor + 共同scale）
calibrate_fake_quant(snn_qat, train_loader)

for epoch in range(EPOCHES):
    epoch_start = time.time()

    # —— 训练阶段 ——
    snn_qat.train()
    train_losses = []
    train_correct = 0
    train_total   = 0
    tbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]", unit="batch")
    train_start = time.time()
    for batch_idx, (data, label) in enumerate(tbar):
        x = data.reshape(-1, 2, 128, 128).to(dtype=torch.float, device=DEVICE)
        y = label.to(dtype=torch.long, device=DEVICE)

        optimizer.zero_grad(set_to_none=True)
        out = snn_qat(x)
        out = out.reshape(BATCH_SIZE, N_TIME_BINS, -1).sum(dim=1)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

        # detach the neuron states and activations from current computation graph(necessary)
        for layer in snn_qat.modules():
            if isinstance(layer, sl.StatefulLayer):
                for name, buffer in layer.named_buffers():
                    buffer.detach_()

        preds = out.argmax(dim=1)
        train_correct += (preds == y).sum().item()
        train_total   += y.size(0)

    avg_train_loss = sum(train_losses) / len(train_losses)
    train_acc = train_correct / train_total * 100
    tqdm.write(f"Epoch {epoch} Train Loss: {avg_train_loss:.4f}, Acc: {train_acc:.2f}%")
    time.sleep(0.1)

    # —— 验证阶段（仍用带假量化的 QAT 图做粗评）——
    snn_qat.eval()
    val_losses = []
    val_correct = 0
    val_total   = 0
    vbar = tqdm(val_loader, desc=f"Epoch {epoch} [Val]", unit="batch")
    val_start = time.time()
    with torch.no_grad():
        for i, (data, label) in enumerate(vbar):
            x = data.reshape(-1, 2, 128, 128).to(dtype=torch.float, device=DEVICE)
            y = label.to(dtype=torch.long, device=DEVICE)

            out = snn_qat(x)
            out = out.reshape(BATCH_SIZE, N_TIME_BINS, -1).sum(dim=1)
            loss = criterion(out, y)
            val_losses.append(loss.item())

            preds = out.argmax(dim=1)
            val_correct += (preds == y).sum().item()
            val_total   += y.size(0)

    avg_val_loss = sum(val_losses) / len(val_losses)
    val_acc = val_correct / val_total * 100
    tqdm.write(f"Epoch {epoch} Val   Loss: {avg_val_loss:.4f}, Acc: {val_acc:.2f}%")
    time.sleep(0.1)

    # 早停 & 保存
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        counter = 0
        torch.save(snn_qat, 'best_qat_model.pth')
        tqdm.write("🔖 Saved best QAT model")
        time.sleep(0.1)
    else:
        counter += 1
        tqdm.write(f"EarlyStopping: {counter}/{PATIENCE}")
        time.sleep(0.1)
        if counter >= PATIENCE:
            tqdm.write("⏹ EarlyStopping triggered")
            time.sleep(0.1)
            break

    tqdm.write(f"Epoch duration: {(time.time()-epoch_start)/60:.2f} min\n")
    time.sleep(0.1)

    # 每 3 个 epoch 重新校准一次（共同scale随权重/阈值微调而更新）
    if (epoch + 1) % 3 == 0:
        calibrate_fake_quant(snn_qat, train_loader)

# -----------------------------------------------------------------------------
# 测试
# -----------------------------------------------------------------------------
best_qat_model = torch.load('best_qat_model.pth', map_location=DEVICE, weights_only=False)
best_qat_model.eval()
correct = 0
total   = 0
tbar = tqdm(test_loader, desc="QAT后的模型 Test", unit="batch")
with torch.no_grad():
    for data, label in tbar:
        x = data.reshape(-1, 2, 128, 128).to(dtype=torch.float, device=DEVICE)
        y = label.to(dtype=torch.long, device=DEVICE)

        out = best_qat_model(x)
        out = out.reshape(BATCH_SIZE, N_TIME_BINS, -1).sum(dim=1)
        preds = out.argmax(dim=1)

        correct += (preds == y).sum().item()
        total   += y.size(0)
tqdm.write(f"🎯 Test Accuracy: {correct/total*100:.2f}%")
time.sleep(0.1)

# -----------------------------------------------------------------------------
# 提取/映射：从 QAT 模型中提取 **per-tensor + 共同scale** 的 int8 权重
# -----------------------------------------------------------------------------
def extract_int8_weights(qat_model: nn.Module):
    """
    从 QAT 模型中提取 int8 权重及 scale/zero_point
    返回 dict: { module_name: { 'w_int8', 'scale', 'zero_point' } }
    （此处 scale 已是“共同scale”，因为校准已写回）
    """
    quantized = {}
    for name, module in qat_model.named_modules():
        if hasattr(module, 'w_q'):
            scale = module.w_q.scale.detach().cpu()           # scalar（per-tensor）
            zero_point = module.w_q.zero_point.detach().cpu() # 0
            qmin, qmax = module.w_q.qmin, module.w_q.qmax
            w_fp = module.weight.detach().cpu()

            w_int = ((w_fp / scale).round() + zero_point)
            w_int = w_int.clamp(qmin, qmax).to(torch.int8)

            quantized[name] = {
                'w_int8': w_int,
                'scale': scale,
                'zero_point': zero_point
            }
    return quantized


def apply_quant_dict_to_model(orig_model: nn.Module,
                               quant_dict: dict) -> nn.Module:
    """
    用 extract_int8_weights 的结果更新原始模型的权重为 dequantized 值（per-tensor）
    """
    for name, module in orig_model.named_modules():
        if name in quant_dict and hasattr(module, 'weight'):
            info = quant_dict[name]
            w_int8 = info['w_int8'].to(module.weight.device)
            scale = info['scale'].to(module.weight.device)             # scalar
            zero_point = info['zero_point'].to(module.weight.device)   # 0
            w_dequant = (w_int8.float() - zero_point) * scale          # 广播到权重形状
            module.weight.data.copy_(w_dequant)
    return orig_model

# -----------------------------------------------------------------------------
# 构造原始模型 & 加载最佳 QAT → 校准共同scale → 提取 & 映射 → 测试
# -----------------------------------------------------------------------------
orig_model = nn.Sequential(
    # [2, 128, 128] -> [4, 32, 32]
    nn.Conv2d(2, 4, stride=2, kernel_size=3, padding=1, bias=False),
    sl.IAFSqueeze(batch_size=BATCH_SIZE, min_v_mem=-1.0, surrogate_grad_fn=PeriodicExponential()),
    nn.AvgPool2d(2, 2),
    # [4, 32, 32] -> [8, 16, 16]
    nn.Conv2d(4, 8, stride=1, kernel_size=3, padding=1, bias=False),
    sl.IAFSqueeze(batch_size=BATCH_SIZE, min_v_mem=-1.0, surrogate_grad_fn=PeriodicExponential()),
    nn.AvgPool2d(2, 2),
    # [8, 16, 16] -> [16, 4, 4]
    nn.Conv2d(8, 16, stride=2, kernel_size=3, padding=1, bias=False),
    sl.IAFSqueeze(batch_size=BATCH_SIZE, min_v_mem=-1.0, surrogate_grad_fn=PeriodicExponential()),
    nn.AvgPool2d(2, 2),
    # [16, 4, 4] -> [15]
    nn.Flatten(),
    nn.Linear(16 * 4 * 4, 15, bias=False),
    sl.IAFSqueeze(batch_size=BATCH_SIZE, min_v_mem=-1.0, surrogate_grad_fn=PeriodicExponential()),
)

# 重新构建 QAT 结构并加载最佳
qat_arch = torch.load('best_qat_model.pth', map_location=DEVICE, weights_only=False)
qat_arch.eval()

# 再做一次校准（确保共同scale已应用）
calibrate_fake_quant(qat_arch, train_loader)

# 提取 int8（per-tensor, 共同scale）并映射到原始模型（反量化）
quant_dict = extract_int8_weights(qat_arch)
orig_model_qsim = apply_quant_dict_to_model(orig_model, quant_dict)

# ---- 测试：查看“共同scale约束后的离线仿真”精度 ----
orig_model_qsim.to(DEVICE).eval()
correct = 0
total   = 0
tbar = tqdm(test_loader, desc="参数映射到原始模型 Test（per-tensor+共同scale）", unit="batch")
with torch.no_grad():
    for data, label in tbar:
        x = data.reshape(-1, 2, 128, 128).to(dtype=torch.float, device=DEVICE)
        y = label.to(dtype=torch.long, device=DEVICE)

        out = orig_model_qsim(x)
        out = out.reshape(BATCH_SIZE, N_TIME_BINS, -1).sum(dim=1)
        preds = out.argmax(dim=1)
        correct += (preds == y).sum().item()
        total   += y.size(0)
tqdm.write(f"🎯 Test Accuracy: {correct/total*100:.2f}%")
time.sleep(0.1)

torch.save(orig_model_qsim, 'final_model.pth')
print("最终模型已保存（per-tensor + 阈值共同scale约束）")

# -----------------------------------------------------------------------------
# 板上部署阶段
# -----------------------------------------------------------------------------
from sinabs.backend.dynapcnn import DynapcnnNetwork
import samna
from collections import Counter
from sklearn.metrics import confusion_matrix

final_model = torch.load("final_model.pth", map_location="cpu", weights_only=False)
dynapcnn = DynapcnnNetwork(snn=final_model, input_shape=(2, 128, 128), discretize=True, dvs_input=False)
devkit_name = "speck2fdevkit"

# use the `to` method of DynapcnnNetwork to deploy the SNN to the devkit
dynapcnn.to(device=devkit_name, chip_layers_ordering="auto")
print(f"The SNN is deployed on the core: {dynapcnn.chip_layers_ordering}")
time.sleep(0.1)

from torch.utils.data import Subset
snn_test_dataset = SpeckTacEventWindowDataset("data_all_surface_split_planE/test", window_us=(300_000, 800_000))
# subset_indices = list(range(0, len(snn_test_dataset), 30))
# snn_test_dataset = Subset(snn_test_dataset, subset_indices)

inference_p_bar = tqdm(snn_test_dataset)
total_angle_error = 0.0
test_samples = 0
correct_samples = 0
y_true, y_pred = [], []
for events, label in inference_p_bar:
    label = int(label)
    # create samna Spike events stream
    samna_event_stream = []
    t0 = events[0]['t']
    for ev in events:
        spk = samna.speck2f.event.Spike()
        spk.x = ev['x']
        spk.y = ev['y']
        spk.timestamp = ev['t'] - t0
        spk.feature = ev['p']
        # Spikes will be sent to layer/core #0, since the SNN is deployed on core: [0, 1, 2, 3]
        spk.layer = 0
        samna_event_stream.append(spk)

    # inference on chip
    # output_events is also a list of Spike, but each Spike.layer is 3, since layer#3 is the output layer
    output_events = dynapcnn(samna_event_stream)

    # use the most frequent output neruon index as the final prediction
    neuron_index = [each.feature for each in output_events]
    if len(neuron_index) != 0:
        frequent_counter = Counter(neuron_index)
        prediction = frequent_counter.most_common(1)[0][0]
        per_angle_error = abs(float(prediction*12) - float(label*12))
        per_angle_error = min(per_angle_error, 180.0 - per_angle_error)
    else:
        prediction = -1
        per_angle_error = 90.0

    if prediction == label:
        correct_samples += 1

    if prediction != -1:
        # 这里默认 prediction 与数据集标签一一对应（0..14）
        y_true.append(label)
        y_pred.append(int(prediction))

    total_angle_error += per_angle_error
    test_samples += 1
    inference_p_bar.set_description(f"label: {label}, prediction: {prediction}， current ACC：{correct_samples / test_samples}， current Avg_Angle_Err：{total_angle_error / test_samples}， output spikes num: {len(output_events)}")

print(f"On chip inference accuracy: {correct_samples / test_samples}， Avg_Angle_Err：{total_angle_error / test_samples}")

# ====== 计算与绘制混淆矩阵（忽略 -1 的样本）======
NUM_CLASSES = 15
ANGLE_STEP_DEG = 12
DISPLAY_LABELS = [f"{i*ANGLE_STEP_DEG}°" for i in range(NUM_CLASSES)]
if len(y_true) > 0:
    labels_idx = list(range(NUM_CLASSES))
    cm = confusion_matrix(y_true, y_pred, labels=labels_idx)

    # 行归一化（按真实类归一化）
    cm_norm = cm.astype(float)
    row_sums = cm_norm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    cm_norm = cm_norm / row_sums

    # ------ 计数版 ------
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    im1 = ax1.imshow(cm, interpolation="nearest")
    ax1.set_title("Confusion Matrix (Counts)")
    ax1.set_xlabel("Predicted")
    ax1.set_ylabel("True")
    ax1.set_xticks(np.arange(NUM_CLASSES))
    ax1.set_yticks(np.arange(NUM_CLASSES))
    ax1.set_xticklabels(DISPLAY_LABELS, rotation=45, ha="right")
    ax1.set_yticklabels(DISPLAY_LABELS)

    # 在格子中标注计数
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax1.text(j, i, int(cm[i, j]), ha="center", va="center")

    plt.tight_layout()
    plt.savefig("confusion_matrix_counts.png", dpi=200)
    plt.show()
else:
    print("没有可用于混淆矩阵的有效预测（全部为 -1），已跳过绘图。")