#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import sinabs
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from MyDataset import SpeckTacDataset, SpeckTacWindowDataset, SpeckTacEventWindowDataset
import torch
from torch import nn
import sinabs.layers as sl
from sinabs.activation.surrogate_gradient_fn import PeriodicExponential
from tqdm import tqdm
from torch.utils.data import Subset

# -----------------------------------------------------------------------------
# 超参
# -----------------------------------------------------------------------------
BATCH_SIZE   = 4
N_TIME_BINS  = 50
NUM_WORKERS = 16
EPOCHES      = 50
LR           = 1e-3
PATIENCE     = 5
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.cuda.empty_cache()

# 1. 固定种子
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # 如果用 GPU

# 2. 为了更强的可复现（可选，但常用）
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

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
# 模型构建
# -----------------------------------------------------------------------------
snn_model = nn.Sequential(
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
# Xavier 初始化
for layer in snn_model.modules():
    if isinstance(layer, (nn.Conv2d, nn.Linear)):
        nn.init.xavier_normal_(layer.weight.data)

snn_model = snn_model.to(device=DEVICE)
optimizer = torch.optim.Adam(snn_model.parameters(), lr=LR)
criterion = CrossEntropyLoss()

best_val_loss = float('inf')
counter       = 0

# -----------------------------------------------------------------------------
# 训练 & 验证
# -----------------------------------------------------------------------------
for epoch in range(EPOCHES):
    epoch_start = time.time()

    # ---------- 训练阶段 ----------
    train_start = time.time()
    train_pbar  = tqdm(train_loader, desc=f"Epoch {epoch} [Train]", unit="batch")
    train_losses = []
    train_correct = 0
    train_total = 0
    for batch_idx, (data, label) in enumerate(train_pbar):
        # 准备
        data  = data.reshape(-1, 2, 128, 128).to(dtype=torch.float, device=DEVICE)
        label = label.to(dtype=torch.long, device=DEVICE)

        # 前向 + 反向
        optimizer.zero_grad()
        output = snn_model(data)
        output = output.reshape(BATCH_SIZE, N_TIME_BINS, -1).sum(dim=1)
        loss   = criterion(output, label)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

        # detach the neuron states and activations from current computation graph(necessary)
        for layer in snn_model.modules():
            if isinstance(layer, sl.StatefulLayer):
                for name, buffer in layer.named_buffers():
                    buffer.detach_()

        # 计算训练准确率
        preds = output.argmax(dim=1)
        train_correct += (preds == label).sum().item()
        train_total += label.size(0)

    train_time = time.time() - train_start
    avg_train_loss = sum(train_losses) / len(train_losses)
    train_acc = train_correct / train_total * 100
    tqdm.write(f"Epoch {epoch} Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%  (阶段耗时 {train_time:.1f}s)")
    time.sleep(0.1)

    # ---------- 验证阶段 ----------
    val_start = time.time()
    val_pbar  = tqdm(val_loader, desc=f"Epoch {epoch} [Val]", unit="batch")
    val_losses = []
    corrects   = []
    snn_model.eval()
    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(val_pbar):
            data = data.reshape(-1, 2, 128, 128).to(dtype=torch.float, device=DEVICE)
            label = label.to(dtype=torch.long, device=DEVICE)

            output = snn_model(data)
            output = output.reshape(BATCH_SIZE, N_TIME_BINS, -1).sum(dim=1)
            loss   = criterion(output, label)
            val_losses.append(loss.item())

            pred = output.argmax(dim=1, keepdims=True)
            corrects.append(pred.eq(label.view_as(pred)))

    avg_val_loss = sum(val_losses) / len(val_losses)
    val_time     = time.time() - val_start
    corrects = torch.cat(corrects)
    val_acc  = corrects.sum().item() / corrects.numel() * 100

    # 模型保存 & 早停
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        counter = 0
        torch.save(snn_model, 'best_model_all_surface_planE.pth')
        tqdm.write("🔖 保存了新的最佳模型")
        time.sleep(0.1)
    else:
        counter += 1
        tqdm.write(f"EarlyStopping counter: {counter}/{PATIENCE}")
        time.sleep(0.1)
        if counter >= PATIENCE:
            tqdm.write("⏹ 触发 EarlyStopping，退出训练")
            break

    tqdm.write(f"Epoch {epoch} Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%  (阶段耗时 {val_time:.1f}s)")
    time.sleep(0.1)

# -----------------------------------------------------------------------------
# 测试阶段
# -----------------------------------------------------------------------------
best_model = torch.load('best_model_all_surface_planE.pth', map_location=DEVICE, weights_only=False)

test_start = time.time()
test_pbar  = tqdm(test_loader, desc="Test Evaluation", unit="batch")
correct = 0
total   = 0

with torch.no_grad():
    for batch_idx, (data, label) in enumerate(test_pbar):
        data = data.reshape(-1, 2, 128, 128).to(dtype=torch.float, device=DEVICE)
        label = label.to(dtype=torch.long, device=DEVICE)

        output = best_model(data)
        output = output.reshape(BATCH_SIZE, N_TIME_BINS, -1).sum(dim=1)
        preds  = output.argmax(dim=1)

        correct += (preds == label).sum().item()
        total   += label.size(0)

test_acc   = correct / total * 100
test_time  = time.time() - test_start
print(f"\n🎯 Test Accuracy: {test_acc:.2f}%  (测试耗时 {test_time:.1f}s)")

# -----------------------------------------------------------------------------
# 板上部署阶段
# -----------------------------------------------------------------------------
from sinabs.backend.dynapcnn import DynapcnnNetwork
import samna
from collections import Counter

dynapcnn = DynapcnnNetwork(snn=best_model.to(device="cpu"), input_shape=(2, 128, 128), discretize=True, dvs_input=False)
devkit_name = "speck2fdevkit"

# use the `to` method of DynapcnnNetwork to deploy the SNN to the devkit
dynapcnn.to(device=devkit_name, chip_layers_ordering="auto")
print(f"The SNN is deployed on the core: {dynapcnn.chip_layers_ordering}")

snn_test_dataset = SpeckTacEventWindowDataset("data_all_surface_split_planE/test", window_us=(300_000, 800_000))
# subset_indices = list(range(0, len(snn_test_dataset), 30))
# snn_test_dataset = Subset(snn_test_dataset, subset_indices)

inference_p_bar = tqdm(snn_test_dataset)
test_samples = 0
correct_samples = 0
for events, label in inference_p_bar:
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
    else:
        prediction = -1

    if prediction == label:
        correct_samples += 1

    test_samples += 1
    inference_p_bar.set_description(f"label: {label}, prediction: {prediction}， current ACC：{correct_samples / test_samples}， output spikes num: {len(output_events)}")

print(f"On chip inference accuracy: {correct_samples / test_samples}")