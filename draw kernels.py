import os
import math
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

# ============ 基础工具 ============
def to_numpy(t):
    return t.detach().cpu().numpy()

def minmax_norm(arr):
    arr = arr.astype(np.float32)
    mn, mx = arr.min(), arr.max()
    if mx - mn < 1e-12:
        return np.zeros_like(arr)
    return (arr - mn) / (mx - mn)

def make_grid(images, ncols=None, pad=2, pad_val=1.0):
    """
    images: list of 2D numpy arrays (H,W), 已经 [0,1]
    """
    if ncols is None:
        ncols = int(math.ceil(math.sqrt(len(images))))
    nrows = int(math.ceil(len(images) / ncols))
    if len(images) == 0:
        return np.zeros((10,10), dtype=np.float32)
    H, W = images[0].shape
    grid = np.full((nrows*H + pad*(nrows-1), ncols*W + pad*(ncols-1)),
                   pad_val, dtype=np.float32)
    idx = 0
    for r in range(nrows):
        for c in range(ncols):
            if idx >= len(images): break
            top = r*(H+pad)
            left = c*(W+pad)
            grid[top:top+H, left:left+W] = images[idx]
            idx += 1
    return grid

def save_img(arr, path, cmap='gray', dpi=300):
    plt.figure()
    plt.axis('off')
    plt.imshow(arr, cmap=cmap, interpolation='nearest')
    plt.tight_layout(pad=0.0)
    plt.savefig(path, dpi=dpi, bbox_inches='tight', pad_inches=0.0)
    plt.close()

# ============ 1) 加载整体模型 ============
ckpt_path = "final_model.pth"  # 你的完整模型
device = "cpu"
obj = torch.load(ckpt_path, map_location=device, weights_only=False)

# 如果你是直接保存的模型本体（不是state_dict），torch.load会得到nn.Sequential
# 若得到的是包含'state_dict'的字典，请按需改为：model.load_state_dict(obj['state_dict'])
model = obj
if isinstance(obj, dict) and 'state_dict' in obj:
    # 需要重建结构才可load_state_dict；这里给出简单重建法：
    raise RuntimeError("你的final_model.pth貌似是state_dict格式；请提供构建网络的代码后再load_state_dict。")

# DataParallel 兼容
if hasattr(model, "module"):
    model = model.module

model.eval()

# ============ 2) 提取卷积层 ============
conv_layers = []
for m in model.modules():
    if isinstance(m, nn.Conv2d):
        conv_layers.append(m)

print(f"Found {len(conv_layers)} Conv2d layers")

out_dir = "kernel_viz"
os.makedirs(out_dir, exist_ok=True)

# ============ 3) 可视化每层卷积核 ============
# 规则：
# - 每个 out_channel 生成一张“合成”核：对 in_channel 做 L2 聚合或求平均（这里用L2）
# - 如果 in_channels == 2（双极性），额外各画一张：正极通道核、负极通道核，以及“正-负”的差分核
for li, conv in enumerate(conv_layers, start=1):
    w = conv.weight  # [out_c, in_c, k, k]
    W = to_numpy(w)
    out_c, in_c, k, _ = W.shape
    print(f"Layer {li}: weight shape = {W.shape}")

    # 3.1 合成展示（跨输入通道聚合）：L2范数聚合
    merged_imgs = []
    for oc in range(out_c):
        # L2 across in_channels -> (k,k)
        merged = np.sqrt(np.sum(W[oc]**2, axis=0))
        merged = minmax_norm(merged)
        merged_imgs.append(merged)

    grid = make_grid(merged_imgs, ncols=None)
    save_img(grid, os.path.join(out_dir, f"layer{li}_merged_kernels.png"))
    save_img(grid, os.path.join(out_dir, f"layer{li}_merged_kernels.pdf"))

    # 3.2 若是双通道（正/负极性），分别展示两个通道 + 差分
    if in_c == 2:
        ch0_imgs, ch1_imgs, diff_imgs = [], [], []
        for oc in range(out_c):
            ch0 = minmax_norm(W[oc, 0])
            ch1 = minmax_norm(W[oc, 1])
            diff = minmax_norm(W[oc, 0] - W[oc, 1])  # “正-负”强调极性对比
            ch0_imgs.append(ch0)
            ch1_imgs.append(ch1)
            diff_imgs.append(diff)

        save_img(make_grid(ch0_imgs), os.path.join(out_dir, f"layer{li}_in0_polarity.png"))
        save_img(make_grid(ch1_imgs), os.path.join(out_dir, f"layer{li}_in1_polarity.png"))
        save_img(make_grid(diff_imgs), os.path.join(out_dir, f"layer{li}_diff_polarity.png"))

    # 3.3 （可选）只挑“最强”的若干核（按权重范数排序）
    strengths = []
    for oc in range(out_c):
        strengths.append(np.linalg.norm(W[oc].reshape(in_c, -1), ord=2))
    order = np.argsort(strengths)[::-1]
    topN = min(16, out_c)
    top_imgs = [merged_imgs[i] for i in order[:topN]]
    save_img(make_grid(top_imgs), os.path.join(out_dir, f"layer{li}_top{topN}_merged.png"))

print(f"Saved kernel visualizations to: {out_dir}")

# ============ 4) （可选）中间特征图可视化 ============
# 需要一条输入样本 x: [1, 2, 128, 128]；这里给出伪造输入示例
do_feature_viz = False  # 你有真实样本时改成True
if do_feature_viz:
    # TODO: 用真实事件张量替换这里的随机数，比如从你的数据管线里拿一条
    x = torch.rand(1, 2, 128, 128)

    # 注册 hook 抓取每个Conv层的输出
    activations = []
    handles = []
    def hook_fn(m, inp, out):
        # out: [B, C, H, W]
        activations.append(out.detach().cpu())

    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            handles.append(m.register_forward_hook(hook_fn))

    with torch.no_grad():
        _ = model(x)

    for h in handles:
        h.remove()

    # 把每层的特征图里“响应最强”的前N个通道可视化
    feat_dir = "feature_viz"
    os.makedirs(feat_dir, exist_ok=True)
    for i, act in enumerate(activations, start=1):
        # 取batch里第0个样本
        A = act[0]  # [C,H,W]
        C, H, W = A.shape
        # 用通道范数排序
        mags = torch.linalg.vector_norm(A.reshape(C, -1), dim=1).numpy()
        idx = np.argsort(mags)[::-1]
        topN = min(16, C)
        imgs = [minmax_norm(A[idx[j]].numpy()) for j in range(topN)]
        save_img(make_grid(imgs), os.path.join(feat_dir, f"layer{i}_top{topN}_featuremaps.png"))

    print(f"Saved feature maps to: {feat_dir}")
