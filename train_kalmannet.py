import os
import glob
import warnings
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

try:
    from yolox.tracker.kalmannet_model import KalmanNetNN
except ImportError:
    print("Error: Could not import KalmanNetNN.")
    raise SystemExit(1)


# ================= 数据集解析模块 (新增) =================
def parse_mot_dataset(base_dir, min_seq_len=20):
    """
    直接扫描并解析 MOT 规范格式下的 gt.txt
    返回: tracks_obs (带噪观测序列), tracks_gt (纯净真实序列)
    """
    tracks_obs = []
    tracks_gt = []

    # 递归查找所有的 gt.txt
    search_pattern = os.path.join(base_dir, '**', 'gt.txt')
    gt_files = glob.glob(search_pattern, recursive=True)

    if not gt_files:
        print(f"❌ 错误: 在 {base_dir} 中未找到任何 gt.txt 文件！")
        return [], []

    print(f"🔍 找到 {len(gt_files)} 个视频序列，正在解析...")

    for gt_file in gt_files:
        try:
            data = np.loadtxt(gt_file, delimiter=',')
        except Exception as e:
            print(f"无法读取 {gt_file}: {e}")
            continue

        if len(data) == 0: continue

        # 按照 track_id 分组
        track_ids = np.unique(data[:, 1])
        for tid in track_ids:
            track_data = data[data[:, 1] == tid]
            track_data = track_data[track_data[:, 0].argsort()]  # 按帧号排序

            # 分割断开的轨迹 (防止中间丢帧跨度太大)
            frames = track_data[:, 0]
            step_diffs = np.diff(frames)
            split_indices = np.where(step_diffs > 1)[0] + 1
            segments = np.split(track_data, split_indices)

            for seg in segments:
                if len(seg) < min_seq_len:
                    continue  # 过滤掉太短的轨迹片段

                # 提取 [left, top, w, h] 并转换为 [cx, cy, w, h]
                left = seg[:, 2]
                top = seg[:, 3]
                w = seg[:, 4]
                h = seg[:, 5]

                cx = left + w / 2
                cy = top + h / 2
                gt_boxes = np.stack([cx, cy, w, h], axis=1)

                # 动态生成模拟观测序列 (obs_boxes)
                # 模拟 YOLOX 检测器的抖动噪声和置信度
                obs_boxes = np.zeros((len(seg), 5))
                noise_x = np.random.normal(0, 3.0, size=len(seg))
                noise_y = np.random.normal(0, 3.0, size=len(seg))
                noise_w = np.random.normal(0, 1.5, size=len(seg))
                noise_h = np.random.normal(0, 2.5, size=len(seg))

                obs_boxes[:, 0] = cx + noise_x
                obs_boxes[:, 1] = cy + noise_y
                obs_boxes[:, 2] = w + noise_w
                obs_boxes[:, 3] = h + noise_h
                obs_boxes[:, 4] = np.random.uniform(0.6, 0.95, size=len(seg))  # 模拟 Conf

                tracks_obs.append(torch.tensor(obs_boxes, dtype=torch.float32))
                tracks_gt.append(torch.tensor(gt_boxes, dtype=torch.float32))

    return tracks_obs, tracks_gt


# ================= 协方差矩阵与增益构建 =================
def build_init_cov(measurement):
    h = measurement[:, 3].clamp_min(1e-3)
    std = torch.stack(
        [
            2.0 * (1.0 / 20.0) * h,
            2.0 * (1.0 / 20.0) * h,
            torch.full_like(h, 1e-2),
            2.0 * (1.0 / 20.0) * h,
            10.0 * (1.0 / 160.0) * h,
            10.0 * (1.0 / 160.0) * h,
            torch.full_like(h, 1e-5),
            10.0 * (1.0 / 160.0) * h,
        ],
        dim=1,
    )
    return torch.diag_embed(std.pow(2))


def build_motion_cov(state):
    h = state[:, 3].clamp_min(1e-3)
    std = torch.stack(
        [
            (1.0 / 20.0) * h,
            (1.0 / 20.0) * h,
            torch.full_like(h, 1e-2),
            (1.0 / 20.0) * h,
            (1.0 / 160.0) * h,
            (1.0 / 160.0) * h,
            torch.full_like(h, 1e-5),
            (1.0 / 160.0) * h,
        ],
        dim=1,
    )
    return torch.diag_embed(std.pow(2))


def build_meas_cov(pred_state, conf):
    h = pred_state[:, 3].clamp_min(1e-3)
    std = torch.stack(
        [
            (1.0 / 20.0) * h,
            (1.0 / 20.0) * h,
            torch.full_like(h, 1e-1),
            (1.0 / 20.0) * h,
        ],
        dim=1,
    )
    cov = torch.diag_embed(std.pow(2))
    conf = conf.clamp(0.1, 0.99).view(-1, 1, 1)
    return cov * (1.0 / conf)


def classical_gain(cov_pred, h_mat, s_mat):
    cross_cov = torch.matmul(cov_pred, h_mat.t())
    s_inv = torch.linalg.inv(s_mat)
    return torch.matmul(cross_cov, s_inv)


# ================= Dataset 构建模块 =================
class FixedWindowDataset(Dataset):
    def __init__(self, tracks_obs, tracks_gt, seq_len=20, step=2):
        self.samples = []
        for obs, gt in zip(tracks_obs, tracks_gt):
            track_len = obs.shape[0]
            if track_len < seq_len:
                continue
            last_start = track_len - seq_len
            for start in range(0, last_start + 1, step):
                self.samples.append((obs[start:start + seq_len], gt[start:start + seq_len]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class FullTrackDataset(Dataset):
    def __init__(self, tracks_obs, tracks_gt):
        self.tracks_obs = tracks_obs
        self.tracks_gt = tracks_gt

    def __len__(self):
        return len(self.tracks_obs)

    def __getitem__(self, idx):
        return self.tracks_obs[idx], self.tracks_gt[idx]


def collate_tracks(batch):
    obs_list, gt_list = zip(*batch)
    return list(obs_list), list(gt_list)


def normalize_tracks(tracks_obs, tracks_gt, scale):
    norm_obs = []
    norm_gt = []
    for obs, gt in zip(tracks_obs, tracks_gt):
        obs_clone = obs.clone()
        gt_clone = gt.clone()
        obs_clone[:, :4] /= scale
        gt_clone /= scale
        norm_obs.append(obs_clone)
        norm_gt.append(gt_clone)
    return norm_obs, norm_gt


def split_tracks(tracks_obs, tracks_gt, val_split=0.1, seed=42):
    total_size = len(tracks_obs)
    val_size = max(1, int(total_size * val_split))
    if total_size - val_size < 1:
        val_size = max(0, total_size - 1)

    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(total_size, generator=generator).tolist()

    val_indices = perm[:val_size]
    train_indices = perm[val_size:]
    if not train_indices:
        train_indices = val_indices[:1]
        val_indices = val_indices[1:]

    train_obs = [tracks_obs[i] for i in train_indices]
    train_gt = [tracks_gt[i] for i in train_indices]
    val_obs = [tracks_obs[i] for i in val_indices]
    val_gt = [tracks_gt[i] for i in val_indices]
    return train_obs, train_gt, val_obs, val_gt


# ================= 训练核心模块 =================
def run_sequence_batch(
        model, batch_obs, batch_gt, device, f_mat, h_mat,
        residual_gain_limit, criterion, add_noise=False
):
    b_obs = batch_obs.to(device)
    b_gt = batch_gt.to(device)

    batch_n = b_obs.size(0)
    identity = torch.eye(8, device=device).unsqueeze(0).expand(batch_n, -1, -1)
    h_batch = h_mat.unsqueeze(0).expand(batch_n, -1, -1)
    f_batch = f_mat.unsqueeze(0).expand(batch_n, -1, -1)

    current_state = torch.zeros(batch_n, 8, device=device)
    current_state[:, :4] = b_obs[:, 0, :4]
    covariance = build_init_cov(b_obs[:, 0, :4]).to(device)
    hidden = None
    prev_update = torch.zeros(batch_n, 8, device=device)
    total_loss = 0.0

    seq_len = b_obs.size(1)
    for t in range(1, seq_len):
        pred_state = torch.matmul(current_state, f_mat.t())
        motion_cov = build_motion_cov(current_state).to(device)
        cov_pred = torch.matmul(torch.matmul(f_batch, covariance), f_mat.t().unsqueeze(0))
        cov_pred = cov_pred + motion_cov

        z_meas = b_obs[:, t, :4]
        conf = b_obs[:, t, 4:5]

        # 如果开启运行期附加噪声，增加难度
        if add_noise and torch.rand(1).item() < 0.15:
            conf = conf * 0.01
            z_meas = z_meas + torch.randn_like(z_meas) * 0.02

        pred_meas = torch.matmul(pred_state, h_mat.t())
        innovation = z_meas - pred_meas
        meas_cov = build_meas_cov(pred_state, conf).to(device)
        s_mat = torch.matmul(torch.matmul(h_batch, cov_pred), h_mat.t().unsqueeze(0))
        s_mat = s_mat + meas_cov
        k_classic = classical_gain(cov_pred, h_mat, s_mat)

        net_input = torch.cat([innovation, prev_update, conf], dim=1).unsqueeze(1)
        delta_k, hidden = model(net_input, hidden)
        gain_span = torch.clamp(k_classic.abs(), min=1e-3)
        k_gain = k_classic + torch.tanh(delta_k) * (residual_gain_limit * gain_span)

        update_term = torch.bmm(k_gain, innovation.unsqueeze(2)).squeeze(2)
        current_state = pred_state + update_term
        prev_update = update_term

        innovation_factor = identity - torch.matmul(k_gain, h_batch)
        covariance = torch.matmul(torch.matmul(innovation_factor, cov_pred), innovation_factor.transpose(1, 2))
        covariance = covariance + torch.matmul(torch.matmul(k_gain, meas_cov), k_gain.transpose(1, 2))
        covariance = 0.5 * (covariance + covariance.transpose(1, 2))

        loss_pos = criterion(current_state[:, :4], b_gt[:, t, :])
        gt_vel = b_gt[:, t, :] - b_gt[:, t - 1, :]
        loss_vel = criterion(current_state[:, 4:8], gt_vel)
        total_loss = total_loss + loss_pos + 2.0 * loss_vel

    return total_loss / (seq_len - 1)


def train_stage(
        model, train_loader, val_loader, device, optimizer, scheduler,
        epochs, save_path, stage_name, residual_gain_limit, criterion,
        start_best=float("inf")
):
    f_mat = torch.eye(8, device=device)
    for i in range(4):
        f_mat[i, 4 + i] = 1.0
    h_mat = torch.eye(4, 8, device=device)

    best_val_loss = start_best
    print(f"\n🚀 开始 {stage_name} 阶段训练 ...")

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0.0
        train_steps = 0

        for batch_obs, batch_gt in train_loader:
            optimizer.zero_grad()

            if isinstance(batch_obs, list):
                batch_loss = 0.0
                for obs, gt in zip(batch_obs, batch_gt):
                    loss = run_sequence_batch(
                        model, obs.unsqueeze(0), gt.unsqueeze(0), device,
                        f_mat, h_mat, residual_gain_limit, criterion, add_noise=False
                    )
                    batch_loss = batch_loss + loss
                batch_loss = batch_loss / max(1, len(batch_obs))
            else:
                batch_loss = run_sequence_batch(
                    model, batch_obs, batch_gt, device,
                    f_mat, h_mat, residual_gain_limit, criterion, add_noise=True
                )

            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_train_loss += batch_loss.item()
            train_steps += 1

        if scheduler is not None:
            scheduler.step()

        avg_train_loss = total_train_loss / max(1, train_steps)

        # Validation
        model.eval()
        total_val_loss = 0.0
        val_steps = 0
        with torch.no_grad():
            for batch_obs, batch_gt in val_loader:
                if isinstance(batch_obs, list):
                    for obs, gt in zip(batch_obs, batch_gt):
                        loss = run_sequence_batch(
                            model, obs.unsqueeze(0), gt.unsqueeze(0), device,
                            f_mat, h_mat, residual_gain_limit, criterion, add_noise=False
                        )
                        total_val_loss += loss.item()
                        val_steps += 1
                else:
                    loss = run_sequence_batch(
                        model, batch_obs, batch_gt, device,
                        f_mat, h_mat, residual_gain_limit, criterion, add_noise=False
                    )
                    total_val_loss += loss.item()
                    val_steps += 1

        avg_val_loss = total_val_loss / max(1, val_steps)
        is_best = avg_val_loss < best_val_loss
        if is_best:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), save_path)

        current_lr = optimizer.param_groups[0]["lr"]
        best_flag = " 🌟 新高！" if is_best else ""
        if (epoch + 1) % 2 == 0 or epoch == 0 or epoch == epochs - 1:
            print(
                f"[{stage_name}] Epoch {epoch + 1}/{epochs} | "
                f"Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f} | "
                f"LR: {current_lr:.6f}{best_flag}"
            )

    return best_val_loss


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  运算设备: {device}")

    # ================= 训练超参数 =================
    short_batch_size = 256
    short_epochs = 40
    long_epochs = 20
    short_lr = 1e-3
    long_lr = 2e-4
    val_split = 0.1
    residual_gain_limit = 0.30

    # 🎯 指定我们刚生成的数据集路径
    # 如果找不到这个路径，程序会自动尝试寻找其他含有 gt.txt 的文件夹
    data_dir = r"datasets\MOT17_Missile"
    if not os.path.exists(data_dir):
        data_dir = r"datasets\mot"

    print(f"📂 正在从 {data_dir} 加载数据集...")
    tracks_obs, tracks_gt = parse_mot_dataset(data_dir, min_seq_len=20)

    if len(tracks_obs) == 0:
        print("❌ 提取轨迹失败，请检查数据集路径是否正确！")
        return

    print(f"✅ 成功从文本中提取并生成了 {len(tracks_obs)} 条连续的导弹飞行轨迹。")

    # 坐标归一化 (X, Y, W, H 分别除以 1920, 1080, 1920, 1080 将尺度映射到 0~1)
    scale = torch.tensor([1920, 1080, 1920, 1080], dtype=torch.float32)
    tracks_obs, tracks_gt = normalize_tracks(tracks_obs, tracks_gt, scale)

    # 划分训练集和验证集
    train_obs, train_gt, val_obs, val_gt = split_tracks(
        tracks_obs, tracks_gt, val_split=val_split, seed=42
    )

    short_seq_len = 20
    short_seq_step = 2

    # 阶段 1：构建定长截断序列数据集 (短窗口)
    short_train_dataset = FixedWindowDataset(train_obs, train_gt, seq_len=short_seq_len, step=short_seq_step)
    short_val_dataset = FixedWindowDataset(val_obs, val_gt, seq_len=short_seq_len, step=short_seq_step)

    # 阶段 2：构建全长序列数据集 (长序列微调)
    long_train_dataset = FullTrackDataset(train_obs, train_gt)
    long_val_dataset = FullTrackDataset(val_obs, val_gt)

    # 修改前：
    # short_train_loader = DataLoader(short_train_dataset, batch_size=short_batch_size, shuffle=True)

    # ⚡ 修改后：
    # num_workers: 开启4个子进程加速读取数据 (如果你的CPU核心多，可以调到 8 或 16)
    # pin_memory: 直接将数据锁在内存中，向 GPU 传输时速度极快！
    short_train_loader = DataLoader(short_train_dataset, batch_size=short_batch_size, shuffle=True, num_workers=4,
                                    pin_memory=True)
    short_val_loader = DataLoader(short_val_dataset, batch_size=short_batch_size, shuffle=False, num_workers=4,
                                  pin_memory=True)

    long_train_loader = DataLoader(long_train_dataset, batch_size=1, shuffle=True, collate_fn=collate_tracks,
                                   num_workers=4, pin_memory=True)
    long_val_loader = DataLoader(long_val_dataset, batch_size=1, shuffle=False, collate_fn=collate_tracks,
                                 num_workers=4, pin_memory=True)

    short_val_loader = DataLoader(short_val_dataset, batch_size=short_batch_size, shuffle=False)
    long_train_loader = DataLoader(long_train_dataset, batch_size=1, shuffle=True, collate_fn=collate_tracks)
    long_val_loader = DataLoader(long_val_dataset, batch_size=1, shuffle=False, collate_fn=collate_tracks)

    print(
        f"📊 数据规模:\n - 长轨迹: 训练集={len(train_obs)}, 验证集={len(val_obs)}\n - 短窗口截断: 训练集={len(short_train_dataset)}, 验证集={len(short_val_dataset)}")

    model = KalmanNetNN().to(device)
    criterion = nn.MSELoss()

    os.makedirs("pretrained", exist_ok=True)
    save_path = "pretrained/kalmannet_best.pth"

    # ============ 阶段 1: 短窗口截断训练 (Stage 1) ============
    short_optimizer = optim.Adam(model.parameters(), lr=short_lr)
    short_scheduler = optim.lr_scheduler.CosineAnnealingLR(short_optimizer, T_max=short_epochs, eta_min=1e-5)

    best_val_loss = train_stage(
        model=model, train_loader=short_train_loader, val_loader=short_val_loader,
        device=device, optimizer=short_optimizer, scheduler=short_scheduler,
        epochs=short_epochs, save_path=save_path, stage_name="Stage-1 (定长短序列训练)",
        residual_gain_limit=residual_gain_limit, criterion=criterion,
    )

    # 加载第一阶段最好的一组权重
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.load_state_dict(torch.load(save_path, map_location=device))

    # ============ 阶段 2: 全序列长程微调 (Stage 2) ============
    long_optimizer = optim.Adam(model.parameters(), lr=long_lr)
    long_scheduler = optim.lr_scheduler.CosineAnnealingLR(long_optimizer, T_max=long_epochs, eta_min=1e-5)

    best_val_loss = train_stage(
        model=model, train_loader=long_train_loader, val_loader=long_val_loader,
        device=device, optimizer=long_optimizer, scheduler=long_scheduler,
        epochs=long_epochs, save_path=save_path, stage_name="Stage-2 (可变长全序列微调)",
        residual_gain_limit=residual_gain_limit, criterion=criterion,
        start_best=best_val_loss,
    )

    print(f"🎉 训练圆满完成！最终最优 Val Loss = {best_val_loss:.6f}")
    print(f"📦 专属神经网络滤波权重已保存至: {save_path}")


if __name__ == "__main__":
    train()