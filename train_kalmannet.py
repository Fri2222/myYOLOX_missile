import argparse
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


# ================= 1. 命令行参数解析 =================
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=60, help="总训练轮数(将自动按 6:4 分配给两阶段)")
    parser.add_argument("--seq-len", type=int, default=20, help="BPTT截断序列长度")
    parser.add_argument("--seq-step", type=int, default=2, help="滑动窗口采样步长")
    parser.add_argument("--residual-gain-limit", type=float, default=0.30)
    return parser.parse_args()


# ================= 2. 数据集解析与修复 =================
def parse_mot_dataset(base_dir, min_seq_len=20):
    tracks_obs = []
    tracks_gt = []
    search_pattern = os.path.join(base_dir, '**', 'gt.txt')
    gt_files = glob.glob(search_pattern, recursive=True)

    if not gt_files: return [], []

    for gt_file in gt_files:
        try:
            data = np.loadtxt(gt_file, delimiter=',')
        except Exception:
            continue
        if len(data) == 0: continue

        track_ids = np.unique(data[:, 1])
        for tid in track_ids:
            track_data = data[data[:, 1] == tid]
            track_data = track_data[track_data[:, 0].argsort()]
            frames = track_data[:, 0]
            split_indices = np.where(np.diff(frames) > 1)[0] + 1
            segments = np.split(track_data, split_indices)

            for seg in segments:
                if len(seg) < min_seq_len: continue

                # 👇 【修复 1】: 将 w/h 转换为 xyah 格式
                left, top, w, h = seg[:, 2], seg[:, 3], seg[:, 4], seg[:, 5]
                cx = left + w / 2
                cy = top + h / 2
                a = w / h

                gt_boxes = np.stack([cx, cy, a, h], axis=1)

                obs_boxes = np.zeros((len(seg), 5))
                obs_boxes[:, 0] = cx + np.random.normal(0, 3.0, size=len(seg))
                obs_boxes[:, 1] = cy + np.random.normal(0, 3.0, size=len(seg))
                obs_boxes[:, 2] = a + np.random.normal(0, 0.05, size=len(seg))
                obs_boxes[:, 3] = h + np.random.normal(0, 2.5, size=len(seg))
                obs_boxes[:, 4] = np.random.uniform(0.6, 0.95, size=len(seg))

                tracks_obs.append(torch.tensor(obs_boxes, dtype=torch.float32))
                tracks_gt.append(torch.tensor(gt_boxes, dtype=torch.float32))
    return tracks_obs, tracks_gt


# ================= 3. 协方差矩阵构建 =================
def build_init_cov(measurement):
    h = measurement[:, 3].clamp_min(1e-3)
    std = torch.stack([
        2.0 * (1.0 / 20.0) * h, 2.0 * (1.0 / 20.0) * h, torch.full_like(h, 1e-2), 2.0 * (1.0 / 20.0) * h,
        10.0 * (1.0 / 160.0) * h, 10.0 * (1.0 / 160.0) * h, torch.full_like(h, 1e-5), 10.0 * (1.0 / 160.0) * h,
    ], dim=1)
    return torch.diag_embed(std.pow(2))


def build_motion_cov(state):
    h = state[:, 3].clamp_min(1e-3)
    std = torch.stack([
        (1.0 / 20.0) * h, (1.0 / 20.0) * h, torch.full_like(h, 1e-2), (1.0 / 20.0) * h,
        (1.0 / 160.0) * h, (1.0 / 160.0) * h, torch.full_like(h, 1e-5), (1.0 / 160.0) * h,
    ], dim=1)
    return torch.diag_embed(std.pow(2))


def build_meas_cov(pred_state, conf):
    h = pred_state[:, 3].clamp_min(1e-3)
    std = torch.stack([
        (1.0 / 20.0) * h, (1.0 / 20.0) * h, torch.full_like(h, 1e-1), (1.0 / 20.0) * h,
    ], dim=1)
    return torch.diag_embed(std.pow(2)) * (1.0 / conf.clamp(0.1, 0.99).view(-1, 1, 1))


def classical_gain(cov_pred, h_mat, s_mat):
    return torch.matmul(torch.matmul(cov_pred, h_mat.t()), torch.linalg.inv(s_mat))


# ================= 4. Dataset 与归一化 =================
class FixedWindowDataset(Dataset):
    def __init__(self, tracks_obs, tracks_gt, seq_len=20, step=2):
        self.samples = []
        for obs, gt in zip(tracks_obs, tracks_gt):
            if obs.shape[0] < seq_len: continue
            for start in range(0, obs.shape[0] - seq_len + 1, step):
                self.samples.append((obs[start:start + seq_len], gt[start:start + seq_len]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class FullTrackDataset(Dataset):
    def __init__(self, tracks_obs, tracks_gt):
        self.tracks_obs, self.tracks_gt = tracks_obs, tracks_gt

    def __len__(self): return len(self.tracks_obs)

    def __getitem__(self, idx): return self.tracks_obs[idx], self.tracks_gt[idx]


def collate_tracks(batch):
    obs_list, gt_list = zip(*batch)
    return list(obs_list), list(gt_list)


def normalize_tracks(tracks_obs, tracks_gt, scale):
    norm_obs, norm_gt = [], []
    for obs, gt in zip(tracks_obs, tracks_gt):
        obs_c, gt_c = obs.clone(), gt.clone()
        obs_c[:, :4] /= scale
        gt_c /= scale
        norm_obs.append(obs_c)
        norm_gt.append(gt_c)
    return norm_obs, norm_gt


def split_tracks(tracks_obs, tracks_gt, val_split=0.1):
    total = len(tracks_obs)
    val_size = max(1, int(total * val_split))
    perm = torch.randperm(total, generator=torch.Generator().manual_seed(42)).tolist()
    v_idx, t_idx = perm[:val_size], perm[val_size:]
    return [tracks_obs[i] for i in t_idx], [tracks_gt[i] for i in t_idx], [tracks_obs[i] for i in v_idx], [tracks_gt[i]
                                                                                                           for i in
                                                                                                           v_idx]


# ================= 5. 核心前向传播 =================
def run_sequence_batch(model, batch_obs, batch_gt, device, f_mat, h_mat, residual_gain_limit, criterion,
                       add_noise=False):
    b_obs, b_gt = batch_obs.to(device), batch_gt.to(device)
    batch_n, seq_len = b_obs.size(0), b_obs.size(1)

    identity = torch.eye(8, device=device).unsqueeze(0).expand(batch_n, -1, -1)
    h_batch = h_mat.unsqueeze(0).expand(batch_n, -1, -1)
    f_batch = f_mat.unsqueeze(0).expand(batch_n, -1, -1)

    current_state = torch.zeros(batch_n, 8, device=device)
    current_state[:, :4] = b_obs[:, 0, :4]
    covariance = build_init_cov(b_obs[:, 0, :4]).to(device)
    hidden, prev_update = None, torch.zeros(batch_n, 8, device=device)
    total_loss = 0.0

    for t in range(1, seq_len):
        pred_state = torch.matmul(current_state, f_mat.t())
        cov_pred = torch.matmul(torch.matmul(f_batch, covariance), f_mat.t().unsqueeze(0)) + build_motion_cov(
            current_state).to(device)

        z_meas, conf = b_obs[:, t, :4], b_obs[:, t, 4:5]
        if add_noise and torch.rand(1).item() < 0.15:
            conf = conf * 0.01
            z_meas = z_meas + torch.randn_like(z_meas) * 0.02

        pred_meas = torch.matmul(pred_state, h_mat.t())
        innovation = z_meas - pred_meas
        meas_cov = build_meas_cov(pred_state, conf).to(device)
        s_mat = torch.matmul(torch.matmul(h_batch, cov_pred), h_mat.t().unsqueeze(0)) + meas_cov
        k_classic = classical_gain(cov_pred, h_mat, s_mat)

        delta_k, hidden = model(torch.cat([innovation, prev_update, conf], dim=1).unsqueeze(1), hidden)
        k_gain = k_classic + torch.tanh(delta_k) * (residual_gain_limit * torch.clamp(k_classic.abs(), min=1e-3))

        update_term = torch.bmm(k_gain, innovation.unsqueeze(2)).squeeze(2)
        current_state = pred_state + update_term
        prev_update = update_term

        innovation_factor = identity - torch.matmul(k_gain, h_batch)
        covariance = torch.matmul(torch.matmul(innovation_factor, cov_pred),
                                  innovation_factor.transpose(1, 2)) + torch.matmul(torch.matmul(k_gain, meas_cov),
                                                                                    k_gain.transpose(1, 2))
        covariance = 0.5 * (covariance + covariance.transpose(1, 2))

        total_loss += criterion(current_state[:, :4], b_gt[:, t, :]) + 2.0 * criterion(current_state[:, 4:8],
                                                                                       b_gt[:, t, :] - b_gt[:, t - 1,
                                                                                                       :])

    return total_loss / (seq_len - 1)


# ================= 6. 单阶段训练逻辑 =================
def train_stage(model, train_loader, val_loader, device, optimizer, scheduler, epochs, save_path, stage_name,
                residual_gain_limit, criterion, start_best=float("inf")):
    f_mat = torch.eye(8, device=device)
    for i in range(4): f_mat[i, 4 + i] = 1.0
    h_mat = torch.eye(4, 8, device=device)
    best_val = start_best
    print(f"\n🚀 开始 {stage_name} 阶段训练 ...")

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for b_obs, b_gt in train_loader:
            optimizer.zero_grad()
            if isinstance(b_obs, list):
                loss = sum(
                    run_sequence_batch(model, o.unsqueeze(0), g.unsqueeze(0), device, f_mat, h_mat, residual_gain_limit,
                                       criterion, False) for o, g in zip(b_obs, b_gt)) / max(1, len(b_obs))
            else:
                loss = run_sequence_batch(model, b_obs, b_gt, device, f_mat, h_mat, residual_gain_limit, criterion,
                                          True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
        if scheduler: scheduler.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for b_obs, b_gt in val_loader:
                if isinstance(b_obs, list):
                    val_loss += sum(run_sequence_batch(model, o.unsqueeze(0), g.unsqueeze(0), device, f_mat, h_mat,
                                                       residual_gain_limit, criterion, False).item() for o, g in
                                    zip(b_obs, b_gt))
                else:
                    val_loss += run_sequence_batch(model, b_obs, b_gt, device, f_mat, h_mat, residual_gain_limit,
                                                   criterion, False).item()

        avg_val = val_loss / max(1, len(val_loader))
        is_best = avg_val < best_val
        if is_best:
            best_val = avg_val
            torch.save(model.state_dict(), save_path)

        print(
            f"[{stage_name}] Epoch {epoch + 1}/{epochs} | Train Loss: {train_loss / max(1, len(train_loader)):.6f} | Val Loss: {avg_val:.6f} | LR: {optimizer.param_groups[0]['lr']:.6f}{' 🌟 新高！' if is_best else ''}")
    return best_val


# ================= 7. 主控函数 =================
def train():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 👇 【关键修复】：将总轮数按 6:4 分配给两阶段，若传入 2，则每阶段只跑 1 轮！
    short_epochs = max(1, int(args.epochs * 0.6))
    long_epochs = max(1, args.epochs - short_epochs)

    data_dir = r"datasets\MOT17_Missile"
    if not os.path.exists(data_dir): data_dir = r"datasets\mot"
    tracks_obs, tracks_gt = parse_mot_dataset(data_dir, min_seq_len=args.seq_len)
    if not tracks_obs: return

    # 👇 【关键修复】: 第三维 a 尺度设为 1，确保 Loss 降回正常水平！
    scale = torch.tensor([1920., 1080., 1., 1080.], dtype=torch.float32)
    tracks_obs, tracks_gt = normalize_tracks(tracks_obs, tracks_gt, scale)

    train_o, train_g, val_o, val_g = split_tracks(tracks_obs, tracks_gt)

    # 👇 使用传入的步长动态切分
    short_train_dataset = FixedWindowDataset(train_o, train_g, seq_len=args.seq_len, step=args.seq_step)
    short_val_dataset = FixedWindowDataset(val_o, val_g, seq_len=args.seq_len, step=args.seq_step)
    long_train_dataset = FullTrackDataset(train_o, train_g)
    long_val_dataset = FullTrackDataset(val_o, val_g)

    short_train_loader = DataLoader(short_train_dataset, batch_size=256, shuffle=True, num_workers=4, pin_memory=True)
    short_val_loader = DataLoader(short_val_dataset, batch_size=256, shuffle=False, num_workers=4, pin_memory=True)
    long_train_loader = DataLoader(long_train_dataset, batch_size=1, shuffle=True, collate_fn=collate_tracks,
                                   num_workers=4, pin_memory=True)
    long_val_loader = DataLoader(long_val_dataset, batch_size=1, shuffle=False, collate_fn=collate_tracks,
                                 num_workers=4, pin_memory=True)

    print(
        f"📊 数据规模:\n - 长轨迹: 训练集={len(train_o)}, 验证集={len(val_o)}\n - 短窗口截断: 训练集={len(short_train_dataset)}, 验证集={len(short_val_dataset)}")

    model = KalmanNetNN().to(device)
    criterion = nn.MSELoss()

    save_dir = r"H:\Code\Byte\ByteTrack-main\ByteTrack-main\pretrained\exp0031"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "kalmannet_best.pth")

    short_opt = optim.Adam(model.parameters(), lr=1e-3)
    short_sch = optim.lr_scheduler.CosineAnnealingLR(short_opt, T_max=short_epochs, eta_min=1e-5)
    best_val = train_stage(model, short_train_loader, short_val_loader, device, short_opt, short_sch, short_epochs,
                           save_path, "Stage-1", args.residual_gain_limit, criterion)

    # 阶段 2：加载最好的权重微调
    model.load_state_dict(torch.load(save_path, map_location=device))
    long_opt = optim.Adam(model.parameters(), lr=2e-4)
    long_sch = optim.lr_scheduler.CosineAnnealingLR(long_opt, T_max=long_epochs, eta_min=1e-5)
    train_stage(model, long_train_loader, long_val_loader, device, long_opt, long_sch, long_epochs, save_path,
                "Stage-2", args.residual_gain_limit, criterion, best_val)
    print(f"🎉 训练圆满完成！最终权重已保存至: {save_path}")


if __name__ == "__main__":
    train()