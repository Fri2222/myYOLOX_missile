import numpy as np
import os
import random
import math

# ================= ⚙️ 配置区域 =================
DATASET_OUT_DIR = r"H:\Code\YOLOX\datasets\MOT17_Missile"
IMG_W, IMG_H = 1920, 1080
FOV = 90  # 摄像机视野角度

# 物理空间参照 UE4 (NED: X前, Y右, Z下)
CAM_POS = np.array([50, 80, -30])
SHARED_TARGET_POS = np.array([50, 10, -30])


# ===============================================

class VirtualCamera:
    """虚拟针孔摄像机：负责将 3D 物理坐标投影为 2D 图像像素框"""

    def __init__(self, cam_pos, fov=90, img_w=1920, img_h=1080):
        self.cam_pos = np.array(cam_pos)
        self.img_w = img_w
        self.img_h = img_h
        # 计算焦距
        self.focal_length = (img_w / 2) / math.tan(math.radians(fov / 2))

    def project_to_2d_bbox(self, pos_3d, phys_w=4.0, phys_h=1.5):
        """
        输入导弹 3D 坐标，输出 2D 图像上的 bbox: [left, top, w, h]
        """
        # 相机向 -Y 看去
        dx = pos_3d[0] - self.cam_pos[0]
        dy = pos_3d[1] - self.cam_pos[1]
        dz = pos_3d[2] - self.cam_pos[2]

        depth = -dy  # 相机前方的深度
        if depth <= 1.0:  # 太近或跑到了相机后面，从画面中消失
            return None

        # 针孔成像：算出画面中心点偏移
        u = (dx * self.focal_length) / depth
        v = (dz * self.focal_length) / depth

        # 映射到 1920x1080 左上角体系
        pixel_x = u + self.img_w / 2
        pixel_y = v + self.img_h / 2

        # 透视形变：算出像素宽高 (近大远小)
        pixel_w = (phys_w * self.focal_length) / depth
        pixel_h = (phys_h * self.focal_length) / depth

        left = pixel_x - pixel_w / 2
        top = pixel_y - pixel_h / 2
        return left, top, pixel_w, pixel_h


def write_seqinfo(path, seq_name, seq_length):
    """生成标准 MOT17 seqinfo.ini"""
    content = f"[Sequence]\nname={seq_name}\nimDir=img1\nframeRate=60\nseqLength={seq_length}\nimWidth={IMG_W}\nimHeight={IMG_H}\nimExt=.jpg\n"
    with open(os.path.join(path, "seqinfo.ini"), "w") as f:
        f.write(content)


def generate_mot_dataset(base_dir, total_seqs=20, frames_per_seq=180):
    train_dir = os.path.join(base_dir, "train")
    test_dir = os.path.join(base_dir, "test")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    camera = VirtualCamera(CAM_POS, FOV, IMG_W, IMG_H)

    for seq_id in range(1, total_seqs + 1):
        split_dir = train_dir if seq_id <= int(total_seqs * 0.8) else test_dir
        seq_name = f"MOT17-{seq_id:02d}-Missile3D"
        seq_path = os.path.join(split_dir, seq_name)
        os.makedirs(os.path.join(seq_path, "gt"), exist_ok=True)
        os.makedirs(os.path.join(seq_path, "det"), exist_ok=True)
        write_seqinfo(seq_path, seq_name, frames_per_seq)

        gt_data, det_data = [], []
        num_missiles = random.randint(1, 5)  # 蜂群数量 1-5 枚

        # 预先为每枚导弹生成 3D 机动参数
        missiles_params = []
        for _ in range(num_missiles):
            # 给定随机起终点，但保证必过中心 SHARED_TARGET_POS
            p0 = np.array([0, random.uniform(-20, 20), random.uniform(-10, 0)])
            p2 = np.array([150, random.uniform(-30, 30), random.uniform(-60, -20)])
            p1 = 2 * SHARED_TARGET_POS - 0.5 * p0 - 0.5 * p2  # 反推必须穿过中心的贝塞尔控制点

            missiles_params.append({
                "p0": p0, "p1": p1, "p2": p2,
                "warp": random.uniform(-0.15, 0.15),
                "wobble1": np.array([random.uniform(-10, 10), random.uniform(-10, 10), random.uniform(-5, 5)]),
                "wobble2": np.array([random.uniform(-4, 4), random.uniform(-4, 4), random.uniform(-2, 2)])
            })

        for f in range(frames_per_seq):
            t = f / (frames_per_seq - 1)

            for m_id, mp in enumerate(missiles_params, start=1):
                # 1. 在 3D 绝对空间计算真实的物理坐标 (遵循 UE4 的控制逻辑)
                warp_t = t + mp["warp"] * math.sin(2 * math.pi * t)
                base_3d = (1 - warp_t) ** 2 * mp["p0"] + 2 * (1 - warp_t) * warp_t * mp["p1"] + warp_t ** 2 * mp["p2"]
                offset = mp["wobble1"] * math.sin(2 * math.pi * t) + mp["wobble2"] * math.sin(4 * math.pi * t)
                true_pos_3d = base_3d + offset

                # 2. 扔给虚拟相机，计算真实的 2D Bbox 投影
                bbox = camera.project_to_2d_bbox(true_pos_3d)
                if not bbox: continue  # 如果飞出界了就不记录
                left, top, w, h = bbox

                # 只保留画面内的坐标 (过滤掉明显飞出 1920x1080 边界的数据)
                if left > IMG_W or top > IMG_H or (left + w) < 0 or (top + h) < 0:
                    continue

                # 3. 写入 GT 真值
                # 格式: frame, track_id, left, top, w, h, 1, 1, 1.0
                gt_data.append(f"{f + 1},{m_id},{left:.2f},{top:.2f},{w:.2f},{h:.2f},1,1,1.0\n")

                # 4. 模拟 YOLOX 缺陷写入 Det
                if random.random() > 0.10:  # 10%的自然丢帧
                    # 加入高斯噪声模拟检测框不稳
                    noisy_l = left + np.random.normal(0, 2.0)
                    noisy_t = top + np.random.normal(0, 2.0)
                    noisy_w = w + np.random.normal(0, 1.0)
                    noisy_h = h + np.random.normal(0, 1.0)
                    conf = random.uniform(0.6, 0.95)
                    # 格式: frame, -1, left, top, w, h, conf, -1, -1, -1
                    det_data.append(
                        f"{f + 1},-1,{noisy_l:.2f},{noisy_t:.2f},{noisy_w:.2f},{noisy_h:.2f},{conf:.2f},-1,-1,-1\n")

        # 保存
        with open(os.path.join(seq_path, "gt", "gt.txt"), "w") as f_gt:
            f_gt.writelines(gt_data)
        with open(os.path.join(seq_path, "det", "det.txt"), "w") as f_det:
            f_det.writelines(det_data)

    print(f"✅ 成功生成 3D 物理投影数据集！共 {total_seqs} 组视频序列。")
    print(f"💾 数据格式兼容 MOT17，存放在: {os.path.abspath(base_dir)}")


if __name__ == "__main__":
    # 生成 100 组数据序列用于训练和测试
    generate_mot_dataset(DATASET_OUT_DIR, total_seqs=100)