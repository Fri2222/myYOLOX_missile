import os
import torch
import numpy as np
from loguru import logger
import argparse

# 导入你改进过的追踪器
from yolox.tracker.byte_tracker import BYTETracker

def make_parser():
    parser = argparse.ArgumentParser("纯数据流 Tracker 评估引擎")
    parser.add_argument("--det_dir", type=str, default=r"H:\Code\YOLOX\datasets\MOT17_Missile\train", help="你的数据集根目录")
    parser.add_argument("--seq", type=str, default="MOT17-01-Missile3D", help="要评估的序列名")
    
    # 核心追踪参数 (和实战保持一致)
    parser.add_argument("--track_thresh", type=float, default=0.5)
    parser.add_argument("--track_buffer", type=int, default=30)
    parser.add_argument("--match_thresh", type=float, default=0.8)
    parser.add_argument("--mot20", dest="mot20", default=False, action="store_true")
    return parser

def main():
    args = make_parser().parse_args()
    
    det_txt_path = os.path.join(args.det_dir, args.seq, "det", "det.txt")
    out_txt_path = f"{args.seq}.txt"
    
    if not os.path.exists(det_txt_path):
        logger.error(f"找不到检测文件: {det_txt_path}")
        return

    logger.info(f"🚀 开始纯数据流仿真跟踪: {args.seq}")
    tracker = BYTETracker(args, frame_rate=60)

    # 1. 预读取所有帧的检测数据 (模拟 YOLOX 输出)
    dets_by_frame = {}
    with open(det_txt_path, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            frame_id = int(parts[0])
            # det.txt 格式: frame, -1, x, y, w, h, conf
            x, y, w, h, conf = map(float, parts[2:7])
            
            if frame_id not in dets_by_frame:
                dets_by_frame[frame_id] = []
                
            # ByteTracker 接收的 Tensor 格式要求是: [x1, y1, x2, y2, score]
            dets_by_frame[frame_id].append([x, y, x + w, y + h, conf])

    results = []
    max_frame = max(dets_by_frame.keys()) if dets_by_frame else 0
    
    # 模拟输入图像尺寸 (基于你之前生成数据的配置 1920x1080)
    img_info = [1080, 1920]
    test_size = [1080, 1920]

    # 2. 逐帧喂给 Tracker
    for frame_id in range(1, max_frame + 1):
        if frame_id in dets_by_frame:
            dets = np.array(dets_by_frame[frame_id])
            # 转换为 tensor 喂入
            dets_tensor = torch.tensor(dets, dtype=torch.float32)
        else:
            # 如果这一帧完全漏检 (模拟 YOLOX 瞎了)
            dets_tensor = torch.empty((0, 5), dtype=torch.float32)

        # 挂载神经元大脑更新轨迹
        online_targets = tracker.update(dets_tensor, img_info, test_size)

        # 3. 收集标准 MOT 格式结果
        for t in online_targets:
            tlwh = t.tlwh
            tid = t.track_id
            score = t.score
            results.append(f"{frame_id},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{score:.2f},-1,-1,-1\n")

    # 4. 保存文件
    with open(out_txt_path, 'w', encoding='utf-8') as f:
        f.writelines(results)
        
    logger.info(f"🎉 跟踪完成！已生成 TrackEval 专用文件: {out_txt_path}")

if __name__ == "__main__":
    main()