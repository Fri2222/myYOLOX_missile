import argparse
import os
import cv2
import torch
import numpy as np
from loguru import logger

from yolox.exp import get_exp
from yolox.utils import fuse_model, postprocess
from yolox.data.data_augment import ValTransform
from yolox.tracker.byte_tracker import BYTETracker


def make_parser():
    parser = argparse.ArgumentParser("KalmanNet+ByteTrack 导弹实战推理与溯源引擎")
    parser.add_argument("-f", "--exp_file", default="exps/default/yolox_missile.py", type=str)
    parser.add_argument("-c", "--ckpt", default="YOLOX_outputs/yolox_s_missile/best_ckpt.pth", type=str)
    parser.add_argument("--path", required=True, type=str, help="输入导弹视频的绝对路径")
    parser.add_argument("--device", default="gpu", type=str)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--fuse", action="store_true")

    # 追踪超参
    parser.add_argument("--track_thresh", type=float, default=0.5)
    parser.add_argument("--track_buffer", type=int, default=30)
    parser.add_argument("--match_thresh", type=float, default=0.8)
    parser.add_argument("--mot20", dest="mot20", default=False, action="store_true")
    return parser


def main():
    args = make_parser().parse_args()
    exp = get_exp(args.exp_file)

    # =========================================================
    # 1. 核心路径解析与归档规范逻辑
    # =========================================================
    abs_video_path = os.path.abspath(args.path)
    # 获取视频所在的父文件夹名称 (如: Dataset_FixedView_20260416_100030)
    source_folder_name = os.path.basename(os.path.dirname(abs_video_path))

    output_base_root = r"H:\Code\YOLOX\YOLOX_outputs\yolox_s_missile\tracked_missile_videos"
    final_save_dir = os.path.join(output_base_root, source_folder_name)
    os.makedirs(final_save_dir, exist_ok=True)

    video_save_path = os.path.join(final_save_dir, "tracked_" + os.path.basename(abs_video_path))
    txt_save_path = os.path.join(final_save_dir, "original_source.txt")

    # =========================================================
    # 2. 初始化视觉模型 (YOLOX)
    # =========================================================
    device = torch.device("cuda" if args.device == "gpu" else "cpu")
    model = exp.get_model().to(device).eval()

    logger.info(f"⏳ 正在加载权重: {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(ckpt["model"])

    if args.fuse:
        model = fuse_model(model)
    if args.fp16:
        model = model.half()

    # =========================================================
    # 3. 挂载神经元大脑 (BYTETracker + KalmanNet)
    # =========================================================
    tracker = BYTETracker(args, frame_rate=60)
    preproc = ValTransform(legacy=False)

    # 4. 视频流准备
    cap = cv2.VideoCapture(abs_video_path)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    vid_writer = cv2.VideoWriter(video_save_path, cv2.VideoWriter_fourcc(*"XVID"), fps, (int(width), int(height)))

    logger.info(f"🚀 开始渲染！输出目录: {final_save_dir}")

    np.random.seed(42)
    colors = np.random.randint(0, 255, size=(100, 3), dtype=np.uint8)

    frame_id = 0
    while True:
        ret, frame = cap.read()
        if not ret: break

        # --- A. YOLOX 视觉提取 ---
        img, _ = preproc(frame, None, exp.test_size)
        img = torch.from_numpy(img).unsqueeze(0).to(device)
        img = img.half() if args.fp16 else img.float()

        with torch.no_grad():
            outputs = model(img)
            # outputs 包含了被 NMS 过滤后的预测框，此时是 640x640 尺度
            outputs = postprocess(outputs, exp.num_classes, exp.test_conf, exp.nmsthre, class_agnostic=True)

        # --- B. Tracker 状态更新 (极简纯净模式) ---
        if outputs[0] is not None:
            # 💡 核心奥义：直接将 YOLOX 原始 Tensor 喂给 tracker！
            # tracker 内部自动拉伸到 1920x1080 -> 传给 KalmanFilter
            # KalmanFilter 内部自动归一化 0~1 -> 神经网络前向传播 -> 还原为 1920x1080 像素误差
            online_targets = tracker.update(outputs[0], [height, width], exp.test_size)
        else:
            online_targets = []

        # --- C. 画图渲染 ---
        for t in online_targets:
            tlwh = t.tlwh  # 这里吐出来的直接就是 1920x1080 的真实像素坐标
            tid = t.track_id
            score = t.score

            color = colors[tid % 100].tolist()
            # 由于已经是真实像素，直接强转 int() 即可画图
            cv2.rectangle(frame, (int(tlwh[0]), int(tlwh[1])), (int(tlwh[0] + tlwh[2]), int(tlwh[1] + tlwh[3])), color,
                          3)
            cv2.putText(frame, f"ID:{tid} [{score:.2f}]", (int(tlwh[0]), int(tlwh[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, color, 2)

        vid_writer.write(frame)
        frame_id += 1
        if frame_id % 30 == 0: logger.info(f"▶️ 已处理 {frame_id} 帧")

    # =========================================================
    # 5. 生成原始路径溯源文件
    # =========================================================
    with open(txt_save_path, "w", encoding="utf-8") as f:
        f.write("Original Video Absolute Path:\n")
        f.write(abs_video_path)

    vid_writer.release()
    cap.release()
    logger.info("-" * 60)
    logger.info(f"🎉 任务完成！")
    logger.info(f"📹 追踪视频: {video_save_path}")
    logger.info(f"📄 溯源文件: {txt_save_path}")


if __name__ == '__main__':
    main()