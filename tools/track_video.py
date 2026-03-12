import argparse
import os
import cv2
import torch
import time

from yolox.exp import get_exp
from yolox.utils import get_model_info, postprocess
from demo import Predictor  # 直接复用你已经跑通的预测器

# 导入刚才移植过来的 ByteTrack 追踪器
from yolox.tracker.byte_tracker import BYTETracker


class TrackArgs:
    def __init__(self, track_thresh=0.1):
        self.track_thresh = track_thresh  # 极低门槛，绝不漏掉导弹
        self.track_buffer = 30  # 允许导弹消失几帧再找回
        self.match_thresh = 0.8
        self.mot20 = False


def make_parser():
    """定义命令行参数"""
    parser = argparse.ArgumentParser("ByteTrack+YOLOX Missile Tracker!")
    parser.add_argument(
        "-p", "--path", type=str, required=True, help="要测试的视频文件绝对路径"
    )
    return parser


def main(args):
    # --- 1. 配置参数 ---
    video_path = args.path  # 从命令行获取视频路径
    ckpt_path = r"H:\Code\YOLOX\YOLOX_outputs\yolox_s_missile\best_ckpt.pth"
    exp_file = r"exps\default\yolox_missile.py"

    if not os.path.exists(video_path):
        print(f"❌ 错误: 找不到视频文件 {video_path}")
        return

    # --- 2. 初始化 YOLOX 模型 ---
    exp = get_exp(exp_file, None)
    exp.test_conf = 0.1  # 检测门槛设为0.1
    exp.nmsthre = 0.4

    model = exp.get_model()
    model.cuda()
    model.eval()
    ckpt = torch.load(ckpt_path, map_location="cuda")
    model.load_state_dict(ckpt["model"])

    predictor = Predictor(model, exp, ("Missile",), None, None, "gpu")

    # --- 3. 初始化 ByteTrack 追踪器 ---
    tracker = BYTETracker(TrackArgs(track_thresh=0.1), frame_rate=30)

    # --- 4. 视频流处理准备 ---
    cap = cv2.VideoCapture(video_path)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = cap.get(cv2.CAP_PROP_FPS)

    # 💥 【核心修改区域】：构建全新的层级目录结构和元数据TXT记录
    # 获取类似 Dataset_FixedView_20260312_112629 这样的父文件夹名
    parent_dir_name = os.path.basename(os.path.dirname(video_path))

    # 定义基础输出主目录
    base_output_dir = r"H:\Code\YOLOX\YOLOX_outputs\yolox_s_missile\tracked_missile_videos"

    # 拼接出本次专属的独立文件夹，并自动创建（包含多级目录）
    target_dir = os.path.join(base_output_dir, parent_dir_name)
    os.makedirs(target_dir, exist_ok=True)

    # 视频保存路径
    save_path = os.path.join(target_dir, f"tracked_video.avi")

    # 写入含有原视频路径的 txt 文件
    txt_path = os.path.join(target_dir, "original_source.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"Original Video Absolute Path:\n{os.path.abspath(video_path)}")

    vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*"XVID"), fps, (int(width), int(height)))

    frame_id = 0
    print(f"🚀 开始追踪视频: {video_path}")
    print(f"📁 结果文件夹已创建在: {target_dir}")

    while True:
        ret_val, frame = cap.read()
        if not ret_val:
            break

        # [检测阶段]
        outputs, img_info = predictor.inference(frame)

        # [追踪阶段]
        if outputs[0] is not None:
            online_targets = tracker.update(outputs[0], [img_info['height'], img_info['width']], exp.test_size)

            # 画框和 ID
            for t in online_targets:
                tlwh = t.tlwh  # 取出边界框坐标
                tid = t.track_id  # 取出专属 ID
                score = t.score  # 置信度

                x1, y1, w, h = int(tlwh[0]), int(tlwh[1]), int(tlwh[2]), int(tlwh[3])

                # 画矩形框
                cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), (0, 0, 255), 2)
                # 写上 Missile 和 ID 号
                cv2.putText(frame, f"Missile ID:{tid} {score:.2f}", (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # 写入视频
        vid_writer.write(frame)
        frame_id += 1
        if frame_id % 30 == 0:
            print(f"✅ 已处理 {frame_id} 帧...")

    cap.release()
    vid_writer.release()
    print(f"🎉 追踪完成！结果已保存至: {target_dir}")


if __name__ == "__main__":
    args = make_parser().parse_args()
    main(args)