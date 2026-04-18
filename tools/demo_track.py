import os
import torch
import numpy as np
import argparse
import motmetrics as mm
import cv2
from loguru import logger

# 导入你自己的 ByteTracker (确保里面的卡尔曼滤波已指向你训练好的 KalmanNet)
from yolox.tracker.byte_tracker import BYTETracker
from yolox.utils import setup_logger


def make_parser():
    parser = argparse.ArgumentParser("ByteTrack+KalmanNet 视频追踪与评测引擎")

    # --- 统一对齐 YOLOX 标准参数格式 ---
    parser.add_argument("-expn", "--experiment-name", type=str, default="yolox_s_missile")
    parser.add_argument("-d", "--dataset_dir", type=str, default=r"datasets\MOT17_Missile\test", help="测试数据集目录")

    # Tracker 的三大核心超参
    parser.add_argument("--track_thresh", type=float, default=0.5, help="检测置信度阈值")
    parser.add_argument("--track_buffer", type=int, default=30, help="丢失找回的最大缓冲帧数")
    parser.add_argument("--match_thresh", type=float, default=0.8, help="前后帧框匹配容忍度")
    parser.add_argument("--mot20", dest="mot20", default=False, action="store_true", help="是否使用 MOT20 标准")

    # 附加可视化参数
    parser.add_argument("--save_video", action="store_true", default=True, help="是否同时渲染并保存带框视频")
    return parser


@logger.catch
def main():
    args = make_parser().parse_args()

    if not os.path.exists(args.dataset_dir):
        print(f"❌ 错误: 找不到测试集目录 {args.dataset_dir}")
        return

    # =========================================================
    # 💥 核心路径规范：与 eval.py 保持完全一致
    # 根目录形如: YOLOX_outputs/yolox_s_missile/tracked_missile_videos
    # =========================================================
    output_root = os.path.join("YOLOX_outputs", args.experiment_name)
    tracked_videos_dir = os.path.join(output_root, "tracked_missile_videos")
    os.makedirs(tracked_videos_dir, exist_ok=True)

    # 设置日志
    setup_logger(output_root, filename="track_eval_log.txt", mode="a")
    logger.info("Args: {}".format(args))

    accs = []
    seq_names = []

    logger.info(f"🚀 开始加载测试集: {args.dataset_dir}")
    logger.info("-" * 60)

    # 遍历 test 目录下的所有视频序列 (如 Dataset_FixedView_20260312_112629)
    for seq_name in sorted(os.listdir(args.dataset_dir)):
        seq_dir = os.path.join(args.dataset_dir, seq_name)
        det_path = os.path.join(seq_dir, "det", "det.txt")
        gt_path = os.path.join(seq_dir, "gt", "gt.txt")

        if not os.path.isdir(seq_dir) or not os.path.exists(det_path) or not os.path.exists(gt_path):
            continue

        logger.info(f"⏳ 正在预测序列: {seq_name} ...")

        # 💥 为当前视频序列创建专属子目录
        # 形如: .../tracked_missile_videos/Dataset_FixedView_20260312_112629
        seq_output_dir = os.path.join(tracked_videos_dir, seq_name)
        os.makedirs(seq_output_dir, exist_ok=True)
        pred_path = os.path.join(seq_output_dir, "pred.txt")

        tracker = BYTETracker(args, frame_rate=60)

        # 读取检测数据
        dets = np.loadtxt(det_path, delimiter=',')
        frames = np.unique(dets[:, 0]) if len(dets) > 0 else []
        results = []

        # 存储每一帧的跟踪目标用于后续画图
        track_vis_data = {}

        # 逐帧推进追踪
        for frame_id in range(1, int(max(frames)) + 1) if len(frames) > 0 else []:
            frame_dets = dets[dets[:, 0] == frame_id]

            if len(frame_dets) > 0:
                bboxes = frame_dets[:, 2:6]
                scores = frame_dets[:, 6]

                detections = np.stack([
                    bboxes[:, 0], bboxes[:, 1],
                    bboxes[:, 0] + bboxes[:, 2], bboxes[:, 1] + bboxes[:, 3],
                    scores
                ], axis=1)

                online_targets = tracker.update(detections, [1080, 1920], [1080, 1920])
            else:
                online_targets = []

            track_vis_data[frame_id] = []
            for t in online_targets:
                tlwh = t.tlwh
                results.append(
                    f"{frame_id},{t.track_id},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{t.score:.2f},-1,-1,-1\n")
                track_vis_data[frame_id].append((t.track_id, tlwh, t.score))

        # 1. 保存 pred.txt 预测结果
        with open(pred_path, 'w') as f:
            f.writelines(results)

        # 2. [新增] 视频渲染模块：寻找原视频并渲染追踪框
        if args.save_video:
            original_video_path = os.path.join(seq_dir, "flight_video.avi")
            if os.path.exists(original_video_path):
                out_video_path = os.path.join(seq_output_dir, f"tracked_{seq_name}.avi")
                cap = cv2.VideoCapture(original_video_path)
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                vid_writer = cv2.VideoWriter(out_video_path, cv2.VideoWriter_fourcc(*"XVID"), fps, (width, height))

                # 随机生成颜色表
                np.random.seed(42)
                colors = np.random.randint(0, 255, size=(100, 3), dtype=np.uint8)

                f_id = 1
                while True:
                    ret, frame = cap.read()
                    if not ret: break

                    if f_id in track_vis_data:
                        for tid, tlwh, score in track_vis_data[f_id]:
                            color = colors[tid % 100].tolist()
                            cv2.rectangle(frame, (int(tlwh[0]), int(tlwh[1])),
                                          (int(tlwh[0] + tlwh[2]), int(tlwh[1] + tlwh[3])), color, 3)
                            label = f"ID:{tid} [{score:.2f}]"
                            cv2.putText(frame, label, (int(tlwh[0]), int(tlwh[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                                        color, 2)

                    vid_writer.write(frame)
                    f_id += 1

                vid_writer.release()
                cap.release()
                logger.info(f"🎞️ 渲染完成，视频存放于: {out_video_path}")

        # 3. 立即计算当前序列的 MOTA / IDF1 差异矩阵
        gt = mm.io.loadtxt(gt_path, fmt="mot15-2D", min_confidence=1)
        ts = mm.io.loadtxt(pred_path, fmt="mot15-2D")
        acc = mm.utils.compare_to_groundtruth(gt, ts, 'iou', distth=0.5)

        accs.append(acc)
        seq_names.append(seq_name)

    # 4. 生成最终的全局性能评估报告
    if len(accs) == 0:
        logger.error("❌ 没有找到任何有效的测试序列！")
        return

    logger.info("\n" + "=" * 90)
    logger.info("🎯 KalmanNet 导弹目标跟踪全局性能报告")
    logger.info("=" * 90)

    mh = mm.metrics.create()
    metrics_list = ['num_frames', 'mota', 'idf1', 'motp', 'num_switches', 'num_false_positives', 'num_misses']

    summary = mh.compute_many(accs, metrics=metrics_list, names=seq_names, generate_overall=True)
    strsummary = mm.io.render_summary(summary, formatters=mh.formatters, namemap=mm.io.motchallenge_metric_names)

    # 打印给用户看
    print(strsummary)
    # 写入日志
    logger.info("\n" + strsummary)


if __name__ == "__main__":
    main()