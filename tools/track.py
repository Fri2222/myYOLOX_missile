import os
import torch
import numpy as np
import argparse
import motmetrics as mm
from loguru import logger

# 导入你自己的 ByteTracker (确保里面的卡尔曼滤波已指向你训练好的 KalmanNet)
from yolox.tracker.byte_tracker import BYTETracker


def make_parser():
    parser = argparse.ArgumentParser("ByteTrack+KalmanNet 视频追踪与评测引擎")

    # 核心测试数据目录
    parser.add_argument(
        "-d", "--dataset_dir",
        type=str,
        default=r"datasets\MOT17_Missile\test",
        help="测试数据集目录，内部包含多个如 Dataset_FixedView_xxx 的子序列"
    )

    # 指定专门的视频输出根目录 (根据你的需求硬编码或作为参数)
    parser.add_argument(
        "-o", "--output_dir",
        type=str,
        default=r"H:\Code\YOLOX\YOLOX_outputs\yolox_s_missile\tracked_missile_videos",
        help="追踪结果保存的根目录"
    )

    # Tracker 的三大核心超参 (与 eval.py 规范保持一致)
    parser.add_argument("--track_thresh", type=float, default=0.5, help="检测置信度阈值")
    parser.add_argument("--track_buffer", type=int, default=30, help="丢失找回的最大缓冲帧数")
    parser.add_argument("--match_thresh", type=float, default=0.8, help="前后帧框匹配容忍度")
    parser.add_argument("--mot20", dest="mot20", default=False, action="store_true", help="是否使用 MOT20 标准")

    return parser


@logger.catch
def main():
    args = make_parser().parse_args()

    if not os.path.exists(args.dataset_dir):
        logger.error(f"❌ 找不到测试集目录: {args.dataset_dir}")
        return

    # 准备收集所有视频序列的测试结果
    accs = []
    seq_names = []

    logger.info(f"🚀 开始加载测试集: {args.dataset_dir}")
    logger.info(f"⚙️ 追踪超参: thresh={args.track_thresh}, buffer={args.track_buffer}, match={args.match_thresh}")
    logger.info("-" * 60)

    # 遍历 test 目录下的所有视频序列 (例如 Dataset_FixedView_20260312_112629)
    for seq_name in sorted(os.listdir(args.dataset_dir)):
        seq_dir = os.path.join(args.dataset_dir, seq_name)
        det_path = os.path.join(seq_dir, "det", "det.txt")
        gt_path = os.path.join(seq_dir, "gt", "gt.txt")

        if not os.path.isdir(seq_dir) or not os.path.exists(det_path) or not os.path.exists(gt_path):
            continue

        logger.info(f"⏳ 正在处理序列: {seq_name} ...")

        # =========================================================
        # 💥 核心路径构造：根据规范生成保存目录
        # 例如: H:\Code\YOLOX\YOLOX_outputs\yolox_s_missile\tracked_missile_videos\Dataset_FixedView_20260312_112629
        # =========================================================
        target_save_dir = os.path.join(args.output_dir, seq_name)
        os.makedirs(target_save_dir, exist_ok=True)

        # 预测结果 txt 文件保存在这个专属目录下
        pred_path = os.path.join(target_save_dir, "pred.txt")

        # 为了追溯，你可以选择记录原始视频所在的路径 (可选功能，符合你的数据闭环习惯)
        source_record_path = os.path.join(target_save_dir, "original_source.txt")
        with open(source_record_path, "w") as f:
            f.write(f"Source Sequence Directory: {os.path.abspath(seq_dir)}\n")

        # 1. 初始化独立的追踪器
        tracker = BYTETracker(args, frame_rate=60)

        # 2. 读取检测数据
        try:
            dets = np.loadtxt(det_path, delimiter=',')
        except Exception as e:
            logger.warning(f"无法读取检测文件 {det_path}: {e}")
            continue

        frames = np.unique(dets[:, 0]) if len(dets) > 0 else []
        results = []

        # 3. 逐帧推进追踪
        for frame_id in range(1, int(max(frames)) + 1) if len(frames) > 0 else []:
            frame_dets = dets[dets[:, 0] == frame_id]

            if len(frame_dets) > 0:
                bboxes = frame_dets[:, 2:6]
                scores = frame_dets[:, 6]

                # 转换为 [x1, y1, x2, y2, score]
                detections = np.stack([
                    bboxes[:, 0],
                    bboxes[:, 1],
                    bboxes[:, 0] + bboxes[:, 2],
                    bboxes[:, 1] + bboxes[:, 3],
                    scores
                ], axis=1)

                # 直接将 NumPy 数组传给 tracker (传入宽高参数)
                online_targets = tracker.update(detections, [1080, 1920], [1080, 1920])
            else:
                online_targets = []

            for t in online_targets:
                tlwh = t.tlwh
                results.append(
                    f"{frame_id},{t.track_id},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{t.score:.2f},-1,-1,-1\n")

        # 保存 pred.txt 备查
        with open(pred_path, 'w') as f:
            f.writelines(results)

        logger.info(f"✅ 预测结果已保存至: {pred_path}")

        # 4. 立即计算当前序列的 MOTA / IDF1 差异矩阵
        try:
            gt = mm.io.loadtxt(gt_path, fmt="mot15-2D", min_confidence=1)
            ts = mm.io.loadtxt(pred_path, fmt="mot15-2D")
            # distth=0.5 表示框重合度(IoU)须大于 50%
            acc = mm.utils.compare_to_groundtruth(gt, ts, 'iou', distth=0.5)
            accs.append(acc)
            seq_names.append(seq_name)
        except Exception as e:
            logger.error(f"评估计算时发生错误: {e}")

    # 5. 生成最终的全局性能评估报告
    if len(accs) == 0:
        logger.error("❌ 没有找到任何有效的测试序列或评估失败！")
        return

    print("\n" + "=" * 90)
    print("🎯 KalmanNet 导弹目标跟踪全局性能报告 (MOT Global Performance)")
    print("=" * 90)

    mh = mm.metrics.create()
    metrics_list = ['num_frames', 'mota', 'idf1', 'motp', 'num_switches', 'num_false_positives', 'num_misses']

    # generate_overall=True 会在底部追加一行 OVERALL 全局平均分
    summary = mh.compute_many(accs, metrics=metrics_list, names=seq_names, generate_overall=True)

    strsummary = mm.io.render_summary(
        summary,
        formatters=mh.formatters,
        namemap=mm.io.motchallenge_metric_names
    )
    print(strsummary)
    print("=" * 90)


if __name__ == "__main__":
    main()