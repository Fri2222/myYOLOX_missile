###################################################################################################################

项目名称:yolox
环境名称:yolox_env

####################################################################################################################

# yolox 实验记录日志

| 实验ID | 日期 | Commit Hash | 关键改动 (Parameters) | 备注 |
| :--- | :--- | :--- | :--- | :--- |
| **Exp-001** | 2026-03-02 | `导弹数据集训练 | "H:\Dataset\Missle\Missile Subset.v3i.coco" |
| **Exp-002** | 2026-04-16 | `增加自己生成的导弹数据集 | " H:\Code\YOLOX\datasets\MOT17_Missile" |
| **Exp-003** | 2026-04-18 | `Exp-003 将yolox_missile工程的Exp-0015进行移植；使用自己生成的数据集进行训练与性能评估 | | | |
| **Exp-004** | 2026-04-18 | `Exp-004 使用demo_track追踪视频，效果不佳；修改视频跟踪后保存目录 | | | |
| **Exp-005** | 2026-04-19 | `Exp-005 重新训练yolox检测器，检测效果一般，小目标容易漏检 | | | |
| **Exp-006** | 2026-04-19 | `Exp-006 重新训练yolox检测器，检测效果一般，目标会重复检测 | | | |
| **Exp-007** | 2026-04-19 | `Exp-007 重新训练yolox检测器，检测效果一般，容易漏检，1.4--.v1i.coco | | | |
| **Exp-007** | 2026-04-20 | `Exp-008 重新训练yolox检测器，roboflow增加图片，1.5--.v1i.coco  | | | |
| **** | 2026-05-1 | `进行导弹数据集一致性实验前的备份"  | | | |
| **Exp-009** | 2026-05-01 | `Exp-009 在个人导弹数据集中评估kalmannet| 81.877 | 88.928 |67.777 |
| **Exp-010** | 2026-05-01 | `Exp-010 在个人导弹数据集中评估原版KF| 82.848  | 89.667  |67.814 |
| **Exp-011** | 2026-05-01 | `Exp-011 在个人导弹数据集中评估yolo_missile工程的kalmannet 架构3| -34.951  | 0.94787  |0.94787 |



###################################################################################################################


## 详细数据备份
### Exp-001 
- **Code Version**: `导弹数据集训练`
					备注：数据集地址 "H:\Dataset\Missle\Missile Subset.v3i.coco"
- **Command**: `python tools\train.py -f exps\default\yolox_missile.py -d 1 -b 8 --fp16 -c yolox_s.pth`
- **Output**:
			权重文件:H:\Code\YOLOX\YOLOX_outputs\yolox_s_missile\best_ckpt.pth
			
			1.交互比：AP @[ IoU=0.50 ] = 0.840	//预测框和真实框重叠 50% 以上就算识别正确，模型准确率达到了 84.0%
					AP @[ IoU=0.50:0.95 ] = 0.497  //要求框的重叠度从 50% 到 95% 来计算平均分
			
			2.召回率:Average Recall (AR) @[ maxDets=100 ] = 0.589  //漏检
			
			3.精确度:AP @[ area= small ] = 0.229   //小目标精度：22.9%
					AP @[ area=medium ] = 0.413  //中目标精度：41.3%
					AP @[ area= large ] = 0.556  //大目标精度：55.6%




Average forward time: 7.21 ms, Average NMS time: 1.55 ms, Average inference time: 8.76 ms

 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.497

 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.840

 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.514

 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.229

 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.413

 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.556

 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.490

 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.587

 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.589

 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.298

 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.523

 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.645

per class AP:

| class   | AP     |

|:--------|:-------|

| missile | 49.696 |

per class AR:

| class   | AR     |

|:--------|:-------|

| missile | 58.899 |




###################################################################################################################


## 详细数据备份
### Exp-003 
- **Code Version**: `Exp-003 将yolox_missile工程的Exp-0015进行移植；使用自己生成的数据集进行训练与性能评估`
					备注：数据集地址 "H:\Dataset\Missle\Missile Subset.v3i.coco"
- **Command**: `
				python gen_kalman_data.py      //生成数据集
				python train_kalmannet.py
				//进行性能评估
				python tools/track.py --track_thresh 0.5 --track_buffer 30 --match_thresh 0.8
				

==========================================================================================
🎯 KalmanNet 导弹目标跟踪全局性能报告 (MOT Global Performance)
==========================================================================================
                    num_frames  MOTA  IDF1  MOTP IDs FP  FN
MOT17-100-Missile3D        145 90.1% 94.8% 0.218   0  2  26
MOT17-81-Missile3D         180 87.8% 93.5% 0.218   0  0  22
MOT17-82-Missile3D         169 87.6% 92.2% 0.197   4  0  17
MOT17-83-Missile3D         170 88.2% 93.2% 0.201   2  0  18
MOT17-84-Missile3D         168 86.4% 77.0% 0.200   4  0  39
MOT17-85-Missile3D         180 87.0% 69.2% 0.205   2  4  58
MOT17-86-Missile3D         151 91.2% 95.4% 0.215   0  1  36
MOT17-87-Missile3D         180 87.9% 93.6% 0.205   0  4  58
MOT17-88-Missile3D         166 84.3% 72.8% 0.204   5  3  68
MOT17-89-Missile3D         170 88.4% 71.3% 0.209   5  1  49
MOT17-90-Missile3D         164 89.6% 94.5% 0.199   0  0  50
MOT17-91-Missile3D         172 87.5% 53.9% 0.201   4  0  59
MOT17-92-Missile3D         165 87.9% 68.2% 0.198   4  0  16
MOT17-93-Missile3D         142 89.5% 94.5% 0.202   0  1  28
MOT17-94-Missile3D         163 87.4% 80.3% 0.200   1  1  57
MOT17-95-Missile3D         180 89.9% 70.5% 0.204   2  1  46
MOT17-96-Missile3D         163 91.6% 95.3% 0.189   2  0  25
MOT17-97-Missile3D         180 86.6% 73.6% 0.212   3  6  71
MOT17-98-Missile3D         180 87.1% 55.6% 0.220   2  3  35
MOT17-99-Missile3D         178 89.8% 94.6% 0.189   0  2  32
OVERALL                   3366 88.2% 80.1% 0.205  40 29 810
==========================================================================================

###################################################################################################################


## 详细数据备份
### Exp-004 
- **Code Version**: `Exp-004 使用demo_track追踪视频，效果不佳；修改视频跟踪后保存目录 `
					备注：数据集地址 "H:\Dataset\Missle\Missile Subset.v3i.coco"
- **Command**: `
				python gen_kalman_data.py      //生成数据集
				python train_kalmannet.py
				//进行性能评估
				python tools/track.py --track_thresh 0.5 --track_buffer 30 --match_thresh 0.8
				python tools/demo_track.py -f exps/default/yolox_missile.py -c YOLOX_outputs/yolox_s_missile/best_ckpt.pth --path "视频地址" --device gpu --fp16 --track_thresh 0.5 --track_buffer 30 --match_thresh 0.9
				
###################################################################################################################


## 详细数据备份
### Exp-005 
- **Code Version**: `Exp-005 重新训练yolox检测器，检测效果一般，小目标容易漏检 `
					备注：数据集地址 "H:\Dataset\Missle\--.v1i.coco"
- **Command**: `
				python gen_kalman_data.py      //生成数据集
				python train_kalmannet.py
				//进行性能评估
				python fix_labels.py 
				python clean_dataset.py
				python tools/track.py --track_thresh 0.5 --track_buffer 30 --match_thresh 0.8
				python tools/demo_track.py -f exps/default/yolox_missile.py -c YOLOX_outputs/yolox_s_missile/best_ckpt.pth --path "视频地址" --device gpu --fp16 --track_thresh 0.5 --track_buffer 30 --match_thresh 0.9
				
###################################################################################################################


## 详细数据备份
### Exp-006 
- **Code Version**: `Exp-006 重新训练yolox检测器，检测效果一般，目标会重复检测 `
					备注：数据集地址 "H:\Dataset\Missle\--.v1i.coco"
- **Command**: `
				python gen_kalman_data.py      //生成数据集
				python train_kalmannet.py
				//进行性能评估
				python fix_labels.py 
				python clean_dataset.py
				python tools/track.py --track_thresh 0.5 --track_buffer 30 --match_thresh 0.8
				python tools/demo_track.py -f exps/default/yolox_missile.py -c YOLOX_outputs/yolox_s_missile/best_ckpt.pth --path "视频地址" --device gpu --fp16 --track_thresh 0.5 --track_buffer 30 --match_thresh 0.9
		
###################################################################################################################


## 详细数据备份
### Exp-007 
- **Code Version**: `Exp-007 重新训练yolox检测器，检测效果一般，容易漏检，1.4--.v1i.coco | | | |
					备注：数据集地址 "H:\Dataset\Missle\--.v1i.coco"
- **Command**: `
				python gen_kalman_data.py      //生成数据集
				python train_kalmannet.py
				//进行性能评估
				python fix_labels.py 
				python clean_dataset.py
				python tools/track.py --track_thresh 0.5 --track_buffer 30 --match_thresh 0.8
				python tools/demo_track.py -f exps/default/yolox_missile.py -c YOLOX_outputs/yolox_s_missile/best_ckpt.pth --path "视频地址" --device gpu --fp16 --track_thresh 0.5 --track_buffer 30 --match_thresh 0.9

###################################################################################################################


## 详细数据备份
### Exp-008
- **Code Version**: `Exp-008 重新训练yolox检测器，roboflow增加图片，1.5--.v1i.coco | | | |
					备注：数据集地址 "H:\Dataset\Missle\--.v1i.coco"
- **Command**: `
				python gen_kalman_data.py      //生成数据集
				python train_kalmannet.py
				//进行性能评估
				python fix_labels.py 
				python clean_dataset.py
				python tools/track.py --track_thresh 0.5 --track_buffer 30 --match_thresh 0.8
				python tools/demo_track.py -f exps/default/yolox_missile.py -c YOLOX_outputs/yolox_s_missile/best_ckpt.pth --path "视频地址" --device gpu --fp16 --track_thresh 0.5 --track_buffer 30 --match_thresh 0.9
		
		
		
Average forward time: 8.57 ms, Average NMS time: 3.09 ms, Average inference time: 11.66 ms
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.500
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.871
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.528
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.253
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.416
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.552
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.440
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.589
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.592
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.412
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.482
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.644
per class AP:
| class   | AP     |
|:--------|:-------|
| missile | 49.978 |
per class AR:
| class   | AR     |
|:--------|:-------|
| missile | 59.181 |



###################################################################################################################


## 详细数据备份
### Exp-009
- **Code Version**: `Exp-009 在个人导弹数据集中评估kalmannet| 81.877 | 88.928 |67.777 |
					备注：数据集地址 "H:\Code\YOLOX\datasets\MOT17_Missile"
- **Command**: 
				python tools/run_tracker_sim.py --seq MOT17-01-Missile3D
				python TrackEval/scripts/run_mot_challenge.py --BENCHMARK MOT17-Missile --SPLIT_TO_EVAL train --TRACKERS_TO_EVAL YOLOX_ByteTrack --METRICS HOTA CLEAR Identity --USE_PARALLEL False --NUM_PARALLEL_CORES 1

HOTA: YOLOX_ByteTrack-pedestrian   HOTA      DetA      AssA      DetRe     DetPr     AssRe     AssPr     LocA      OWTA      HOTA(0)   LocA(0)   HOTALocA(0)
MOT17-01-Missile3D                 67.777    68.629    66.944    71.419    84.879    69.38     85.72     85.319    69.101    82.405    83.137    68.51
COMBINED                           67.777    68.629    66.944    71.419    84.879    69.38     85.72     85.319    69.101    82.405    83.137    68.51

CLEAR: YOLOX_ByteTrack-pedestrian  MOTA      MOTP      MODA      CLR_Re    CLR_Pr    MTR       PTR       MLR       sMOTA     CLR_TP    CLR_FN    CLR_FP    IDSW      MT        PT        ML        Frag
MOT17-01-Missile3D                 81.877    83.323    83.495    83.819    99.615    50        50        0         67.899    259       50        1         5         1         1         0         16
COMBINED                           81.877    83.323    83.495    83.819    99.615    50        50        0         67.899    259       50        1         5         1         1         0         16

Identity: YOLOX_ByteTrack-pedestrianIDF1      IDR       IDP       IDTP      IDFN      IDFP
MOT17-01-Missile3D                 88.928    81.877    97.308    253       56        7
COMBINED                           88.928    81.877    97.308    253       56        7

Count: YOLOX_ByteTrack-pedestrian  Dets      GT_Dets   IDs       GT_IDs
MOT17-01-Missile3D                 260       309       8         2
COMBINED                           260       309       8         2

Timing analysis:
MotChallenge2DBox.get_raw_seq_data                                     0.0210 sec
MotChallenge2DBox.get_preprocessed_seq_data                            0.0474 sec
HOTA.eval_sequence                                                     0.0453 sec
CLEAR.eval_sequence                                                    0.0140 sec
Identity.eval_sequence                                                 0.0033 sec
Count.eval_sequence                                                    0.0000 sec
eval_sequence                                                          0.1325 sec
Evaluator.evaluate                                                     1.5261 sec


###################################################################################################################


## 详细数据备份
### Exp-010
- **Code Version**: `Exp-010 在个人导弹数据集中评估原版KF| 82.848  | 89.667  |67.814 |
					备注：数据集地址 "H:\Code\YOLOX\datasets\MOT17_Missile"
- **Command**: 
				python tools/run_tracker_sim.py --seq MOT17-01-Missile3D 
				>copy /Y H:\Code\YOLOX\MOT17-01-Missile3D.txt H:\Code\YOLOX\TrackEval\data\trackers\mot_challenge\MOT17-Missile-train\YOLOX_ByteTrack\data\
				python TrackEval/scripts/run_mot_challenge.py --BENCHMARK MOT17-Missile --SPLIT_TO_EVAL train --TRACKERS_TO_EVAL YOLOX_ByteTrack --METRICS HOTA CLEAR Identity --USE_PARALLEL False --NUM_PARALLEL_CORES 1


HOTA: YOLOX_ByteTrack-pedestrian   HOTA      DetA      AssA      DetRe     DetPr     AssRe     AssPr     LocA      OWTA      HOTA(0)   LocA(0)   HOTALocA(0)
MOT17-01-Missile3D                 67.814    68.577    67.064    71.3      84.09     69.587    84.635    84.694    69.093    83.467    82.371    68.753
COMBINED                           67.814    68.577    67.064    71.3      84.09     69.587    84.635    84.694    69.093    83.467    82.371    68.753

CLEAR: YOLOX_ByteTrack-pedestrian  MOTA      MOTP      MODA      CLR_Re    CLR_Pr    MTR       PTR       MLR       sMOTA     CLR_TP    CLR_FN    CLR_FP    IDSW      MT        PT        ML        Frag
MOT17-01-Missile3D                 82.848    82.546    84.142    84.466    99.618    50        50        0         68.105    261       48        1         4         1         1         0         15
COMBINED                           82.848    82.546    84.142    84.466    99.618    50        50        0         68.105    261       48        1         4         1         1         0         15

Identity: YOLOX_ByteTrack-pedestrianIDF1      IDR       IDP       IDTP      IDFN      IDFP
MOT17-01-Missile3D                 89.667    82.848    97.71     256       53        6
COMBINED                           89.667    82.848    97.71     256       53        6

Count: YOLOX_ByteTrack-pedestrian  Dets      GT_Dets   IDs       GT_IDs
MOT17-01-Missile3D                 262       309       7         2
COMBINED                           262       309       7         2

Timing analysis:
MotChallenge2DBox.get_raw_seq_data                                     0.0149 sec
MotChallenge2DBox.get_preprocessed_seq_data                            0.0461 sec
HOTA.eval_sequence                                                     0.0545 sec
CLEAR.eval_sequence                                                    0.0091 sec
Identity.eval_sequence                                                 0.0041 sec
Count.eval_sequence                                                    0.0000 sec
eval_sequence                                                          0.1303 sec
Evaluator.evaluate    


###################################################################################################################


## 详细数据备份
### Exp-010
- **Code Version**: `Exp-010 在个人导弹数据集中评估yolo_missile工程的kalmannet 架构3| -34.951  | 0.94787  |0.94787 |
					备注：数据集地址 "H:\Code\YOLOX\datasets\MOT17_Missile"
- **Command**: 
				python tools/run_tracker_sim.py --seq MOT17-01-Missile3D 
				>copy /Y H:\Code\YOLOX\MOT17-01-Missile3D.txt H:\Code\YOLOX\TrackEval\data\trackers\mot_challenge\MOT17-Missile-train\YOLOX_ByteTrack\data\
				python TrackEval/scripts/run_mot_challenge.py --BENCHMARK MOT17-Missile --SPLIT_TO_EVAL train --TRACKERS_TO_EVAL YOLOX_ByteTrack --METRICS HOTA CLEAR Identity --USE_PARALLEL False --NUM_PARALLEL_CORES 1



HOTA: YOLOX_ByteTrack-pedestrian   HOTA      DetA      AssA      DetRe     DetPr     AssRe     AssPr     LocA      OWTA      HOTA(0)   LocA(0)   HOTALocA(0)
MOT17-01-Missile3D                 0.94787   0.92358   0.62003   1.2434    3.4001    0.62327   56.551    69.718    0.82624   1.1493    47.464    0.54549
COMBINED                           0.71145   0.92358   0.62003   1.2434    3.4001    0.62327   56.551    69.718    0.82624   1.1493    47.464    0.54549

CLEAR: YOLOX_ByteTrack-pedestrian  MOTA      MOTP      MODA      CLR_Re    CLR_Pr    MTR       PTR       MLR       sMOTA     CLR_TP    CLR_FN    CLR_FP    IDSW      MT        PT        ML        Frag
MOT17-01-Missile3D                 -34.951   70.066    -34.628   0.97087   2.6549    0         0         100       -35.242   3         306       110       1         0         0         2         1
COMBINED                           -34.951   70.066    -34.628   0.97087   2.6549    0         0         100       -35.242   3         306       110       1         0         0         2         1

Identity: YOLOX_ByteTrack-pedestrianIDF1      IDR       IDP       IDTP      IDFN      IDFP
MOT17-01-Missile3D                 0.94787   0.64725   1.7699    2         307       111
COMBINED                           0.94787   0.64725   1.7699    2         307       111

Count: YOLOX_ByteTrack-pedestrian  Dets      GT_Dets   IDs       GT_IDs
MOT17-01-Missile3D                 113       309       110       2
COMBINED                           113       309       110       2

Timing analysis:
MotChallenge2DBox.get_raw_seq_data                                     0.0141 sec
MotChallenge2DBox.get_preprocessed_seq_data                            0.0273 sec
HOTA.eval_sequence                                                     0.0325 sec
CLEAR.eval_sequence                                                    0.0044 sec
Identity.eval_sequence                                                 0.0040 sec
Count.eval_sequence                                                    0.0000 sec
eval_sequence                                                          0.0839 sec
Evaluator.evaluate                                                     1.1135 sec