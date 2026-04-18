###################################################################################################################

项目名称:yolox
环境名称:yolox_env

####################################################################################################################

# yolox 实验记录日志

| 实验ID | 日期 | Commit Hash | 关键改动 (Parameters) | 备注 |
| :--- | :--- | :--- | :--- | :--- |
| **Exp-001** | 2026-03-02 | `导弹数据集训练 | "H:\Dataset\Missle\Missile Subset.v3i.coco" |
| **Exp-002** | 2026-04-16 | `增加自己生成的导弹数据集 | " H:\Code\YOLOX\datasets\MOT17_Missile" |
| **Exp-002** | 2026-04-18 | `| **Exp-002** | 2026-04-16 | `增加自己生成的导弹数据集 | " H:\Code\YOLOX\datasets\MOT17_Missile" | | |



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
				python gen_kalman_data.py 
				python train_kalmannet.py
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
