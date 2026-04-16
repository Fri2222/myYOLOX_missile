###################################################################################################################

项目名称:yolox
环境名称:yolox_env

####################################################################################################################

# yolox 实验记录日志

| 实验ID | 日期 | Commit Hash | 关键改动 (Parameters) | 备注 |
| :--- | :--- | :--- | :--- | :--- |
| **Exp-001** | 2026-03-02 | `导弹数据集训练 | "H:\Dataset\Missle\Missile Subset.v3i.coco" |
| **Exp-001** | 2026-04-16 | `增加自己生成的导弹数据集 | " H:\Code\YOLOX\datasets\MOT17_Missile" |



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