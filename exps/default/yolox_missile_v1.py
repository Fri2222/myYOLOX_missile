import os
from yolox.exp import Exp as MyExp

class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        # 模型大小配置 (对应 yolox_s)
        self.depth = 0.33
        self.width = 0.50
        self.exp_name = "yolox_missile_v1"

        # 🎯 核心设置：你的专属类别数量 (导弹)
        self.num_classes = 1

        # 📂 数据集绝对路径 (指向根目录即可)
        self.data_dir = r"H:\Dataset\Missle\--.v1i.coco"

        # 🎯 指定标签文件名 (YOLOX 会自动去 data_dir/annotations/ 目录下找这俩文件)
        self.train_ann = "instances_train2017.json"
        self.val_ann = "instances_val2017.json"

        # 训练参数微调
        self.max_epoch = 100
        self.data_num_workers = 4
        self.eval_interval = 5