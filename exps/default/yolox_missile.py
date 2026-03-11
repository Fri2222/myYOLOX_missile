import os
from yolox.exp import Exp as MyExp

class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.num_classes = 1
        self.depth = 0.33
        self.width = 0.50
        self.exp_name = "yolox_s_missile"

        # 使用 r 前缀防止 Windows 路径转义出错
        self.data_dir = r"H:\Dataset\Missle\Missile Detection.v9-sanket-edited.coco"
        self.train_ann = "instances_train2017.json"
        self.val_ann = "instances_val2017.json"

        self.max_epoch = 100
        self.input_size = (640, 640)
        self.test_size = (640, 640)