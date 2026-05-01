import cv2
import os
import shutil
import torch
from yolox.exp import get_exp
from yolox.utils import postprocess
from yolox.data.data_augment import preproc  # ⬅️ 【新增引入】专用的推理预处理函数

# ========== 📂 核心路径配置 ==========
INPUT_BASE_DIR = r"H:\Missile_Picture_Dataset"
OUTPUT_DIR = r"H:\Identify_Missile_Picture_Dataset\dataset"

# ========== 🧠 模型配置 ==========
EXP_FILE = "exps/default/yolox_missile_v1.py"
CKPT_FILE = "YOLOX_outputs/yolox_missile_v1/best_ckpt.pth"
CONF_THRESH = 0.45  # 预测阈值


# ==================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("🧠 正在加载 YOLOX 模型，准备流水线作业...")
    exp = get_exp(EXP_FILE, None)
    model = exp.get_model()
    model.eval()
    model.cuda()

    # 💡 顺手修复警告：显式声明 weights_only=False
    model.load_state_dict(torch.load(CKPT_FILE, map_location="cuda", weights_only=False)["model"])

    print(f"🚀 模型加载完毕，开始横扫所有子文件夹...")
    total_processed = 0

    for folder_name in os.listdir(INPUT_BASE_DIR):
        folder_path = os.path.join(INPUT_BASE_DIR, folder_name)

        if os.path.isdir(folder_path) and folder_name.startswith("Images_"):
            img_files = [f for f in os.listdir(folder_path) if f.endswith('.jpg')]

            for img_name in img_files:
                img_path = os.path.join(folder_path, img_name)

                unique_name_base = f"{folder_name}_{img_name.replace('.jpg', '')}"
                new_img_name = f"{unique_name_base}.jpg"
                new_txt_name = f"{unique_name_base}.txt"

                out_img_path = os.path.join(OUTPUT_DIR, new_img_name)
                out_txt_path = os.path.join(OUTPUT_DIR, new_txt_name)

                img = cv2.imread(img_path)
                if img is None:
                    continue

                img_info = {"height": img.shape[0], "width": img.shape[1]}
                test_size = exp.test_size

                # 💡 【核心修复】：直接使用 YOLOX 的独立 preproc 进行推理预处理
                # 这样既不会触发报错，速度也比之前调用 DataLoader 快无数倍！
                img_resized, ratio = preproc(img, test_size)
                img_tensor = torch.from_numpy(img_resized).unsqueeze(0).float().cuda()

                with torch.no_grad():
                    outputs = model(img_tensor)
                    outputs = postprocess(outputs, exp.num_classes, exp.test_conf, exp.nmsthre, class_agnostic=True)

                shutil.copy2(img_path, out_img_path)

                with open(out_txt_path, 'w') as f:
                    if outputs[0] is not None:
                        bboxes = outputs[0][:, 0:4] / ratio
                        cls_ids = outputs[0][:, 6]
                        scores = outputs[0][:, 4] * outputs[0][:, 5]

                        for i in range(len(bboxes)):
                            if scores[i] < CONF_THRESH:
                                continue

                            box = bboxes[i]
                            w = (box[2] - box[0]) / img_info["width"]
                            h = (box[3] - box[1]) / img_info["height"]
                            x_center = box[0] / img_info["width"] + w / 2
                            y_center = box[1] / img_info["height"] + h / 2

                            f.write(
                                f"{int(cls_ids[i])} {x_center.item():.6f} {y_center.item():.6f} {w.item():.6f} {h.item():.6f}\n")

                total_processed += 1
                if total_processed % 200 == 0:
                    print(f"  ...流水线轰鸣中：已极速处理并融合 {total_processed} 张图片")

    print(f"\n✅ 批量打标完美竣工！共处理了 {total_processed} 张图片。")
    print(f"📁 所有图片和标签已贴心存放于同一目录: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()