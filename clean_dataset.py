import json
import os

def clean_missing_images(json_path, img_dir):
    print(f"正在扫描检查: {json_path}")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    valid_images = []
    valid_image_ids = set()
    missing_count = 0

    # 1. 检查图片文件是否真的存在，且不是 0 字节的坏文件
    for img in data['images']:
        img_path = os.path.join(img_dir, img['file_name'])
        if os.path.exists(img_path) and os.path.getsize(img_path) > 0:
            valid_images.append(img)
            valid_image_ids.add(img['id'])
        else:
            missing_count += 1
            print(f"  [-] 发现丢失或损坏的图片: {img['file_name']}")

    # 2. 把对应丢失图片的标注框也一起删掉
    valid_annotations = [ann for ann in data['annotations'] if ann['image_id'] in valid_image_ids]

    data['images'] = valid_images
    data['annotations'] = valid_annotations

    # 3. 保存干净的 JSON
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f)

    print(f"✅ 清理完毕！共删除了 {missing_count} 个丢失/损坏的图片记录。\n")

# 执行清理
base_dir = r"H:\Dataset\Missle\--.v1i.coco"
clean_missing_images(
    os.path.join(base_dir, "annotations", "instances_train2017.json"),
    os.path.join(base_dir, "train2017")
)
clean_missing_images(
    os.path.join(base_dir, "annotations", "instances_val2017.json"),
    os.path.join(base_dir, "val2017")
)

# 顺手再次清理旧缓存，防止 YOLOX 偷懒
for file in os.listdir(os.path.join(base_dir, "annotations")):
    if file.endswith(".cache"):
        os.remove(os.path.join(base_dir, "annotations", file))
        print(f"🗑️ 已清理缓存: {file}")