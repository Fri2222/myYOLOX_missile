import json
import os

base_path = r"H:\Dataset\Missle\--.v1i.coco\annotations"
files = ["instances_train2017.json", "instances_val2017.json"]

for file in files:
    path = os.path.join(base_path, file)
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 强制将类别注册表精简为 1 个（ID为0）
    data['categories'] = [{'id': 0, 'name': 'missile', 'supercategory': 'none'}]

    # 强制将所有标注框的类别 ID 重置为 0
    for ann in data['annotations']:
        ann['category_id'] = 0

    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f)

print("Bingo！标签修复完成！所有的 category_id 已经对齐为 0。")