import json
import os

base_path = r"H:\Dataset\Missle\Missile Subset.v3i.coco\annotations"
files = ["instances_train2017.json", "instances_val2017.json"]

for file in files:
    path = os.path.join(base_path, file)
    print(f"正在处理: {path}")

    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 不管三七二十一，强行注入官方严格要求的完整格式
    data['info'] = {
        "description": "Missile Dataset",
        "url": "",
        "version": "1.0",
        "year": 2026,
        "contributor": "",
        "date_created": "2026-03-10"
    }
    data['licenses'] = [{"url": "", "id": 1, "name": "Unknown License"}]

    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f)

print("💥 强行覆写完毕！这回绝对有 info 字段了！")