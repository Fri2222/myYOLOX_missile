import os

target_file = r"H:\Code\YOLOX\yolox\evaluators\coco_evaluator.py"

with open(target_file, 'r', encoding='utf-8') as f:
    code = f.read()

# 把尝试导入 C++ 加速模块的代码，强行替换为导入纯 Python 模块
code = code.replace(
    "from yolox.layers import COCOeval_opt as COCOeval",
    "from pycocotools.cocoeval import COCOeval"
)

with open(target_file, 'w', encoding='utf-8') as f:
    f.write(code)

print("🛡️ C++ 依赖已解除！YOLOX 现在将使用安全的纯 Python 评估。")