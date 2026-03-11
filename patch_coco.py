import os

# 精准定位到你报错的那个底层文件
target_file = r"F:\Software\Anaconda\envs\yolox_env\lib\site-packages\pycocotools\coco.py"

with open(target_file, 'r', encoding='utf-8') as f:
    code = f.read()

# 把强制要求 info 的代码，替换成“如果有就用，没有就拉倒”的宽容代码
code = code.replace(
    "res.dataset['info'] = copy.deepcopy(self.dataset['info'])",
    "res.dataset['info'] = copy.deepcopy(self.dataset.get('info', {}))"
)
code = code.replace(
    "res.dataset['licenses'] = copy.deepcopy(self.dataset['licenses'])",
    "res.dataset['licenses'] = copy.deepcopy(self.dataset.get('licenses', []))"
)

with open(target_file, 'w', encoding='utf-8') as f:
    f.write(code)

print("🎯 底层源码修改完成！pycocotools 再也不会因为 info 报错了。")