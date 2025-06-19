import os
from devdeploy.inference.class_infer import Inference

# 模型路径和图片目录
CHECKPOINT_PATH = 'charCls/works/fullmodel_best.pth'
IMG_DIR = 'data/classify/val/25'

# 创建推理器实例
infer = Inference(CHECKPOINT_PATH, device='cpu', class_names=['20', '25', '50'])

# 批量推理
results = infer.infer_batch(IMG_DIR)

# 打印结果
for img_name, result in results.items():
    print(f'class_id: {result["class_id"]}  class_name: {result["class_name"]} 置信度: {result["confidence"]:.3f}      {img_name}')

