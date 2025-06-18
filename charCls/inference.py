import os
import torch
from torchvision import transforms
from PIL import Image
from mmengine.config import Config
from mmpretrain.registry import MODELS
from mmengine.runner import load_checkpoint

# 路径
CONFIG_PATH = 'charCls/resnet18_finetune.py'
CHECKPOINT_PATH = 'charCls/works/fullmodel_epoch50.pth'
IMG_DIR = 'data/classify/val/20'

# 1. 定义前处理流程（和训练一致）
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[123.675/255, 116.28/255, 103.53/255],
        std=[58.395/255, 57.12/255, 57.375/255]
    )
])


# 2. 用配置文件构建模型
# cfg = Config.fromfile(CONFIG_PATH)
# model = MODELS.build(cfg.model)
# load_checkpoint(model, CHECKPOINT_PATH, map_location='cpu')
# model.eval()


model = torch.load(CHECKPOINT_PATH, map_location='cpu') #load_state_dict(torch.load(CHECKPOINT_PATH, map_location='cpu'))
model.eval()



# # 4. 遍历图片并推理
image_files = [f for f in os.listdir(IMG_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]

for img_name in image_files:
    img_path = os.path.join(IMG_DIR, img_name)
    img = Image.open(img_path).convert('RGB')
    input_tensor = preprocess(img).unsqueeze(0)  # 增加batch维
    with torch.no_grad():
        output = model(input_tensor)
        pred = output.argmax(dim=1).item()
    print(f'{img_name}: class {pred}')

# 如果有类别名文件，可以在这里加载并替换输出类别编号
# 例如：
# with open('classes.txt') as f:
#     class_names = [line.strip() for line in f]
#     print(f'{img_name}: {class_names[pred]}') 