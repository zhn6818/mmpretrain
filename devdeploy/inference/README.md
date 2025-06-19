# DevDeploy Inference - 模型推理接口

这个模块提供了独立的模型加载和推理接口，与mmpretrain完全分离。

## 快速开始

### 1. 导入模块

```python
from devdeploy.inference import (
    load_fullmodel_with_configs,
    load_fullmodel_only,
    get_model_configs,
    get_task_type
)
```

### 2. 加载完整模型

```python
# 加载模型和所有配置信息
model, configs = load_fullmodel_with_configs('fullmodel_best.pth')

# 查看配置信息
print(f"任务类型: {configs['task_type']}")
print(f"训练轮次: {configs['epoch']}/{configs['max_epochs']}")
print(f"数据预处理器: {configs['data_preprocessor']}")
print(f"测试流水线: {configs['test_pipeline']}")
```

### 3. 只加载模型

```python
# 只加载模型，不关心配置信息
model = load_fullmodel_only('fullmodel_best.pth')
model.eval()
```

### 4. 只获取配置信息

```python
# 只获取配置信息，不加载模型（节省内存）
configs = get_model_configs('fullmodel_best.pth')
print(f"任务类型: {configs['task_type']}")
```

### 5. 只获取任务类型

```python
# 快速获取任务类型
task_type = get_task_type('fullmodel_best.pth')
print(f"任务类型: {task_type}")
```

## 函数详细说明

### load_fullmodel_with_configs()

加载包含网络结构、权重和配置信息的完整模型。

**参数:**
- `checkpoint_path` (str): 模型文件路径
- `device` (str): 加载设备，默认为'cpu'
- `map_location` (str, optional): 指定加载设备

**返回:**
- `Tuple[torch.nn.Module, Dict[str, Any]]`: 模型和配置信息字典

**配置信息包含:**
- `data_preprocessor`: 数据预处理器
- `test_pipeline`: 测试数据流水线
- `task_type`: 任务类型（classification/segmentation/detection等）
- `epoch`: 训练轮次
- `max_epochs`: 最大训练轮次

### load_fullmodel_only()

只加载模型，不返回配置信息。

**参数:**
- `checkpoint_path` (str): 模型文件路径
- `device` (str): 加载设备，默认为'cpu'
- `map_location` (str, optional): 指定加载设备

**返回:**
- `torch.nn.Module`: 加载的模型

### get_model_configs()

只获取配置信息，不加载模型。

**参数:**
- `checkpoint_path` (str): 模型文件路径
- `device` (str): 加载设备，默认为'cpu'
- `map_location` (str, optional): 指定加载设备

**返回:**
- `Dict[str, Any]`: 配置信息字典

### get_task_type()

只获取任务类型信息。

**参数:**
- `checkpoint_path` (str): 模型文件路径
- `device` (str): 加载设备，默认为'cpu'
- `map_location` (str, optional): 指定加载设备

**返回:**
- `str`: 任务类型（classification/segmentation/detection/unknown）

## 使用示例

### 分类任务推理

```python
from devdeploy.inference import load_fullmodel_with_configs
import torch

# 加载模型
model, configs = load_fullmodel_with_configs('classification_model.pth')

# 准备输入数据
input_data = torch.randn(1, 3, 224, 224)

# 推理
model.eval()
with torch.no_grad():
    output = model(input_data)
    probabilities = torch.softmax(output, dim=1)
    predicted_class = torch.argmax(probabilities, dim=1)
    
print(f"预测类别: {predicted_class.item()}")
```

### 根据任务类型进行不同处理

```python
from devdeploy.inference import load_fullmodel_with_configs
import torch

model, configs = load_fullmodel_with_configs('model.pth')
task_type = configs['task_type']

if task_type == 'classification':
    # 分类任务处理
    input_size = (224, 224)
    print("处理分类任务")
elif task_type == 'segmentation':
    # 分割任务处理
    input_size = (512, 512)
    print("处理分割任务")
elif task_type == 'detection':
    # 检测任务处理
    input_size = (640, 640)
    print("处理检测任务")
else:
    print(f"未知任务类型: {task_type}")
```

## 错误处理

```python
from devdeploy.inference import load_fullmodel_with_configs

try:
    model, configs = load_fullmodel_with_configs('model.pth')
    print("模型加载成功")
except FileNotFoundError:
    print("模型文件不存在")
except Exception as e:
    print(f"加载模型时出错: {str(e)}")
```

## 注意事项

1. 确保模型文件是由`SaveFullModelHook`保存的完整模型
2. 加载模型时需要确保Python环境中有相同的依赖包
3. 跨环境使用时注意PyTorch版本的兼容性
4. 如果模型文件不包含配置信息，会返回默认值 