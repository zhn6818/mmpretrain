# DevDeploy Inference - 模型推理接口

这个模块提供了独立的模型加载和推理接口，与mmpretrain完全分离。

## 快速开始

### 1. 导入模块

```python
from devdeploy.inference import FullModelLoader, FullModelInference
```

### 2. 使用FullModelInference类（推荐）

`FullModelInference`是一个高级推理类，自动处理模型加载、预处理流水线构建和推理过程。

**重要：** 此推理器要求使用由`SaveFullModelHook`保存的完整模型，包含test_pipeline和data_preprocessor配置。

```python
# 创建推理器
inferencer = FullModelInference('fullmodel_best.pth', device='cpu')

# 单张图像推理
result = inferencer.predict_single('image.jpg')
print(f"预测类别: {result['pred_label']}")
print(f"置信度: {result['pred_score'].max():.3f}")

# 批量推理
results = inferencer.predict_batch(['image1.jpg', 'image2.jpg'])

# 目录推理
results = inferencer.predict_directory('image_folder/')
```

### 3. 使用FullModelLoader类

```python
# 创建模型加载器
loader = FullModelLoader('fullmodel_best.pth', device='cpu')

# 打印模型信息
loader.print_info()

# 加载模型
model = loader.load_model()

# 获取配置信息
configs = loader.get_configs()
task_type = loader.get_task_type()
data_preprocessor = loader.get_data_preprocessor()
test_pipeline = loader.get_test_pipeline()

# 验证模型
if loader.validate_model():
    print("模型验证通过")
```

## FullModelInference类详细说明

`FullModelInference`是一个完整的推理类，自动处理以下功能：

### 初始化

```python
inferencer = FullModelInference(
    checkpoint_path='fullmodel_best.pth',
    device='cpu'
)
```

**参数:**
- `checkpoint_path` (str): 模型文件路径（必须是由SaveFullModelHook保存的完整模型）
- `device` (str): 推理设备，默认为'cpu'

**要求:**
- 模型文件必须包含test_pipeline配置
- 模型文件必须包含data_preprocessor配置
- 如果缺少任一配置，将抛出ValueError异常

### 主要方法

#### 1. 单张图像推理
```python
result = inferencer.predict_single('image.jpg')
# 返回: {'image_path': 'image.jpg', 'pred_label': 0, 'pred_score': array([...]), 'task_type': 'classification'}
```

#### 2. 批量推理
```python
results = inferencer.predict_batch(['image1.jpg', 'image2.jpg'])
# 返回: [result1, result2, ...]
```

#### 3. 目录推理
```python
results = inferencer.predict_directory('image_folder/', extensions=['.jpg', '.png'])
# 返回: [result1, result2, ...]
```

### 自动预处理功能

`FullModelInference`会自动：

1. **加载test_pipeline配置**：从保存的模型中读取test_pipeline配置
2. **构建预处理流水线**：根据test_pipeline自动构建Compose预处理流水线
3. **应用data_preprocessor**：使用保存的data_preprocessor进行最终处理
4. **错误处理**：如果test_pipeline或data_preprocessor不存在，抛出明确的错误信息

### 使用示例

#### 基本推理

```python
from devdeploy.inference import FullModelInference

# 创建推理器
inferencer = FullModelInference('charCls/works/fullmodel_epoch50.pth', device='cpu')

# 单张图像推理
result = inferencer.predict_single('data/classify/val/20/test.jpg')
print(f"图像: {os.path.basename(result['image_path'])}")
print(f"预测类别: {result['pred_label']}")
print(f"置信度: {result['pred_score'].max():.3f}")
```

#### 批量推理

```python
# 批量推理
image_paths = ['image1.jpg', 'image2.jpg', 'image3.jpg']
results = inferencer.predict_batch(image_paths)

for result in results:
    if 'error' not in result:
        print(f"{os.path.basename(result['image_path'])}: 类别 {result['pred_label']}")
    else:
        print(f"{os.path.basename(result['image_path'])}: 错误 - {result['error']}")
```

#### 目录推理

```python
# 对整个目录进行推理
results = inferencer.predict_directory('data/classify/val/20/')

# 统计结果
success_count = sum(1 for r in results if 'error' not in r)
print(f"成功处理: {success_count}/{len(results)} 张图像")

# 显示前几个结果
for result in results[:5]:
    if 'error' not in result:
        print(f"{os.path.basename(result['image_path'])}: "
              f"类别 {result['pred_label']} (置信度: {result['pred_score'].max():.3f})")
```

#### 自定义扩展名

```python
# 只处理特定格式的图像
results = inferencer.predict_directory(
    'image_folder/', 
    extensions=['.jpg', '.jpeg', '.png']
)
```

### 错误处理

`FullModelInference`具有完善的错误处理机制：

```python
try:
    inferencer = FullModelInference('model.pth', device='cpu')
    result = inferencer.predict_single('image.jpg')
    print(f"推理成功: 类别 {result['pred_label']}")
except ValueError as e:
    print(f"配置错误: {e}")
except FileNotFoundError as e:
    print(f"文件不存在: {e}")
except Exception as e:
    print(f"其他错误: {e}")
```

### 常见错误及解决方案

#### 1. 缺少test_pipeline配置
```
ValueError: 模型文件中未找到test_pipeline配置，无法进行推理。请确保模型是由SaveFullModelHook保存的完整模型。
```
**解决方案：** 确保使用SaveFullModelHook保存模型，或检查模型文件是否完整。

#### 2. 缺少data_preprocessor配置
```
ValueError: 模型文件中未找到data_preprocessor配置，无法进行推理。请确保模型是由SaveFullModelHook保存的完整模型。
```
**解决方案：** 确保使用SaveFullModelHook保存模型，或检查模型文件是否完整。

#### 3. 预处理流水线构建失败
```
ValueError: 构建预处理流水线失败: ...
```
**解决方案：** 检查test_pipeline配置是否正确，确保所有变换都可用。

## FullModelLoader类详细说明

`FullModelLoader`是一个功能完整的模型加载器类，提供了丰富的功能来加载和管理由`SaveFullModelHook`保存的完整模型。

### 初始化

```python
loader = FullModelLoader(
    checkpoint_path='fullmodel_best.pth',
    device='cpu',
    map_location=None
)
```

**参数:**
- `checkpoint_path` (str): 模型文件路径
- `device` (str): 加载设备，默认为'cpu'
- `map_location` (str, optional): 指定加载设备，如果为None则使用device参数

### 主要方法

#### 1. 模型加载
- `load_model()`: 加载模型
- `validate_model()`: 验证模型是否可用

#### 2. 配置访问
- `get_configs()`: 获取所有配置信息
- `get_task_type()`: 获取任务类型
- `get_data_preprocessor()`: 获取数据预处理器
- `get_test_pipeline()`: 获取测试流水线
- `get_training_info()`: 获取训练信息
- `has_config(config_name)`: 检查是否包含指定配置

#### 3. 信息显示
- `print_info()`: 打印模型和配置信息
- `get_model_info()`: 获取模型信息摘要

#### 4. 保存操作
- `save_model_only(output_path)`: 只保存模型
- `save_configs_only(output_path)`: 只保存配置信息

#### 5. 推理流水线
- `create_inference_pipeline()`: 创建推理流水线配置

### 使用示例

#### 基本使用

```python
from devdeploy.inference import FullModelLoader

# 创建加载器
loader = FullModelLoader('charCls/works/fullmodel_best.pth', device='cpu')

# 打印信息
loader.print_info()

# 加载模型
model = loader.load_model()
model.eval()

# 获取配置
task_type = loader.get_task_type()
print(f"任务类型: {task_type}")
```

#### 高级使用

```python
# 获取模型信息摘要
info = loader.get_model_info()
print(f"模型类型: {info['model_type']}")
print(f"参数数量: {info['total_parameters']:,}")

# 检查配置是否存在
if loader.has_config('data_preprocessor'):
    data_preprocessor = loader.get_data_preprocessor()
    print(f"数据预处理器: {type(data_preprocessor).__name__}")

# 创建推理流水线
pipeline = loader.create_inference_pipeline()
print(f"推理流水线: {pipeline}")
```

#### 推理示例

```python
# 根据任务类型准备输入
task_type = loader.get_task_type()

if task_type == 'classification':
    input_data = torch.randn(1, 3, 224, 224)
elif task_type == 'segmentation':
    input_data = torch.randn(1, 3, 512, 512)
elif task_type == 'detection':
    input_data = torch.randn(1, 3, 640, 640)
else:
    input_data = torch.randn(1, 3, 224, 224)

# 推理
with torch.no_grad():
    output = model(input_data)
    print(f"输出形状: {output.shape}")
```

#### 保存操作

```python
# 只保存模型
loader.save_model_only('model_only.pth')

# 只保存配置
loader.save_configs_only('configs_only.pth')
```

## 使用示例

### 分类任务推理

```python
from devdeploy.inference import FullModelLoader
import torch

# 创建加载器
loader = FullModelLoader('classification_model.pth', device='cpu')

# 加载模型
model = loader.load_model()
model.eval()

# 准备输入数据
input_data = torch.randn(1, 3, 224, 224)

# 推理
with torch.no_grad():
    output = model(input_data)
    probabilities = torch.softmax(output, dim=1)
    predicted_class = torch.argmax(probabilities, dim=1)
    
print(f"预测类别: {predicted_class.item()}")
```

### 根据任务类型进行不同处理

```python
from devdeploy.inference import FullModelLoader
import torch

loader = FullModelLoader('model.pth', device='cpu')
task_type = loader.get_task_type()

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

### 批处理推理

```python
from devdeploy.inference import FullModelLoader
import torch

loader = FullModelLoader('model.pth', device='cpu')
model = loader.load_model()
model.eval()

# 准备批处理输入
batch_size = 4
task_type = loader.get_task_type()

if task_type == 'classification':
    batch_input = torch.randn(batch_size, 3, 224, 224)
else:
    batch_input = torch.randn(batch_size, 3, 224, 224)

# 批处理推理
with torch.no_grad():
    batch_output = model(batch_input)
    print(f"批处理输出形状: {batch_output.shape}")
```

## 错误处理

```python
from devdeploy.inference import FullModelLoader

try:
    loader = FullModelLoader('model.pth', device='cpu')
    
    # 验证模型
    if loader.validate_model():
        print("模型验证通过")
        model = loader.load_model()
    else:
        print("模型验证失败")
        
except FileNotFoundError:
    print("模型文件不存在")
except Exception as e:
    print(f"加载模型时出错: {str(e)}")
```

## 测试

运行测试脚本来验证功能：

```bash
python devdeploy/inference/test_fullmodel_loader.py
```

## 注意事项

1. 确保模型文件是由`SaveFullModelHook`保存的完整模型
2. 加载模型时需要确保Python环境中有相同的依赖包
3. 跨环境使用时注意PyTorch版本的兼容性
4. 如果模型文件不包含配置信息，会返回默认值
5. 使用`validate_model()`方法可以验证模型是否可用
6. 对于大型模型，可以使用`get_configs()`只获取配置信息以节省内存 