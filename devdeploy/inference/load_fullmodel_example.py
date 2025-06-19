# Copyright (c) OpenMMLab. All rights reserved.
"""
使用示例：如何加载包含配置信息的完整模型

这个示例展示了如何使用SaveFullModelHook保存的完整模型，
以及如何使用load_fullmodel_with_configs函数加载模型和配置信息。
"""

import torch
from devdeploy.inference import (load_fullmodel_with_configs, load_fullmodel_only, 
                                get_model_configs, get_task_type)

def example_load_fullmodel():
    """示例：加载完整模型和配置信息"""
    
    # 假设你已经训练并保存了完整模型
    checkpoint_path = 'work_dir/fullmodel_best.pth'
    
    print("=== 加载完整模型和配置信息 ===")
    try:
        # 加载模型和配置
        model, configs = load_fullmodel_with_configs(checkpoint_path, device='cpu')
        
        print(f"模型加载成功！")
        print(f"任务类型: {configs['task_type']}")
        print(f"训练轮次: {configs['epoch']}/{configs['max_epochs']}")
        
        # 检查配置信息
        if configs['data_preprocessor'] is not None:
            print(f"数据预处理器: {type(configs['data_preprocessor']).__name__}")
        else:
            print("未找到数据预处理器配置")
            
        if configs['test_pipeline'] is not None:
            print(f"测试流水线: {len(configs['test_pipeline'])} 个步骤")
        else:
            print("未找到测试流水线配置")
            
        # 根据任务类型进行不同的处理
        if configs['task_type'] == 'classification':
            print("这是一个分类模型")
            # 这里可以添加分类任务特定的处理逻辑
        elif configs['task_type'] == 'segmentation':
            print("这是一个分割模型")
            # 这里可以添加分割任务特定的处理逻辑
        elif configs['task_type'] == 'detection':
            print("这是一个检测模型")
            # 这里可以添加检测任务特定的处理逻辑
        else:
            print(f"未知任务类型: {configs['task_type']}")
            
        # 使用模型进行推理
        model.eval()
        with torch.no_grad():
            # 这里可以添加你的推理代码
            print("模型已准备就绪，可以进行推理")
            
    except FileNotFoundError:
        print(f"错误：找不到模型文件 {checkpoint_path}")
        print("请先运行训练脚本生成完整模型文件")
    except Exception as e:
        print(f"加载模型时出错: {str(e)}")

def example_load_model_only():
    """示例：只加载模型，不关心配置信息"""
    
    checkpoint_path = 'work_dir/fullmodel_best.pth'
    
    print("\n=== 只加载模型 ===")
    try:
        model = load_fullmodel_only(checkpoint_path, device='cpu')
        print("模型加载成功！")
        print(f"模型类型: {type(model).__name__}")
        
    except Exception as e:
        print(f"加载模型时出错: {str(e)}")

def example_get_configs_only():
    """示例：只获取配置信息，不加载模型"""
    
    checkpoint_path = 'work_dir/fullmodel_best.pth'
    
    print("\n=== 只获取配置信息 ===")
    try:
        configs = get_model_configs(checkpoint_path, device='cpu')
        
        print("配置信息获取成功！")
        print(f"任务类型: {configs['task_type']}")
        print(f"训练轮次: {configs['epoch']}/{configs['max_epochs']}")
        
        if configs['data_preprocessor'] is not None:
            print("包含数据预处理器配置")
        if configs['test_pipeline'] is not None:
            print("包含测试流水线配置")
            
    except Exception as e:
        print(f"获取配置信息时出错: {str(e)}")

def example_get_task_type_only():
    """示例：只获取任务类型信息"""
    
    checkpoint_path = 'work_dir/fullmodel_best.pth'
    
    print("\n=== 只获取任务类型 ===")
    try:
        task_type = get_task_type(checkpoint_path, device='cpu')
        print(f"任务类型: {task_type}")
        
        # 根据任务类型进行不同的处理
        if task_type == 'classification':
            print("这是一个分类任务模型")
        elif task_type == 'segmentation':
            print("这是一个分割任务模型")
        elif task_type == 'detection':
            print("这是一个检测任务模型")
        else:
            print("未知任务类型")
            
    except Exception as e:
        print(f"获取任务类型时出错: {str(e)}")

def example_use_configs_for_inference():
    """示例：使用配置信息进行推理"""
    
    checkpoint_path = 'work_dir/fullmodel_best.pth'
    
    print("\n=== 使用配置信息进行推理 ===")
    try:
        model, configs = load_fullmodel_with_configs(checkpoint_path, device='cpu')
        model.eval()
        
        # 根据任务类型准备不同的输入数据
        task_type = configs['task_type']
        
        if task_type == 'classification':
            # 分类任务：准备图像输入
            dummy_input = torch.randn(1, 3, 224, 224)
            print("准备分类任务输入数据...")
        elif task_type == 'segmentation':
            # 分割任务：准备图像输入
            dummy_input = torch.randn(1, 3, 512, 512)
            print("准备分割任务输入数据...")
        elif task_type == 'detection':
            # 检测任务：准备图像输入
            dummy_input = torch.randn(1, 3, 640, 640)
            print("准备检测任务输入数据...")
        else:
            # 默认输入
            dummy_input = torch.randn(1, 3, 224, 224)
            print("准备默认输入数据...")
        
        # 如果有数据预处理器，使用它处理输入
        if configs['data_preprocessor'] is not None:
            print("使用数据预处理器处理输入...")
            # 这里可以添加数据预处理代码
            # processed_input = configs['data_preprocessor'](dummy_input)
        
        # 进行推理
        with torch.no_grad():
            output = model(dummy_input)
            print(f"推理输出形状: {output.shape}")
            
            # 根据任务类型处理输出
            if task_type == 'classification':
                print("分类任务：输出为类别概率")
            elif task_type == 'segmentation':
                print("分割任务：输出为分割掩码")
            elif task_type == 'detection':
                print("检测任务：输出为检测框和类别")
            
    except Exception as e:
        print(f"推理时出错: {str(e)}")

if __name__ == '__main__':
    print("完整模型加载示例")
    print("=" * 50)
    
    # 运行所有示例
    example_load_fullmodel()
    example_load_model_only()
    example_get_configs_only()
    example_get_task_type_only()
    example_use_configs_for_inference()
    
    print("\n" + "=" * 50)
    print("示例完成！")
    print("\n使用说明：")
    print("1. 确保已经运行训练脚本并生成了完整模型文件")
    print("2. 修改checkpoint_path为你的实际模型文件路径")
    print("3. 根据需要选择合适的加载函数")
    print("4. 根据task_type进行任务特定的处理") 