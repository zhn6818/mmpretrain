# Copyright (c) OpenMMLab. All rights reserved.
import torch
from typing import Dict, Any, Optional, Tuple
from mmengine.logging import MMLogger

def load_fullmodel_with_configs(
    checkpoint_path: str,
    device: str = 'cpu',
    map_location: Optional[str] = None
) -> Tuple[torch.nn.Module, Dict[str, Any]]:
    """
    加载包含网络结构、权重和配置信息的完整模型。
    
    Args:
        checkpoint_path (str): 模型文件路径
        device (str): 加载设备，默认为'cpu'
        map_location (str, optional): 指定加载设备，如果为None则使用device参数
    
    Returns:
        Tuple[torch.nn.Module, Dict[str, Any]]: 
            - 加载的模型
            - 包含配置信息的字典，包括：
                - 'data_preprocessor': 数据预处理器
                - 'test_pipeline': 测试数据流水线
                - 'task_type': 任务类型（classification/segmentation/detection等）
                - 'epoch': 训练轮次
                - 'max_epochs': 最大训练轮次
    
    Example:
        >>> model, configs = load_fullmodel_with_configs('fullmodel_best.pth')
        >>> print(f"Model trained for {configs['epoch']} epochs")
        >>> print(f"Task type: {configs.get('task_type')}")
        >>> print(f"Data preprocessor: {configs.get('data_preprocessor')}")
        >>> print(f"Test pipeline: {configs.get('test_pipeline')}")
    """
    logger = MMLogger.get_current_instance()
    
    # 设置加载设备
    if map_location is None:
        map_location = device
    
    try:
        # 加载保存的数据
        checkpoint = torch.load(checkpoint_path, map_location=map_location)
        
        # 检查是否是新的保存格式（包含配置信息）
        if isinstance(checkpoint, dict) and 'model' in checkpoint:
            model = checkpoint['model']
            configs = {
                'epoch': checkpoint.get('epoch', 'unknown'),
                'max_epochs': checkpoint.get('max_epochs', 'unknown'),
                'data_preprocessor': checkpoint.get('data_preprocessor', None),
                'test_pipeline': checkpoint.get('test_pipeline', None),
                'task_type': checkpoint.get('task_type', 'unknown'),
            }
            logger.info(f'Loaded full model with configs from {checkpoint_path}')
            logger.info(f'Model was trained for {configs["epoch"]}/{configs["max_epochs"]} epochs')
            logger.info(f'Task type: {configs["task_type"]}')
            
            # 记录配置信息
            if configs['data_preprocessor'] is not None:
                logger.info('Data preprocessor found in checkpoint')
            if configs['test_pipeline'] is not None:
                logger.info('Test pipeline found in checkpoint')
            if configs['task_type'] != 'unknown':
                logger.info(f'Task type: {configs["task_type"]}')
                
        else:
            # 兼容旧的保存格式（只包含模型）
            model = checkpoint
            configs = {
                'epoch': 'unknown',
                'max_epochs': 'unknown',
                'data_preprocessor': None,
                'test_pipeline': None,
                'task_type': 'unknown',
            }
            logger.info(f'Loaded model only (no configs) from {checkpoint_path}')
            logger.warning('This checkpoint does not contain configuration information')
        
        return model, configs
        
    except Exception as e:
        logger.error(f'Failed to load model from {checkpoint_path}: {str(e)}')
        raise

def load_fullmodel_only(
    checkpoint_path: str,
    device: str = 'cpu',
    map_location: Optional[str] = None
) -> torch.nn.Module:
    """
    只加载模型，不返回配置信息。
    
    Args:
        checkpoint_path (str): 模型文件路径
        device (str): 加载设备，默认为'cpu'
        map_location (str, optional): 指定加载设备
    
    Returns:
        torch.nn.Module: 加载的模型
    """
    model, _ = load_fullmodel_with_configs(checkpoint_path, device, map_location)
    return model

def get_model_configs(
    checkpoint_path: str,
    device: str = 'cpu',
    map_location: Optional[str] = None
) -> Dict[str, Any]:
    """
    只获取配置信息，不加载模型。
    
    Args:
        checkpoint_path (str): 模型文件路径
        device (str): 加载设备，默认为'cpu'
        map_location (str, optional): 指定加载设备
    
    Returns:
        Dict[str, Any]: 配置信息字典
    """
    _, configs = load_fullmodel_with_configs(checkpoint_path, device, map_location)
    return configs

def get_task_type(
    checkpoint_path: str,
    device: str = 'cpu',
    map_location: Optional[str] = None
) -> str:
    """
    只获取任务类型信息。
    
    Args:
        checkpoint_path (str): 模型文件路径
        device (str): 加载设备，默认为'cpu'
        map_location (str, optional): 指定加载设备
    
    Returns:
        str: 任务类型（classification/segmentation/detection/unknown）
    """
    configs = get_model_configs(checkpoint_path, device, map_location)
    return configs.get('task_type', 'unknown') 