# Copyright (c) OpenMMLab. All rights reserved.
"""
完整模型加载器类

这个模块提供了一个FullModelLoader类，用于加载由SaveFullModelHook保存的完整模型，
包括模型结构、权重以及相关的配置信息（task_type、data_preprocessor、test_pipeline等）。
"""

import os
import torch
from typing import Dict, Any, Optional, Union, List
from mmengine.logging import MMLogger
# from mmengine.registry import MODELS
# from mmengine.config import Config


class FullModelLoader:
    """
    完整模型加载器类
    
    用于加载由SaveFullModelHook保存的完整模型，包括：
    - 模型结构和权重
    - 任务类型（task_type）
    - 数据预处理器（data_preprocessor）
    - 测试流水线（test_pipeline）
    - 训练信息（epoch、max_epochs等）
    
    Args:
        checkpoint_path (str): 模型文件路径
        device (str): 加载设备，默认为'cpu'
        map_location (str, optional): 指定加载设备，如果为None则使用device参数
    
    Example:
        >>> # 创建加载器实例
        >>> loader = FullModelLoader('charCls/works/fullmodel_best.pth', device='cuda')
        >>> 
        >>> # 加载模型和配置
        >>> model = loader.load_model()
        >>> configs = loader.get_configs()
        >>> 
        >>> # 获取特定配置
        >>> task_type = loader.get_task_type()
        >>> data_preprocessor = loader.get_data_preprocessor()
        >>> test_pipeline = loader.get_test_pipeline()
        >>> 
        >>> # 使用模型进行推理
        >>> model.eval()
        >>> with torch.no_grad():
        >>>     output = model(input_data)
    """
    
    def __init__(self, checkpoint_path: str, device: str = 'cpu', 
                 map_location: Optional[str] = None):
        """
        初始化模型加载器
        
        Args:
            checkpoint_path (str): 模型文件路径
            device (str): 加载设备，默认为'cpu'
            map_location (str, optional): 指定加载设备
        """
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.map_location = map_location or device
        self.logger = MMLogger.get_current_instance()
        
        # 初始化内部状态
        self._checkpoint_data = None
        self._model = None
        self._configs = None
        self._is_loaded = False
        
        # 验证文件是否存在
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"模型文件不存在: {checkpoint_path}")
    
    def _load_checkpoint(self) -> None:
        """
        加载检查点文件到内存
        
        Raises:
            Exception: 加载失败时抛出异常
        """
        if self._checkpoint_data is not None:
            return
            
        try:
            self.logger.info(f'正在加载模型文件: {self.checkpoint_path}')
            self._checkpoint_data = torch.load(self.checkpoint_path, 
                                             map_location=self.map_location)
            self.logger.info('模型文件加载成功')
        except Exception as e:
            self.logger.error(f'加载模型文件失败: {str(e)}')
            raise
    
    def _parse_checkpoint(self) -> None:
        """
        解析检查点数据，提取模型和配置信息
        """
        if self._is_loaded:
            return
            
        self._load_checkpoint()
        
        # 检查是否是新的保存格式（包含配置信息）
        if isinstance(self._checkpoint_data, dict) and 'model' in self._checkpoint_data:
            self._model = self._checkpoint_data['model']
            self._configs = {
                'epoch': self._checkpoint_data.get('epoch', 'unknown'),
                'max_epochs': self._checkpoint_data.get('max_epochs', 'unknown'),
                'data_preprocessor': self._checkpoint_data.get('data_preprocessor', None),
                'test_pipeline': self._checkpoint_data.get('test_pipeline', None),
                'task_type': self._checkpoint_data.get('task_type', 'unknown'),
            }
            self.logger.info(f'检测到完整模型格式，包含配置信息')
            self.logger.info(f'训练轮次: {self._configs["epoch"]}/{self._configs["max_epochs"]}')
            self.logger.info(f'任务类型: {self._configs["task_type"]}')
            
        else:
            # 兼容旧的保存格式（只包含模型）
            self._model = self._checkpoint_data
            self._configs = {
                'epoch': 'unknown',
                'max_epochs': 'unknown',
                'data_preprocessor': None,
                'test_pipeline': None,
                'task_type': 'unknown',
            }
            self.logger.warning('检测到旧格式模型，不包含配置信息')
        
        self._is_loaded = True
    
    def load_model(self) -> torch.nn.Module:
        """
        加载模型
        
        Returns:
            torch.nn.Module: 加载的模型
        
        Raises:
            Exception: 加载失败时抛出异常
        """
        self._parse_checkpoint()
        return self._model
    
    def get_configs(self) -> Dict[str, Any]:
        """
        获取所有配置信息
        
        Returns:
            Dict[str, Any]: 配置信息字典，包含：
                - 'epoch': 训练轮次
                - 'max_epochs': 最大训练轮次
                - 'data_preprocessor': 数据预处理器
                - 'test_pipeline': 测试流水线
                - 'task_type': 任务类型
        """
        self._parse_checkpoint()
        return self._configs.copy()
    
    def get_task_type(self) -> str:
        """
        获取任务类型
        
        Returns:
            str: 任务类型（classification/segmentation/detection/unknown）
        """
        self._parse_checkpoint()
        return self._configs['task_type']
    
    def get_data_preprocessor(self) -> Optional[Any]:
        """
        获取数据预处理器
        
        Returns:
            Any: 数据预处理器对象，如果不存在则返回None
        """
        self._parse_checkpoint()
        return self._configs['data_preprocessor']
    
    def get_test_pipeline(self) -> Optional[List[Dict[str, Any]]]:
        """
        获取测试流水线
        
        Returns:
            List[Dict[str, Any]]: 测试流水线配置列表，如果不存在则返回None
        """
        self._parse_checkpoint()
        return self._configs['test_pipeline']
    
    def get_training_info(self) -> Dict[str, Any]:
        """
        获取训练信息
        
        Returns:
            Dict[str, Any]: 训练信息字典，包含epoch和max_epochs
        """
        self._parse_checkpoint()
        return {
            'epoch': self._configs['epoch'],
            'max_epochs': self._configs['max_epochs']
        }
    
    def has_config(self, config_name: str) -> bool:
        """
        检查是否包含指定的配置
        
        Args:
            config_name (str): 配置名称
        
        Returns:
            bool: 是否包含该配置
        """
        self._parse_checkpoint()
        return config_name in self._configs and self._configs[config_name] is not None
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        获取模型信息摘要
        
        Returns:
            Dict[str, Any]: 模型信息摘要
        """
        self._parse_checkpoint()
        
        info = {
            'checkpoint_path': self.checkpoint_path,
            'device': self.device,
            'model_type': type(self._model).__name__,
            'task_type': self._configs['task_type'],
            'training_epochs': f"{self._configs['epoch']}/{self._configs['max_epochs']}",
            'has_data_preprocessor': self.has_config('data_preprocessor'),
            'has_test_pipeline': self.has_config('test_pipeline'),
        }
        
        # 添加模型参数数量信息
        if hasattr(self._model, 'parameters'):
            total_params = sum(p.numel() for p in self._model.parameters())
            trainable_params = sum(p.numel() for p in self._model.parameters() if p.requires_grad)
            info['total_parameters'] = total_params
            info['trainable_parameters'] = trainable_params
        
        return info
    
    def print_info(self) -> None:
        """
        打印模型和配置信息
        """
        info = self.get_model_info()
        
        print("=" * 60)
        print("完整模型信息")
        print("=" * 60)
        print(f"模型文件: {info['checkpoint_path']}")
        print(f"设备: {info['device']}")
        print(f"模型类型: {info['model_type']}")
        print(f"任务类型: {info['task_type']}")
        print(f"训练轮次: {info['training_epochs']}")
        print(f"数据预处理器: {'✓' if info['has_data_preprocessor'] else '✗'}")
        print(f"测试流水线: {'✓' if info['has_test_pipeline'] else '✗'}")
        
        if 'total_parameters' in info:
            print(f"总参数数量: {info['total_parameters']:,}")
            print(f"可训练参数数量: {info['trainable_parameters']:,}")
        
        print("=" * 60)
    
    def save_model_only(self, output_path: str) -> None:
        """
        只保存模型（不包含配置信息）
        
        Args:
            output_path (str): 输出文件路径
        """
        self._parse_checkpoint()
        torch.save(self._model, output_path)
        self.logger.info(f'模型已保存到: {output_path}')
    
    def save_configs_only(self, output_path: str) -> None:
        """
        只保存配置信息（不包含模型）
        
        Args:
            output_path (str): 输出文件路径
        """
        self._parse_checkpoint()
        torch.save(self._configs, output_path)
        self.logger.info(f'配置信息已保存到: {output_path}')
    
    def create_inference_pipeline(self) -> Dict[str, Any]:
        """
        创建推理流水线配置
        
        Returns:
            Dict[str, Any]: 推理流水线配置字典
        """
        self._parse_checkpoint()
        
        pipeline = {
            'model': self._model,
            'task_type': self._configs['task_type'],
            'data_preprocessor': self._configs['data_preprocessor'],
            'test_pipeline': self._configs['test_pipeline'],
        }
        
        self.logger.info(f'创建推理流水线，任务类型: {pipeline["task_type"]}')
        return pipeline
    
    def validate_model(self) -> bool:
        """
        验证模型是否可用
        
        Returns:
            bool: 模型是否有效
        """
        try:
            self._parse_checkpoint()
            
            # 检查模型是否有forward方法
            if not hasattr(self._model, 'forward'):
                self.logger.error('模型缺少forward方法')
                return False
            
            # 检查模型是否可以移动到指定设备
            try:
                test_model = self._model.to(self.device)
                # 尝试创建一个简单的测试输入
                if hasattr(self._model, 'data_preprocessor'):
                    # 如果有数据预处理器，使用它来创建测试输入
                    pass
                else:
                    # 创建默认测试输入
                    dummy_input = torch.randn(1, 3, 224, 224).to(self.device)
                    with torch.no_grad():
                        _ = test_model(dummy_input)
                
                self.logger.info('模型验证通过')
                return True
                
            except Exception as e:
                self.logger.error(f'模型验证失败: {str(e)}')
                return False
                
        except Exception as e:
            self.logger.error(f'模型验证过程中出错: {str(e)}')
            return False 