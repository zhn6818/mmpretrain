"""
通用推理类

这个模块提供了一个Inference类，用于进行多种任务的推理。
支持图像分类、图像分割等任务。
"""

import os
import torch
from torchvision import transforms
from PIL import Image
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import numpy as np
from mmengine.logging import MMLogger
from devdeploy.inference.fullmodel_loader import FullModelLoader


class Inference:
    """
    通用推理类
    
    用于加载训练好的模型并进行多种任务的推理。
    支持图像分类、图像分割等任务。
    
    Args:
        model_path (str): 模型文件路径
        device (str): 推理设备，默认为'cpu'
        class_names (Optional[List[str]]): 类别名称列表（仅分类任务需要）
    
    Example:
        >>> # 创建推理器实例
        >>> infer = Inference('charCls/works/fullmodel_best.pth', device='cuda')
        >>> 
        >>> # 单张图片推理
        >>> result = infer.infer_single('path/to/image.jpg')
        >>> print(f"预测类别: {result['class_name']}, 置信度: {result['confidence']:.2f}")
        >>> 
        >>> # 批量推理
        >>> results = infer.infer_batch('path/to/image/directory')
        >>> for img_name, result in results.items():
        >>>     print(f"{img_name}: {result['class_name']} ({result['confidence']:.2f})")
    """
    
    def __init__(self, model_path: str, device: str = 'cpu', 
                 class_names: Optional[List[str]] = None):
        """
        初始化推理器
        
        Args:
            model_path (str): 模型文件路径
            device (str): 推理设备，默认为'cpu'
            class_names (Optional[List[str]]): 类别名称列表
        """
        self.model_path = model_path
        self.device = device
        self.class_names = class_names
        self.logger = MMLogger.get_current_instance()
        
        # 初始化模型和预处理器
        self.model = None
        self.preprocess = None
        self.model_loader = None
        self.task_type = None
        
        # 默认的预处理参数（ImageNet标准）
        self.default_mean = [123.675/255, 116.28/255, 103.53/255]
        self.default_std = [58.395/255, 57.12/255, 57.375/255]
        
        # 加载模型
        self._load_model()
        self._setup_preprocessor()
    
    def _load_model(self) -> None:
        """
        加载模型并获取任务类型
        """
        try:
            self.logger.info(f'正在加载模型: {self.model_path}')
            self.model_loader = FullModelLoader(self.model_path, device=self.device)
            self.model = self.model_loader.load_model()
            self.model.eval()
            
            # 获取任务类型
            configs = self.model_loader.get_configs()
            self.task_type = configs.get('task_type', 'classification')  # 默认为分类任务
            
            self.logger.info('模型加载成功')
            self.logger.info(f'任务类型: {self.task_type}')
            self.model_loader.print_info()
            
        except Exception as e:
            self.logger.error(f'模型加载失败: {str(e)}')
            raise
    
    def _setup_preprocessor(self) -> None:
        """
        设置数据预处理器
        """
        # 尝试从模型配置中获取预处理器
        configs = self.model_loader.get_configs()
        data_preprocessor = configs.get('data_preprocessor')
        test_pipeline = configs.get('test_pipeline')
        
        if data_preprocessor is not None:
            # 使用模型自带的数据预处理器
            self.logger.info('使用模型自带的数据预处理器')
            self.preprocess = data_preprocessor
        elif test_pipeline is not None:
            # 从测试流水线构建预处理器
            self.logger.info('从测试流水线构建预处理器')
            self.preprocess = self._build_preprocessor_from_pipeline(test_pipeline)
        else:
            # 使用默认预处理器
            self.logger.info('使用默认预处理器')
            self.preprocess = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.default_mean, std=self.default_std)
            ])
    
    def _build_preprocessor_from_pipeline(self, test_pipeline: List[Dict[str, Any]]) -> transforms.Compose:
        """
        从测试流水线构建预处理器
        
        Args:
            test_pipeline (List[Dict[str, Any]]): 测试流水线配置
            
        Returns:
            transforms.Compose: 预处理器
        """
        transform_list = []
        
        for step in test_pipeline:
            if step['type'] == 'Resize':
                size = step.get('size', 256)
                transform_list.append(transforms.Resize(size))
            elif step['type'] == 'CenterCrop':
                crop_size = step.get('crop_size', 224)
                transform_list.append(transforms.CenterCrop(crop_size))
            elif step['type'] == 'ToTensor':
                transform_list.append(transforms.ToTensor())
            elif step['type'] == 'Normalize':
                mean = step.get('mean', self.default_mean)
                std = step.get('std', self.default_std)
                transform_list.append(transforms.Normalize(mean=mean, std=std))
        
        return transforms.Compose(transform_list)
    
    def _get_class_name(self, class_id: int) -> str:
        """
        获取类别名称
        
        Args:
            class_id (int): 类别ID
            
        Returns:
            str: 类别名称
        """
        if self.class_names is not None and 0 <= class_id < len(self.class_names):
            return self.class_names[class_id]
        else:
            return f"class_{class_id}"
    
    def _preprocess_image(self, image: Union[str, Image.Image]) -> torch.Tensor:
        """
        预处理图片

        Args:
            image (Union[str, Image.Image]): 图片路径或PIL Image对象
            
        Returns:
            torch.Tensor: 预处理后的图片张量
        """
        try:
            # 如果输入是路径，则打开图片
            if isinstance(image, str):
                img = Image.open(image).convert('RGB')
            elif isinstance(image, Image.Image):
                img = image.convert('RGB')
            else:
                raise ValueError(f'不支持的图片输入类型: {type(image)}，必须是字符串路径或PIL Image对象')
            
            if isinstance(self.preprocess, transforms.Compose):
                # 使用torchvision transforms
                input_tensor = self.preprocess(img).unsqueeze(0)
            else:
                # 使用模型自带的数据预处理器
                try:
                    # 尝试直接调用预处理器
                    if hasattr(self.preprocess, '__call__'):
                        # 如果预处理器是可调用的，尝试不同的调用方式
                        try:
                            # 方式1：直接传入图片
                            input_tensor = self.preprocess(img)
                        except:
                            try:
                                # 方式2：传入字典格式
                                input_tensor = self.preprocess({'inputs': img, 'data_samples': None})
                            except:
                                # 方式3：传入列表格式
                                input_tensor = self.preprocess([img])
                        
                        # 处理返回结果
                        if isinstance(input_tensor, dict):
                            if 'inputs' in input_tensor:
                                input_tensor = input_tensor['inputs']
                            else:
                                input_tensor = list(input_tensor.values())[0]
                        elif isinstance(input_tensor, (list, tuple)):
                            input_tensor = input_tensor[0]
                        
                        # 确保是tensor格式
                        if not isinstance(input_tensor, torch.Tensor):
                            input_tensor = torch.tensor(input_tensor)
                        
                        # 确保有batch维度
                        if input_tensor.dim() == 3:
                            input_tensor = input_tensor.unsqueeze(0)
                    else:
                        # 如果不是可调用的，回退到默认预处理器
                        self.logger.warning('数据预处理器不可调用，使用默认预处理器')
                        input_tensor = transforms.Compose([
                            transforms.Resize(256),
                            transforms.CenterCrop(224),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=self.default_mean, std=self.default_std)
                        ])(img).unsqueeze(0)
                        
                except Exception as e:
                    self.logger.warning(f'使用模型预处理器失败，回退到默认预处理器: {str(e)}')
                    # 回退到默认预处理器
                    input_tensor = transforms.Compose([
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=self.default_mean, std=self.default_std)
                    ])(img).unsqueeze(0)
            
            return input_tensor.to(self.device)
            
        except Exception as e:
            error_msg = f'图片预处理失败: {str(e)}'
            if isinstance(image, str):
                error_msg = f'图片预处理失败 {image}: {str(e)}'
            self.logger.error(error_msg)
            raise
    
    def _process_classification_output(self, output: Any) -> Dict[str, Any]:
        """
        处理分类任务的输出
        
        Args:
            output: 模型输出
            
        Returns:
            Dict[str, Any]: 分类结果
        """
        # 获取预测结果
        if isinstance(output, dict):
            # 如果输出是字典格式
            if 'logits' in output:
                logits = output['logits']
            elif 'pred' in output:
                logits = output['pred']
            else:
                logits = list(output.values())[0]
        else:
            logits = output
        
        # 计算概率
        probabilities = torch.softmax(logits, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
        
        # 构建结果
        result = {
            'class_id': predicted.item(),
            'class_name': self._get_class_name(predicted.item()),
            'confidence': confidence.item(),
            'probabilities': probabilities.cpu().numpy()[0]
        }
        
        return result
    
    def _process_segmentation_output(self, output: Any) -> Dict[str, Any]:
        """
        处理分割任务的输出
        
        Args:
            output: 模型输出
            
        Returns:
            Dict[str, Any]: 分割结果
        """
        # 获取分割结果
        if isinstance(output, dict):
            if 'pred_sem_seg' in output:
                seg_logits = output['pred_sem_seg']
            elif 'pred' in output:
                seg_logits = output['pred']
            else:
                seg_logits = list(output.values())[0]
        else:
            seg_logits = output
        
        # 获取预测的分割掩码
        if seg_logits.dim() == 4:  # [B, C, H, W]
            seg_pred = torch.argmax(seg_logits, dim=1)  # [B, H, W]
        else:
            seg_pred = seg_logits
        
        result = {
            'segmentation_mask': seg_pred.cpu().numpy()[0],  # [H, W]
            'num_classes': seg_logits.shape[1] if seg_logits.dim() == 4 else 1
        }
        
        return result
    
    def infer_single(self, image: Union[str, Image.Image], return_prob: bool = True) -> Dict[str, Any]:
        """
        对单张图片进行推理

        Args:
            image (Union[str, Image.Image]): 图片路径或PIL Image对象
            return_prob (bool): 是否返回概率分布（仅分类任务有效），默认为True
            
        Returns:
            Dict[str, Any]: 推理结果
        """
        try:
            # 预处理图片
            input_tensor = self._preprocess_image(image)
            
            # 进行推理
            with torch.no_grad():
                output = self.model(input_tensor)
                
                # 根据任务类型处理输出
                if self.task_type == 'classification':
                    result = self._process_classification_output(output)
                    if not return_prob:
                        result.pop('probabilities', None)
                elif self.task_type == 'segmentation':
                    result = self._process_segmentation_output(output)
                else:
                    # 默认按分类任务处理
                    self.logger.warning(f'未知任务类型 {self.task_type}，按分类任务处理')
                    result = self._process_classification_output(output)
                    if not return_prob:
                        result.pop('probabilities', None)
                
                return result
                
        except Exception as e:
            error_msg = f'推理失败: {str(e)}'
            if isinstance(image, str):
                error_msg = f'推理失败 {image}: {str(e)}'
            self.logger.error(error_msg)
            raise
    
    def infer_batch(self, image_dir: str, return_prob: bool = False) -> Dict[str, Dict[str, Any]]:
        """
        对目录中的所有图片进行批量推理
        
        Args:
            image_dir (str): 图片目录路径
            return_prob (bool): 是否返回概率分布（仅分类任务有效），默认为False
            
        Returns:
            Dict[str, Dict[str, Any]]: 批量推理结果，键为图片文件名，值为推理结果
        """
        try:
            # 获取所有图片文件
            image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')
            image_files = [
                f for f in os.listdir(image_dir) 
                if f.lower().endswith(image_extensions)
            ]
            
            if not image_files:
                self.logger.warning(f'目录中没有找到图片文件: {image_dir}')
                return {}
            
            self.logger.info(f'找到 {len(image_files)} 张图片，开始批量推理')
            
            results = {}
            for img_name in image_files:
                img_path = os.path.join(image_dir, img_name)
                try:
                    result = self.infer_single(img_path, return_prob=return_prob)
                    results[img_name] = result
                except Exception as e:
                    self.logger.error(f'处理图片失败 {img_name}: {str(e)}')
                    results[img_name] = {
                        'error': str(e),
                        'class_id': -1,
                        'class_name': 'error',
                        'confidence': 0.0
                    }
            
            self.logger.info(f'批量推理完成，成功处理 {len(results)} 张图片')
            return results
            
        except Exception as e:
            self.logger.error(f'批量推理失败: {str(e)}')
            raise
    
    def get_task_type(self) -> str:
        """
        获取任务类型
        
        Returns:
            str: 任务类型
        """
        return self.task_type
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        获取模型信息
        
        Returns:
            Dict[str, Any]: 模型信息
        """
        return self.model_loader.get_model_info()
    
    def validate_model(self) -> bool:
        """
        验证模型是否可用
        
        Returns:
            bool: 模型是否有效
        """
        return self.model_loader.validate_model()
    
    def set_class_names(self, class_names: List[str]) -> None:
        """
        设置类别名称列表
        
        Args:
            class_names (List[str]): 类别名称列表
        """
        self.class_names = class_names
        self.logger.info(f'设置类别名称，共 {len(class_names)} 个类别')
    
    def get_top_k_predictions(self, image_path: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        获取top-k预测结果（仅分类任务）
        
        Args:
            image_path (str): 图片路径
            k (int): 返回前k个预测结果，默认为5
            
        Returns:
            List[Dict[str, Any]]: top-k预测结果列表
        """
        if self.task_type != 'classification':
            raise ValueError(f'get_top_k_predictions方法仅支持分类任务，当前任务类型: {self.task_type}')
        
        try:
            # 预处理图片
            input_tensor = self._preprocess_image(image_path)
            
            # 进行推理
            with torch.no_grad():
                output = self.model(input_tensor)
                
                # 获取预测结果
                if isinstance(output, dict):
                    if 'logits' in output:
                        logits = output['logits']
                    elif 'pred' in output:
                        logits = output['pred']
                    else:
                        logits = list(output.values())[0]
                else:
                    logits = output
                
                # 计算概率
                probabilities = torch.softmax(logits, dim=1)
                
                # 获取top-k结果
                top_k_probs, top_k_indices = torch.topk(probabilities, k, dim=1)
                
                results = []
                for i in range(k):
                    class_id = top_k_indices[0][i].item()
                    confidence = top_k_probs[0][i].item()
                    results.append({
                        'rank': i + 1,
                        'class_id': class_id,
                        'class_name': self._get_class_name(class_id),
                        'confidence': confidence
                    })
                
                return results
                
        except Exception as e:
            self.logger.error(f'获取top-k预测失败 {image_path}: {str(e)}')
            raise 