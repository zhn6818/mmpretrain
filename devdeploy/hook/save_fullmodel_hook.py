# Copyright (c) OpenMMLab. All rights reserved.
import os
import torch
from mmengine.hooks import Hook
from mmengine.registry import HOOKS

@HOOKS.register_module()
class SaveFullModelHook(Hook):
    """
    保存包含网络结构和权重的完整模型（.pth文件），同时保存data_preprocessor、test_pipeline和task_type配置。
    
    Args:
        out_dir (str, optional): 保存目录，默认使用runner的work_dir
        interval (int, optional): 保存间隔。如果设置，则每interval个epoch保存一次。
            如果为None，则只在最后一个epoch和最优模型时保存。默认为None。
        by_epoch (bool): 是否按epoch计数。默认为True。
        save_config (bool): 是否同时保存data_preprocessor、test_pipeline和task_type配置。默认为True。
    """
    def __init__(self, out_dir=None, interval=None, by_epoch=True, save_config=True):
        self.out_dir = out_dir
        self.interval = interval
        self.by_epoch = by_epoch
        self.save_config = save_config
        # 用于追踪最优模型
        self._last_best_path = None

    def after_train_epoch(self, runner):
        # 获取保存目录
        out_dir = self.out_dir or runner.work_dir
        os.makedirs(out_dir, exist_ok=True)
        
        should_save = False
        save_name = None

        # 情况1: 如果设置了interval，按interval保存
        if self.interval is not None:
            if self.by_epoch and (runner.epoch + 1) % self.interval == 0:
                should_save = True
                save_name = f'fullmodel_epoch{runner.epoch+1}.pth'
        
        # 情况2: 如果是最后一个epoch，保存
        if runner.epoch + 1 == runner.max_epochs:
            should_save = True
            save_name = 'fullmodel_final.pth'
        
        # 情况3: 如果是最优模型，保存
        best_ckpt = runner.message_hub.get_info('best_ckpt')
        if best_ckpt is not None and best_ckpt != self._last_best_path:
            should_save = True
            save_name = 'fullmodel_best.pth'
            self._last_best_path = best_ckpt

        # 执行保存
        if should_save and save_name:
            save_path = os.path.join(out_dir, save_name)
            
            # 准备保存的数据
            save_data = {
                'model': runner.model,
                'epoch': runner.epoch + 1,
                'max_epochs': runner.max_epochs,
            }
            
            # 如果启用配置保存，添加各种配置信息
            if self.save_config:
                # 获取data_preprocessor配置
                if hasattr(runner.model, 'data_preprocessor'):
                    save_data['data_preprocessor'] = runner.model.data_preprocessor
                
                # 获取test_pipeline配置（从dataloader中获取）
                if hasattr(runner, 'val_loop') and hasattr(runner.val_loop, 'dataloader'):
                    val_dataset = runner.val_loop.dataloader.dataset
                    if hasattr(val_dataset, 'pipeline'):
                        save_data['test_pipeline'] = val_dataset.pipeline
                
                # 如果val_loop不存在，尝试从test_loop获取
                elif hasattr(runner, 'test_loop') and hasattr(runner.test_loop, 'dataloader'):
                    test_dataset = runner.test_loop.dataloader.dataset
                    if hasattr(test_dataset, 'pipeline'):
                        save_data['test_pipeline'] = test_dataset.pipeline
                
                # 获取task_type配置（从runner的cfg中获取）
                if hasattr(runner, 'cfg') and hasattr(runner.cfg, 'task_type'):
                    save_data['task_type'] = runner.cfg.task_type
                elif hasattr(runner, 'cfg') and 'task_type' in runner.cfg:
                    save_data['task_type'] = runner.cfg.task_type
                else:
                    # 尝试从模型类型推断任务类型
                    model_type = type(runner.model).__name__
                    if 'Classifier' in model_type:
                        save_data['task_type'] = 'classification'
                    elif 'Segmentor' in model_type:
                        save_data['task_type'] = 'segmentation'
                    elif 'Detector' in model_type:
                        save_data['task_type'] = 'detection'
                    else:
                        save_data['task_type'] = 'unknown'
            
            # 保存完整数据
            torch.save(save_data, save_path)
            runner.logger.info(f'Saved full model with configs to {save_path}')
            
            # 记录保存的配置信息
            if self.save_config:
                config_info = []
                if 'data_preprocessor' in save_data:
                    config_info.append('data_preprocessor')
                if 'test_pipeline' in save_data:
                    config_info.append('test_pipeline')
                if 'task_type' in save_data:
                    config_info.append(f'task_type({save_data["task_type"]})')
                if config_info:
                    runner.logger.info(f'Configs saved: {", ".join(config_info)}') 