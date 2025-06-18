# Copyright (c) OpenMMLab. All rights reserved.
import os
import torch
from mmengine.hooks import Hook
from mmpretrain.registry import HOOKS

@HOOKS.register_module()
class SaveFullModelHook(Hook):
    """
    保存包含网络结构和权重的完整模型（.pth文件）。
    
    Args:
        out_dir (str, optional): 保存目录，默认使用runner的work_dir
        interval (int, optional): 保存间隔。如果设置，则每interval个epoch保存一次。
            如果为None，则只在最后一个epoch和最优模型时保存。默认为None。
        by_epoch (bool): 是否按epoch计数。默认为True。
    """
    def __init__(self, out_dir=None, interval=None, by_epoch=True):
        self.out_dir = out_dir
        self.interval = interval
        self.by_epoch = by_epoch
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
            torch.save(runner.model, save_path)
            runner.logger.info(f'Saved full model to {save_path}')