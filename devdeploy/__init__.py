# Copyright (c) OpenMMLab. All rights reserved.
"""
DevDeploy - 独立的推理和部署工具包

这个包包含了与mmpretrain独立的推理接口和自定义hook，
用于模型训练、保存和推理部署。
"""

from . import hook
from . import inference

__all__ = ['hook', 'inference'] 