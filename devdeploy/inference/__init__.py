# Copyright (c) OpenMMLab. All rights reserved.
from .load_fullmodel import (get_model_configs, get_task_type, load_fullmodel_only,
                            load_fullmodel_with_configs)

__all__ = [
    'load_fullmodel_with_configs', 'load_fullmodel_only', 'get_model_configs',
    'get_task_type'
] 