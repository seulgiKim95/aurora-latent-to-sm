from omegaconf import DictConfig
config = DictConfig({})

import os
import sys

_current_dir = os.path.dirname(os.path.abspath(__file__))
_workspace_dir = os.path.dirname(os.path.dirname(os.path.dirname(_current_dir)))
if _workspace_dir not in sys.path:
    sys.path.append(_workspace_dir)

from aurora.normalisation import locations
from aurora.normalisation import scales

from common_utils import (
    load_nc_list,
    AuroraDataset,
    init_logger,
    load_aurora_pretrained_model,
    load_aurora_pretrained_small_model,
    print_trainable_params,
    get_now,
    LogTransformer,
    HyperbolicArcSineTransformer,
    parse_datetime,
)

from .main import (
    load_model,
)
from .train import (
    load_lr_scheduler,
    load_optimizer,
    load_scaler,
    aurora_collate_fn,
)
from .data import (
    AuroraDatasetLiteDecoder,
    load_data,
    dict_to_device,
    dict_to_numpy,
)
from .rollout import rollout 

import logging

init_logger()
logger = logging.getLogger('log')


__all__ = [
    "config",
    "logger",
    "locations",
    "scales",
]
