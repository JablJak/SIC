import importlib
import os
from copy import deepcopy
from typing import Any, TypeAlias, Union, Iterable

import sophia.sophia
import torch
import yaml
from sophia import SophiaG
from torch import nn
from torchvision import transforms
from torch.nn import Module, init
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from src.data.base_dataset import BaseDataset
from src.losses.rdloss import RDLoss
from src.utils.const import MODEL_OUTPUT_PATH

# Based on pyTorch implementation
ParamsT: TypeAlias = Union[
    Iterable[torch.Tensor], Iterable[dict[str, Any]], Iterable[tuple[str, torch.Tensor]]
]
key_mapping_dict = {
    'encoder.0.0.weight': 'encoder.patch_embed.0.weight',
    'encoder.0.0.bias': 'encoder.patch_embed.0.bias',
    'encoder.0.2.weight': 'encoder.patch_embed.2.weight',
    'encoder.0.2.bias': 'encoder.patch_embed.2.bias',    
    'encoder.1.0.norm1.weight': 'encoder.stages.0.0.norm1.weight',
    'encoder.1.0.norm1.bias': 'encoder.stages.0.0.norm1.bias',
    'encoder.1.0.attn.logit_scale': 'encoder.stages.0.0.attn.logit_scale',
    'encoder.1.0.attn.relative_coords_table': 'encoder.stages.0.0.attn.relative_coords_table',
    'encoder.1.0.attn.relative_position_index': 'encoder.stages.0.0.attn.relative_position_index',
    'encoder.1.0.attn.qkv.weight': 'encoder.stages.0.0.attn.qkv.weight',
    'encoder.1.0.attn.qkv.bias': 'encoder.stages.0.0.attn.qkv.bias',
    'encoder.1.0.attn.proj.weight': 'encoder.stages.0.0.attn.proj.weight',
    'encoder.1.0.attn.proj.bias': 'encoder.stages.0.0.attn.proj.bias',
    'encoder.1.0.attn.cpb_mlp.0.weight': 'encoder.stages.0.0.attn.cpb_mlp.0.weight',
    'encoder.1.0.attn.cpb_mlp.0.bias': 'encoder.stages.0.0.attn.cpb_mlp.0.bias',
    'encoder.1.0.attn.cpb_mlp.2.weight': 'encoder.stages.0.0.attn.cpb_mlp.2.weight',
    'encoder.1.0.norm2.weight': 'encoder.stages.0.0.norm2.weight',
    'encoder.1.0.norm2.bias': 'encoder.stages.0.0.norm2.bias',
    'encoder.1.0.mlp.0.weight': 'encoder.stages.0.0.mlp.0.weight',
    'encoder.1.0.mlp.0.bias': 'encoder.stages.0.0.mlp.0.bias',
    'encoder.1.0.mlp.3.weight': 'encoder.stages.0.0.mlp.3.weight',
    'encoder.1.0.mlp.3.bias': 'encoder.stages.0.0.mlp.3.bias',
    'encoder.1.1.norm1.weight': 'encoder.stages.0.1.norm1.weight',
    'encoder.1.1.norm1.bias': 'encoder.stages.0.1.norm1.bias',
    'encoder.1.1.attn.logit_scale': 'encoder.stages.0.1.attn.logit_scale',
    'encoder.1.1.attn.relative_coords_table': 'encoder.stages.0.1.attn.relative_coords_table',
    'encoder.1.1.attn.relative_position_index': 'encoder.stages.0.1.attn.relative_position_index',
    'encoder.1.1.attn.qkv.weight': 'encoder.stages.0.1.attn.qkv.weight',
    'encoder.1.1.attn.qkv.bias': 'encoder.stages.0.1.attn.qkv.bias',
    'encoder.1.1.attn.proj.weight': 'encoder.stages.0.1.attn.proj.weight',
    'encoder.1.1.attn.proj.bias': 'encoder.stages.0.1.attn.proj.bias',
    'encoder.1.1.attn.cpb_mlp.0.weight': 'encoder.stages.0.1.attn.cpb_mlp.0.weight',
    'encoder.1.1.attn.cpb_mlp.0.bias': 'encoder.stages.0.1.attn.cpb_mlp.0.bias',
    'encoder.1.1.attn.cpb_mlp.2.weight': 'encoder.stages.0.1.attn.cpb_mlp.2.weight',
    'encoder.1.1.norm2.weight': 'encoder.stages.0.1.norm2.weight',
    'encoder.1.1.norm2.bias': 'encoder.stages.0.1.norm2.bias',
    'encoder.1.1.mlp.0.weight': 'encoder.stages.0.1.mlp.0.weight',
    'encoder.1.1.mlp.0.bias': 'encoder.stages.0.1.mlp.0.bias',
    'encoder.1.1.mlp.3.weight': 'encoder.stages.0.1.mlp.3.weight',
    'encoder.1.1.mlp.3.bias': 'encoder.stages.0.1.mlp.3.bias',
    'encoder.2.reduction.weight': 'encoder.downsamplers.0.reduction.weight',
    'encoder.2.norm.weight': 'encoder.downsamplers.0.norm.weight',
    'encoder.2.norm.bias': 'encoder.downsamplers.0.norm.bias',
    'encoder.3.0.norm1.weight': 'encoder.stages.1.0.norm1.weight',
    'encoder.3.0.norm1.bias': 'encoder.stages.1.0.norm1.bias',
    'encoder.3.0.attn.logit_scale': 'encoder.stages.1.0.attn.logit_scale',
    'encoder.3.0.attn.relative_coords_table': 'encoder.stages.1.0.attn.relative_coords_table',
    'encoder.3.0.attn.relative_position_index': 'encoder.stages.1.0.attn.relative_position_index',
    'encoder.3.0.attn.qkv.weight': 'encoder.stages.1.0.attn.qkv.weight',
    'encoder.3.0.attn.qkv.bias': 'encoder.stages.1.0.attn.qkv.bias',
    'encoder.3.0.attn.proj.weight': 'encoder.stages.1.0.attn.proj.weight',
    'encoder.3.0.attn.proj.bias': 'encoder.stages.1.0.attn.proj.bias',
    'encoder.3.0.attn.cpb_mlp.0.weight': 'encoder.stages.1.0.attn.cpb_mlp.0.weight',
    'encoder.3.0.attn.cpb_mlp.0.bias': 'encoder.stages.1.0.attn.cpb_mlp.0.bias',
    'encoder.3.0.attn.cpb_mlp.2.weight': 'encoder.stages.1.0.attn.cpb_mlp.2.weight',
    'encoder.3.0.norm2.weight': 'encoder.stages.1.0.norm2.weight',
    'encoder.3.0.norm2.bias': 'encoder.stages.1.0.norm2.bias',
    'encoder.3.0.mlp.0.weight': 'encoder.stages.1.0.mlp.0.weight',
    'encoder.3.0.mlp.0.bias': 'encoder.stages.1.0.mlp.0.bias',
    'encoder.3.0.mlp.3.weight': 'encoder.stages.1.0.mlp.3.weight',
    'encoder.3.0.mlp.3.bias': 'encoder.stages.1.0.mlp.3.bias',
    'encoder.3.1.norm1.weight': 'encoder.stages.1.1.norm1.weight',
    'encoder.3.1.norm1.bias': 'encoder.stages.1.1.norm1.bias',
    'encoder.3.1.attn.logit_scale': 'encoder.stages.1.1.attn.logit_scale',
    'encoder.3.1.attn.relative_coords_table': 'encoder.stages.1.1.attn.relative_coords_table',
    'encoder.3.1.attn.relative_position_index': 'encoder.stages.1.1.attn.relative_position_index',
    'encoder.3.1.attn.qkv.weight': 'encoder.stages.1.1.attn.qkv.weight',
    'encoder.3.1.attn.qkv.bias': 'encoder.stages.1.1.attn.qkv.bias',
    'encoder.3.1.attn.proj.weight': 'encoder.stages.1.1.attn.proj.weight',
    'encoder.3.1.attn.proj.bias': 'encoder.stages.1.1.attn.proj.bias',
    'encoder.3.1.attn.cpb_mlp.0.weight': 'encoder.stages.1.1.attn.cpb_mlp.0.weight',
    'encoder.3.1.attn.cpb_mlp.0.bias': 'encoder.stages.1.1.attn.cpb_mlp.0.bias',
    'encoder.3.1.attn.cpb_mlp.2.weight': 'encoder.stages.1.1.attn.cpb_mlp.2.weight',
    'encoder.3.1.norm2.weight': 'encoder.stages.1.1.norm2.weight',
    'encoder.3.1.norm2.bias': 'encoder.stages.1.1.norm2.bias',
    'encoder.3.1.mlp.0.weight': 'encoder.stages.1.1.mlp.0.weight',
    'encoder.3.1.mlp.0.bias': 'encoder.stages.1.1.mlp.0.bias',
    'encoder.3.1.mlp.3.weight': 'encoder.stages.1.1.mlp.3.weight',
    'encoder.3.1.mlp.3.bias': 'encoder.stages.1.1.mlp.3.bias',
    'encoder.4.reduction.weight': 'encoder.downsamplers.1.reduction.weight',
    'encoder.4.norm.weight': 'encoder.downsamplers.1.norm.weight',
    'encoder.4.norm.bias': 'encoder.downsamplers.1.norm.bias',
    'encoder.5.0.norm1.weight': 'encoder.stages.2.0.norm1.weight',
    'encoder.5.0.norm1.bias': 'encoder.stages.2.0.norm1.bias',
    'encoder.5.0.attn.logit_scale': 'encoder.stages.2.0.attn.logit_scale',
    'encoder.5.0.attn.relative_coords_table': 'encoder.stages.2.0.attn.relative_coords_table',
    'encoder.5.0.attn.relative_position_index': 'encoder.stages.2.0.attn.relative_position_index',
    'encoder.5.0.attn.qkv.weight': 'encoder.stages.2.0.attn.qkv.weight',
    'encoder.5.0.attn.qkv.bias': 'encoder.stages.2.0.attn.qkv.bias',
    'encoder.5.0.attn.proj.weight': 'encoder.stages.2.0.attn.proj.weight',
    'encoder.5.0.attn.proj.bias': 'encoder.stages.2.0.attn.proj.bias',
    'encoder.5.0.attn.cpb_mlp.0.weight': 'encoder.stages.2.0.attn.cpb_mlp.0.weight',
    'encoder.5.0.attn.cpb_mlp.0.bias': 'encoder.stages.2.0.attn.cpb_mlp.0.bias',
    'encoder.5.0.attn.cpb_mlp.2.weight': 'encoder.stages.2.0.attn.cpb_mlp.2.weight',
    'encoder.5.0.norm2.weight': 'encoder.stages.2.0.norm2.weight',
    'encoder.5.0.norm2.bias': 'encoder.stages.2.0.norm2.bias',
    'encoder.5.0.mlp.0.weight': 'encoder.stages.2.0.mlp.0.weight',
    'encoder.5.0.mlp.0.bias': 'encoder.stages.2.0.mlp.0.bias',
    'encoder.5.0.mlp.3.weight': 'encoder.stages.2.0.mlp.3.weight',
    'encoder.5.0.mlp.3.bias': 'encoder.stages.2.0.mlp.3.bias',
    'encoder.5.1.norm1.weight': 'encoder.stages.2.1.norm1.weight',
    'encoder.5.1.norm1.bias': 'encoder.stages.2.1.norm1.bias',
    'encoder.5.1.attn.logit_scale': 'encoder.stages.2.1.attn.logit_scale',
    'encoder.5.1.attn.relative_coords_table': 'encoder.stages.2.1.attn.relative_coords_table',
    'encoder.5.1.attn.relative_position_index': 'encoder.stages.2.1.attn.relative_position_index',
    'encoder.5.1.attn.qkv.weight': 'encoder.stages.2.1.attn.qkv.weight',
    'encoder.5.1.attn.qkv.bias': 'encoder.stages.2.1.attn.qkv.bias',
    'encoder.5.1.attn.proj.weight': 'encoder.stages.2.1.attn.proj.weight',
    'encoder.5.1.attn.proj.bias': 'encoder.stages.2.1.attn.proj.bias',
    'encoder.5.1.attn.cpb_mlp.0.weight': 'encoder.stages.2.1.attn.cpb_mlp.0.weight',
    'encoder.5.1.attn.cpb_mlp.0.bias': 'encoder.stages.2.1.attn.cpb_mlp.0.bias',
    'encoder.5.1.attn.cpb_mlp.2.weight': 'encoder.stages.2.1.attn.cpb_mlp.2.weight',
    'encoder.5.1.norm2.weight': 'encoder.stages.2.1.norm2.weight',
    'encoder.5.1.norm2.bias': 'encoder.stages.2.1.norm2.bias',
    'encoder.5.1.mlp.0.weight': 'encoder.stages.2.1.mlp.0.weight',
    'encoder.5.1.mlp.0.bias': 'encoder.stages.2.1.mlp.0.bias',
    'encoder.5.1.mlp.3.weight': 'encoder.stages.2.1.mlp.3.weight',
    'encoder.5.1.mlp.3.bias': 'encoder.stages.2.1.mlp.3.bias',
    'encoder.5.2.norm1.weight': 'encoder.stages.2.2.norm1.weight',
    'encoder.5.2.norm1.bias': 'encoder.stages.2.2.norm1.bias',
    'encoder.5.2.attn.logit_scale': 'encoder.stages.2.2.attn.logit_scale',
    'encoder.5.2.attn.relative_coords_table': 'encoder.stages.2.2.attn.relative_coords_table',
    'encoder.5.2.attn.relative_position_index': 'encoder.stages.2.2.attn.relative_position_index',
    'encoder.5.2.attn.qkv.weight': 'encoder.stages.2.2.attn.qkv.weight',
    'encoder.5.2.attn.qkv.bias': 'encoder.stages.2.2.attn.qkv.bias',
    'encoder.5.2.attn.proj.weight': 'encoder.stages.2.2.attn.proj.weight',
    'encoder.5.2.attn.proj.bias': 'encoder.stages.2.2.attn.proj.bias',
    'encoder.5.2.attn.cpb_mlp.0.weight': 'encoder.stages.2.2.attn.cpb_mlp.0.weight',
    'encoder.5.2.attn.cpb_mlp.0.bias': 'encoder.stages.2.2.attn.cpb_mlp.0.bias',
    'encoder.5.2.attn.cpb_mlp.2.weight': 'encoder.stages.2.2.attn.cpb_mlp.2.weight',
    'encoder.5.2.norm2.weight': 'encoder.stages.2.2.norm2.weight',
    'encoder.5.2.norm2.bias': 'encoder.stages.2.2.norm2.bias',
    'encoder.5.2.mlp.0.weight': 'encoder.stages.2.2.mlp.0.weight',
    'encoder.5.2.mlp.0.bias': 'encoder.stages.2.2.mlp.0.bias',
    'encoder.5.2.mlp.3.weight': 'encoder.stages.2.2.mlp.3.weight',
    'encoder.5.2.mlp.3.bias': 'encoder.stages.2.2.mlp.3.bias',
    'encoder.5.3.norm1.weight': 'encoder.stages.2.3.norm1.weight',
    'encoder.5.3.norm1.bias': 'encoder.stages.2.3.norm1.bias',
    'encoder.5.3.attn.logit_scale': 'encoder.stages.2.3.attn.logit_scale',
    'encoder.5.3.attn.relative_coords_table': 'encoder.stages.2.3.attn.relative_coords_table',
    'encoder.5.3.attn.relative_position_index': 'encoder.stages.2.3.attn.relative_position_index',
    'encoder.5.3.attn.qkv.weight': 'encoder.stages.2.3.attn.qkv.weight',
    'encoder.5.3.attn.qkv.bias': 'encoder.stages.2.3.attn.qkv.bias',
    'encoder.5.3.attn.proj.weight': 'encoder.stages.2.3.attn.proj.weight',
    'encoder.5.3.attn.proj.bias': 'encoder.stages.2.3.attn.proj.bias',
    'encoder.5.3.attn.cpb_mlp.0.weight': 'encoder.stages.2.3.attn.cpb_mlp.0.weight',
    'encoder.5.3.attn.cpb_mlp.0.bias': 'encoder.stages.2.3.attn.cpb_mlp.0.bias',
    'encoder.5.3.attn.cpb_mlp.2.weight': 'encoder.stages.2.3.attn.cpb_mlp.2.weight',
    'encoder.5.3.norm2.weight': 'encoder.stages.2.3.norm2.weight',
    'encoder.5.3.norm2.bias': 'encoder.stages.2.3.norm2.bias',
    'encoder.5.3.mlp.0.weight': 'encoder.stages.2.3.mlp.0.weight',
    'encoder.5.3.mlp.0.bias': 'encoder.stages.2.3.mlp.0.bias',
    'encoder.5.3.mlp.3.weight': 'encoder.stages.2.3.mlp.3.weight',
    'encoder.5.3.mlp.3.bias': 'encoder.stages.2.3.mlp.3.bias',
    'encoder.5.4.norm1.weight': 'encoder.stages.2.4.norm1.weight',
    'encoder.5.4.norm1.bias': 'encoder.stages.2.4.norm1.bias',
    'encoder.5.4.attn.logit_scale': 'encoder.stages.2.4.attn.logit_scale',
    'encoder.5.4.attn.relative_coords_table': 'encoder.stages.2.4.attn.relative_coords_table',
    'encoder.5.4.attn.relative_position_index': 'encoder.stages.2.4.attn.relative_position_index',
    'encoder.5.4.attn.qkv.weight': 'encoder.stages.2.4.attn.qkv.weight',
    'encoder.5.4.attn.qkv.bias': 'encoder.stages.2.4.attn.qkv.bias',
    'encoder.5.4.attn.proj.weight': 'encoder.stages.2.4.attn.proj.weight',
    'encoder.5.4.attn.proj.bias': 'encoder.stages.2.4.attn.proj.bias',
    'encoder.5.4.attn.cpb_mlp.0.weight': 'encoder.stages.2.4.attn.cpb_mlp.0.weight',
    'encoder.5.4.attn.cpb_mlp.0.bias': 'encoder.stages.2.4.attn.cpb_mlp.0.bias',
    'encoder.5.4.attn.cpb_mlp.2.weight': 'encoder.stages.2.4.attn.cpb_mlp.2.weight',
    'encoder.5.4.norm2.weight': 'encoder.stages.2.4.norm2.weight',
    'encoder.5.4.norm2.bias': 'encoder.stages.2.4.norm2.bias',
    'encoder.5.4.mlp.0.weight': 'encoder.stages.2.4.mlp.0.weight',
    'encoder.5.4.mlp.0.bias': 'encoder.stages.2.4.mlp.0.bias',
    'encoder.5.4.mlp.3.weight': 'encoder.stages.2.4.mlp.3.weight',
    'encoder.5.4.mlp.3.bias': 'encoder.stages.2.4.mlp.3.bias',
    'encoder.5.5.norm1.weight': 'encoder.stages.2.5.norm1.weight',
    'encoder.5.5.norm1.bias': 'encoder.stages.2.5.norm1.bias',
    'encoder.5.5.attn.logit_scale': 'encoder.stages.2.5.attn.logit_scale',
    'encoder.5.5.attn.relative_coords_table': 'encoder.stages.2.5.attn.relative_coords_table',
    'encoder.5.5.attn.relative_position_index': 'encoder.stages.2.5.attn.relative_position_index',
    'encoder.5.5.attn.qkv.weight': 'encoder.stages.2.5.attn.qkv.weight',
    'encoder.5.5.attn.qkv.bias': 'encoder.stages.2.5.attn.qkv.bias',
    'encoder.5.5.attn.proj.weight': 'encoder.stages.2.5.attn.proj.weight',
    'encoder.5.5.attn.proj.bias': 'encoder.stages.2.5.attn.proj.bias',
    'encoder.5.5.attn.cpb_mlp.0.weight': 'encoder.stages.2.5.attn.cpb_mlp.0.weight',
    'encoder.5.5.attn.cpb_mlp.0.bias': 'encoder.stages.2.5.attn.cpb_mlp.0.bias',
    'encoder.5.5.attn.cpb_mlp.2.weight': 'encoder.stages.2.5.attn.cpb_mlp.2.weight',
    'encoder.5.5.norm2.weight': 'encoder.stages.2.5.norm2.weight',
    'encoder.5.5.norm2.bias': 'encoder.stages.2.5.norm2.bias',
    'encoder.5.5.mlp.0.weight': 'encoder.stages.2.5.mlp.0.weight',
    'encoder.5.5.mlp.0.bias': 'encoder.stages.2.5.mlp.0.bias',
    'encoder.5.5.mlp.3.weight': 'encoder.stages.2.5.mlp.3.weight',
    'encoder.5.5.mlp.3.bias': 'encoder.stages.2.5.mlp.3.bias',
    'encoder.5.6.norm1.weight': 'encoder.stages.2.6.norm1.weight',
    'encoder.5.6.norm1.bias': 'encoder.stages.2.6.norm1.bias',
    'encoder.5.6.attn.logit_scale': 'encoder.stages.2.6.attn.logit_scale',
    'encoder.5.6.attn.relative_coords_table': 'encoder.stages.2.6.attn.relative_coords_table',
    'encoder.5.6.attn.relative_position_index': 'encoder.stages.2.6.attn.relative_position_index',
    'encoder.5.6.attn.qkv.weight': 'encoder.stages.2.6.attn.qkv.weight',
    'encoder.5.6.attn.qkv.bias': 'encoder.stages.2.6.attn.qkv.bias',
    'encoder.5.6.attn.proj.weight': 'encoder.stages.2.6.attn.proj.weight',
    'encoder.5.6.attn.proj.bias': 'encoder.stages.2.6.attn.proj.bias',
    'encoder.5.6.attn.cpb_mlp.0.weight': 'encoder.stages.2.6.attn.cpb_mlp.0.weight',
    'encoder.5.6.attn.cpb_mlp.0.bias': 'encoder.stages.2.6.attn.cpb_mlp.0.bias',
    'encoder.5.6.attn.cpb_mlp.2.weight': 'encoder.stages.2.6.attn.cpb_mlp.2.weight',
    'encoder.5.6.norm2.weight': 'encoder.stages.2.6.norm2.weight',
    'encoder.5.6.norm2.bias': 'encoder.stages.2.6.norm2.bias',
    'encoder.5.6.mlp.0.weight': 'encoder.stages.2.6.mlp.0.weight',
    'encoder.5.6.mlp.0.bias': 'encoder.stages.2.6.mlp.0.bias',
    'encoder.5.6.mlp.3.weight': 'encoder.stages.2.6.mlp.3.weight',
    'encoder.5.6.mlp.3.bias': 'encoder.stages.2.6.mlp.3.bias',
    'encoder.5.7.norm1.weight': 'encoder.stages.2.7.norm1.weight',
    'encoder.5.7.norm1.bias': 'encoder.stages.2.7.norm1.bias',
    'encoder.5.7.attn.logit_scale': 'encoder.stages.2.7.attn.logit_scale',
    'encoder.5.7.attn.relative_coords_table': 'encoder.stages.2.7.attn.relative_coords_table',
    'encoder.5.7.attn.relative_position_index': 'encoder.stages.2.7.attn.relative_position_index',
    'encoder.5.7.attn.qkv.weight': 'encoder.stages.2.7.attn.qkv.weight',
    'encoder.5.7.attn.qkv.bias': 'encoder.stages.2.7.attn.qkv.bias',
    'encoder.5.7.attn.proj.weight': 'encoder.stages.2.7.attn.proj.weight',
    'encoder.5.7.attn.proj.bias': 'encoder.stages.2.7.attn.proj.bias',
    'encoder.5.7.attn.cpb_mlp.0.weight': 'encoder.stages.2.7.attn.cpb_mlp.0.weight',
    'encoder.5.7.attn.cpb_mlp.0.bias': 'encoder.stages.2.7.attn.cpb_mlp.0.bias',
    'encoder.5.7.attn.cpb_mlp.2.weight': 'encoder.stages.2.7.attn.cpb_mlp.2.weight',
    'encoder.5.7.norm2.weight': 'encoder.stages.2.7.norm2.weight',
    'encoder.5.7.norm2.bias': 'encoder.stages.2.7.norm2.bias',
    'encoder.5.7.mlp.0.weight': 'encoder.stages.2.7.mlp.0.weight',
    'encoder.5.7.mlp.0.bias': 'encoder.stages.2.7.mlp.0.bias',
    'encoder.5.7.mlp.3.weight': 'encoder.stages.2.7.mlp.3.weight',
    'encoder.5.7.mlp.3.bias': 'encoder.stages.2.7.mlp.3.bias',
    'encoder.5.8.norm1.weight': 'encoder.stages.2.8.norm1.weight',
    'encoder.5.8.norm1.bias': 'encoder.stages.2.8.norm1.bias',
    'encoder.5.8.attn.logit_scale': 'encoder.stages.2.8.attn.logit_scale',
    'encoder.5.8.attn.relative_coords_table': 'encoder.stages.2.8.attn.relative_coords_table',
    'encoder.5.8.attn.relative_position_index': 'encoder.stages.2.8.attn.relative_position_index',
    'encoder.5.8.attn.qkv.weight': 'encoder.stages.2.8.attn.qkv.weight',
    'encoder.5.8.attn.qkv.bias': 'encoder.stages.2.8.attn.qkv.bias',
    'encoder.5.8.attn.proj.weight': 'encoder.stages.2.8.attn.proj.weight',
    'encoder.5.8.attn.proj.bias': 'encoder.stages.2.8.attn.proj.bias',
    'encoder.5.8.attn.cpb_mlp.0.weight': 'encoder.stages.2.8.attn.cpb_mlp.0.weight',
    'encoder.5.8.attn.cpb_mlp.0.bias': 'encoder.stages.2.8.attn.cpb_mlp.0.bias',
    'encoder.5.8.attn.cpb_mlp.2.weight': 'encoder.stages.2.8.attn.cpb_mlp.2.weight',
    'encoder.5.8.norm2.weight': 'encoder.stages.2.8.norm2.weight',
    'encoder.5.8.norm2.bias': 'encoder.stages.2.8.norm2.bias',
    'encoder.5.8.mlp.0.weight': 'encoder.stages.2.8.mlp.0.weight',
    'encoder.5.8.mlp.0.bias': 'encoder.stages.2.8.mlp.0.bias',
    'encoder.5.8.mlp.3.weight': 'encoder.stages.2.8.mlp.3.weight',
    'encoder.5.8.mlp.3.bias': 'encoder.stages.2.8.mlp.3.bias',
    'encoder.5.9.norm1.weight': 'encoder.stages.2.9.norm1.weight',
    'encoder.5.9.norm1.bias': 'encoder.stages.2.9.norm1.bias',
    'encoder.5.9.attn.logit_scale': 'encoder.stages.2.9.attn.logit_scale',
    'encoder.5.9.attn.relative_coords_table': 'encoder.stages.2.9.attn.relative_coords_table',
    'encoder.5.9.attn.relative_position_index': 'encoder.stages.2.9.attn.relative_position_index',
    'encoder.5.9.attn.qkv.weight': 'encoder.stages.2.9.attn.qkv.weight',
    'encoder.5.9.attn.qkv.bias': 'encoder.stages.2.9.attn.qkv.bias',
    'encoder.5.9.attn.proj.weight': 'encoder.stages.2.9.attn.proj.weight',
    'encoder.5.9.attn.proj.bias': 'encoder.stages.2.9.attn.proj.bias',
    'encoder.5.9.attn.cpb_mlp.0.weight': 'encoder.stages.2.9.attn.cpb_mlp.0.weight',
    'encoder.5.9.attn.cpb_mlp.0.bias': 'encoder.stages.2.9.attn.cpb_mlp.0.bias',
    'encoder.5.9.attn.cpb_mlp.2.weight': 'encoder.stages.2.9.attn.cpb_mlp.2.weight',
    'encoder.5.9.norm2.weight': 'encoder.stages.2.9.norm2.weight',
    'encoder.5.9.norm2.bias': 'encoder.stages.2.9.norm2.bias',
    'encoder.5.9.mlp.0.weight': 'encoder.stages.2.9.mlp.0.weight',
    'encoder.5.9.mlp.0.bias': 'encoder.stages.2.9.mlp.0.bias',
    'encoder.5.9.mlp.3.weight': 'encoder.stages.2.9.mlp.3.weight',
    'encoder.5.9.mlp.3.bias': 'encoder.stages.2.9.mlp.3.bias',
    'encoder.5.10.norm1.weight': 'encoder.stages.2.10.norm1.weight',
    'encoder.5.10.norm1.bias': 'encoder.stages.2.10.norm1.bias',
    'encoder.5.10.attn.logit_scale': 'encoder.stages.2.10.attn.logit_scale',
    'encoder.5.10.attn.relative_coords_table': 'encoder.stages.2.10.attn.relative_coords_table',
    'encoder.5.10.attn.relative_position_index': 'encoder.stages.2.10.attn.relative_position_index',
    'encoder.5.10.attn.qkv.weight': 'encoder.stages.2.10.attn.qkv.weight',
    'encoder.5.10.attn.qkv.bias': 'encoder.stages.2.10.attn.qkv.bias',
    'encoder.5.10.attn.proj.weight': 'encoder.stages.2.10.attn.proj.weight',
    'encoder.5.10.attn.proj.bias': 'encoder.stages.2.10.attn.proj.bias',
    'encoder.5.10.attn.cpb_mlp.0.weight': 'encoder.stages.2.10.attn.cpb_mlp.0.weight',
    'encoder.5.10.attn.cpb_mlp.0.bias': 'encoder.stages.2.10.attn.cpb_mlp.0.bias',
    'encoder.5.10.attn.cpb_mlp.2.weight': 'encoder.stages.2.10.attn.cpb_mlp.2.weight',
    'encoder.5.10.norm2.weight': 'encoder.stages.2.10.norm2.weight',
    'encoder.5.10.norm2.bias': 'encoder.stages.2.10.norm2.bias',
    'encoder.5.10.mlp.0.weight': 'encoder.stages.2.10.mlp.0.weight',
    'encoder.5.10.mlp.0.bias': 'encoder.stages.2.10.mlp.0.bias',
    'encoder.5.10.mlp.3.weight': 'encoder.stages.2.10.mlp.3.weight',
    'encoder.5.10.mlp.3.bias': 'encoder.stages.2.10.mlp.3.bias',
    'encoder.5.11.norm1.weight': 'encoder.stages.2.11.norm1.weight',
    'encoder.5.11.norm1.bias': 'encoder.stages.2.11.norm1.bias',
    'encoder.5.11.attn.logit_scale': 'encoder.stages.2.11.attn.logit_scale',
    'encoder.5.11.attn.relative_coords_table': 'encoder.stages.2.11.attn.relative_coords_table',
    'encoder.5.11.attn.relative_position_index': 'encoder.stages.2.11.attn.relative_position_index',
    'encoder.5.11.attn.qkv.weight': 'encoder.stages.2.11.attn.qkv.weight',
    'encoder.5.11.attn.qkv.bias': 'encoder.stages.2.11.attn.qkv.bias',
    'encoder.5.11.attn.proj.weight': 'encoder.stages.2.11.attn.proj.weight',
    'encoder.5.11.attn.proj.bias': 'encoder.stages.2.11.attn.proj.bias',
    'encoder.5.11.attn.cpb_mlp.0.weight': 'encoder.stages.2.11.attn.cpb_mlp.0.weight',
    'encoder.5.11.attn.cpb_mlp.0.bias': 'encoder.stages.2.11.attn.cpb_mlp.0.bias',
    'encoder.5.11.attn.cpb_mlp.2.weight': 'encoder.stages.2.11.attn.cpb_mlp.2.weight',
    'encoder.5.11.norm2.weight': 'encoder.stages.2.11.norm2.weight',
    'encoder.5.11.norm2.bias': 'encoder.stages.2.11.norm2.bias',
    'encoder.5.11.mlp.0.weight': 'encoder.stages.2.11.mlp.0.weight',
    'encoder.5.11.mlp.0.bias': 'encoder.stages.2.11.mlp.0.bias',
    'encoder.5.11.mlp.3.weight': 'encoder.stages.2.11.mlp.3.weight',
    'encoder.5.11.mlp.3.bias': 'encoder.stages.2.11.mlp.3.bias',
    'encoder.5.12.norm1.weight': 'encoder.stages.2.12.norm1.weight',
    'encoder.5.12.norm1.bias': 'encoder.stages.2.12.norm1.bias',
    'encoder.5.12.attn.logit_scale': 'encoder.stages.2.12.attn.logit_scale',
    'encoder.5.12.attn.relative_coords_table': 'encoder.stages.2.12.attn.relative_coords_table',
    'encoder.5.12.attn.relative_position_index': 'encoder.stages.2.12.attn.relative_position_index',
    'encoder.5.12.attn.qkv.weight': 'encoder.stages.2.12.attn.qkv.weight',
    'encoder.5.12.attn.qkv.bias': 'encoder.stages.2.12.attn.qkv.bias',
    'encoder.5.12.attn.proj.weight': 'encoder.stages.2.12.attn.proj.weight',
    'encoder.5.12.attn.proj.bias': 'encoder.stages.2.12.attn.proj.bias',
    'encoder.5.12.attn.cpb_mlp.0.weight': 'encoder.stages.2.12.attn.cpb_mlp.0.weight',
    'encoder.5.12.attn.cpb_mlp.0.bias': 'encoder.stages.2.12.attn.cpb_mlp.0.bias',
    'encoder.5.12.attn.cpb_mlp.2.weight': 'encoder.stages.2.12.attn.cpb_mlp.2.weight',
    'encoder.5.12.norm2.weight': 'encoder.stages.2.12.norm2.weight',
    'encoder.5.12.norm2.bias': 'encoder.stages.2.12.norm2.bias',
    'encoder.5.12.mlp.0.weight': 'encoder.stages.2.12.mlp.0.weight',
    'encoder.5.12.mlp.0.bias': 'encoder.stages.2.12.mlp.0.bias',
    'encoder.5.12.mlp.3.weight': 'encoder.stages.2.12.mlp.3.weight',
    'encoder.5.12.mlp.3.bias': 'encoder.stages.2.12.mlp.3.bias',
    'encoder.5.13.norm1.weight': 'encoder.stages.2.13.norm1.weight',
    'encoder.5.13.norm1.bias': 'encoder.stages.2.13.norm1.bias',
    'encoder.5.13.attn.logit_scale': 'encoder.stages.2.13.attn.logit_scale',
    'encoder.5.13.attn.relative_coords_table': 'encoder.stages.2.13.attn.relative_coords_table',
    'encoder.5.13.attn.relative_position_index': 'encoder.stages.2.13.attn.relative_position_index',
    'encoder.5.13.attn.qkv.weight': 'encoder.stages.2.13.attn.qkv.weight',
    'encoder.5.13.attn.qkv.bias': 'encoder.stages.2.13.attn.qkv.bias',
    'encoder.5.13.attn.proj.weight': 'encoder.stages.2.13.attn.proj.weight',
    'encoder.5.13.attn.proj.bias': 'encoder.stages.2.13.attn.proj.bias',
    'encoder.5.13.attn.cpb_mlp.0.weight': 'encoder.stages.2.13.attn.cpb_mlp.0.weight',
    'encoder.5.13.attn.cpb_mlp.0.bias': 'encoder.stages.2.13.attn.cpb_mlp.0.bias',
    'encoder.5.13.attn.cpb_mlp.2.weight': 'encoder.stages.2.13.attn.cpb_mlp.2.weight',
    'encoder.5.13.norm2.weight': 'encoder.stages.2.13.norm2.weight',
    'encoder.5.13.norm2.bias': 'encoder.stages.2.13.norm2.bias',
    'encoder.5.13.mlp.0.weight': 'encoder.stages.2.13.mlp.0.weight',
    'encoder.5.13.mlp.0.bias': 'encoder.stages.2.13.mlp.0.bias',
    'encoder.5.13.mlp.3.weight': 'encoder.stages.2.13.mlp.3.weight',
    'encoder.5.13.mlp.3.bias': 'encoder.stages.2.13.mlp.3.bias',
    'encoder.5.14.norm1.weight': 'encoder.stages.2.14.norm1.weight',
    'encoder.5.14.norm1.bias': 'encoder.stages.2.14.norm1.bias',
    'encoder.5.14.attn.logit_scale': 'encoder.stages.2.14.attn.logit_scale',
    'encoder.5.14.attn.relative_coords_table': 'encoder.stages.2.14.attn.relative_coords_table',
    'encoder.5.14.attn.relative_position_index': 'encoder.stages.2.14.attn.relative_position_index',
    'encoder.5.14.attn.qkv.weight': 'encoder.stages.2.14.attn.qkv.weight',
    'encoder.5.14.attn.qkv.bias': 'encoder.stages.2.14.attn.qkv.bias',
    'encoder.5.14.attn.proj.weight': 'encoder.stages.2.14.attn.proj.weight',
    'encoder.5.14.attn.proj.bias': 'encoder.stages.2.14.attn.proj.bias',
    'encoder.5.14.attn.cpb_mlp.0.weight': 'encoder.stages.2.14.attn.cpb_mlp.0.weight',
    'encoder.5.14.attn.cpb_mlp.0.bias': 'encoder.stages.2.14.attn.cpb_mlp.0.bias',
    'encoder.5.14.attn.cpb_mlp.2.weight': 'encoder.stages.2.14.attn.cpb_mlp.2.weight',
    'encoder.5.14.norm2.weight': 'encoder.stages.2.14.norm2.weight',
    'encoder.5.14.norm2.bias': 'encoder.stages.2.14.norm2.bias',
    'encoder.5.14.mlp.0.weight': 'encoder.stages.2.14.mlp.0.weight',
    'encoder.5.14.mlp.0.bias': 'encoder.stages.2.14.mlp.0.bias',
    'encoder.5.14.mlp.3.weight': 'encoder.stages.2.14.mlp.3.weight',
    'encoder.5.14.mlp.3.bias': 'encoder.stages.2.14.mlp.3.bias',
    'encoder.5.15.norm1.weight': 'encoder.stages.2.15.norm1.weight',
    'encoder.5.15.norm1.bias': 'encoder.stages.2.15.norm1.bias',
    'encoder.5.15.attn.logit_scale': 'encoder.stages.2.15.attn.logit_scale',
    'encoder.5.15.attn.relative_coords_table': 'encoder.stages.2.15.attn.relative_coords_table',
    'encoder.5.15.attn.relative_position_index': 'encoder.stages.2.15.attn.relative_position_index',
    'encoder.5.15.attn.qkv.weight': 'encoder.stages.2.15.attn.qkv.weight',
    'encoder.5.15.attn.qkv.bias': 'encoder.stages.2.15.attn.qkv.bias',
    'encoder.5.15.attn.proj.weight': 'encoder.stages.2.15.attn.proj.weight',
    'encoder.5.15.attn.proj.bias': 'encoder.stages.2.15.attn.proj.bias',
    'encoder.5.15.attn.cpb_mlp.0.weight': 'encoder.stages.2.15.attn.cpb_mlp.0.weight',
    'encoder.5.15.attn.cpb_mlp.0.bias': 'encoder.stages.2.15.attn.cpb_mlp.0.bias',
    'encoder.5.15.attn.cpb_mlp.2.weight': 'encoder.stages.2.15.attn.cpb_mlp.2.weight',
    'encoder.5.15.norm2.weight': 'encoder.stages.2.15.norm2.weight',
    'encoder.5.15.norm2.bias': 'encoder.stages.2.15.norm2.bias',
    'encoder.5.15.mlp.0.weight': 'encoder.stages.2.15.mlp.0.weight',
    'encoder.5.15.mlp.0.bias': 'encoder.stages.2.15.mlp.0.bias',
    'encoder.5.15.mlp.3.weight': 'encoder.stages.2.15.mlp.3.weight',
    'encoder.5.15.mlp.3.bias': 'encoder.stages.2.15.mlp.3.bias',
    'encoder.5.16.norm1.weight': 'encoder.stages.2.16.norm1.weight',
    'encoder.5.16.norm1.bias': 'encoder.stages.2.16.norm1.bias',
    'encoder.5.16.attn.logit_scale': 'encoder.stages.2.16.attn.logit_scale',
    'encoder.5.16.attn.relative_coords_table': 'encoder.stages.2.16.attn.relative_coords_table',
    'encoder.5.16.attn.relative_position_index': 'encoder.stages.2.16.attn.relative_position_index',
    'encoder.5.16.attn.qkv.weight': 'encoder.stages.2.16.attn.qkv.weight',
    'encoder.5.16.attn.qkv.bias': 'encoder.stages.2.16.attn.qkv.bias',
    'encoder.5.16.attn.proj.weight': 'encoder.stages.2.16.attn.proj.weight',
    'encoder.5.16.attn.proj.bias': 'encoder.stages.2.16.attn.proj.bias',
    'encoder.5.16.attn.cpb_mlp.0.weight': 'encoder.stages.2.16.attn.cpb_mlp.0.weight',
    'encoder.5.16.attn.cpb_mlp.0.bias': 'encoder.stages.2.16.attn.cpb_mlp.0.bias',
    'encoder.5.16.attn.cpb_mlp.2.weight': 'encoder.stages.2.16.attn.cpb_mlp.2.weight',
    'encoder.5.16.norm2.weight': 'encoder.stages.2.16.norm2.weight',
    'encoder.5.16.norm2.bias': 'encoder.stages.2.16.norm2.bias',
    'encoder.5.16.mlp.0.weight': 'encoder.stages.2.16.mlp.0.weight',
    'encoder.5.16.mlp.0.bias': 'encoder.stages.2.16.mlp.0.bias',
    'encoder.5.16.mlp.3.weight': 'encoder.stages.2.16.mlp.3.weight',
    'encoder.5.16.mlp.3.bias': 'encoder.stages.2.16.mlp.3.bias',
    'encoder.5.17.norm1.weight': 'encoder.stages.2.17.norm1.weight',
    'encoder.5.17.norm1.bias': 'encoder.stages.2.17.norm1.bias',
    'encoder.5.17.attn.logit_scale': 'encoder.stages.2.17.attn.logit_scale',
    'encoder.5.17.attn.relative_coords_table': 'encoder.stages.2.17.attn.relative_coords_table',
    'encoder.5.17.attn.relative_position_index': 'encoder.stages.2.17.attn.relative_position_index',
    'encoder.5.17.attn.qkv.weight': 'encoder.stages.2.17.attn.qkv.weight',
    'encoder.5.17.attn.qkv.bias': 'encoder.stages.2.17.attn.qkv.bias',
    'encoder.5.17.attn.proj.weight': 'encoder.stages.2.17.attn.proj.weight',
    'encoder.5.17.attn.proj.bias': 'encoder.stages.2.17.attn.proj.bias',
    'encoder.5.17.attn.cpb_mlp.0.weight': 'encoder.stages.2.17.attn.cpb_mlp.0.weight',
    'encoder.5.17.attn.cpb_mlp.0.bias': 'encoder.stages.2.17.attn.cpb_mlp.0.bias',
    'encoder.5.17.attn.cpb_mlp.2.weight': 'encoder.stages.2.17.attn.cpb_mlp.2.weight',
    'encoder.5.17.norm2.weight': 'encoder.stages.2.17.norm2.weight',
    'encoder.5.17.norm2.bias': 'encoder.stages.2.17.norm2.bias',
    'encoder.5.17.mlp.0.weight': 'encoder.stages.2.17.mlp.0.weight',
    'encoder.5.17.mlp.0.bias': 'encoder.stages.2.17.mlp.0.bias',
    'encoder.5.17.mlp.3.weight': 'encoder.stages.2.17.mlp.3.weight',
    'encoder.5.17.mlp.3.bias': 'encoder.stages.2.17.mlp.3.bias',
    'encoder.6.reduction.weight': 'encoder.downsamplers.2.reduction.weight',
    'encoder.6.norm.weight': 'encoder.downsamplers.2.norm.weight',
    'encoder.6.norm.bias': 'encoder.downsamplers.2.norm.bias',
    'encoder.7.0.norm1.weight': 'encoder.stages.3.0.norm1.weight',
    'encoder.7.0.norm1.bias': 'encoder.stages.3.0.norm1.bias',
    'encoder.7.0.attn.logit_scale': 'encoder.stages.3.0.attn.logit_scale',
    'encoder.7.0.attn.relative_coords_table': 'encoder.stages.3.0.attn.relative_coords_table',
    'encoder.7.0.attn.relative_position_index': 'encoder.stages.3.0.attn.relative_position_index',
    'encoder.7.0.attn.qkv.weight': 'encoder.stages.3.0.attn.qkv.weight',
    'encoder.7.0.attn.qkv.bias': 'encoder.stages.3.0.attn.qkv.bias',
    'encoder.7.0.attn.proj.weight': 'encoder.stages.3.0.attn.proj.weight',
    'encoder.7.0.attn.proj.bias': 'encoder.stages.3.0.attn.proj.bias',
    'encoder.7.0.attn.cpb_mlp.0.weight': 'encoder.stages.3.0.attn.cpb_mlp.0.weight',
    'encoder.7.0.attn.cpb_mlp.0.bias': 'encoder.stages.3.0.attn.cpb_mlp.0.bias',
    'encoder.7.0.attn.cpb_mlp.2.weight': 'encoder.stages.3.0.attn.cpb_mlp.2.weight',
    'encoder.7.0.norm2.weight': 'encoder.stages.3.0.norm2.weight',
    'encoder.7.0.norm2.bias': 'encoder.stages.3.0.norm2.bias',
    'encoder.7.0.mlp.0.weight': 'encoder.stages.3.0.mlp.0.weight',
    'encoder.7.0.mlp.0.bias': 'encoder.stages.3.0.mlp.0.bias',
    'encoder.7.0.mlp.3.weight': 'encoder.stages.3.0.mlp.3.weight',
    'encoder.7.0.mlp.3.bias': 'encoder.stages.3.0.mlp.3.bias',
    'encoder.7.1.norm1.weight': 'encoder.stages.3.1.norm1.weight',
    'encoder.7.1.norm1.bias': 'encoder.stages.3.1.norm1.bias',
    'encoder.7.1.attn.logit_scale': 'encoder.stages.3.1.attn.logit_scale',
    'encoder.7.1.attn.relative_coords_table': 'encoder.stages.3.1.attn.relative_coords_table',
    'encoder.7.1.attn.relative_position_index': 'encoder.stages.3.1.attn.relative_position_index',
    'encoder.7.1.attn.qkv.weight': 'encoder.stages.3.1.attn.qkv.weight',
    'encoder.7.1.attn.qkv.bias': 'encoder.stages.3.1.attn.qkv.bias',
    'encoder.7.1.attn.proj.weight': 'encoder.stages.3.1.attn.proj.weight',
    'encoder.7.1.attn.proj.bias': 'encoder.stages.3.1.attn.proj.bias',
    'encoder.7.1.attn.cpb_mlp.0.weight': 'encoder.stages.3.1.attn.cpb_mlp.0.weight',
    'encoder.7.1.attn.cpb_mlp.0.bias': 'encoder.stages.3.1.attn.cpb_mlp.0.bias',
    'encoder.7.1.attn.cpb_mlp.2.weight': 'encoder.stages.3.1.attn.cpb_mlp.2.weight',
    'encoder.7.1.norm2.weight': 'encoder.stages.3.1.norm2.weight',
    'encoder.7.1.norm2.bias': 'encoder.stages.3.1.norm2.bias',
    'encoder.7.1.mlp.0.weight': 'encoder.stages.3.1.mlp.0.weight',
    'encoder.7.1.mlp.0.bias': 'encoder.stages.3.1.mlp.0.bias',
    'encoder.7.1.mlp.3.weight': 'encoder.stages.3.1.mlp.3.weight',
    'encoder.7.1.mlp.3.bias': 'encoder.stages.3.1.mlp.3.bias',

    'g_a.0.0.weight': 'g_a.patch_embed.0.weight',
    'g_a.0.0.bias': 'g_a.patch_embed.0.bias',
    'g_a.0.2.weight': 'g_a.patch_embed.2.weight',
    'g_a.0.2.bias': 'g_a.patch_embed.2.bias',    
    'g_a.1.0.norm1.weight': 'g_a.stages.0.0.norm1.weight',
    'g_a.1.0.norm1.bias': 'g_a.stages.0.0.norm1.bias',
    'g_a.1.0.attn.logit_scale': 'g_a.stages.0.0.attn.logit_scale',
    'g_a.1.0.attn.relative_coords_table': 'g_a.stages.0.0.attn.relative_coords_table',
    'g_a.1.0.attn.relative_position_index': 'g_a.stages.0.0.attn.relative_position_index',
    'g_a.1.0.attn.qkv.weight': 'g_a.stages.0.0.attn.qkv.weight',
    'g_a.1.0.attn.qkv.bias': 'g_a.stages.0.0.attn.qkv.bias',
    'g_a.1.0.attn.proj.weight': 'g_a.stages.0.0.attn.proj.weight',
    'g_a.1.0.attn.proj.bias': 'g_a.stages.0.0.attn.proj.bias',
    'g_a.1.0.attn.cpb_mlp.0.weight': 'g_a.stages.0.0.attn.cpb_mlp.0.weight',
    'g_a.1.0.attn.cpb_mlp.0.bias': 'g_a.stages.0.0.attn.cpb_mlp.0.bias',
    'g_a.1.0.attn.cpb_mlp.2.weight': 'g_a.stages.0.0.attn.cpb_mlp.2.weight',
    'g_a.1.0.norm2.weight': 'g_a.stages.0.0.norm2.weight',
    'g_a.1.0.norm2.bias': 'g_a.stages.0.0.norm2.bias',
    'g_a.1.0.mlp.0.weight': 'g_a.stages.0.0.mlp.0.weight',
    'g_a.1.0.mlp.0.bias': 'g_a.stages.0.0.mlp.0.bias',
    'g_a.1.0.mlp.3.weight': 'g_a.stages.0.0.mlp.3.weight',
    'g_a.1.0.mlp.3.bias': 'g_a.stages.0.0.mlp.3.bias',
    'g_a.1.1.norm1.weight': 'g_a.stages.0.1.norm1.weight',
    'g_a.1.1.norm1.bias': 'g_a.stages.0.1.norm1.bias',
    'g_a.1.1.attn.logit_scale': 'g_a.stages.0.1.attn.logit_scale',
    'g_a.1.1.attn.relative_coords_table': 'g_a.stages.0.1.attn.relative_coords_table',
    'g_a.1.1.attn.relative_position_index': 'g_a.stages.0.1.attn.relative_position_index',
    'g_a.1.1.attn.qkv.weight': 'g_a.stages.0.1.attn.qkv.weight',
    'g_a.1.1.attn.qkv.bias': 'g_a.stages.0.1.attn.qkv.bias',
    'g_a.1.1.attn.proj.weight': 'g_a.stages.0.1.attn.proj.weight',
    'g_a.1.1.attn.proj.bias': 'g_a.stages.0.1.attn.proj.bias',
    'g_a.1.1.attn.cpb_mlp.0.weight': 'g_a.stages.0.1.attn.cpb_mlp.0.weight',
    'g_a.1.1.attn.cpb_mlp.0.bias': 'g_a.stages.0.1.attn.cpb_mlp.0.bias',
    'g_a.1.1.attn.cpb_mlp.2.weight': 'g_a.stages.0.1.attn.cpb_mlp.2.weight',
    'g_a.1.1.norm2.weight': 'g_a.stages.0.1.norm2.weight',
    'g_a.1.1.norm2.bias': 'g_a.stages.0.1.norm2.bias',
    'g_a.1.1.mlp.0.weight': 'g_a.stages.0.1.mlp.0.weight',
    'g_a.1.1.mlp.0.bias': 'g_a.stages.0.1.mlp.0.bias',
    'g_a.1.1.mlp.3.weight': 'g_a.stages.0.1.mlp.3.weight',
    'g_a.1.1.mlp.3.bias': 'g_a.stages.0.1.mlp.3.bias',
    'g_a.2.reduction.weight': 'g_a.downsamplers.0.reduction.weight',
    'g_a.2.norm.weight': 'g_a.downsamplers.0.norm.weight',
    'g_a.2.norm.bias': 'g_a.downsamplers.0.norm.bias',
    'g_a.3.0.norm1.weight': 'g_a.stages.1.0.norm1.weight',
    'g_a.3.0.norm1.bias': 'g_a.stages.1.0.norm1.bias',
    'g_a.3.0.attn.logit_scale': 'g_a.stages.1.0.attn.logit_scale',
    'g_a.3.0.attn.relative_coords_table': 'g_a.stages.1.0.attn.relative_coords_table',
    'g_a.3.0.attn.relative_position_index': 'g_a.stages.1.0.attn.relative_position_index',
    'g_a.3.0.attn.qkv.weight': 'g_a.stages.1.0.attn.qkv.weight',
    'g_a.3.0.attn.qkv.bias': 'g_a.stages.1.0.attn.qkv.bias',
    'g_a.3.0.attn.proj.weight': 'g_a.stages.1.0.attn.proj.weight',
    'g_a.3.0.attn.proj.bias': 'g_a.stages.1.0.attn.proj.bias',
    'g_a.3.0.attn.cpb_mlp.0.weight': 'g_a.stages.1.0.attn.cpb_mlp.0.weight',
    'g_a.3.0.attn.cpb_mlp.0.bias': 'g_a.stages.1.0.attn.cpb_mlp.0.bias',
    'g_a.3.0.attn.cpb_mlp.2.weight': 'g_a.stages.1.0.attn.cpb_mlp.2.weight',
    'g_a.3.0.norm2.weight': 'g_a.stages.1.0.norm2.weight',
    'g_a.3.0.norm2.bias': 'g_a.stages.1.0.norm2.bias',
    'g_a.3.0.mlp.0.weight': 'g_a.stages.1.0.mlp.0.weight',
    'g_a.3.0.mlp.0.bias': 'g_a.stages.1.0.mlp.0.bias',
    'g_a.3.0.mlp.3.weight': 'g_a.stages.1.0.mlp.3.weight',
    'g_a.3.0.mlp.3.bias': 'g_a.stages.1.0.mlp.3.bias',
    'g_a.3.1.norm1.weight': 'g_a.stages.1.1.norm1.weight',
    'g_a.3.1.norm1.bias': 'g_a.stages.1.1.norm1.bias',
    'g_a.3.1.attn.logit_scale': 'g_a.stages.1.1.attn.logit_scale',
    'g_a.3.1.attn.relative_coords_table': 'g_a.stages.1.1.attn.relative_coords_table',
    'g_a.3.1.attn.relative_position_index': 'g_a.stages.1.1.attn.relative_position_index',
    'g_a.3.1.attn.qkv.weight': 'g_a.stages.1.1.attn.qkv.weight',
    'g_a.3.1.attn.qkv.bias': 'g_a.stages.1.1.attn.qkv.bias',
    'g_a.3.1.attn.proj.weight': 'g_a.stages.1.1.attn.proj.weight',
    'g_a.3.1.attn.proj.bias': 'g_a.stages.1.1.attn.proj.bias',
    'g_a.3.1.attn.cpb_mlp.0.weight': 'g_a.stages.1.1.attn.cpb_mlp.0.weight',
    'g_a.3.1.attn.cpb_mlp.0.bias': 'g_a.stages.1.1.attn.cpb_mlp.0.bias',
    'g_a.3.1.attn.cpb_mlp.2.weight': 'g_a.stages.1.1.attn.cpb_mlp.2.weight',
    'g_a.3.1.norm2.weight': 'g_a.stages.1.1.norm2.weight',
    'g_a.3.1.norm2.bias': 'g_a.stages.1.1.norm2.bias',
    'g_a.3.1.mlp.0.weight': 'g_a.stages.1.1.mlp.0.weight',
    'g_a.3.1.mlp.0.bias': 'g_a.stages.1.1.mlp.0.bias',
    'g_a.3.1.mlp.3.weight': 'g_a.stages.1.1.mlp.3.weight',
    'g_a.3.1.mlp.3.bias': 'g_a.stages.1.1.mlp.3.bias',
    'g_a.4.reduction.weight': 'g_a.downsamplers.1.reduction.weight',
    'g_a.4.norm.weight': 'g_a.downsamplers.1.norm.weight',
    'g_a.4.norm.bias': 'g_a.downsamplers.1.norm.bias',
    'g_a.5.0.norm1.weight': 'g_a.stages.2.0.norm1.weight',
    'g_a.5.0.norm1.bias': 'g_a.stages.2.0.norm1.bias',
    'g_a.5.0.attn.logit_scale': 'g_a.stages.2.0.attn.logit_scale',
    'g_a.5.0.attn.relative_coords_table': 'g_a.stages.2.0.attn.relative_coords_table',
    'g_a.5.0.attn.relative_position_index': 'g_a.stages.2.0.attn.relative_position_index',
    'g_a.5.0.attn.qkv.weight': 'g_a.stages.2.0.attn.qkv.weight',
    'g_a.5.0.attn.qkv.bias': 'g_a.stages.2.0.attn.qkv.bias',
    'g_a.5.0.attn.proj.weight': 'g_a.stages.2.0.attn.proj.weight',
    'g_a.5.0.attn.proj.bias': 'g_a.stages.2.0.attn.proj.bias',
    'g_a.5.0.attn.cpb_mlp.0.weight': 'g_a.stages.2.0.attn.cpb_mlp.0.weight',
    'g_a.5.0.attn.cpb_mlp.0.bias': 'g_a.stages.2.0.attn.cpb_mlp.0.bias',
    'g_a.5.0.attn.cpb_mlp.2.weight': 'g_a.stages.2.0.attn.cpb_mlp.2.weight',
    'g_a.5.0.norm2.weight': 'g_a.stages.2.0.norm2.weight',
    'g_a.5.0.norm2.bias': 'g_a.stages.2.0.norm2.bias',
    'g_a.5.0.mlp.0.weight': 'g_a.stages.2.0.mlp.0.weight',
    'g_a.5.0.mlp.0.bias': 'g_a.stages.2.0.mlp.0.bias',
    'g_a.5.0.mlp.3.weight': 'g_a.stages.2.0.mlp.3.weight',
    'g_a.5.0.mlp.3.bias': 'g_a.stages.2.0.mlp.3.bias',
    'g_a.5.1.norm1.weight': 'g_a.stages.2.1.norm1.weight',
    'g_a.5.1.norm1.bias': 'g_a.stages.2.1.norm1.bias',
    'g_a.5.1.attn.logit_scale': 'g_a.stages.2.1.attn.logit_scale',
    'g_a.5.1.attn.relative_coords_table': 'g_a.stages.2.1.attn.relative_coords_table',
    'g_a.5.1.attn.relative_position_index': 'g_a.stages.2.1.attn.relative_position_index',
    'g_a.5.1.attn.qkv.weight': 'g_a.stages.2.1.attn.qkv.weight',
    'g_a.5.1.attn.qkv.bias': 'g_a.stages.2.1.attn.qkv.bias',
    'g_a.5.1.attn.proj.weight': 'g_a.stages.2.1.attn.proj.weight',
    'g_a.5.1.attn.proj.bias': 'g_a.stages.2.1.attn.proj.bias',
    'g_a.5.1.attn.cpb_mlp.0.weight': 'g_a.stages.2.1.attn.cpb_mlp.0.weight',
    'g_a.5.1.attn.cpb_mlp.0.bias': 'g_a.stages.2.1.attn.cpb_mlp.0.bias',
    'g_a.5.1.attn.cpb_mlp.2.weight': 'g_a.stages.2.1.attn.cpb_mlp.2.weight',
    'g_a.5.1.norm2.weight': 'g_a.stages.2.1.norm2.weight',
    'g_a.5.1.norm2.bias': 'g_a.stages.2.1.norm2.bias',
    'g_a.5.1.mlp.0.weight': 'g_a.stages.2.1.mlp.0.weight',
    'g_a.5.1.mlp.0.bias': 'g_a.stages.2.1.mlp.0.bias',
    'g_a.5.1.mlp.3.weight': 'g_a.stages.2.1.mlp.3.weight',
    'g_a.5.1.mlp.3.bias': 'g_a.stages.2.1.mlp.3.bias',
    'g_a.5.2.norm1.weight': 'g_a.stages.2.2.norm1.weight',
    'g_a.5.2.norm1.bias': 'g_a.stages.2.2.norm1.bias',
    'g_a.5.2.attn.logit_scale': 'g_a.stages.2.2.attn.logit_scale',
    'g_a.5.2.attn.relative_coords_table': 'g_a.stages.2.2.attn.relative_coords_table',
    'g_a.5.2.attn.relative_position_index': 'g_a.stages.2.2.attn.relative_position_index',
    'g_a.5.2.attn.qkv.weight': 'g_a.stages.2.2.attn.qkv.weight',
    'g_a.5.2.attn.qkv.bias': 'g_a.stages.2.2.attn.qkv.bias',
    'g_a.5.2.attn.proj.weight': 'g_a.stages.2.2.attn.proj.weight',
    'g_a.5.2.attn.proj.bias': 'g_a.stages.2.2.attn.proj.bias',
    'g_a.5.2.attn.cpb_mlp.0.weight': 'g_a.stages.2.2.attn.cpb_mlp.0.weight',
    'g_a.5.2.attn.cpb_mlp.0.bias': 'g_a.stages.2.2.attn.cpb_mlp.0.bias',
    'g_a.5.2.attn.cpb_mlp.2.weight': 'g_a.stages.2.2.attn.cpb_mlp.2.weight',
    'g_a.5.2.norm2.weight': 'g_a.stages.2.2.norm2.weight',
    'g_a.5.2.norm2.bias': 'g_a.stages.2.2.norm2.bias',
    'g_a.5.2.mlp.0.weight': 'g_a.stages.2.2.mlp.0.weight',
    'g_a.5.2.mlp.0.bias': 'g_a.stages.2.2.mlp.0.bias',
    'g_a.5.2.mlp.3.weight': 'g_a.stages.2.2.mlp.3.weight',
    'g_a.5.2.mlp.3.bias': 'g_a.stages.2.2.mlp.3.bias',
    'g_a.5.3.norm1.weight': 'g_a.stages.2.3.norm1.weight',
    'g_a.5.3.norm1.bias': 'g_a.stages.2.3.norm1.bias',
    'g_a.5.3.attn.logit_scale': 'g_a.stages.2.3.attn.logit_scale',
    'g_a.5.3.attn.relative_coords_table': 'g_a.stages.2.3.attn.relative_coords_table',
    'g_a.5.3.attn.relative_position_index': 'g_a.stages.2.3.attn.relative_position_index',
    'g_a.5.3.attn.qkv.weight': 'g_a.stages.2.3.attn.qkv.weight',
    'g_a.5.3.attn.qkv.bias': 'g_a.stages.2.3.attn.qkv.bias',
    'g_a.5.3.attn.proj.weight': 'g_a.stages.2.3.attn.proj.weight',
    'g_a.5.3.attn.proj.bias': 'g_a.stages.2.3.attn.proj.bias',
    'g_a.5.3.attn.cpb_mlp.0.weight': 'g_a.stages.2.3.attn.cpb_mlp.0.weight',
    'g_a.5.3.attn.cpb_mlp.0.bias': 'g_a.stages.2.3.attn.cpb_mlp.0.bias',
    'g_a.5.3.attn.cpb_mlp.2.weight': 'g_a.stages.2.3.attn.cpb_mlp.2.weight',
    'g_a.5.3.norm2.weight': 'g_a.stages.2.3.norm2.weight',
    'g_a.5.3.norm2.bias': 'g_a.stages.2.3.norm2.bias',
    'g_a.5.3.mlp.0.weight': 'g_a.stages.2.3.mlp.0.weight',
    'g_a.5.3.mlp.0.bias': 'g_a.stages.2.3.mlp.0.bias',
    'g_a.5.3.mlp.3.weight': 'g_a.stages.2.3.mlp.3.weight',
    'g_a.5.3.mlp.3.bias': 'g_a.stages.2.3.mlp.3.bias',
    'g_a.5.4.norm1.weight': 'g_a.stages.2.4.norm1.weight',
    'g_a.5.4.norm1.bias': 'g_a.stages.2.4.norm1.bias',
    'g_a.5.4.attn.logit_scale': 'g_a.stages.2.4.attn.logit_scale',
    'g_a.5.4.attn.relative_coords_table': 'g_a.stages.2.4.attn.relative_coords_table',
    'g_a.5.4.attn.relative_position_index': 'g_a.stages.2.4.attn.relative_position_index',
    'g_a.5.4.attn.qkv.weight': 'g_a.stages.2.4.attn.qkv.weight',
    'g_a.5.4.attn.qkv.bias': 'g_a.stages.2.4.attn.qkv.bias',
    'g_a.5.4.attn.proj.weight': 'g_a.stages.2.4.attn.proj.weight',
    'g_a.5.4.attn.proj.bias': 'g_a.stages.2.4.attn.proj.bias',
    'g_a.5.4.attn.cpb_mlp.0.weight': 'g_a.stages.2.4.attn.cpb_mlp.0.weight',
    'g_a.5.4.attn.cpb_mlp.0.bias': 'g_a.stages.2.4.attn.cpb_mlp.0.bias',
    'g_a.5.4.attn.cpb_mlp.2.weight': 'g_a.stages.2.4.attn.cpb_mlp.2.weight',
    'g_a.5.4.norm2.weight': 'g_a.stages.2.4.norm2.weight',
    'g_a.5.4.norm2.bias': 'g_a.stages.2.4.norm2.bias',
    'g_a.5.4.mlp.0.weight': 'g_a.stages.2.4.mlp.0.weight',
    'g_a.5.4.mlp.0.bias': 'g_a.stages.2.4.mlp.0.bias',
    'g_a.5.4.mlp.3.weight': 'g_a.stages.2.4.mlp.3.weight',
    'g_a.5.4.mlp.3.bias': 'g_a.stages.2.4.mlp.3.bias',
    'g_a.5.5.norm1.weight': 'g_a.stages.2.5.norm1.weight',
    'g_a.5.5.norm1.bias': 'g_a.stages.2.5.norm1.bias',
    'g_a.5.5.attn.logit_scale': 'g_a.stages.2.5.attn.logit_scale',
    'g_a.5.5.attn.relative_coords_table': 'g_a.stages.2.5.attn.relative_coords_table',
    'g_a.5.5.attn.relative_position_index': 'g_a.stages.2.5.attn.relative_position_index',
    'g_a.5.5.attn.qkv.weight': 'g_a.stages.2.5.attn.qkv.weight',
    'g_a.5.5.attn.qkv.bias': 'g_a.stages.2.5.attn.qkv.bias',
    'g_a.5.5.attn.proj.weight': 'g_a.stages.2.5.attn.proj.weight',
    'g_a.5.5.attn.proj.bias': 'g_a.stages.2.5.attn.proj.bias',
    'g_a.5.5.attn.cpb_mlp.0.weight': 'g_a.stages.2.5.attn.cpb_mlp.0.weight',
    'g_a.5.5.attn.cpb_mlp.0.bias': 'g_a.stages.2.5.attn.cpb_mlp.0.bias',
    'g_a.5.5.attn.cpb_mlp.2.weight': 'g_a.stages.2.5.attn.cpb_mlp.2.weight',
    'g_a.5.5.norm2.weight': 'g_a.stages.2.5.norm2.weight',
    'g_a.5.5.norm2.bias': 'g_a.stages.2.5.norm2.bias',
    'g_a.5.5.mlp.0.weight': 'g_a.stages.2.5.mlp.0.weight',
    'g_a.5.5.mlp.0.bias': 'g_a.stages.2.5.mlp.0.bias',
    'g_a.5.5.mlp.3.weight': 'g_a.stages.2.5.mlp.3.weight',
    'g_a.5.5.mlp.3.bias': 'g_a.stages.2.5.mlp.3.bias',
    'g_a.5.6.norm1.weight': 'g_a.stages.2.6.norm1.weight',
    'g_a.5.6.norm1.bias': 'g_a.stages.2.6.norm1.bias',
    'g_a.5.6.attn.logit_scale': 'g_a.stages.2.6.attn.logit_scale',
    'g_a.5.6.attn.relative_coords_table': 'g_a.stages.2.6.attn.relative_coords_table',
    'g_a.5.6.attn.relative_position_index': 'g_a.stages.2.6.attn.relative_position_index',
    'g_a.5.6.attn.qkv.weight': 'g_a.stages.2.6.attn.qkv.weight',
    'g_a.5.6.attn.qkv.bias': 'g_a.stages.2.6.attn.qkv.bias',
    'g_a.5.6.attn.proj.weight': 'g_a.stages.2.6.attn.proj.weight',
    'g_a.5.6.attn.proj.bias': 'g_a.stages.2.6.attn.proj.bias',
    'g_a.5.6.attn.cpb_mlp.0.weight': 'g_a.stages.2.6.attn.cpb_mlp.0.weight',
    'g_a.5.6.attn.cpb_mlp.0.bias': 'g_a.stages.2.6.attn.cpb_mlp.0.bias',
    'g_a.5.6.attn.cpb_mlp.2.weight': 'g_a.stages.2.6.attn.cpb_mlp.2.weight',
    'g_a.5.6.norm2.weight': 'g_a.stages.2.6.norm2.weight',
    'g_a.5.6.norm2.bias': 'g_a.stages.2.6.norm2.bias',
    'g_a.5.6.mlp.0.weight': 'g_a.stages.2.6.mlp.0.weight',
    'g_a.5.6.mlp.0.bias': 'g_a.stages.2.6.mlp.0.bias',
    'g_a.5.6.mlp.3.weight': 'g_a.stages.2.6.mlp.3.weight',
    'g_a.5.6.mlp.3.bias': 'g_a.stages.2.6.mlp.3.bias',
    'g_a.5.7.norm1.weight': 'g_a.stages.2.7.norm1.weight',
    'g_a.5.7.norm1.bias': 'g_a.stages.2.7.norm1.bias',
    'g_a.5.7.attn.logit_scale': 'g_a.stages.2.7.attn.logit_scale',
    'g_a.5.7.attn.relative_coords_table': 'g_a.stages.2.7.attn.relative_coords_table',
    'g_a.5.7.attn.relative_position_index': 'g_a.stages.2.7.attn.relative_position_index',
    'g_a.5.7.attn.qkv.weight': 'g_a.stages.2.7.attn.qkv.weight',
    'g_a.5.7.attn.qkv.bias': 'g_a.stages.2.7.attn.qkv.bias',
    'g_a.5.7.attn.proj.weight': 'g_a.stages.2.7.attn.proj.weight',
    'g_a.5.7.attn.proj.bias': 'g_a.stages.2.7.attn.proj.bias',
    'g_a.5.7.attn.cpb_mlp.0.weight': 'g_a.stages.2.7.attn.cpb_mlp.0.weight',
    'g_a.5.7.attn.cpb_mlp.0.bias': 'g_a.stages.2.7.attn.cpb_mlp.0.bias',
    'g_a.5.7.attn.cpb_mlp.2.weight': 'g_a.stages.2.7.attn.cpb_mlp.2.weight',
    'g_a.5.7.norm2.weight': 'g_a.stages.2.7.norm2.weight',
    'g_a.5.7.norm2.bias': 'g_a.stages.2.7.norm2.bias',
    'g_a.5.7.mlp.0.weight': 'g_a.stages.2.7.mlp.0.weight',
    'g_a.5.7.mlp.0.bias': 'g_a.stages.2.7.mlp.0.bias',
    'g_a.5.7.mlp.3.weight': 'g_a.stages.2.7.mlp.3.weight',
    'g_a.5.7.mlp.3.bias': 'g_a.stages.2.7.mlp.3.bias',
    'g_a.5.8.norm1.weight': 'g_a.stages.2.8.norm1.weight',
    'g_a.5.8.norm1.bias': 'g_a.stages.2.8.norm1.bias',
    'g_a.5.8.attn.logit_scale': 'g_a.stages.2.8.attn.logit_scale',
    'g_a.5.8.attn.relative_coords_table': 'g_a.stages.2.8.attn.relative_coords_table',
    'g_a.5.8.attn.relative_position_index': 'g_a.stages.2.8.attn.relative_position_index',
    'g_a.5.8.attn.qkv.weight': 'g_a.stages.2.8.attn.qkv.weight',
    'g_a.5.8.attn.qkv.bias': 'g_a.stages.2.8.attn.qkv.bias',
    'g_a.5.8.attn.proj.weight': 'g_a.stages.2.8.attn.proj.weight',
    'g_a.5.8.attn.proj.bias': 'g_a.stages.2.8.attn.proj.bias',
    'g_a.5.8.attn.cpb_mlp.0.weight': 'g_a.stages.2.8.attn.cpb_mlp.0.weight',
    'g_a.5.8.attn.cpb_mlp.0.bias': 'g_a.stages.2.8.attn.cpb_mlp.0.bias',
    'g_a.5.8.attn.cpb_mlp.2.weight': 'g_a.stages.2.8.attn.cpb_mlp.2.weight',
    'g_a.5.8.norm2.weight': 'g_a.stages.2.8.norm2.weight',
    'g_a.5.8.norm2.bias': 'g_a.stages.2.8.norm2.bias',
    'g_a.5.8.mlp.0.weight': 'g_a.stages.2.8.mlp.0.weight',
    'g_a.5.8.mlp.0.bias': 'g_a.stages.2.8.mlp.0.bias',
    'g_a.5.8.mlp.3.weight': 'g_a.stages.2.8.mlp.3.weight',
    'g_a.5.8.mlp.3.bias': 'g_a.stages.2.8.mlp.3.bias',
    'g_a.5.9.norm1.weight': 'g_a.stages.2.9.norm1.weight',
    'g_a.5.9.norm1.bias': 'g_a.stages.2.9.norm1.bias',
    'g_a.5.9.attn.logit_scale': 'g_a.stages.2.9.attn.logit_scale',
    'g_a.5.9.attn.relative_coords_table': 'g_a.stages.2.9.attn.relative_coords_table',
    'g_a.5.9.attn.relative_position_index': 'g_a.stages.2.9.attn.relative_position_index',
    'g_a.5.9.attn.qkv.weight': 'g_a.stages.2.9.attn.qkv.weight',
    'g_a.5.9.attn.qkv.bias': 'g_a.stages.2.9.attn.qkv.bias',
    'g_a.5.9.attn.proj.weight': 'g_a.stages.2.9.attn.proj.weight',
    'g_a.5.9.attn.proj.bias': 'g_a.stages.2.9.attn.proj.bias',
    'g_a.5.9.attn.cpb_mlp.0.weight': 'g_a.stages.2.9.attn.cpb_mlp.0.weight',
    'g_a.5.9.attn.cpb_mlp.0.bias': 'g_a.stages.2.9.attn.cpb_mlp.0.bias',
    'g_a.5.9.attn.cpb_mlp.2.weight': 'g_a.stages.2.9.attn.cpb_mlp.2.weight',
    'g_a.5.9.norm2.weight': 'g_a.stages.2.9.norm2.weight',
    'g_a.5.9.norm2.bias': 'g_a.stages.2.9.norm2.bias',
    'g_a.5.9.mlp.0.weight': 'g_a.stages.2.9.mlp.0.weight',
    'g_a.5.9.mlp.0.bias': 'g_a.stages.2.9.mlp.0.bias',
    'g_a.5.9.mlp.3.weight': 'g_a.stages.2.9.mlp.3.weight',
    'g_a.5.9.mlp.3.bias': 'g_a.stages.2.9.mlp.3.bias',
    'g_a.5.10.norm1.weight': 'g_a.stages.2.10.norm1.weight',
    'g_a.5.10.norm1.bias': 'g_a.stages.2.10.norm1.bias',
    'g_a.5.10.attn.logit_scale': 'g_a.stages.2.10.attn.logit_scale',
    'g_a.5.10.attn.relative_coords_table': 'g_a.stages.2.10.attn.relative_coords_table',
    'g_a.5.10.attn.relative_position_index': 'g_a.stages.2.10.attn.relative_position_index',
    'g_a.5.10.attn.qkv.weight': 'g_a.stages.2.10.attn.qkv.weight',
    'g_a.5.10.attn.qkv.bias': 'g_a.stages.2.10.attn.qkv.bias',
    'g_a.5.10.attn.proj.weight': 'g_a.stages.2.10.attn.proj.weight',
    'g_a.5.10.attn.proj.bias': 'g_a.stages.2.10.attn.proj.bias',
    'g_a.5.10.attn.cpb_mlp.0.weight': 'g_a.stages.2.10.attn.cpb_mlp.0.weight',
    'g_a.5.10.attn.cpb_mlp.0.bias': 'g_a.stages.2.10.attn.cpb_mlp.0.bias',
    'g_a.5.10.attn.cpb_mlp.2.weight': 'g_a.stages.2.10.attn.cpb_mlp.2.weight',
    'g_a.5.10.norm2.weight': 'g_a.stages.2.10.norm2.weight',
    'g_a.5.10.norm2.bias': 'g_a.stages.2.10.norm2.bias',
    'g_a.5.10.mlp.0.weight': 'g_a.stages.2.10.mlp.0.weight',
    'g_a.5.10.mlp.0.bias': 'g_a.stages.2.10.mlp.0.bias',
    'g_a.5.10.mlp.3.weight': 'g_a.stages.2.10.mlp.3.weight',
    'g_a.5.10.mlp.3.bias': 'g_a.stages.2.10.mlp.3.bias',
    'g_a.5.11.norm1.weight': 'g_a.stages.2.11.norm1.weight',
    'g_a.5.11.norm1.bias': 'g_a.stages.2.11.norm1.bias',
    'g_a.5.11.attn.logit_scale': 'g_a.stages.2.11.attn.logit_scale',
    'g_a.5.11.attn.relative_coords_table': 'g_a.stages.2.11.attn.relative_coords_table',
    'g_a.5.11.attn.relative_position_index': 'g_a.stages.2.11.attn.relative_position_index',
    'g_a.5.11.attn.qkv.weight': 'g_a.stages.2.11.attn.qkv.weight',
    'g_a.5.11.attn.qkv.bias': 'g_a.stages.2.11.attn.qkv.bias',
    'g_a.5.11.attn.proj.weight': 'g_a.stages.2.11.attn.proj.weight',
    'g_a.5.11.attn.proj.bias': 'g_a.stages.2.11.attn.proj.bias',
    'g_a.5.11.attn.cpb_mlp.0.weight': 'g_a.stages.2.11.attn.cpb_mlp.0.weight',
    'g_a.5.11.attn.cpb_mlp.0.bias': 'g_a.stages.2.11.attn.cpb_mlp.0.bias',
    'g_a.5.11.attn.cpb_mlp.2.weight': 'g_a.stages.2.11.attn.cpb_mlp.2.weight',
    'g_a.5.11.norm2.weight': 'g_a.stages.2.11.norm2.weight',
    'g_a.5.11.norm2.bias': 'g_a.stages.2.11.norm2.bias',
    'g_a.5.11.mlp.0.weight': 'g_a.stages.2.11.mlp.0.weight',
    'g_a.5.11.mlp.0.bias': 'g_a.stages.2.11.mlp.0.bias',
    'g_a.5.11.mlp.3.weight': 'g_a.stages.2.11.mlp.3.weight',
    'g_a.5.11.mlp.3.bias': 'g_a.stages.2.11.mlp.3.bias',
    'g_a.5.12.norm1.weight': 'g_a.stages.2.12.norm1.weight',
    'g_a.5.12.norm1.bias': 'g_a.stages.2.12.norm1.bias',
    'g_a.5.12.attn.logit_scale': 'g_a.stages.2.12.attn.logit_scale',
    'g_a.5.12.attn.relative_coords_table': 'g_a.stages.2.12.attn.relative_coords_table',
    'g_a.5.12.attn.relative_position_index': 'g_a.stages.2.12.attn.relative_position_index',
    'g_a.5.12.attn.qkv.weight': 'g_a.stages.2.12.attn.qkv.weight',
    'g_a.5.12.attn.qkv.bias': 'g_a.stages.2.12.attn.qkv.bias',
    'g_a.5.12.attn.proj.weight': 'g_a.stages.2.12.attn.proj.weight',
    'g_a.5.12.attn.proj.bias': 'g_a.stages.2.12.attn.proj.bias',
    'g_a.5.12.attn.cpb_mlp.0.weight': 'g_a.stages.2.12.attn.cpb_mlp.0.weight',
    'g_a.5.12.attn.cpb_mlp.0.bias': 'g_a.stages.2.12.attn.cpb_mlp.0.bias',
    'g_a.5.12.attn.cpb_mlp.2.weight': 'g_a.stages.2.12.attn.cpb_mlp.2.weight',
    'g_a.5.12.norm2.weight': 'g_a.stages.2.12.norm2.weight',
    'g_a.5.12.norm2.bias': 'g_a.stages.2.12.norm2.bias',
    'g_a.5.12.mlp.0.weight': 'g_a.stages.2.12.mlp.0.weight',
    'g_a.5.12.mlp.0.bias': 'g_a.stages.2.12.mlp.0.bias',
    'g_a.5.12.mlp.3.weight': 'g_a.stages.2.12.mlp.3.weight',
    'g_a.5.12.mlp.3.bias': 'g_a.stages.2.12.mlp.3.bias',
    'g_a.5.13.norm1.weight': 'g_a.stages.2.13.norm1.weight',
    'g_a.5.13.norm1.bias': 'g_a.stages.2.13.norm1.bias',
    'g_a.5.13.attn.logit_scale': 'g_a.stages.2.13.attn.logit_scale',
    'g_a.5.13.attn.relative_coords_table': 'g_a.stages.2.13.attn.relative_coords_table',
    'g_a.5.13.attn.relative_position_index': 'g_a.stages.2.13.attn.relative_position_index',
    'g_a.5.13.attn.qkv.weight': 'g_a.stages.2.13.attn.qkv.weight',
    'g_a.5.13.attn.qkv.bias': 'g_a.stages.2.13.attn.qkv.bias',
    'g_a.5.13.attn.proj.weight': 'g_a.stages.2.13.attn.proj.weight',
    'g_a.5.13.attn.proj.bias': 'g_a.stages.2.13.attn.proj.bias',
    'g_a.5.13.attn.cpb_mlp.0.weight': 'g_a.stages.2.13.attn.cpb_mlp.0.weight',
    'g_a.5.13.attn.cpb_mlp.0.bias': 'g_a.stages.2.13.attn.cpb_mlp.0.bias',
    'g_a.5.13.attn.cpb_mlp.2.weight': 'g_a.stages.2.13.attn.cpb_mlp.2.weight',
    'g_a.5.13.norm2.weight': 'g_a.stages.2.13.norm2.weight',
    'g_a.5.13.norm2.bias': 'g_a.stages.2.13.norm2.bias',
    'g_a.5.13.mlp.0.weight': 'g_a.stages.2.13.mlp.0.weight',
    'g_a.5.13.mlp.0.bias': 'g_a.stages.2.13.mlp.0.bias',
    'g_a.5.13.mlp.3.weight': 'g_a.stages.2.13.mlp.3.weight',
    'g_a.5.13.mlp.3.bias': 'g_a.stages.2.13.mlp.3.bias',
    'g_a.5.14.norm1.weight': 'g_a.stages.2.14.norm1.weight',
    'g_a.5.14.norm1.bias': 'g_a.stages.2.14.norm1.bias',
    'g_a.5.14.attn.logit_scale': 'g_a.stages.2.14.attn.logit_scale',
    'g_a.5.14.attn.relative_coords_table': 'g_a.stages.2.14.attn.relative_coords_table',
    'g_a.5.14.attn.relative_position_index': 'g_a.stages.2.14.attn.relative_position_index',
    'g_a.5.14.attn.qkv.weight': 'g_a.stages.2.14.attn.qkv.weight',
    'g_a.5.14.attn.qkv.bias': 'g_a.stages.2.14.attn.qkv.bias',
    'g_a.5.14.attn.proj.weight': 'g_a.stages.2.14.attn.proj.weight',
    'g_a.5.14.attn.proj.bias': 'g_a.stages.2.14.attn.proj.bias',
    'g_a.5.14.attn.cpb_mlp.0.weight': 'g_a.stages.2.14.attn.cpb_mlp.0.weight',
    'g_a.5.14.attn.cpb_mlp.0.bias': 'g_a.stages.2.14.attn.cpb_mlp.0.bias',
    'g_a.5.14.attn.cpb_mlp.2.weight': 'g_a.stages.2.14.attn.cpb_mlp.2.weight',
    'g_a.5.14.norm2.weight': 'g_a.stages.2.14.norm2.weight',
    'g_a.5.14.norm2.bias': 'g_a.stages.2.14.norm2.bias',
    'g_a.5.14.mlp.0.weight': 'g_a.stages.2.14.mlp.0.weight',
    'g_a.5.14.mlp.0.bias': 'g_a.stages.2.14.mlp.0.bias',
    'g_a.5.14.mlp.3.weight': 'g_a.stages.2.14.mlp.3.weight',
    'g_a.5.14.mlp.3.bias': 'g_a.stages.2.14.mlp.3.bias',
    'g_a.5.15.norm1.weight': 'g_a.stages.2.15.norm1.weight',
    'g_a.5.15.norm1.bias': 'g_a.stages.2.15.norm1.bias',
    'g_a.5.15.attn.logit_scale': 'g_a.stages.2.15.attn.logit_scale',
    'g_a.5.15.attn.relative_coords_table': 'g_a.stages.2.15.attn.relative_coords_table',
    'g_a.5.15.attn.relative_position_index': 'g_a.stages.2.15.attn.relative_position_index',
    'g_a.5.15.attn.qkv.weight': 'g_a.stages.2.15.attn.qkv.weight',
    'g_a.5.15.attn.qkv.bias': 'g_a.stages.2.15.attn.qkv.bias',
    'g_a.5.15.attn.proj.weight': 'g_a.stages.2.15.attn.proj.weight',
    'g_a.5.15.attn.proj.bias': 'g_a.stages.2.15.attn.proj.bias',
    'g_a.5.15.attn.cpb_mlp.0.weight': 'g_a.stages.2.15.attn.cpb_mlp.0.weight',
    'g_a.5.15.attn.cpb_mlp.0.bias': 'g_a.stages.2.15.attn.cpb_mlp.0.bias',
    'g_a.5.15.attn.cpb_mlp.2.weight': 'g_a.stages.2.15.attn.cpb_mlp.2.weight',
    'g_a.5.15.norm2.weight': 'g_a.stages.2.15.norm2.weight',
    'g_a.5.15.norm2.bias': 'g_a.stages.2.15.norm2.bias',
    'g_a.5.15.mlp.0.weight': 'g_a.stages.2.15.mlp.0.weight',
    'g_a.5.15.mlp.0.bias': 'g_a.stages.2.15.mlp.0.bias',
    'g_a.5.15.mlp.3.weight': 'g_a.stages.2.15.mlp.3.weight',
    'g_a.5.15.mlp.3.bias': 'g_a.stages.2.15.mlp.3.bias',
    'g_a.5.16.norm1.weight': 'g_a.stages.2.16.norm1.weight',
    'g_a.5.16.norm1.bias': 'g_a.stages.2.16.norm1.bias',
    'g_a.5.16.attn.logit_scale': 'g_a.stages.2.16.attn.logit_scale',
    'g_a.5.16.attn.relative_coords_table': 'g_a.stages.2.16.attn.relative_coords_table',
    'g_a.5.16.attn.relative_position_index': 'g_a.stages.2.16.attn.relative_position_index',
    'g_a.5.16.attn.qkv.weight': 'g_a.stages.2.16.attn.qkv.weight',
    'g_a.5.16.attn.qkv.bias': 'g_a.stages.2.16.attn.qkv.bias',
    'g_a.5.16.attn.proj.weight': 'g_a.stages.2.16.attn.proj.weight',
    'g_a.5.16.attn.proj.bias': 'g_a.stages.2.16.attn.proj.bias',
    'g_a.5.16.attn.cpb_mlp.0.weight': 'g_a.stages.2.16.attn.cpb_mlp.0.weight',
    'g_a.5.16.attn.cpb_mlp.0.bias': 'g_a.stages.2.16.attn.cpb_mlp.0.bias',
    'g_a.5.16.attn.cpb_mlp.2.weight': 'g_a.stages.2.16.attn.cpb_mlp.2.weight',
    'g_a.5.16.norm2.weight': 'g_a.stages.2.16.norm2.weight',
    'g_a.5.16.norm2.bias': 'g_a.stages.2.16.norm2.bias',
    'g_a.5.16.mlp.0.weight': 'g_a.stages.2.16.mlp.0.weight',
    'g_a.5.16.mlp.0.bias': 'g_a.stages.2.16.mlp.0.bias',
    'g_a.5.16.mlp.3.weight': 'g_a.stages.2.16.mlp.3.weight',
    'g_a.5.16.mlp.3.bias': 'g_a.stages.2.16.mlp.3.bias',
    'g_a.5.17.norm1.weight': 'g_a.stages.2.17.norm1.weight',
    'g_a.5.17.norm1.bias': 'g_a.stages.2.17.norm1.bias',
    'g_a.5.17.attn.logit_scale': 'g_a.stages.2.17.attn.logit_scale',
    'g_a.5.17.attn.relative_coords_table': 'g_a.stages.2.17.attn.relative_coords_table',
    'g_a.5.17.attn.relative_position_index': 'g_a.stages.2.17.attn.relative_position_index',
    'g_a.5.17.attn.qkv.weight': 'g_a.stages.2.17.attn.qkv.weight',
    'g_a.5.17.attn.qkv.bias': 'g_a.stages.2.17.attn.qkv.bias',
    'g_a.5.17.attn.proj.weight': 'g_a.stages.2.17.attn.proj.weight',
    'g_a.5.17.attn.proj.bias': 'g_a.stages.2.17.attn.proj.bias',
    'g_a.5.17.attn.cpb_mlp.0.weight': 'g_a.stages.2.17.attn.cpb_mlp.0.weight',
    'g_a.5.17.attn.cpb_mlp.0.bias': 'g_a.stages.2.17.attn.cpb_mlp.0.bias',
    'g_a.5.17.attn.cpb_mlp.2.weight': 'g_a.stages.2.17.attn.cpb_mlp.2.weight',
    'g_a.5.17.norm2.weight': 'g_a.stages.2.17.norm2.weight',
    'g_a.5.17.norm2.bias': 'g_a.stages.2.17.norm2.bias',
    'g_a.5.17.mlp.0.weight': 'g_a.stages.2.17.mlp.0.weight',
    'g_a.5.17.mlp.0.bias': 'g_a.stages.2.17.mlp.0.bias',
    'g_a.5.17.mlp.3.weight': 'g_a.stages.2.17.mlp.3.weight',
    'g_a.5.17.mlp.3.bias': 'g_a.stages.2.17.mlp.3.bias',
    'g_a.6.reduction.weight': 'g_a.downsamplers.2.reduction.weight',
    'g_a.6.norm.weight': 'g_a.downsamplers.2.norm.weight',
    'g_a.6.norm.bias': 'g_a.downsamplers.2.norm.bias',
    'g_a.7.0.norm1.weight': 'g_a.stages.3.0.norm1.weight',
    'g_a.7.0.norm1.bias': 'g_a.stages.3.0.norm1.bias',
    'g_a.7.0.attn.logit_scale': 'g_a.stages.3.0.attn.logit_scale',
    'g_a.7.0.attn.relative_coords_table': 'g_a.stages.3.0.attn.relative_coords_table',
    'g_a.7.0.attn.relative_position_index': 'g_a.stages.3.0.attn.relative_position_index',
    'g_a.7.0.attn.qkv.weight': 'g_a.stages.3.0.attn.qkv.weight',
    'g_a.7.0.attn.qkv.bias': 'g_a.stages.3.0.attn.qkv.bias',
    'g_a.7.0.attn.proj.weight': 'g_a.stages.3.0.attn.proj.weight',
    'g_a.7.0.attn.proj.bias': 'g_a.stages.3.0.attn.proj.bias',
    'g_a.7.0.attn.cpb_mlp.0.weight': 'g_a.stages.3.0.attn.cpb_mlp.0.weight',
    'g_a.7.0.attn.cpb_mlp.0.bias': 'g_a.stages.3.0.attn.cpb_mlp.0.bias',
    'g_a.7.0.attn.cpb_mlp.2.weight': 'g_a.stages.3.0.attn.cpb_mlp.2.weight',
    'g_a.7.0.norm2.weight': 'g_a.stages.3.0.norm2.weight',
    'g_a.7.0.norm2.bias': 'g_a.stages.3.0.norm2.bias',
    'g_a.7.0.mlp.0.weight': 'g_a.stages.3.0.mlp.0.weight',
    'g_a.7.0.mlp.0.bias': 'g_a.stages.3.0.mlp.0.bias',
    'g_a.7.0.mlp.3.weight': 'g_a.stages.3.0.mlp.3.weight',
    'g_a.7.0.mlp.3.bias': 'g_a.stages.3.0.mlp.3.bias',
    'g_a.7.1.norm1.weight': 'g_a.stages.3.1.norm1.weight',
    'g_a.7.1.norm1.bias': 'g_a.stages.3.1.norm1.bias',
    'g_a.7.1.attn.logit_scale': 'g_a.stages.3.1.attn.logit_scale',
    'g_a.7.1.attn.relative_coords_table': 'g_a.stages.3.1.attn.relative_coords_table',
    'g_a.7.1.attn.relative_position_index': 'g_a.stages.3.1.attn.relative_position_index',
    'g_a.7.1.attn.qkv.weight': 'g_a.stages.3.1.attn.qkv.weight',
    'g_a.7.1.attn.qkv.bias': 'g_a.stages.3.1.attn.qkv.bias',
    'g_a.7.1.attn.proj.weight': 'g_a.stages.3.1.attn.proj.weight',
    'g_a.7.1.attn.proj.bias': 'g_a.stages.3.1.attn.proj.bias',
    'g_a.7.1.attn.cpb_mlp.0.weight': 'g_a.stages.3.1.attn.cpb_mlp.0.weight',
    'g_a.7.1.attn.cpb_mlp.0.bias': 'g_a.stages.3.1.attn.cpb_mlp.0.bias',
    'g_a.7.1.attn.cpb_mlp.2.weight': 'g_a.stages.3.1.attn.cpb_mlp.2.weight',
    'g_a.7.1.norm2.weight': 'g_a.stages.3.1.norm2.weight',
    'g_a.7.1.norm2.bias': 'g_a.stages.3.1.norm2.bias',
    'g_a.7.1.mlp.0.weight': 'g_a.stages.3.1.mlp.0.weight',
    'g_a.7.1.mlp.0.bias': 'g_a.stages.3.1.mlp.0.bias',
    'g_a.7.1.mlp.3.weight': 'g_a.stages.3.1.mlp.3.weight',
    'g_a.7.1.mlp.3.bias': 'g_a.stages.3.1.mlp.3.bias',
}

def read_config(config_file_path: str) -> dict[str, Any]:
    """
    Reads a YAML configuration file and returns its content as a dictionary.

    Args:
        config_file_path (str): Path to the YAML file.

    Returns:
        dict[str, Any]: Parsed YAML content.

    Example:
        config = read_config("config.yaml")
        print(config)  # -> {'module': '...', 'args': {...}, ...}
    """
    data = yaml.safe_load(open(config_file_path))
    return data

def transform_from_config(config: dict[str, Any]) -> Module:
    args = config['args']
    package, module = config['module'].rsplit('.', 1)
    package = importlib.import_module(package)
    type: Module = getattr(package, module)
    return type(**args)

def dataset_from_config(config: dict[str, Any]) -> BaseDataset:
    """
    Initializes a dataset based on the configuration provided.

    Args:
        config (dict[str, Any]): A dictionary containing the dataset type and its arguments.
            Example format:
            {
                "module": "torchvision.datasets.CIFAR10",
                "args": {
                    "root": "./data",
                    "train": True,
                    "download": True,
                    "transform": transforms.Compose([...])
                }
            }

    Returns:
        Dataset: An instance of the dataset.

    Example:
        config = {
            "module": "torchvision.datasets.CIFAR10",
            "args": {
                "root": "./data",
                "train": True,
                "download": True,
                "transform": transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,), (0.5,))
                ])
            }
        }
        dataset = init_dataset(config)
    """
    args = deepcopy(config['args'])
    package, module = config['module'].rsplit('.', 1)
    package = importlib.import_module(package)
    type = getattr(package, module)
    if 'transform' in config.keys():
        tsfs = transforms.Compose([transform_from_config(tsf) for tsf in config['transform']])
        args['transform'] = tsfs
    if 'target_transform' in config.keys():
        tg_tsfs = transforms.Compose([transform_from_config(tsf) for tsf in config['target_transform']])
        args['target_transform'] = tg_tsfs
    if 'subset_classes' in args and 'subset_samples' in args:
        num_classes = int(args.pop('subset_classes'))
        num_samples = int(args.pop('subset_samples'))
        return type(**args).subset(num_classes, num_samples)

    return type(**args)


def dataloader_from_config(dataset: Dataset, config: dict[str, Any]) -> DataLoader:
    """
    Initializes a DataLoader for a given dataset and configuration.

    Args:
        dataset (Dataset): The dataset for which the DataLoader will be created.
        config (dict[str, Any]): A dictionary containing DataLoader arguments.
            Example format:
            {
                "batch_size": 64,
                "shuffle": True,
                "num_workers": 4
            }

    Returns:
        DataLoader: An instance of the DataLoader.

    Example:
        dataset = datasets.CIFAR10(...)
        config = {
            "batch_size": 64,
            "shuffle": True,
            "num_workers": 4
        }
        dataloader = init_dataloader(dataset, config)
    """
    return DataLoader(dataset=dataset, **config['args'])


def model_from_config(config: dict[str, Any]) -> Module:
    # TODO: Correct docstring
    """
    Initializes a PyTorch model based on the configuration provided.

    Args:
        config (dict[str, Any]): A dictionary containing the model type and its arguments.
            Example format:
            {
                "module": "torchvision.models.resnet18",
                "weights": "torchvision.models.resnet.ResNet18_Weights.DEFAULT",
                "args": {}
            }

    Returns:
        Module: An instance of the PyTorch model.

    Example:
        config = {
            "module": "torchvision.models.resnet18",
            "weights": "torchvision.models.resnet.ResNet18_Weights.DEFAULT",
            "args": {}
        }
        model = init_model(config)
    """
    args = config['args'].copy() # Kopiuj, aby uniknąć modyfikacji oryginalnego configu przez pop
    try:
        weights = config['weights']
    except KeyError:
        weights = None

    state_dict = None # Inicjalizuj na None
    if weights is not None:
        model_path = os.path.join(MODEL_OUTPUT_PATH, f'{weights}.pth')
        if os.path.exists(model_path):
            state_dict = torch.load(model_path, map_location='cpu')
             # Dostosuj, jeśli state_dict jest zagnieżdżony
            if isinstance(state_dict, dict) and 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']
        else:
            print(f"Warning: Weights file not found at {model_path}")
            weights = None # Traktuj jak brak wag, jeśli plik nie istnieje

    model_package, model_module = config['module'].rsplit('.', 1)
    model_package = importlib.import_module(model_package)
    model_type = getattr(model_package, model_module)

    # Sprawdź, czy 'pretrained_encoder' jest w args przed użyciem pop
    pretrained_encoder = bool(args.pop('pretrained_encoder', False)) # Domyślnie False jeśli brak
    args['encoder_pretrained'] = pretrained_encoder # Czy ta flaga jest nadal potrzebna w konstruktorze?

    if pretrained_encoder:
        # Jeśli 'pretrained_encoder' jest True, zakładamy, że model sam ładuje wagi w __init__
        # lub jest to obsłużone inaczej (np. przez bibliotekę jak timm)
        print("Initializing model with pretrained encoder flag set to True (external handling assumed).")
        return model_type(**args)
    elif weights is not None and state_dict is not None:
        # Tworzymy model bez wczytywania wag w konstruktorze (zakładając, że args tego nie robią)
        model: Module = model_type(**args)
        print(f"Loading weights from specified file: {weights}.pth")

        if weights.startswith('SWIN-T-IC_0.3'): # Specjalna obsługa dla konkretnych wag
            print("Applying specific mapping for SWIN-T-IC_0.3 weights...")
            mapped = {
                k.replace("encoder", "g_a")
                .replace("decoder", "g_s")
                .replace("stage1", "stages.0")
                .replace("stage2", "stages.1")
                .replace("stage3", "stages.2")
                .replace("stage4", "stages.3"): v
                for k, v in state_dict.items()
                if k.startswith("encoder") or k.startswith("decoder")
            }
            model.load_state_dict(mapped, strict=False)
            print("Loaded SWIN-T-IC_0.3 weights with specific mapping.")
        else:
            # Użyj standardowego mapowania za pomocą key_mapping_dict
            print("Applying general mapping using key_mapping_dict...")
            mapped_state_dict = {}
            # Pobierz state_dict nowego modelu raz, aby uniknąć wielokrotnego wywoływania
            new_model_state_dict = model.state_dict()

            for old_key, old_weight in state_dict.items():
                if old_key in key_mapping_dict:
                    new_key = key_mapping_dict[old_key]

                    # --- POPRAWIONY FRAGMENT ---
                    # Sprawdź, czy zmapowany klucz istnieje w state_dict nowego modelu
                    # i porównaj kształty tensorów
                    if new_key in new_model_state_dict:
                        new_weight_shape = new_model_state_dict[new_key].shape
                        if old_weight.shape == new_weight_shape:
                            mapped_state_dict[new_key] = old_weight
                        else:
                            print(f"Shape mismatch! Old key: {old_key} ({old_weight.shape}), "
                                  f"Mapped New key: {new_key} ({new_weight_shape}). Skipping.")
                    else:
                        print(f"Warning: Mapped key '{new_key}' (from old key '{old_key}') not found in the new model's state_dict.")
                    # --- KONIEC POPRAWIONEGO FRAGMENTU ---
                else:
                    mapped_state_dict[old_key] = old_weight

            filtered_state_dict = {}

            for new_key, new_weight in mapped_state_dict.items():
                if new_key not in new_model_state_dict.keys():
                    continue
                elif new_model_state_dict[new_key].shape != new_weight.shape:
                    print(f"Shape mismatch! Key: {new_key} ({new_weight.shape}). Skipping.")
                else:
                    filtered_state_dict[new_key] = new_weight

            # Wczytaj zmapowane wagi do nowego modelu
            missing_keys, unexpected_keys = model.load_state_dict(filtered_state_dict, strict=False)

            print("--- Weight loading process finished ---")
            print(f"Number of keys in mapped state_dict: {len(mapped_state_dict)}")
            if missing_keys:
                print(f"Missing keys ({len(missing_keys)}): {missing_keys[:10]}...") # Pokaż pierwsze 10
                all_missing_are_gdn = all("gdn" in key for key in missing_keys)
                print(f"All missing keys are from GDN layers: {all_missing_are_gdn}")
                if not all_missing_are_gdn:
                    missing_keys = list(filter(lambda key: "gdn" not in key, missing_keys))
                    print(f"Missing keys ({len(missing_keys)}): {missing_keys}") # Pokaż pierwsze 10
            if unexpected_keys:
                 print(f"Unexpected keys ({len(unexpected_keys)}): {unexpected_keys}") # Powinno być puste przy mapowaniu

        return model
    else:
        # Inicjalizuj model bez wczytywania wag (użyje domyślnej inicjalizacji PyTorch)
        print("Initializing model with default random weights (no weights file specified or found, or pretrained_encoder=False).")
        return model_type(**args)


def optimizer_from_config(parameters: ParamsT, config: dict[str, Any]) -> Optimizer:
    """
    Initializes an optimizer based on the configuration provided for a given set of parameters.
    
    Args:
        parameters (ParamsT): The parameters to optimize. It can be:
            - An iterable of torch.Tensor objects (model.parameters())
            - An iterable of dictionaries with parameter groups and optional settings
            - An iterable of tuples where keys are strings and values are torch.Tensor objects.
        config (dict[str, Any]): A dictionary containing the optimizer type and its arguments.
        Example format:
        {
            "module": "torch.optim.SGD",
            "args": {
                "lr": 0.01,
                "momentum": 0.9,
                "weight_decay": 0.0005
            }
        }

    Returns:
        Optimizer: An instance of the optimizer.

    Example:
        model = torchvision.models.resnet18(weights=torchvision.models.resnet.ResNet18_Weights.DEFAULT)
        config = {
            "module": "torch.optim.SGD",
            "args": {
                "lr": 0.01,
                "momentum": 0.9,
                "weight_decay": 0.0005
            }
        }
        init_optimizer(parameters=model.parameters(), config=config)
    """
    args = config['args']
    package, module = config['module'].rsplit('.', 1)
    package = importlib.import_module(package)
    type = getattr(package, module)
    SophiaG

    return type(params=parameters, **args)


def scheduler_from_config(optimizer: Optimizer, config: dict[str, Any]) -> LRScheduler | None:
    """
    Initializes a learning rate scheduler based on the configuration provided.

    Args:
        optimizer (Optimizer): The optimizer for which to schedule the learning rate.
        config (dict[str, Any]): A dictionary containing the scheduler type and its arguments.
            Example format:
            {
                "module": "torch.optim.lr_scheduler.CyclicLR",
                "args": {
                    "base_lr": 1e-3,
                    "max_lr": 1e-2,
                    "gamma": 0.9,
                }
            }
    Returns:
        LRScheduler: An instance of the learning rate scheduler.

    Example:
        model = torchvision.models.resnet18(weights=torchvision.models.resnet.ResNet18_Weights.DEFAULT)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        config = {
            "module": "torch.optim.lr_scheduler.CyclicLR",
            "args": {
                "base_lr": 1e-3,
                "max_lr": 1e-2,
                "gamma": 0.9,
            }
        }
        scheduler = init_scheduler(optimizer, config)
    """
    try:
        enabled = bool(config['enabled'])
    except KeyError:
        enabled = False
    if not enabled:
        return None
    args = config['args']
    package, module = config['module'].rsplit('.', 1)
    package = importlib.import_module(package)
    type = getattr(package, module)
    return type(optimizer=optimizer, **args)


def loss_from_config(config: dict[str, Any]) -> Module:
    args = config['args']
    package, module = config['module'].rsplit('.', 1)
    package = importlib.import_module(package)
    type = getattr(package, module)
    return type(**args)

def rd_loss_wrapper_from_config(loss_function: Module, config: dict[str, Any]) -> Module:
    l = config['lambda']
    return RDLoss(distortion_loss=loss_function, l=l)

def initialize_weights(module):
    for m in module.modules():
        if isinstance(m, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.LayerNorm, nn.GroupNorm)):
            if m.weight is not None:
                init.constant_(m.weight, 1)
            if m.bias is not None:
                init.constant_(m.bias, 0)
        elif isinstance(m, nn.Embedding):
             init.normal_(m.weight, mean=0, std=0.02)
