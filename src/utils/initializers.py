import importlib
import os
from copy import deepcopy
from typing import Any, TypeAlias, Union, Iterable

import torch
import yaml
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

def flatten_collate(batch):

    num_groups = len(batch[0][0])
    patch_batch_size = len(batch)
    drop_last = False
    batch_samples = []
    batch_targets = []

    for g in range(num_groups):
        samples_list = [item[0][g] for item in batch]  # [N_i, C, H, W]
        targets_list = [item[1][g] for item in batch]  # [N_i, C, H, W]

        samples_cat = torch.cat(samples_list, dim=0)
        targets_cat = torch.cat(targets_list, dim=0)

        T = samples_cat.shape[0]
        group_samples = []
        group_targets = []

        for start in range(0, T, patch_batch_size):
            end = start + patch_batch_size
            if end > T and drop_last:
                break
            s_chunk = samples_cat[start:end]  # [k, C, H, W], k<=patch_batch_size
            t_chunk = targets_cat[start:end]  # [k, C, H, W]

            group_samples.append(s_chunk)
            group_targets.append(t_chunk)

        batch_samples.append(group_samples)
        batch_targets.append(group_targets)

    return batch_samples, batch_targets

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


def dataloader_from_config(dataset: BaseDataset, config: dict[str, Any]) -> DataLoader:
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
    # if dataset.train:
    #     return DataLoader(dataset=dataset, collate_fn=flatten_collate, **config['args'])
    # else:
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
    args = config['args'].copy()
    try:
        weights = config['weights']
    except KeyError:
        weights = None

    state_dict = None
    if weights is not None:
        model_path = os.path.join(MODEL_OUTPUT_PATH, f'{weights}.pth')
        if os.path.exists(model_path):
            state_dict = torch.load(model_path, map_location='cpu')
            if isinstance(state_dict, dict) and 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']
        else:
            print(f"Warning: Weights file not found at {model_path}")
            weights = None

    model_package, model_module = config['module'].rsplit('.', 1)
    model_package = importlib.import_module(model_package)
    model_type = getattr(model_package, model_module)

    pretrained_encoder = bool(args.pop('pretrained_encoder', False)) #
    args['encoder_pretrained'] = pretrained_encoder

    if pretrained_encoder:
        print("Initializing model with pretrained encoder flag set to True (external handling assumed).")
        return model_type(**args)
    elif weights is not None and state_dict is not None:
        model: Module = model_type(**args)
        print(f"Loading weights from specified file: {weights}.pth")


        # filtered_state_dict = {}
        #
        # for key, weight in state_dict.items():
        #     if not "latent_codec" in key:
        #         filtered_state_dict[key] = weight
        # print("Loaded filtered weights.")
        # model.load_state_dict(filtered_state_dict, strict=False)
        # filtered_state_dict = {
        #     k: v for k, v in state_dict.items() if 'latent_codec.y' not in k
        # }
        #
        # # Opcjonalnie pokaż usunięte klucze
        # removed_keys = set(state_dict.keys()) - set(filtered_state_dict.keys())
        # print("Removed quantiles keys:")
        # for key in removed_keys:
        #     print(f" - {key}")
        #
        # # Załaduj do modelu
        model.load_state_dict(state_dict, strict=False)
        # model.load_state_dict(state_dict, strict=False)
        return model
    else:
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
            init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='selu')
            if m.bias is not None:
                init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.LayerNorm, nn.GroupNorm)):
            if m.weight is not None:
                init.constant_(m.weight, 1)
            if m.bias is not None:
                init.constant_(m.bias, 0)
        elif isinstance(m, nn.Embedding):
             init.normal_(m.weight, mean=0, std=0.02)
