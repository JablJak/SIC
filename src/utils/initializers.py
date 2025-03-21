import importlib
import os
from typing import Any, TypeAlias, Union, Iterable

import torch
import yaml
from torchvision import transforms
from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from src.utils.const import MODEL_OUTPUT_PATH

# Based on pyTorch implementation
ParamsT: TypeAlias = Union[
    Iterable[torch.Tensor], Iterable[dict[str, Any]], Iterable[tuple[str, torch.Tensor]]
]

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

def dataset_from_config(config: dict[str, Any]) -> Dataset:
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
    args = config['args']
    package, module = config['module'].rsplit('.', 1)
    package = importlib.import_module(package)
    type = getattr(package, module)
    if 'transform' in config.keys():
        tsfs = transforms.Compose([transform_from_config(tsf) for tsf in config['transform']])
        args['transform'] = tsfs
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
    args = config['args']
    try:
        weights = config['weights']
    except KeyError:
        weights = None

    if weights is not None:
        model_path = os.path.join(MODEL_OUTPUT_PATH, f'{weights}.pth')
        state_dict = torch.load(model_path)

    model_package, model_module = config['module'].rsplit('.', 1)
    model_package = importlib.import_module(model_package)
    model_type = getattr(model_package, model_module)
    pretrained_encoder = bool(args.pop('pretrained_encoder'))
    if pretrained_encoder:
        return model_type(**args)
    elif weights is not None:
        model = model_type(**args)
        model.load_state_dict(state_dict)
        return model
    else:
        return model_type(encoder_weights=None, **args)


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