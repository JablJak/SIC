from typing import Any, TypeAlias, Union, Iterable

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

# Based on pyTorch implementation
ParamsT: TypeAlias = Union[
    Iterable[torch.Tensor], Iterable[dict[str, Any]], Iterable[tuple[str, torch.Tensor]]
]


def init_optimizer(parameters: ParamsT, config: dict[str, Any], optimizer_module=torch.optim) -> Optimizer:
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
                "type": "SGD",  # Optimizer class name as a string
                "args": {  # Arguments to pass to the optimizer
                    lr=0.01,
                    momentum=0.9,
                    weight_decay=0.0005
                }
            }
        optimizer_module (module, optional): The module containing the optimizer class. Defaults to torch.optim.

    Returns:
        Optimizer: An instance of the optimizer.

    Example:
        model = torchvision.models.resnet18(pretrained=True)
        config = {
            "type": "SGD",
            "args": {
                "lr": 0.01,
                "momentum": 0.9,
                "weight_decay": 0.0005
            }
        }
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        init_optimizer(parameters=model.parameters(), config=config)
    """
    optimizer_type = config['type']
    args = config['args']
    optimizer_type = getattr(optimizer_module, optimizer_type)
    return optimizer_type(parameters=parameters, **args)


def init_scheduler(optimizer: Optimizer, config: dict[str, Any],
                   scheduler_module=torch.optim.lr_scheduler) -> LRScheduler:
    """
    Initializes a learning rate scheduler based on the configuration provided.

    Args:
        optimizer (Optimizer): The optimizer for which to schedule the learning rate.
        config (dict[str, Any]): A dictionary containing the scheduler type and its arguments.
            Example format:
            {
                "type": "CyclicLR",  # Scheduler class name as a string
                "args": {  # Arguments to pass to the scheduler
                    "base_lr": 1e-3,
                    "max_lr": 1e-2,
                    "gamma": 0.9,
                }
            }
        scheduler_module (module, optional): The module containing the scheduler class. Defaults to torch.optim.lr_scheduler.
    Returns:
        LRScheduler: An instance of the learning rate scheduler.

    Example:
        model = torchvision.models.resnet18(pretrained=True)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        config = {
            "type": "CyclicLR",
            "args": {
                "base_lr": 1e-3,
                "max_lr": 1e-2,
                "gamma": 0.9,
            }
        }
        scheduler = init_scheduler(optimizer, config)
    """
    scheduler_type = config['type']
    args = config['args']
    scheduler_type = getattr(scheduler_module, scheduler_type)
    return scheduler_type(optimizer=optimizer, **args)
