import importlib
from typing import Any, TypeAlias, Union, Iterable

import torch
import yaml
from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet18_Weights, Swin_V2_T_Weights

import src.models.swin_autoencoder

# Based on PyTorch implementation
ParamsT: TypeAlias = Union[
    Iterable[torch.Tensor],
    Iterable[dict[str, Any]],
    Iterable[tuple[str, torch.Tensor]]
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
    with open(config_file_path, 'r') as f:
        data = yaml.safe_load(f)
    return data


def dataset_from_config(config: dict[str, Any]) -> Dataset:
    """
    Initializes a Dataset instance based on the configuration.

    The config must specify:
      - "module": ścieżka do klasy datasetu, np. "torchvision.datasets.CIFAR10"
      - "args": słownik argumentów konstruktora datasetu

    Args:
        config (dict[str, Any]): A dictionary containing at least:
            {
                "module": "package.subpackage.DatasetClass",
                "args": {...}
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
        dataset = dataset_from_config(config)
    """
    module_path = config["module"]
    args = config.get("args", {})

    package_name, class_name = module_path.rsplit('.', 1)
    package = importlib.import_module(package_name)
    dataset_cls = getattr(package, class_name)

    return dataset_cls(**args)


def dataloader_from_config(dataset: Dataset, config: dict[str, Any]) -> DataLoader:
    """
    Initializes a DataLoader instance for a given dataset and configuration.

    The config must specify:
      - "args": słownik argumentów, np. {"batch_size": 64, "shuffle": True, "num_workers": 4}

    Args:
        dataset (Dataset): The dataset object.
        config (dict[str, Any]): A dictionary containing at least:
            {
                "args": {...}
            }

    Returns:
        DataLoader: A DataLoader instance.

    Example:
        dataset = torchvision.datasets.CIFAR10(...)
        config = {
            "args": {
                "batch_size": 64,
                "shuffle": True,
                "num_workers": 4
            }
        }
        dataloader = dataloader_from_config(dataset, config)
    """
    args = config.get("args", {})
    return DataLoader(dataset=dataset, **args)


def model_from_config(config: dict[str, Any]) -> Module:
    """
    Initializes a PyTorch model based on the configuration.

    The config must specify:
      - "module": ścieżka do klasy modelu, np. "torchvision.models.resnet18"
      - optionally "weights": jeśli chcemy użyć gotowych wag (np. z torchvision)
      - "args": słownik argumentów konstruktora (parametry modelu)

    Uwaga: Jeśli model nie obsługuje wbudowanego parametru `weights`,
    a w configu jest on ustawiony, może dojść do błędu w trakcie inicjalizacji.
    W przypadku własnych modeli należy zadbać o to, aby nazwy argumentów pasowały
    do definicji klasy modelu.

    Args:
        config (dict[str, Any]): A dictionary, for example:
            {
                "module": "torchvision.models.resnet18",
                "weights": "torchvision.models.resnet.ResNet18_Weights.DEFAULT",
                "args": {
                    "num_classes": 10
                }
            }

    Returns:
        Module: A PyTorch model instance.

    Example:
        # Model z wbudowanymi wagami z torchvision
        config = {
            "module": "torchvision.models.resnet18",
            "weights": "torchvision.models.resnet.ResNet18_Weights.DEFAULT",
            "args": {
                "num_classes": 10
            }
        }
        model = model_from_config(config)

        # Własny model bez wag pretrained
        config = {
            "module": "my_project.models.custom_model.CustomNet",
            "args": {
                "hidden_size": 128
            }
        }
        model = model_from_config(config)
    """
    module_path = config["module"]
    weights = config.get("weights", None)
    args = config.get("args", {})

    package_name, class_name = module_path.rsplit('.', 1)
    package = importlib.import_module(package_name)
    model_cls = getattr(package, class_name)

    if weights is not None:
        # Jeśli klasa modelu obsługuje argument 'weights', możemy go przekazać:
        return model_cls(weights=weights, **args)
    else:
        # Brak zdefiniowanych wag - zwykła inicjalizacja
        return model_cls(**args)


def optimizer_from_config(parameters: ParamsT, config: dict[str, Any]) -> Optimizer:
    """
    Initializes an optimizer for a given set of parameters based on the configuration.

    The config must specify:
      - "module": ścieżka do klasy optimizera, np. "torch.optim.SGD"
      - "args": słownik argumentów konstruktora (lr, momentum, weight_decay, itp.)

    Args:
        parameters (ParamsT): The parameters to optimize (e.g. model.parameters()).
        config (dict[str, Any]): For example:
            {
                "module": "torch.optim.SGD",
                "args": {
                    "lr": 0.01,
                    "momentum": 0.9,
                    "weight_decay": 0.0005
                }
            }

    Returns:
        Optimizer: A PyTorch optimizer instance.

    Example:
        model = torchvision.models.resnet18(...)
        config = {
            "module": "torch.optim.SGD",
            "args": {
                "lr": 0.01,
                "momentum": 0.9,
                "weight_decay": 0.0005
            }
        }
        optimizer = optimizer_from_config(model.parameters(), config)
    """
    module_path = config["module"]
    args = config.get("args", {})

    package_name, class_name = module_path.rsplit('.', 1)
    package = importlib.import_module(package_name)
    optimizer_cls = getattr(package, class_name)

    return optimizer_cls(params=parameters, **args)


def scheduler_from_config(optimizer: Optimizer, config: dict[str, Any]) -> LRScheduler:
    """
    Initializes a learning rate scheduler based on the configuration.

    The config must specify:
      - "module": ścieżka do klasy scheduler, np. "torch.optim.lr_scheduler.CyclicLR"
      - "args": słownik argumentów konstruktora (base_lr, max_lr, gamma, itp.)

    Args:
        optimizer (Optimizer): The optimizer whose learning rate will be scheduled.
        config (dict[str, Any]): For example:
            {
                "module": "torch.optim.lr_scheduler.CyclicLR",
                "args": {
                    "base_lr": 1e-3,
                    "max_lr": 1e-2,
                    "gamma": 0.9,
                }
            }

    Returns:
        LRScheduler: A PyTorch LR scheduler instance.

    Example:
        model = torchvision.models.resnet18(...)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        config = {
            "module": "torch.optim.lr_scheduler.CyclicLR",
            "args": {
                "base_lr": 1e-3,
                "max_lr": 1e-2,
                "gamma": 0.9,
            }
        }
        scheduler = scheduler_from_config(optimizer, config)
    """
    module_path = config["module"]
    args = config.get("args", {})

    package_name, class_name = module_path.rsplit('.', 1)
    package = importlib.import_module(package_name)
    scheduler_cls = getattr(package, class_name)

    return scheduler_cls(optimizer=optimizer, **args)


# ----- PRZYKŁAD UŻYCIA -----
# config = {
#     "module": "torchvision.models.resnet18",
#     "weights": "torchvision.models.resnet.ResNet18_Weights.DEFAULT",
#     "args": {}
# }
# model = model_from_config(config)
config = {
    "module": "src.models.swin_autoencoder.SwinTransformerAutoencoder",
    "args": {
        "encoder_weights": "Swin_V2_T_Weights.DEFAULT",
    }
}
model = model_from_config(config)
print(model.state_dict())
Swin_V2_T_Weights.DEFAULT
