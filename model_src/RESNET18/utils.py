import importlib
from torch.nn import Module
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
import torch.optim as optim
from torchvision import transforms

def get_criterion(name: str, criterion_params: dict) -> Module:
    """
    Dynamically get the loss function class from torch.nn and instantiate it.

    Parameters:
    - name (str): Name of the loss function (e.g., 'CrossEntropyLoss', 'MSELoss').
    - criterion_params (dict): Parameters for the loss function.

    Returns:
    - criterion (Module): Instantiated loss function.
    """
    criterion_class = getattr(nn, name, None)
    if criterion_class is None:
        raise ValueError(f"Loss function '{name}' is not found in torch.nn")
    return criterion_class(**criterion_params)


def get_optimizer(name: str, model_parameters, optimizer_params: dict) -> Optimizer:
    """
    Dynamically get the optimizer class from torch.optim and instantiate it.

    Parameters:
    - name (str): Name of the optimizer (e.g., 'SGD', 'Adam').
    - model_parameters: Parameters of the model to optimize.
    - optimizer_params (dict): Parameters for the optimizer.

    Returns:
    - optimizer (Optimizer): Instantiated optimizer.
    """
    optimizer_class = getattr(optim, name, None)
    if optimizer_class is None:
        raise ValueError(f"Optimizer '{name}' is not found in torch.optim")
    return optimizer_class(model_parameters, **optimizer_params)


def get_scheduler(name: str, optimizer: Optimizer, scheduler_params: dict) -> _LRScheduler:
    """
    Dynamically get the scheduler class from torch.optim.lr_scheduler and instantiate it.

    Parameters:
    - name (str): Name of the scheduler (e.g., 'StepLR', 'ReduceLROnPlateau').
    - optimizer (Optimizer): Optimizer to which the scheduler will be attached.
    - scheduler_params (dict): Parameters for the scheduler.

    Returns:
    - scheduler (_LRScheduler): Instantiated scheduler.
    """
    scheduler_class = getattr(optim.lr_scheduler, name, None)
    if scheduler_class is None:
        raise ValueError(f"Scheduler '{name}' is not found in torch.optim.lr_scheduler")
    return scheduler_class(optimizer, **scheduler_params)


def build_transforms(transform_configs):
    """
    Build a torchvision.transforms.Compose object from a list of transform configurations.

    Parameters:
    - transform_configs (list): A list of dictionaries, each containing 'name' and 'params' keys.

    Returns:
    - transforms.Compose: The composed transform.
    """
    transform_list = []
    for transform in transform_configs:
        transform_name = transform['name']
        params = transform.get('params', {})
        
        # Dynamically get the transform class from torchvision.transforms
        transform_class = getattr(transforms, transform_name, None)
        if transform_class is None:
            raise ValueError(f"Transform '{transform_name}' is not found in torchvision.transforms.")
        
        # Instantiate the transform with parameters
        try:
            transform_instance = transform_class(**params)
        except TypeError as e:
            raise ValueError(f"Error initializing transform '{transform_name}' with params {params}: {e}")
        
        transform_list.append(transform_instance)
    
    return transforms.Compose(transform_list)
