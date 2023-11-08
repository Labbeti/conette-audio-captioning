#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Callable, Optional, Union

import torch

from torch import nn, Tensor
from torch.nn import functional as F


ACTIVATIONS = ("relu", "gelu")


def get_activation_module(name: str) -> nn.Module:
    if name == "relu":
        activation = nn.ReLU(inplace=True)
    elif name == "gelu":
        activation = nn.GELU()
    else:
        raise ValueError(f"Invalid argument {name=}. (expected one of {ACTIVATIONS})")
    return activation


def get_activation_fn(name: str) -> Callable[[Tensor], Tensor]:
    if name == "relu":
        return F.relu
    elif name == "gelu":
        return F.gelu
    else:
        raise ValueError(f"Invalid argument {name=}. (expected one of {ACTIVATIONS})")


def get_device(
    device: Union[str, torch.device, None] = "auto"
) -> Optional[torch.device]:
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if isinstance(device, str):
        device = torch.device(device)
    return device


def get_device_name(
    device_name: Union[str, torch.device, None] = "auto"
) -> Optional[str]:
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    if isinstance(device_name, torch.device):
        device_name = f"{device_name.type}:{device_name.index}"
    return device_name
