#!/usr/bin/env python
# -*- coding: utf-8 -*-

import copy

from typing import (
    Iterable,
    TypeVar,
    Union,
)

import torch

from torch import nn, Tensor
from torch.nn import functional as F
from torchoutil.nn.functional import (
    crop_dim,
    tensor_to_pad_mask,
    pad_dim,
)


T = TypeVar("T")


def count_params(model: nn.Module, only_trainable: bool = False) -> int:
    return sum(
        param.numel()
        for param in model.parameters()
        if not only_trainable or param.requires_grad
    )
