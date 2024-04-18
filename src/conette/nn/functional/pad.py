#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Iterable

import torch
from torch import Tensor
from torchoutil.nn.functional.pad import pad_dim


def pad_and_stack(x: Iterable[Tensor], dim: int = -1) -> Tensor:
    if isinstance(x, Tensor):
        return x
    max_len = max(xi.shape[dim] for xi in x)
    x = [pad_dim(xi, max_len, dim=dim) for xi in x]
    x = torch.stack(x, dim=0)
    return x
