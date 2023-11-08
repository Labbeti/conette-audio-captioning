#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Iterable

import torch

from torch import Tensor
from torch.nn import functional as F

PAD_ALIGNS = ("left", "right", "center", "random")


def pad_dim(
    x: Tensor,
    target_length: int,
    align: str = "left",
    fill_value: float = 0.0,
    dim: int = -1,
    mode: str = "constant",
) -> Tensor:
    """Generic function for pad a specific tensor dimension."""
    missing = max(target_length - x.shape[dim], 0)

    if missing == 0:
        return x

    if align == "left":
        missing_left = 0
        missing_right = missing
    elif align == "right":
        missing_left = missing
        missing_right = 0
    elif align == "center":
        missing_left = missing // 2 + missing % 2
        missing_right = missing // 2
    elif align == "random":
        missing_left = int(torch.randint(low=0, high=missing + 1, size=()).item())
        missing_right = missing - missing_left
    else:
        raise ValueError(f"Invalid argument {align=}. (expected one of {PAD_ALIGNS})")

    # Note: pad_seq : [pad_left_dim_-1, pad_right_dim_-1, pad_left_dim_-2, pad_right_dim_-2, ...)
    idx = len(x.shape) - (dim % len(x.shape)) - 1
    pad_seq = [0 for _ in range(len(x.shape) * 2)]
    pad_seq[idx * 2] = missing_left
    pad_seq[idx * 2 + 1] = missing_right
    x = F.pad(x, pad_seq, mode=mode, value=fill_value)
    return x


def pad_and_stack(x: Iterable[Tensor], dim: int = -1) -> Tensor:
    if isinstance(x, Tensor):
        return x
    max_len = max(xi.shape[dim] for xi in x)
    x = [pad_dim(xi, max_len, dim=dim) for xi in x]
    x = torch.stack(x, dim=0)
    return x
