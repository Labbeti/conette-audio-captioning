#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
import random

from torch import nn, Tensor

from conette.nn.functional.pad import (
    pad_dim,
)


class PadDim(nn.Module):
    ALIGNS = ("left", "right", "center", "random")

    def __init__(
        self,
        target_length: int,
        align: str = "left",
        fill_value: float = 0.0,
        dim: int = -1,
        mode: str = "constant",
        p: float = 1.0,
    ) -> None:
        """PadDim on tensor with alignment option.

        Example :

        >>> import torch; from torch import tensor
        >>> x = torch.ones(6)
        >>> zero_pad = Pad(10, align='right')
        >>> zero_pad(x)
        ... tensor([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])

        :param target_length: The target length of the dimension.
        :param align: The alignment type. Can be 'left', 'right', 'center' or 'random'. (default: 'left')
        :param fill_value: The fill value used for constant padding. (default: 0.0)
        :param dim: The dimension to pad. (default: -1)
        :param mode: The padding mode. Can be 'constant', 'reflect', 'replicate' or 'circular'. (default: 'constant')
        :param p: The probability to apply the transform. (default: 1.0)
        """
        if align not in PadDim.ALIGNS:
            raise ValueError(
                f"Invalid argument {align=}. (expected one of {PadDim.ALIGNS})"
            )

        super().__init__()
        self.target_length = target_length
        self.align = align
        self.fill_value = fill_value
        self.dim = dim
        self.mode = mode
        self.p = p

    # nn.Module methods
    def extra_repr(self) -> str:
        hparams = {
            "target_length": self.target_length,
            "align": self.align,
            "fill_value": self.fill_value,
            "dim": self.dim,
            "mode": self.mode,
            "p": self.p,
        }
        return ", ".join(f"{k}={v}" for k, v in hparams.items())

    def forward(self, x: Tensor) -> Tensor:
        floor_p = math.floor(self.p)
        for _ in range(floor_p):
            x = self.apply_transform(x)

        rest = self.p - floor_p
        if rest > 0.0 and rest < random.random():
            return self.apply_transform(x)
        else:
            return x

    # Other methods
    def apply_transform(self, x: Tensor) -> Tensor:
        """Apply the transform without taking into account the probability p."""
        return pad_dim(
            x, self.target_length, self.align, self.fill_value, self.dim, self.mode
        )
