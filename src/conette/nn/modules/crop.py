#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
import random

from torch import nn, Tensor

from conette.nn.functional.crop import crop_dim


class CropDim(nn.Module):
    def __init__(
        self,
        target_length: int,
        align: str = "left",
        dim: int = -1,
        p: float = 1.0,
    ) -> None:
        super().__init__()
        self.target_length = target_length
        self.align = align
        self.dim = dim
        self.p = p

    # nn.Module methods
    def extra_repr(self) -> str:
        hparams = {
            "target_length": self.target_length,
            "align": self.align,
            "dim": self.dim,
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
        return crop_dim(x, self.target_length, self.align, self.dim)
