#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
import random

from typing import Any, List, Tuple

import torch

from torch import nn, Tensor
from torch.distributions import Uniform


class ResampleNearest(nn.Module):
    def __init__(
        self,
        rates: Tuple[float, float] = (0.5, 1.5),
        dim: int = -1,
        p: float = 1.0,
    ) -> None:
        """Resample an audio waveform signal.

        :param rates: The rate of the stretch. Ex: use 2.0 for multiply the signal length by 2. (default: (0.5, 1.5))
        :param dim: The dimension to modify. (default: -1)
        :param p: The probability to apply the transform. (default: 1.0)
        """
        assert 0.0 <= p
        super().__init__()
        self.rates = rates
        self.dim = dim
        self.p = p

    # nn.Module methods
    def extra_repr(self) -> str:
        hparams = {
            "rates": self.rates,
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
        if self.rates[0] == self.rates[1]:
            rate = self.rates[0]
        else:
            sampler = Uniform(*self.rates)
            rate = sampler.sample().item()

        x = self._resample_nearest(x, rate)
        return x

    def _resample_nearest(self, x: Tensor, rate: float) -> Tensor:
        length = x.shape[self.dim]
        step = 1.0 / rate
        indexes = torch.arange(0, length, step)
        indexes = indexes.round().long().clamp(max=length - 1)
        slices: List[Any] = [slice(None)] * len(x.shape)
        slices[self.dim] = indexes
        x = x[slices]
        x = x.contiguous()
        return x
