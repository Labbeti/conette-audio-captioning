#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
import random

from typing import Tuple, Union

from torch import nn, Tensor

from conette.nn.modules.crop import CropDim
from conette.nn.modules.pad import PadDim
from conette.transforms.resample import ResampleNearest


class SpeedPerturbation(nn.Module):
    def __init__(
        self,
        rates: Tuple[float, float] = (0.9, 1.1),
        target_length: Union[int, str, None] = "same",
        align: str = "random",
        fill_value: float = 0.0,
        dim: int = -1,
        p: float = 1.0,
    ) -> None:
        """Resample, Pad and Crop a signal.

        :param rates: The ratio of the signal used for resize. defaults to (0.9, 1.1).
        :param target_length: Optional target length of the signal dimension.
                If 'same', the output will have the same shape than the input.
                defaults to "same".
        :param align: Alignment to use for cropping and padding. Can be 'left', 'right', 'center' or 'random'.
                defaults to "random".
        :param fill_value: The fill value when padding the waveform. defaults to 0.0.
        :param dim: The dimension to stretch and pad or crop. defaults to -1.
        :param p: The probability to apply the transform. defaults to 1.0.
        """
        assert 0.0 <= p
        rates = tuple(rates)  #  type: ignore

        super().__init__()
        self.rates = rates
        self._target_length = target_length
        self.align = align
        self.fill_value = fill_value
        self.dim = dim
        self.p = p

        target_length = self.target_length if isinstance(self.target_length, int) else 1
        self.resampler = ResampleNearest(rates, dim=dim)
        self.pad = PadDim(target_length, align, fill_value, dim, mode="constant")
        self.crop = CropDim(target_length, align, dim)

    # nn.Module methods
    def extra_repr(self) -> str:
        hparams = {
            "rates": self.rates,
            "target_length": self.target_length,
            "align": self.align,
            "fill_value": self.fill_value,
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
        if self.target_length == "same":
            target_length = x.shape[self.dim]
            self.pad.target_length = target_length
            self.crop.target_length = target_length

        x = self.resampler(x)

        if self.target_length is not None:
            x = self.pad(x)
            x = self.crop(x)
        return x

    @property
    def target_length(self) -> Union[int, str, None]:
        return self._target_length
