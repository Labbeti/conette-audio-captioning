#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Provides modules to use SpecAugment
BASED ON https://github.com/qiuqiangkong/sound_event_detection_dcase2017_task4
MODIFIED: Yes (typing, spectrogram reshape, add probability of specaugment)
"""

import random

from typing import Union

import torch

from torch import nn, Tensor


class DropStripes(nn.Module):
    def __init__(
        self,
        max_width: int,
        stripes_num: int,
        dim: int,
        fill_value: float = 0.0,
        inplace: bool = True,
        generator: Union[int, torch.Generator, None] = None,
    ) -> None:
        """Drop stripes.

        :param dim: int, dimension along which to drop
        :param drop_width: int, maximum width of stripes to drop
        :param stripes_num: int, how many stripes to drop
        :param fill_value: the value used to mask stripes
        """
        if max_width <= 0:
            raise ValueError(
                f"Invalid argument {max_width=} in {self.__class__.__name__}. (expected a value > 0)"
            )

        if isinstance(generator, int):
            generator = torch.Generator().manual_seed(generator)

        super().__init__()
        self.dim = dim
        self.max_width = max_width
        self.stripes_num = stripes_num
        self.fill_value = fill_value
        self.inplace = inplace
        self.generator = generator

    # nn.Module methods
    def extra_repr(self) -> str:
        hparams = {
            "dim": self.dim,
            "max_width": self.max_width,
            "stripes_num": self.stripes_num,
            "fill_value": self.fill_value,
        }
        return ", ".join(f"{k}={v}" for k, v in hparams.items())

    def forward(self, x: Tensor) -> Tensor:
        total_width = x.shape[self.dim]

        # Add: If audio is empty, do nothing
        if total_width == 0:
            return x

        # Add: If audio is shorter than self.drop_width, clip drop width.
        max_width = min(self.max_width, total_width)

        widths = torch.randint(
            low=0, high=max_width, size=(self.stripes_num,), generator=self.generator
        ).tolist()
        starts = [
            torch.randint(
                low=0, high=total_width - size, size=(), generator=self.generator
            )
            for size in widths
        ]

        if not self.inplace:
            x = x.clone()

        for width, start in zip(widths, starts):
            slices = [slice(None) for _ in range(x.ndim)]
            slices[self.dim] = slice(start, start + width)
            x[slices] = self.fill_value

        return x


class SpecAugment(nn.Module):
    def __init__(
        self,
        time_max_width: int,
        time_stripes_num: int,
        freq_max_width: int,
        freq_stripes_num: int,
        time_dim: int = -2,
        freq_dim: int = -1,
        fill_value: float = 0.0,
        p: float = 1.0,
    ) -> None:
        """Spec augmentation.
        [ref] Park, D.S., Chan, W., Zhang, Y., Chiu, C.C., Zoph, B., Cubuk, E.D.
        and Le, Q.V., 2019. Specaugment: A simple data augmentation method
        for automatic speech recognition. arXiv preprint arXiv:1904.08779.

        Args:
            time_drop_width: int
            time_stripes_num: int
            freq_drop_width: int
            freq_stripes_num: int
        """
        assert 0.0 <= p <= 1.0
        super().__init__()
        self.p = p

        self.time_dropper = DropStripes(
            max_width=time_max_width,
            stripes_num=time_stripes_num,
            dim=time_dim,
            fill_value=fill_value,
        )
        self.freq_dropper = DropStripes(
            max_width=freq_max_width,
            stripes_num=freq_stripes_num,
            dim=freq_dim,
            fill_value=fill_value,
        )

    # nn.Module methods
    def extra_repr(self) -> str:
        hparams = {
            "p": self.p,
        }
        return ", ".join(f"{k}={v}" for k, v in hparams.items())

    def forward(self, x: Tensor) -> Tensor:
        if self.p >= 1.0 or random.random() < self.p:
            return self.apply_transform(x)
        else:
            return x

    # Other methods
    def apply_transform(self, x: Tensor) -> Tensor:
        x = self.time_dropper(x)
        x = self.freq_dropper(x)
        return x


class DropStripesRatio(nn.Module):
    def __init__(
        self,
        ratios: tuple[float, float],
        stripes_num: int,
        dim: int,
        fill_value: float = 0.0,
        generator: Union[int, torch.Generator, None] = None,
        inplace: bool = True,
    ) -> None:
        if not (0.0 <= ratios[0] <= ratios[1] <= 1.0):
            raise ValueError(
                f"Invalid argument {ratios=}. (expected a tuple of two floats in [0, 1], with ratios[0] <= ratios[1])"
            )

        if isinstance(generator, int):
            generator = torch.Generator().manual_seed(generator)

        super().__init__()
        self.ratios = ratios
        self.stripes_num = stripes_num
        self.dim = dim
        self.fill_value = fill_value
        self.generator = generator
        self.inplace = inplace

    # nn.Module methods
    def extra_repr(self) -> str:
        hparams = {
            "dim": self.dim,
            "max_width": self.max_width,
            "stripes_num": self.stripes_num,
            "fill_value": self.fill_value,
        }
        return ", ".join(f"{k}={v}" for k, v in hparams.items())

    def forward(self, x: Tensor) -> Tensor:
        total_width = x.shape[self.dim]
        # If audio is empty, do nothing
        if total_width == 0:
            return x
        imin = round(total_width * self.ratios[0])
        imax = round(total_width * self.ratios[1])

        if imin > imax:
            return x
        elif imin == imax:
            widths = torch.full((self.stripes_num,), imin)
        else:
            widths = torch.randint(
                imin, imax, (self.stripes_num,), generator=self.generator
            )

        starts = [
            torch.randint(low=0, high=total_width - size, size=(), generator=self.generator) for size in widths  # type: ignore
        ]

        if not self.inplace:
            x = x.clone()

        for width, start in zip(widths, starts):
            slices = [slice(None) for _ in range(x.ndim)]
            slices[self.dim] = slice(start, start + width)
            x[slices] = self.fill_value

        return x


class SpecAugmentRatio(nn.Module):
    def __init__(
        self,
        time_ratios: tuple[float, float],
        time_stripes_num: int,
        freq_ratios: tuple[float, float],
        freq_stripes_num: int,
        time_dim: int = -2,
        freq_dim: int = -1,
        fill_value: float = 0.0,
        inplace: bool = True,
        p: float = 1.0,
    ) -> None:
        assert 0.0 <= p <= 1.0
        super().__init__()
        self.p = p

        self.time_dropper = DropStripesRatio(
            ratios=time_ratios,
            stripes_num=time_stripes_num,
            dim=time_dim,
            fill_value=fill_value,
            inplace=inplace,
        )
        self.freq_dropper = DropStripesRatio(
            ratios=freq_ratios,
            stripes_num=freq_stripes_num,
            dim=freq_dim,
            fill_value=fill_value,
            inplace=inplace,
        )

    def forward(self, x: Tensor) -> Tensor:
        if self.p >= 1.0 or random.random() < self.p:
            return self.apply_transform(x)
        else:
            return x

    def apply_transform(self, x: Tensor) -> Tensor:
        x = self.time_dropper(x)
        x = self.freq_dropper(x)
        return x
