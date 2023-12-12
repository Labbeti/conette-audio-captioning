#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
import random

from typing import Iterable, Optional, Union

import torch

from torch import nn, Tensor
from torch.distributions import Uniform


class CutOutSpec(nn.Module):
    def __init__(
        self,
        freq_size_range: Union[tuple[float, float], tuple[int, int]] = (0.1, 0.5),
        time_size_range: Union[tuple[float, float], tuple[int, int]] = (0.1, 0.5),
        fill_value: Union[float, tuple[float, float]] = -100.0,
        fill_mode: Union[str, nn.Module] = "constant",
        freq_dim: int = -1,
        time_dim: int = -2,
        p: float = 1.0,
    ) -> None:
        """
        CutOut transform for spectrogram PyTorch tensors.

        Input must be of shape (..., freq, time), but you can specify frequency and time dimension dim/axis.

        Example 1
        ----------
        >>> from aac.transforms.augments.cutoutspec import CutOutSpec
        >>> spectrogram = torch.rand((32, 1, 160, 64))
        >>> augment = CutOutSpec((0.5, 0.5), (0.5, 0.5))
        >>> # Remove 25% of the spectrogram values in a squared area
        >>> spectrogram_augmented = augment(spectrogram)

        :param freq_size_range: The range of ratios for the frequencies dim. defaults to (0.1, 0.5).
        :param time_size_range: The range of ratios for the time steps dim. defaults to (0.1, 0.5).
        :param fill_value: The value used for fill. Can be a range of values for sampling the fill value. defaults to -100.0.
                This parameter is ignored if fill_mode is a custom Module.
        :param fill_mode: The fill mode. defaults to 'constant'.
                Can be 'constant', 'random' or a custom transform for the data delimited by the rectange.
        :param freq_dim: The dimension index of the spectrogram frequencies. defaults to -1.
        :param time_dim: The dimension index of the spectrogram time steps. defaults to -2.
        :param p: The probability to apply the transform. default to 1.0.
        """
        assert 0.0 <= p <= 1.0
        super().__init__()

        self.freq_size_range = freq_size_range
        self.time_size_range = time_size_range
        self.fill_value = fill_value
        self.fill_mode = fill_mode
        self.freq_dim = freq_dim
        self.time_dim = time_dim
        self.p = p

        self._check_attributes()

    # nn.Module methods
    def extra_repr(self) -> str:
        hparams = {
            "freq_size_range": self.freq_size_range,
            "time_size_range": self.time_size_range,
            "fill_value": self.fill_value,
            "fill_mode": self.fill_mode,
            "freq_dim": self.freq_dim,
            "time_dim": self.time_dim,
            "p": self.p,
        }
        return ", ".join(f"{k}={v}" for k, v in hparams.items())

    def forward(self, x: Tensor) -> Tensor:
        if self.p >= 1.0 or random.random() < self.p:
            return self.apply_transform(x)
        else:
            return x

    # Other methods
    def apply_transform(self, data: Tensor) -> Tensor:
        if not isinstance(data, Tensor) or data.ndim < 2:
            raise ValueError(
                f"Input data must be a PyTorch Tensor with at least 2 dimensions for {self.__class__.__name__} transform, "
                f"found {type(data)}"
                + (f" of shape {data.shape}" if hasattr(data, "shape") else "")
                + "."
            )

        # Prepare slices indexes for frequencies and time dimensions
        slices = [slice(None)] * data.ndim
        slices[self.freq_dim] = gen_range(
            data.shape[self.freq_dim], self.freq_size_range
        )
        slices[self.time_dim] = gen_range(
            data.shape[self.time_dim], self.time_size_range
        )

        if self.fill_mode == "constant":
            data[slices] = self._gen_constant(data[slices])

        elif self.fill_mode == "random":
            data[slices] = self._gen_random(data[slices])

        elif isinstance(self.fill_mode, nn.Module):
            data[slices] = self.fill_mode(data[slices])

        else:
            raise ValueError(
                f'Invalid fill_mode "{self.fill_mode}". '
                f'Must be one of "{("constant", "random")}" or a custom transform Module.'
            )

        return data

    def _gen_constant(self, data: Tensor) -> Tensor:
        if isinstance(self.fill_value, float):
            fill_value = self.fill_value
        else:
            uniform = Uniform(*self.fill_value)  # type: ignore
            fill_value = uniform.sample()
        return torch.full_like(data, fill_value)

    def _gen_random(self, data: Tensor) -> Tensor:
        if isinstance(self.fill_value, float):
            raise ValueError(
                "Invalid fill_value with random fill_mode. Please use a tuple of 2 floats for fill_value or use "
                'fill_mode="constant".'
            )
        else:
            uniform = Uniform(*self.fill_value)  # type: ignore
            return uniform.sample(data.shape)

    def _check_attributes(self) -> None:
        if self.freq_dim == self.time_dim:
            raise ValueError(
                "Frequency dimension index cannot be the same than time dimension index."
            )

        if not isinstance(self.fill_value, float) and not (
            isinstance(self.fill_value, tuple) and len(self.fill_value) == 2
        ):
            raise ValueError(
                f'Invalid fill_value "{self.fill_value}", must be a float or a tuple of 2 floats.'
            )

        if self.fill_mode == "random" and isinstance(self.fill_value, float):
            raise ValueError(
                "Invalid fill_value with random fill_mode. Please use a tuple of 2 floats for fill_value or use "
                'fill_mode="constant".'
            )


def gen_range(
    size: int,
    scales: Union[Iterable[float], Iterable[int]],
    generator: Optional[torch.Generator] = None,
) -> slice:
    """
    Generate an random range in [0, size].
    The position of the range is random.

    :param size: The size of the array.
    :param scales: The scales attributes defined the length of the range.
        If scales is (float, float), the int length will be sampled from [ ceil(size * scales[0]), ceil(size * scales[1]) ].
        If scales is (int, int), the int length will be sampled from scales.

    Example 1
    ----------
    >>> gen_range(size=100, scales=(0.5, 0.5))
    ... slice(10, 60)
    """
    if not isinstance(scales, Iterable):
        raise ValueError(
            f"Invalid argument {scales=}. (expected tuple[int, int] or tuple[float, float])"
        )
    scales = list(scales)
    if len(scales) != 2:
        raise ValueError(
            f"Invalid argument {scales=}. (expected tuple[int, int] or tuple[float, float])"
        )
    if not all(isinstance(s, float) for s in scales) and not all(
        isinstance(s, int) for s in scales
    ):
        raise ValueError(
            f"Invalid argument {scales=}. (expected tuple[int, int] or tuple[float, float])"
        )

    if isinstance(scales[0], float):
        cutout_size_min = math.ceil(scales[0] * size)
        cutout_size_max = max(math.ceil(scales[1] * size), cutout_size_min + 1)
    elif isinstance(scales[0], int):
        cutout_size_min: int = scales[0]  # type: ignore
        cutout_size_max: int = scales[1]  # type: ignore
    else:
        raise ValueError(
            f"Invalid argument {scales=}. (expected tuple[int, int] or tuple[float, float])"
        )

    cutout_size = int(
        torch.randint(cutout_size_min, cutout_size_max, (), generator=generator).item()
    )
    cutout_start = torch.randint(
        0, max(size - cutout_size + 1, 1), (), generator=generator
    )
    cutout_end = cutout_start + cutout_size
    assert (
        cutout_end - cutout_start == cutout_size
    ), f"{cutout_end} - {cutout_start} != {cutout_size}"

    return slice(cutout_start, cutout_end)
