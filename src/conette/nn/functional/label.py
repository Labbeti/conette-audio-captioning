#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Any, Iterable, Optional, Sequence, Union

import torch

from torch import Tensor


def multihots_to_ints(multihots: Tensor, tolist: bool = True) -> Union[list, Tensor]:
    """Convert multihot tensor to list or tensor of ints indexes."""

    if multihots.ndim == 0:
        raise ValueError(f"Invalid number of dimensions {multihots.ndim=}.")
    elif multihots.ndim == 1:
        arange = torch.arange(len(multihots), device=multihots.device)
        indexes = arange[multihots.bool()]
        if tolist:
            indexes = indexes.tolist()
        return indexes
    else:
        indexes = [multihots_to_ints(value, tolist) for value in multihots]
        if not tolist and (
            len(indexes) == 0
            or all(
                isinstance(indexes_i, Tensor) and indexes_i.shape == indexes[0].shape  # type: ignore
                for indexes_i in indexes
            )
        ):
            indexes = torch.stack(indexes)  # type: ignore
        return indexes


def ints_to_multihots(
    ints: Union[Iterable, Tensor],
    n_classes: Optional[int],
    dtype: torch.dtype = torch.bool,
    device: Union[str, torch.device, None] = None,
    clamp_max: Optional[float] = 1.0,
) -> Tensor:
    """Returns multihots encoded version of ints.

    Note: Negative classes indexes are ignored.

    :param ints: The integer list or Tensor to transform to multihot.
    :param n_classes: The maximal number of classes. If None, it will be inferred from max(ints) + 1.
    :param dtype: The output dtype. defaults to torch.bool.
    :param device: The output device. defaults to None.
    :param clamp_max: The maximal output value. Changing this argument requires to change the output dtype. defaults to 1.0.
    """
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if isinstance(device, str):
        device = torch.device(device)

    if isinstance(ints, Tensor):
        if device is None:
            device = ints.device
        else:
            ints = ints.to(device)

    elif isinstance(ints, Iterable):
        ints = list(ints)
        if all(isinstance(v, int) for v in ints):
            ints = torch.as_tensor(ints, dtype=torch.int64, device=device)
        elif all(isinstance(v, Tensor) and v.ndim == 0 for v in ints):
            ints = torch.stack(ints)
            ints = ints.to(device=device)
        else:
            if n_classes is None:
                n_classes = max_rec(ints) + 1

            return torch.stack(
                [
                    ints_to_multihots(ints_i, n_classes, dtype, device, clamp_max)
                    for ints_i in ints
                ]
            )

    if ints.is_floating_point():
        raise TypeError(
            f"Invalid argument type. (expected list[int], list[ScalarIntTensor] or IntTensor but found floating point values in {ints=})"
        )
    if ints.ndim == 0:
        raise ValueError(f"Invalid number of dimensions {ints.ndim=}.")

    if n_classes is None:
        n_classes = int(ints.max().item()) + 1

    ints_shape = tuple(ints.shape)

    contains_neg_ints = ints.lt(0).any()
    if contains_neg_ints:
        ints[ints.lt(0)] = n_classes
        n_classes += 1

    ints = ints.unsqueeze(dim=-1).to(dtype=torch.int64)

    mult_hots = torch.zeros(ints_shape + (n_classes,), device=device)
    mult_hots.scatter_(-1, ints, 1.0)
    # Reduce seq dim
    mult_hots = mult_hots.sum(dim=-2)
    if clamp_max is not None:
        mult_hots.clamp_(max=clamp_max)
    mult_hots = mult_hots.to(dtype=dtype)

    if contains_neg_ints:
        # Remove neg multihot vector
        slices = [slice(None) for _ in range(mult_hots.ndim)]
        slices[-1] = slice(-1)
        mult_hots = mult_hots[slices]

    return mult_hots


def max_rec(x: Sequence) -> Any:
    if all(not isinstance(xi, Iterable) for xi in x):
        return max(x)
    else:
        return max(max_rec(xi) for xi in x)
