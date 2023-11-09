#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import (
    Any,
    Union,
)

import torch

from torch import Tensor


def randperm_diff(
    size: int,
    seed: Union[None, int, torch.Generator] = None,
    device: Union[str, torch.device, None] = "auto",
) -> Tensor:
    """This function ensure that every value i cannot be the element at index i.

    Example 1
    ----------
    >>> torch.randperm(5)
    tensor([1, 4, 2, 5, 0])  # 2 is the element of index 2 !
    >>> randperm_diff(5)  # the function ensure that every value i cannot be the element at index i
    tensor([2, 0, 4, 1, 3])

    :param size: The number of indexes. Cannot be < 2.
    :param seed: The seed or torch.Generator used to generate permutation.
    :param device: The PyTorch device of the output indexes tensor.
    :returns: A tensor of shape (size,).
    """
    if size < 2:
        raise ValueError(f"Invalid argument {size=} < 2 for randperm_diff.")

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if isinstance(device, str):
        device = torch.device(device)

    if isinstance(seed, int):
        generator = torch.Generator().manual_seed(seed)
    else:
        generator = seed

    perm_kws: dict[str, Any] = dict(generator=generator, device=device)
    arange = torch.arange(size, device=device)
    perm = torch.randperm(size, **perm_kws)

    while perm.eq(arange).any():
        perm = torch.randperm(size, **perm_kws)
    return perm
