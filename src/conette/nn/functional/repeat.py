#!/usr/bin/env python
# -*- coding: utf-8 -*-

from torch import Tensor


def repeat_interleave_nd(x: Tensor, n: int, dim: int = 0) -> Tensor:
    """Generalized version of torch.repeat_interleave for N >= 1 dimensions.

    >>> x = torch.as_tensor([[0, 1, 2, 3], [4, 5, 6, 7]])
    >>> repeat_interleave_nd(x, n=2, dim=0)
    tensor([[0, 1, 2, 3],
            [0, 1, 2, 3],
            [4, 5, 6, 7],
            [4, 5, 6, 7]])

    :param x: Any tensor of shape (..., D, ...) with at least 1 dim.
    :param n: Number of repeats.
    :param dim: Dimension to repeat.
    :returns: Tensor of shape (..., D*n, ...), where D is the size of the dimension of the dim parameter.
    """
    assert x.ndim > 0
    dim = dim % x.ndim
    x = x.unsqueeze(dim=dim + 1)
    shape = list(x.shape)
    shape[dim + 1] = n
    return x.expand(*shape).flatten(dim, dim + 1)
