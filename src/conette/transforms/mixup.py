#!/usr/bin/env python
# -*- coding: utf-8 -*-

import random

from typing import Any, Iterable, Union

import torch

from torch import nn, Tensor
from torch.distributions.beta import Beta


def pann_mixup(x: Tensor, mixup_lambda: Tensor) -> Tensor:
    """Mixup x of even indexes (0, 2, 4, ...) with x of odd indexes (1, 3, 5, ...).

    Args:
      x: (batch_size * 2, ...)
      mixup_lambda: (batch_size * 2,)

    Returns:
      out: (batch_size, ...)
    """
    out = (
        x[0::2].transpose(0, -1) * mixup_lambda[0::2]
        + x[1::2].transpose(0, -1) * mixup_lambda[1::2]
    ).transpose(0, -1)
    return out


def sample_lambda(
    alpha: Union[float, Tensor],
    asymmetric: bool,
    size: Iterable[int] = (),
) -> Tensor:
    """
    :param alpha: alpha hp to control the Beta distribution.
        Values closes to 0 means distribution will peak up at 0 and 1, while values closes to 1 means uniform distribution.
    :param asymmetric: If True, lbd value will always be in [0.5, 1], with values close to 1.
    :param size: The size of the sampled lambda(s) value(s). defaults to ().
    :returns: Sampled values of shape defined by size argument.
    """
    tensor_kwds: dict[str, Any] = dict(dtype=torch.get_default_dtype())
    size = torch.Size(size)

    if alpha == 0.0:
        if asymmetric:
            return torch.full(size, 1.0, **tensor_kwds)
        else:
            return torch.rand(size).ge(0.5).to(**tensor_kwds)

    alpha = torch.as_tensor(alpha, **tensor_kwds)
    beta = Beta(alpha, alpha)
    lbd = beta.sample(size)
    if asymmetric:
        lbd = torch.max(lbd, 1.0 - lbd)
    return lbd


class Mixup(nn.Module):
    """
    Mix linearly inputs with coefficient sampled from a Beta distribution.
    """

    def __init__(
        self,
        alpha: float = 0.4,
        asymmetric: bool = False,
        p: float = 1.0,
    ) -> None:
        """
        ```
        lambda ~ Beta(alpha, alpha)
        x = lambda * x + (1.0 - lambda) * shuffle(x)
        y = lambda * y + (1.0 - lambda) * shuffle(y)
        ```

        :param alpha: The parameter used by the beta distribution.
            If alpha -> 0, the value sampled will be close to 0 or 1.
            If alpha -> 1, the value will be sampled from a uniform distribution.
            defaults to 0.4.
        :param asymmetric: If True, the first coefficient will always be the higher one, which means the result will be closer to the input.
            defaults to False.
        :param p: The probability to apply the mixup.
            defaults to 1.0.
        """
        assert 0.0 <= p <= 1.0
        super().__init__()
        self.alpha = alpha
        self.asymmetric = asymmetric
        self.p = p

        self.beta = Beta(alpha, alpha)

    # nn.Module methods
    def extra_repr(self) -> str:
        hparams = {
            "alpha": self.alpha,
            "asymmetric": self.asymmetric,
            "p": self.p,
        }
        return ", ".join(f"{k}={v}" for k, v in hparams.items())

    def __call__(self, x: Tensor, y: Tensor) -> tuple[Tensor, Tensor]:
        # This method is here only for typing
        return super().__call__(x, y)

    def forward(self, x: Tensor, y: Tensor) -> tuple[Tensor, Tensor]:
        if self.p >= 1.0 or random.random() < self.p:
            return self.apply_transform(x, y)
        else:
            return x, y

    # Other methods
    def sample_lambda(self, size: Iterable[int] = ()) -> Tensor:
        return sample_lambda(self.alpha, self.asymmetric, size)

    def apply_transform(self, x: Tensor, y: Tensor) -> tuple[Tensor, Tensor]:
        if x.shape[0] != y.shape[0]:
            raise ValueError(
                f"Data to mix must have the same size along the first dim. ({x.shape[0]=} != {y.shape[0]=})"
            )

        bsize = x.shape[0]
        lbd = self.sample_lambda(())
        indexes = torch.randperm(bsize)

        x = x * lbd + x[indexes] * (1.0 - lbd)
        y = y * lbd + y[indexes] * (1.0 - lbd)
        return x, y
