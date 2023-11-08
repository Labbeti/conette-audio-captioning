#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Optional, Union

from torch import nn, Tensor
from torch.nn import functional as F


class Mean(nn.Module):
    def __init__(self, dim: Optional[int]) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        if self.dim is None:
            return x.mean()
        else:
            return x.mean(dim=self.dim)

    def extra_repr(self) -> str:
        return f"dim={self.dim}"


class Normalize(nn.Module):
    def forward(self, data: Tensor) -> Tensor:
        return F.normalize(data)


class Transpose(nn.Module):
    def __init__(self, dim0: int, dim1: int) -> None:
        super().__init__()
        self.dim0 = dim0
        self.dim1 = dim1

    def forward(self, x: Tensor) -> Tensor:
        return x.transpose(self.dim0, self.dim1)

    def extra_repr(self) -> str:
        return f"{self.dim0}, {self.dim1}"


class Squeeze(nn.Module):
    def __init__(self, dim: Optional[int] = None, inplace: bool = True) -> None:
        super().__init__()
        self.dim = dim
        self.inplace = inplace

    def forward(self, x: Tensor) -> Tensor:
        if not self.inplace:
            if self.dim is None:
                return x.squeeze()
            else:
                return x.squeeze(self.dim)
        else:
            if self.dim is None:
                return x.squeeze_()
            else:
                return x.squeeze_(self.dim)

    def extra_repr(self) -> str:
        return f"dim={self.dim}, inplace={self.inplace}"


class Unsqueeze(nn.Module):
    def __init__(self, dim: int, inplace: bool = True) -> None:
        super().__init__()
        self.dim = dim
        self.inplace = inplace

    def forward(self, x: Tensor) -> Tensor:
        if not self.inplace:
            return x.unsqueeze(self.dim)
        else:
            return x.unsqueeze_(self.dim)

    def extra_repr(self) -> str:
        return f"dim={self.dim}, inplace={self.inplace}"


class TensorTo(nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.kwargs = kwargs

    def forward(self, x: Tensor, **kwargs) -> Tensor:
        kwargs = self.kwargs | kwargs
        return x.to(**kwargs)

    def extra_repr(self) -> str:
        kwargs_str = ", ".join(f"{k}={v}" for k, v in self.kwargs.items())
        return kwargs_str


class Permute(nn.Module):
    def __init__(self, *args: int) -> None:
        super().__init__()
        self._dims = tuple(args)

    def forward(self, x: Tensor) -> Tensor:
        return x.permute(self._dims)

    def extra_repr(self) -> str:
        return ", ".join(map(str, self._dims))


class Div(nn.Module):
    def __init__(
        self,
        divisor: Union[float, Tensor],
        rounding_mode: str = "trunc",
    ) -> None:
        super().__init__()
        self.divisor = divisor
        self.rounding_mode = rounding_mode

    def forward(self, x: Tensor) -> Tensor:
        return x.div(self.divisor, rounding_mode=self.rounding_mode)
