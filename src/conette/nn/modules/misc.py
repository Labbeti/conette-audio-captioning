#!/usr/bin/env python
# -*- coding: utf-8 -*-

import inspect
import tqdm

from dataclasses import asdict, astuple
from typing import Any, Callable, Iterable, Mapping, Optional

import torch

from torch import nn, Tensor


class AmplitudeToLog(nn.Module):
    def __init__(self, eps: float = torch.finfo(torch.float).eps) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, data: Tensor) -> Tensor:
        return torch.log(data + self.eps)


class Lambda(nn.Module):
    def __init__(self, fn: Callable, **default_kwargs) -> None:
        """Wrap a callable function or object to a Module."""
        super().__init__()
        self.fn = fn
        self.default_kwargs = default_kwargs

    def forward(self, *args, **kwargs) -> Any:
        kwargs = self.default_kwargs | kwargs
        return self.fn(*args, **kwargs)

    def extra_repr(self) -> str:
        if isinstance(self.fn, nn.Module):
            return ""
        elif inspect.isfunction(self.fn):
            return self.fn.__name__
        elif inspect.ismethod(self.fn):
            return self.fn.__qualname__
        else:
            return self.fn.__class__.__name__


class Reshape(nn.Module):
    def __init__(self, shape: tuple[int, ...]) -> None:
        super().__init__()
        self.shape = shape

    def forward(self, x: Tensor) -> Tensor:
        return torch.reshape(x, self.shape)


class Print(nn.Module):
    def __init__(
        self,
        preprocess: Optional[Callable] = None,
        prefix: str = "DEBUG - ",
    ) -> None:
        super().__init__()
        self._preprocess = preprocess
        self._prefix = prefix

    def forward(self, x: Any) -> Any:
        x_out = x
        if self._preprocess is not None:
            x = self._preprocess(x)
        print(f"{self._prefix}{x=}")
        return x_out


class AsTensor(nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.kwargs = kwargs

    def forward(self, inp: list, *args, **kwargs) -> Tensor:
        kwargs = self.kwargs | kwargs
        return torch.as_tensor(inp, *args, **kwargs)

    def extra_repr(self) -> str:
        kwargs_str = ",".join(f"{k}={v}" for k, v in self.kwargs.items())
        return f"kwargs=dict({kwargs_str})"


class ParallelDict(nn.ModuleDict):
    """Compute output of each submodule value when forward(.) is called."""

    def __init__(
        self, modules: Optional[dict[str, nn.Module]] = None, verbose: bool = False
    ) -> None:
        super().__init__(modules)
        self._verbose = verbose

    def forward(self, *args, **kwargs) -> dict[str, Any]:
        tqdm_obj = tqdm.tqdm(
            self.items(), desc=f"{self.__class__.__name__}", disable=not self._verbose
        )
        outs = {}
        for name, module in tqdm_obj:
            tqdm_obj.set_description(
                f"{self.__class__.__name__}:{module.__class__.__name__}"
            )
            outs[name] = module(*args, **kwargs)
        return outs


class ParallelList(nn.ModuleList):
    def __init__(
        self, modules: Optional[Iterable[nn.Module]] = (), verbose: bool = False
    ) -> None:
        super().__init__(modules)
        self._verbose = verbose

    def forward(self, *args, **kwargs) -> list[Any]:
        tqdm_obj = tqdm.tqdm(
            self,
            disable=not self._verbose,
            desc=f"{self.__class__.__name__}",
        )
        outs = []
        for module in tqdm_obj:
            tqdm_obj.set_description(
                f"{self.__class__.__name__}:{module.__class__.__name__}"
            )
            outs.append(module(*args, **kwargs))
        return outs


class SequentialArgs(nn.Sequential):
    def forward(self, *args) -> Any:
        x = args
        for module in self:
            if isinstance(x, tuple):
                x = module(*x)
            else:
                x = module(x)
        return x


class SequentialKwargs(nn.Sequential):
    def forward(self, **kwargs) -> Any:
        x = kwargs
        for module in self:
            if isinstance(x, dict):
                x = module(**x)
            else:
                x = module(x)
        return x


class Standardize(nn.Module):
    def __init__(self, unbiased_std: bool = True) -> None:
        super().__init__()
        self.unbiased_std = unbiased_std

    def forward(self, x: Tensor) -> Tensor:
        x = (x - x.mean()) / x.std(unbiased=self.unbiased_std)
        return x


class AsDict(nn.Module):
    def forward(self, x: Any) -> dict[str, Any]:
        return asdict(x)


class AsTuple(nn.Module):
    def forward(self, x: Any) -> tuple[Any, ...]:
        return astuple(x)


class DictTransformModule(nn.ModuleDict):
    """Wrap a dictionary of modules to apply to each value of a dictionary input at a corresponding key.

    Example 1
    ----------
    ```py
    >>> mean_a = DictTransformModule({"a": nn.ReLU()})
    >>> input = {"a": torch.as_tensor([-1., 2, -3]), "b": "something", "c": torch.as_tensor([1, 2, 3])}
    >>> mean_a(input)
    ... {"a": tensor([0.0, 2.0, 0.0]), "b": "something", "c": tensor([1, 2, 3])}
    ```
    """

    def __init__(
        self, modules: Optional[Mapping[str, Optional[nn.Module]]] = None, **kwargs
    ) -> None:
        if modules is None:
            modules = {}
        else:
            modules = dict(modules)
        modules = modules | kwargs
        modules = {k: v for k, v in modules.items() if v is not None}
        super().__init__(modules)

    def forward(self, dic: dict[str, Any]) -> dict[str, Any]:
        for name, module in self.items():
            if name in dic:
                dic[name] = module(dic[name])
        return dic


class IdMapping(nn.Module):
    def __init__(self, mapper: Tensor) -> None:
        super().__init__()
        self.mapper = mapper

    @classmethod
    def from_dic(
        cls, dic: Mapping[int, int], dtype: torch.dtype = torch.long
    ) -> "IdMapping":
        max_src_id = max(dic.keys())
        mapper = torch.zeros((max_src_id,), dtype=dtype)
        for k, v in dic.items():
            mapper[k] = v
        return IdMapping(mapper)

    def forward(self, ids: Tensor) -> Tensor:
        assert not ids.is_floating_point()
        return self.mapper[ids]
