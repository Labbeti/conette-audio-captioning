#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Any, Callable, Iterable, Mapping, Optional

from torch import Tensor

from conette.utils.misc import pass_filter


class DictTransform(dict[str, Optional[Callable]]):
    def __init__(
        self,
        transforms_dict: Optional[Mapping[str, Optional[Callable]]] = None,
        **transforms_kwargs: Optional[Callable],
    ) -> None:
        """Wrap a dictionary of transforms to apply to each value of a dictionary input at a corresponding key.

        Example 1
        ----------
        ```py
        >>> triple_a = DictTransform({"a": lambda x: x * 3})
        >>> input = {"a": 4, "b": 5}
        >>> triple_a(input)
        ... {"a": 12, "b": 5}
        ```
        """
        if transforms_dict is None:
            transforms_dict = {}
        else:
            transforms_dict = dict(transforms_dict)
        transforms = transforms_dict | transforms_kwargs
        super().__init__(transforms)

    def forward(self, item: dict[str, Any]) -> dict[str, Any]:
        for name, transform in self.items():
            if transform is not None and name in item:
                item[name] = transform(item[name])
        return item

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.forward(*args, **kwds)


class ShapesToSizes:
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def __call__(self, x_shapes: Tensor) -> Tensor:
        return x_shapes[:, self.dim]


class SelectColumns:
    def __init__(
        self,
        /,
        include: Optional[Iterable[str]] = None,
        exclude: Optional[Iterable[str]] = None,
    ) -> None:
        super().__init__()
        self._include = include
        self._exclude = exclude

    def __call__(self, item: Mapping[str, Any]) -> dict[str, Any]:
        item = {
            k: v
            for k, v in item.items()
            if pass_filter(k, self._include, self._exclude)
        }
        return item


class Rename:
    def __init__(self, **kwargs: str) -> None:
        super().__init__()
        self.renames = kwargs

    def __call__(self, item: Mapping[str, Any]) -> dict[str, Any]:
        item = {self.renames.get(k, k): v for k, v in item.items()}
        return item


class Compose:
    def __init__(self, *fns: Callable) -> None:
        super().__init__()
        self.fns = fns

    def __call__(self, x: Any) -> Any:
        for fn in self.fns:
            x = fn(x)
        return x
