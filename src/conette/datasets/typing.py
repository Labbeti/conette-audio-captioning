#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Any, Callable, Protocol, runtime_checkable


@runtime_checkable
class AACDatasetLike(Protocol):
    """Protocal abstract class for aac datasets. Used only for typing.

    Methods signatures:
        - column_names: () -> list[str]
        - at: (int, str) -> Any
        - __getitem__: (Any) -> Any
        - __len__: () -> int
    """

    at: Callable[..., Any]
    __getitem__: Callable[..., Any]

    @property
    def column_names(self) -> list[str]:
        raise NotImplementedError("Protocal abstract method.")

    def __len__(self) -> int:
        raise NotImplementedError("Protocal abstract method.")
