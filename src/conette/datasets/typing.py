#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import (
    Any,
    Protocol,
    runtime_checkable,
)

from torchoutil.utils.data.dataset import SizedDatasetLike  # noqa: E402


@runtime_checkable
class AACDatasetLike(Protocol):
    """Protocal abstract class for aac datasets. Used only for typing.

    Methods signatures:
        - column_names: () -> list[str]
        - at: (int, str) -> Any
        - __getitem__: (int, str) -> Any
        - __len__: () -> int
    """

    @property
    def column_names(self) -> list[str]:
        raise NotImplementedError("Protocal abstract method.")

    def at(self, idx: Any, column: Any) -> Any:
        raise NotImplementedError("Protocal abstract method.")

    def __getitem__(self, idx: Any) -> Any:
        raise NotImplementedError("Protocal abstract method.")

    def __len__(self) -> int:
        raise NotImplementedError("Protocal abstract method.")
