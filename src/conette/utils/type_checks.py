#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Any, Iterable, TypeGuard

from torch import Tensor
from torchoutil.utils.type_checks import is_iterable_str


def is_iter_str(x: Any) -> TypeGuard[Iterable[str]]:
    return is_iterable_str(x)


def is_list_tensor(x: Any) -> TypeGuard[list[Tensor]]:
    return isinstance(x, list) and all(isinstance(xi, Tensor) for xi in x)


def is_iter_tensor(x: Any) -> TypeGuard[Iterable[Tensor]]:
    return isinstance(x, Iterable) and all(isinstance(xi, Tensor) for xi in x)
