#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import (
    Callable,
    Iterable,
    Optional,
    TypeVar,
    overload,
)

T = TypeVar("T")
U = TypeVar("U")
V = TypeVar("V")
W = TypeVar("W")


def all_eq(it: Iterable[T], eq_fn: Optional[Callable[[T, T], bool]] = None) -> bool:
    """Returns true if all elements in inputs are equal."""
    it = list(it)
    first = it[0]
    if eq_fn is None:
        return all(first == elt for elt in it)
    else:
        return all(eq_fn(first, elt) for elt in it)


@overload
def unzip(lst: Iterable[tuple[T]]) -> tuple[list[T]]:
    ...


@overload
def unzip(lst: Iterable[tuple[T, U]]) -> tuple[list[T], list[U]]:
    ...


@overload
def unzip(lst: Iterable[tuple[T, U, V]]) -> tuple[list[T], list[U], list[V]]:
    ...


@overload
def unzip(
    lst: Iterable[tuple[T, U, V, W]]
) -> tuple[list[T], list[U], list[V], list[W]]:
    ...


def unzip(lst: Iterable) -> tuple[list, ...]:
    """Invert zip() function.

    .. code-block:: python
        :caption:  Example

        >>> lst1 = [1, 2, 3, 4]
        >>> lst2 = [5, 6, 7, 8]
        >>> lst_zipped = list(zip(lst1, lst2))
        >>> lst_zipped
        ... [(1, 5), (2, 6), (3, 7), (4, 8)]
        >>> unzip(lst_zipped)
        ... ([1, 2, 3, 4], [5, 6, 7, 8])
    """
    return tuple(map(list, zip(*lst)))
