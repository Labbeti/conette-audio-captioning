#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re
from typing import (
    Any,
    Callable,
    Iterable,
    Mapping,
    Optional,
    Sequence,
    TypeVar,
    Union,
    overload,
)

import numpy as np
from torchoutil.utils.collections import all_eq

T = TypeVar("T")
U = TypeVar("U")
V = TypeVar("V")
W = TypeVar("W")


def all_ne(it: Iterable[T], ne_fn: Optional[Callable[[T, T], bool]] = None) -> bool:
    """Returns true if all elements in inputs are differents."""
    it = list(it)
    if ne_fn is None:
        return all(
            it[i] != it[j] for i in range(len(it)) for j in range(i + 1, len(it))
        )
    else:
        return all(
            ne_fn(it[i], it[j]) for i in range(len(it)) for j in range(i + 1, len(it))
        )


def dict_list_to_list_dict(dic: Mapping[str, Sequence[T]]) -> list[dict[str, T]]:
    """Convert dict of lists with same sizes to list of dicts.

    Example 1
    ----------
    ```
    >>> dic = {"a": [1, 2], "b": [3, 4]}
    >>> dict_list_to_list_dict(dic)
    ... [{"a": 1, "b": 3}, {"a": 2, "b": 4}]
    ```
    """
    assert all_eq(map(len, dic.values()))
    length = len(next(iter(dic.values())))
    return [{k: v[i] for k, v in dic.items()} for i in range(length)]


def unflat_dict_of_dict(dic: Mapping[str, Any], sep: str = ".") -> dict[str, Any]:
    """Unflat a dictionary.

    Example 1
    ----------
    ```
    >>> dic = {
        "a.a": 1,
        "b.a": 2,
        "b.b": 3,
        "c": 4,
    }
    >>> unflat_dict(dic)
    ... {"a": {"a": 1}, "b": {"a": 2, "b": 3}, "c": 4}
    ```
    """
    output = {}
    for k, v in dic.items():
        if sep not in k:
            output[k] = v
        else:
            idx = k.index(sep)
            k, kk = k[:idx], k[idx + 1 :]
            if k not in output:
                output[k] = {}
            elif not isinstance(output[k], Mapping):
                raise ValueError(
                    f"Invalid dict argument. (found keys {k} and {k}{sep}{kk})"
                )

            output[k][kk] = v

    output = {
        k: (unflat_dict_of_dict(v) if isinstance(v, Mapping) else v)
        for k, v in output.items()
    }
    return output


def flat_list_rec(
    nested_lst: Union[list, tuple],
    returns_shapes: bool = False,
) -> Union[list, tuple]:
    """Flat nested list to list of scalars."""
    if not isinstance(nested_lst, (list, tuple)):
        output = (nested_lst,), ()
    else:
        flat_lst = []
        shapes = []
        for elt in nested_lst:
            subelt, subshapes = flat_list_rec(elt, True)
            flat_lst += subelt
            shapes.append(subshapes)

        if len(shapes) == 0:
            output = [], (0,)
        elif all(subshapes == shapes[0] for subshapes in shapes):
            output = flat_lst, (len(nested_lst),) + shapes[0]
        else:
            output = flat_lst, shapes

    if returns_shapes:
        return output
    else:
        return output[0]


def unflat_list_rec(flat_lst: list, shapes: Union[list, tuple]) -> list:
    """Unflat list to nested list with given shapes."""
    if isinstance(shapes, tuple):
        if shapes == ():
            return flat_lst[0]
        else:
            array = np.array(flat_lst, dtype=object)
            array = array.reshape(*shapes)
            array = array.tolist()
            return array
    else:
        out = []
        idx = 0
        for shape_i in shapes:
            num_elements = _prod_rec(shape_i)
            unflatten = unflat_list_rec(flat_lst[idx : idx + num_elements], shape_i)
            idx += num_elements
            out.append(unflatten)
        return out


def _prod_rec(x: Union[int, float, Iterable]) -> Union[int, float]:
    if isinstance(x, (int, float)):
        return x
    elif isinstance(x, Iterable):
        out = 1
        for xi in x:
            out *= _prod_rec(xi)
        return out
    else:
        raise TypeError(
            f"Invalid argument type {type(x)=}. (expected int, float or iterable of int floats."
        )


def is_ascending(x: Iterable, strict: bool = False) -> bool:
    x = list(x)
    if len(x) <= 1:
        return True

    if strict:
        return all(xi < x[i + 1] for i, xi in enumerate(x[:-1]))
    else:
        return all(xi <= x[i + 1] for i, xi in enumerate(x[:-1]))


def is_descending(x: Iterable, strict: bool = False) -> bool:
    x = list(x)
    if len(x) <= 1:
        return True

    if strict:
        return all(xi > x[i + 1] for i, xi in enumerate(x[:-1]))
    else:
        return all(xi >= x[i + 1] for i, xi in enumerate(x[:-1]))


def sort_dict_with_patterns(
    dic: dict[str, Any],
    patterns: Iterable[str],
) -> dict[str, Any]:
    patterns = list(patterns)
    compl_patterns = list(map(re.compile, patterns))

    def key_fn(key: str) -> int:
        for i, pattern in enumerate(compl_patterns):
            if re.match(pattern, key):
                return i
        return len(compl_patterns)

    dic = {k: dic[k] for k in sorted(dic.keys(), key=key_fn)}
    return dic


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
