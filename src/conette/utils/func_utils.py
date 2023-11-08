#!/usr/bin/env python
# -*- coding: utf-8 -*-

import inspect
import logging

from typing import (
    Any,
    Callable,
    Iterable,
)

pylog = logging.getLogger(__name__)


def get_argnames(func: Callable) -> list[str]:
    if inspect.ismethod(func):
        # If method, remove 'self' arg
        argnames = func.__code__.co_varnames[1:]  # type: ignore
    elif inspect.isfunction(func):
        argnames = func.__code__.co_varnames
    else:
        argnames = func.__call__.__code__.co_varnames
    argnames = list(argnames)
    return argnames


def filter_and_call(func: Callable, **kwargs: Any) -> Any:
    """Filter kwargs with function arg names and call function."""
    argnames = get_argnames(func)
    kwargs_filtered = {
        name: value for name, value in kwargs.items() if name in argnames
    }
    return func(**kwargs_filtered)


def filter_kwargs(class_, kwargs: dict) -> dict:
    varnames = class_.__init__.__code__.co_varnames
    return {name: value for name, value in kwargs.items() if name in varnames}


def cache_feature(func: Callable) -> Callable:
    """Cache feature decorator for any type of argument that can be converted to string."""

    def decorator(*args, **kwargs):
        key = ",".join(map(str, args + tuple(kwargs.items())))

        if key not in decorator.cache:
            decorator.cache[key] = func(*args, **kwargs)

        return decorator.cache[key]

    decorator.cache = dict()
    decorator.func = func
    return decorator


def decorator_factory(ignore: Iterable[str] = ()) -> Callable:
    if isinstance(ignore, str):
        ignore = [ignore]
    else:
        ignore = list(ignore)

    def decorator(func: Callable) -> Callable:
        ignore_pos = [i for i, name in enumerate(get_argnames(func)) if name in ignore]

        def wrapper(*args, **kwargs):
            hash_args = tuple(arg for i, arg in enumerate(args) if i not in ignore_pos)
            hash_kwargs = {k: v for k, v in kwargs.items() if k not in ignore}
            key = ",".join(map(str, hash_args + tuple(hash_kwargs.items())))

            if key not in wrapper.cache:
                wrapper.cache[key] = func(*args, **kwargs)

            return wrapper.cache[key]

        wrapper.cache = {}
        wrapper.func = func
        return wrapper

    return decorator
