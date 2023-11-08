#!/usr/bin/env python
# -*- coding: utf-8 -*-

import hashlib
import inspect
import itertools
import logging
import pickle
import struct
import zlib

from functools import cache
from types import FunctionType, MethodType, NoneType
from typing import Any, Iterable, Mapping, Union

import numpy as np
import torch

from torch import nn, Tensor
from torch.types import Number as TorchNumber


BYTES_MODES = ("adler32", "md5", "crc32")
FLOAT_MODES = ("tobytes", "tostr", "toint")
TENSOR_MODES = ("sum", "sum_order", "bytes", "tolist")
UNK_MODES = ("error", "pickle", "class")


pylog = logging.getLogger(__name__)


# Non-terminal (recursive) checksums functions
def csum_any(
    x: Any,
    *,
    bytes_mode: str = "crc32",
    tensor_mode: str = "sum_order",
    iter_order: bool = False,
    float_mode: str = "tobytes",
    accumulator: int = 0,
    unk_mode: str = "error",
    **kwargs,
) -> int:
    """
    :param x: The value to reduce.
    :param bytes_mode: ("adler32", "md5")
    :param tensor_mode: ("sum", "sum_order", "bytes")
    :param iter_order: If True, order in iterable will influence the final checksum value.
    :param float_mode: ("tobytes", "tostr", "toint")
    :param accumulator: Accumulated value.
    :param unk_mode: Define behaviour when encountering an unknown class. Can be one of ("error", "pickle", "class"). defaults to "error".
    :returns: The checksum of the x argument as integer.
    """
    kwargs |= dict(
        bytes_mode=bytes_mode,
        tensor_fast=tensor_mode,
        iter_order=iter_order,
        float_mode=float_mode,
        accumulator=accumulator,
        unk_mode=unk_mode,
    )

    if isinstance(x, bool):
        return csum_bool(x, **kwargs)
    elif isinstance(x, bytes):
        return csum_bytes(x, **kwargs)
    elif isinstance(x, float):
        return csum_float(x, **kwargs)
    elif isinstance(x, int):
        return csum_int(x, **kwargs)
    elif isinstance(x, complex):
        return csum_complex(x, **kwargs)
    elif x is None:
        return csum_none(x, **kwargs)
    elif isinstance(x, str):
        return csum_str(x, **kwargs)
    elif isinstance(x, Tensor):
        return csum_tensor(x, **kwargs)
    elif isinstance(x, np.ndarray):
        return csum_ndarray(x, **kwargs)
    elif isinstance(x, type) or inspect.isfunction(x):
        return csum_type_func(x, **kwargs)
    elif inspect.ismethod(x):
        return csum_method(x, **kwargs)
    elif isinstance(x, Mapping):
        return csum_mapping(x, **kwargs)
    # Iterable must be after str and mapping
    elif isinstance(x, Iterable):
        return csum_iter(x, **kwargs)
    elif isinstance(x, nn.Module):
        return csum_module(x, **kwargs)
    elif unk_mode == "pickle":
        return csum_bytes(pickle.dumps(x), **kwargs)
    elif unk_mode == "class":
        return csum_type_func(x.__class__, **kwargs)
    elif unk_mode == "error":
        EXPECTED_TYPES = (
            "bool",
            "bytes",
            "float",
            "int",
            "Mapping",
            "nn.Module",
            "str",
            "Tensor",
            "ndarray",
            "type",
            "FunctionType",
            "MethodType",
            "Iterable",
        )
        raise TypeError(
            f"Invalid argument type {x.__class__.__name__}. (expected one of {EXPECTED_TYPES})"
        )
    else:
        raise ValueError(
            f"Invalid argument {unk_mode=}. (expected one of {UNK_MODES}) "
        )


def csum_iter(x: Iterable, *, iter_order: bool = False, **kwargs) -> int:
    accumulator = kwargs.pop("accumulator", 0)
    return sum(
        csum_any(
            xi,
            accumulator=accumulator + (i + 1) * int(iter_order),
            iter_order=iter_order,
            **kwargs,
        )
        * ((i + 1) if iter_order else 1)
        for i, xi in enumerate(x)
    )


def csum_mapping(x: Mapping, **kwargs) -> int:
    return csum_iter(x.items(), **kwargs)


def csum_method(
    x: MethodType,
    *,
    bytes_mode: str = "adler32",
    **kwargs,
) -> int:
    return csum_iter(
        [
            csum_str(x.__qualname__, bytes_mode=bytes_mode, **kwargs),
            csum_any(x.__self__, bytes_mode=bytes_mode, **kwargs),
        ]
    )


def csum_module(
    x: nn.Module,
    *,
    with_names: bool = True,
    only_trainable: bool = False,
    with_buffers: bool = False,
    **kwargs,
) -> int:
    params = x.named_parameters()
    params = ((n, p) for n, p in params if p.requires_grad or not only_trainable)

    if not with_names:
        params = (n for n, _ in params)

    if with_buffers:
        buffers = x.named_buffers()
        if not with_names:
            buffers = (n for n, _ in buffers)
        content = itertools.chain.from_iterable((params, buffers))
    else:
        content = params

    return csum_iter(
        content,
        with_names=with_names,
        only_trainable=only_trainable,
        with_buffers=with_buffers,
        **kwargs,
    )


def csum_ndarray(
    x: np.ndarray,
    **kwargs,
) -> int:
    return csum_tensor(torch.from_numpy(x), **kwargs)


def csum_tensor(
    x: Tensor,
    *,
    tensor_mode: str = "sum_order",
    only_trainable: bool = False,
    **kwargs,
) -> int:
    if only_trainable and not x.requires_grad:
        return kwargs.get("accumulator", 0)

    if tensor_mode in ("sum", "sum_order"):
        if x.ndim > 0:
            x = x.detach()
            if tensor_mode == "sum_order":
                x = x.flatten()
                dtype = x.dtype if x.dtype != torch.bool else torch.int
                x = x * torch.arange(1, len(x) + 1, device=x.device, dtype=dtype)
            x = x.nansum()

        xitem = x.item()
        return csum_torch_number(
            xitem, tensor_mode=tensor_mode, only_trainable=only_trainable, **kwargs
        )

    elif tensor_mode == "bytes":
        __warn_once(
            f"Bytes checksum on PyTorch tensors is not deterministic. (found {tensor_mode=})"
        )
        tensor_bytes = pickle.dumps(x)
        return csum_bytes(tensor_bytes, fast=tensor_mode, **kwargs)

    elif tensor_mode == "tolist":
        return csum_iter(
            x.tolist(), tensor_mode=tensor_mode, only_trainable=only_trainable, **kwargs
        )

    else:
        raise ValueError(
            f"Invalid argument {tensor_mode=}. (expected one of {TENSOR_MODES})"
        )


# Terminal checksums functions
def csum_bool(x: bool, **kwargs) -> int:
    return int(x) + kwargs.get("accumulator", 0)


def csum_bytes(x: bytes, *, bytes_mode: str = "adler32", **kwargs) -> int:
    if bytes_mode == "adler32":
        return zlib.adler32(x) + kwargs.get("accumulator", 0)
    elif bytes_mode == "md5":
        x = hashlib.md5(x).digest()
        csum = int.from_bytes(x, "big", signed=False)
        return csum + kwargs.get("accumulator", 0)
    elif bytes_mode == "crc32":
        return zlib.crc32(x) % (1 << 32) + kwargs.get("accumulator", 0)
    else:
        raise ValueError(
            f"Invalid argument {bytes_mode=}. (expected one of {BYTES_MODES})"
        )


def csum_complex(x: complex, **kwargs) -> int:
    return csum_iter(
        [csum_float(x.real, **kwargs), csum_float(x.imag, **kwargs)], **kwargs
    )


def csum_float(
    x: float,
    *,
    float_mode: str = "tostr",
    bytes_mode: str = "adler32",
    **kwargs,
) -> int:
    if float_mode == "tostr":
        return csum_str(str(x), float_mode=float_mode, bytes_mode=bytes_mode, **kwargs)
    elif float_mode == "tobytes":
        return csum_bytes(
            struct.pack("f", x), float_mode=float_mode, bytes_mode=bytes_mode, **kwargs
        )
    elif float_mode == "toint":
        return csum_int(int(x), float_mode=float_mode, bytes_mode=bytes_mode, **kwargs)
    else:
        raise ValueError(
            f"Invalid argument {float_mode=}. (expected one of {FLOAT_MODES})"
        )


def csum_int(x: int, **kwargs) -> int:
    return x + kwargs.get("accumulator", 0)


def csum_none(x: NoneType, **kwargs) -> int:
    return kwargs.get("accumulator", 0)


def csum_torch_number(x: TorchNumber, **kwargs) -> int:
    """Csum for int, bool, float or complex."""
    if isinstance(x, int):
        return csum_int(x, **kwargs)
    elif isinstance(x, bool):
        return csum_bool(x, **kwargs)
    elif isinstance(x, float):
        return csum_float(x, **kwargs)
    elif isinstance(x, complex):
        return csum_complex(x, **kwargs)
    else:
        raise TypeError(
            f"Invalid argument type {type(x)=} for csum_number function. (expected int, bool, float or complex)"
        )


def csum_str(x: str, *, bytes_mode: str = "adler32", **kwargs) -> int:
    return csum_bytes(x.encode(), bytes_mode=bytes_mode, **kwargs)


def csum_type_func(
    x: Union[type, FunctionType],
    *,
    bytes_mode: str = "adler32",
    **kwargs,
) -> int:
    return csum_str(x.__qualname__, bytes_mode=bytes_mode, **kwargs)


@cache
def __warn_once(msg: str) -> None:
    pylog.warning(msg)
