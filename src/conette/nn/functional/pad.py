#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Any, Iterable, Sized, Union

import torch

from torch import Tensor
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence


PAD_ALIGNS = ("left", "right", "center", "random")


def pad_sequence_rec(
    sequence: Union[Tensor, int, float, tuple, list],
    pad_value: float,
    dtype: Union[None, torch.dtype] = None,
    device: Union[str, torch.device, None] = None,
) -> Tensor:
    """Recursive version of torch.nn.utils.rnn.pad_sequence, with padding of Tensors.

    :param sequence: The sequence to pad. Must be convertable to tensor by having the correct number of dims in all sublists.
    :param pad_value: The pad value used.
    :param dtype: The dtype of the output Tensor. defaults to None.
    :param device: The device of the output Tensor. defaults to None.
    :returns: The sequence as a padded Tensor.

    Example 1
    ----------
    >>> sequence = [[1, 2], [3], [], [4, 5]]
    >>> output = pad_sequence_rec(sequence, 0)
    tensor([[1, 2], [3, 0], [0, 0], [4, 5]])

    Example 2
    ----------
    >>> invalid_sequence = [[1, 2, 3], 3]
    >>> output = pad_sequence_rec(invalid_sequence, 0)
    ValueError : Cannot pad sequence of tensors of differents number of dims.

    """
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if isinstance(device, str):
        device = torch.device(device)

    if isinstance(sequence, Tensor):
        return sequence.to(dtype=dtype, device=device)

    if isinstance(sequence, (int, float)) or (
        isinstance(sequence, Sized) and len(sequence) == 0
    ):
        return torch.as_tensor(sequence, dtype=dtype, device=device)  # type: ignore

    elif isinstance(sequence, (list, tuple)):
        if all(isinstance(elt, (int, float)) for elt in sequence):
            return torch.as_tensor(sequence, dtype=dtype, device=device)  # type: ignore

        sequence = [pad_sequence_rec(elt, pad_value, dtype, device) for elt in sequence]
        # sequence is now a list[Tensor]
        shapes = [elt.shape for elt in sequence]

        # If all tensors have the same shape
        if all(shape == shapes[0] for shape in shapes):
            return torch.stack(sequence, dim=0)

        # If all tensors have the same number of dims
        elif all(elt.ndim == sequence[0].ndim for elt in sequence):
            if all(shape[1:] == shapes[0][1:] for shape in shapes):
                return pad_sequence(sequence, True, pad_value)
            else:
                max_lens = [
                    max(shape[i] for shape in shapes) for i in range(sequence[0].ndim)
                ]
                paddings = [
                    [
                        (max_lens[i] - elt.shape[i]) * j
                        for i in range(-1, -sequence[0].ndim, -1)
                        for j in range(2)
                    ]
                    for elt in sequence
                ]
                sequence = [
                    F.pad(elt, padding, value=pad_value)
                    for elt, padding in zip(sequence, paddings)
                ]
                return pad_sequence(sequence, True, pad_value)

        else:
            raise ValueError(
                f"Cannot pad sequence of tensors of differents number of dims. ({sequence=}, {shapes=})"
            )

    else:
        raise TypeError(
            f"Invalid type {type(sequence)}. (expected Tensor, int, float, list or tuple)"
        )


def pad_sequence_1d(tensors: list[Tensor], pad_value: float) -> Tensor:
    if not all(tensor.ndim == 1 for tensor in tensors):
        raise ValueError("Invalid argument tensors for pad_sequence_1d.")

    max_len = max(tensor.shape[0] for tensor in tensors)
    output = torch.empty(
        (len(tensors), max_len), device=tensors[0].device, dtype=tensors[0].dtype
    )
    for i, tensor in enumerate(tensors):
        output[i, : tensor.shape[0]] = tensor
        output[i, tensor.shape[0] :] = pad_value
    return output


def pad_sequence_nd(tensors: list[Tensor], pad_value: float) -> Tensor:
    if not all(tensor.ndim >= 1 for tensor in tensors):
        raise ValueError("Invalid argument tensors for pad_sequence_1d.")
    if not all(tensor.shape[1:] == tensors[0].shape[1:] for tensor in tensors[1:]):
        raise ValueError("Invalid argument tensors for pad_sequence_1d.")

    max_len = max(tensor.shape[0] for tensor in tensors)
    output = torch.empty(
        (len(tensors), max_len) + tuple(tensors[0].shape[1:]),
        device=tensors[0].device,
        dtype=tensors[0].dtype,
    )
    for i, tensor in enumerate(tensors):
        output[i, : tensor.shape[0]] = tensor
        output[i, tensor.shape[0] :] = pad_value
    return output


def pad_last_dim(tensor: Tensor, target_length: int, pad_value: float) -> Tensor:
    """Left padding tensor at last dim.

    :param tensor: Tensor of at least 1 dim. (..., T)
    :param target_length: Target length of the last dim. If target_length <= T, the function has no effect.
    :param pad_value: Fill value used to pad tensor.
    :returns: A tensor of shape (..., target_length).
    """
    pad_size = max(target_length - tensor.shape[-1], 0)
    return F.pad(tensor, [0, pad_size], value=pad_value)


def pad_dim(
    x: Tensor,
    target_length: int,
    align: str = "left",
    fill_value: float = 0.0,
    dim: int = -1,
    mode: str = "constant",
) -> Tensor:
    """Generic function for pad a specific tensor dimension."""
    missing = max(target_length - x.shape[dim], 0)

    if missing == 0:
        return x

    if align == "left":
        missing_left = 0
        missing_right = missing
    elif align == "right":
        missing_left = missing
        missing_right = 0
    elif align == "center":
        missing_left = missing // 2 + missing % 2
        missing_right = missing // 2
    elif align == "random":
        missing_left = int(torch.randint(low=0, high=missing + 1, size=()).item())
        missing_right = missing - missing_left
    else:
        raise ValueError(f"Invalid argument {align=}. (expected one of {PAD_ALIGNS})")

    # Note: pad_seq : [pad_left_dim_-1, pad_right_dim_-1, pad_left_dim_-2, pad_right_dim_-2, ...)
    idx = len(x.shape) - (dim % len(x.shape)) - 1
    pad_seq = [0 for _ in range(len(x.shape) * 2)]
    pad_seq[idx * 2] = missing_left
    pad_seq[idx * 2 + 1] = missing_right
    x = F.pad(x, pad_seq, mode=mode, value=fill_value)
    return x


def pad_dims(
    x: Tensor,
    target_lengths: Union[int, Iterable[int]],
    aligns: Union[str, Iterable[str]] = "left",
    fill_value: float = 0.0,
    dims: Iterable[int] = (-1,),
    mode: str = "constant",
) -> Tensor:
    """Generic function to pad multiple dimensions."""
    dims = list(dims)
    if len(dims) == 0:
        raise ValueError(
            f"Invalid argument {dims=}. (cannot use an empty list of dimensions)"
        )

    if isinstance(target_lengths, int):
        target_lengths = [target_lengths] * len(dims)
    else:
        target_lengths = list(target_lengths)

    if isinstance(aligns, str):
        aligns = [aligns] * len(dims)
    else:
        aligns = list(aligns)

    if len(target_lengths) != len(dims):
        raise ValueError(
            f"Invalid number of targets lengths ({len(target_lengths)}) with the number of dimensions ({len(dims)})."
        )

    if len(aligns) != len(dims):
        raise ValueError(
            f"Invalid number of aligns ({len(aligns)}) with the number of dimensions ({len(dims)})."
        )

    pad_seq = [0 for _ in range(len(x.shape) * 2)]
    for target_length, dim, align in zip(target_lengths, dims, aligns):
        missing = max(target_length - x.shape[dim], 0)

        if align == "left":
            missing_left = 0
            missing_right = missing
        elif align == "right":
            missing_left = missing
            missing_right = 0
        elif align == "center":
            missing_left = missing // 2 + missing % 2
            missing_right = missing // 2
        elif align == "random":
            missing_left = int(torch.randint(low=0, high=missing + 1, size=()).item())
            missing_right = missing - missing_left
        else:
            ALIGNS = ("left", "right", "center", "random")
            raise ValueError(f"Invalid argument {align=}. (expected one of {ALIGNS})")

        # Note: pad_seq : [pad_left_dim_-1, pad_right_dim_-1, pad_left_dim_-2, pad_right_dim_-2, ...)
        idx = len(x.shape) - (dim % len(x.shape)) - 1
        assert pad_seq[idx * 2] == 0 and pad_seq[idx * 2 + 1] == 0
        pad_seq[idx * 2] = missing_left
        pad_seq[idx * 2 + 1] = missing_right

    x = F.pad(x, pad_seq, mode=mode, value=fill_value)
    return x


def stack_tensors(tensors: Iterable[Tensor], pad_value: float) -> Tensor:
    tensors = list(tensors)
    if len(tensors) == 0:
        raise ValueError(f"Invalid argument {tensors=}.")

    d0_sum = sum(tensor.shape[0] for tensor in tensors)
    max_shapes = tuple(
        max(tensor.shape[i] for tensor in tensors) for i in range(1, tensors[0].ndim)
    )

    factory_kws: dict[str, Any] = dict(dtype=tensors[0].dtype, device=tensors[0].device)
    output = torch.full((d0_sum,) + max_shapes, pad_value, **factory_kws)

    d0_start = 0
    for tensor in tensors:
        d0_end = d0_start + tensor.shape[0]
        slices = (slice(d0_start, d0_end),) + tuple(
            slice(shape_i) for shape_i in tensor.shape[1:]
        )
        output[slices] = tensor
        d0_start = d0_end
    return output


def pad_and_stack(x: Iterable[Tensor], dim: int = -1) -> Tensor:
    if isinstance(x, Tensor):
        return x
    max_len = max(xi.shape[dim] for xi in x)
    x = [pad_dim(xi, max_len, dim=dim) for xi in x]
    x = torch.stack(x, dim=0)
    return x
