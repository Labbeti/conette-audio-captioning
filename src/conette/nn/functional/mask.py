#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Iterable, Optional, Union

import torch

from torch import Tensor


def masked_mean(
    tensor: Tensor,
    non_pad_mask: Tensor,
    dim: Union[None, int, Iterable[int]] = None,
) -> Tensor:
    """
    :param tensor: (N, ...)
    :param non_pad_mask: Non-padding mask, should be broadcastable with argument tensor and reduced with argument dim.
    :param dim: Optional dim to reduce. defaults to None.
    :returns: The reduced tensor along the specified dim(s).
    """
    if dim is None:
        dim = ()
    elif isinstance(dim, int):
        dim = (dim,)
    else:
        dim = tuple(dim)

    masked_tensor = tensor * non_pad_mask
    return masked_tensor.sum(dim=dim) / non_pad_mask.sum(dim=dim).clamp(min=1.0)


def masked_sum(
    tensor: Tensor,
    non_pad_mask: Tensor,
    dim: Union[None, int, Iterable[int]] = None,
) -> Tensor:
    """
    :param tensor: (N, ...)
    :param non_pad_mask: Non-padding mask, should be broadcastable with argument tensor and reduced with argument dim.
    :param dim: Optional dim to reduce. defaults to None.
    :returns: The reduced tensor along the specified dim(s).
    """
    if dim is None:
        dim = ()
    elif isinstance(dim, int):
        dim = (dim,)
    else:
        dim = tuple(dim)

    masked_tensor = tensor * non_pad_mask
    return masked_tensor.sum(dim=dim)


def generate_square_subsequent_mask(
    size: int,
    device: Union[str, torch.device, None] = None,
) -> Tensor:
    r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
    Unmasked positions are filled with float(0.0).

    NOTE : BASED ON PYTORCH IMPLEMENTATION in nn.Transformer

    :param size: The size of the output tensor.
    :param device: The device of the output tensor.
    :returns: A tensor of shape (size, size)

    Example 1
    ----------
    >>> generate_square_subsequent_mask(6)
    tensor([[0., -inf, -inf, -inf, -inf, -inf],
            [0., 0., -inf, -inf, -inf, -inf],
            [0., 0., 0., -inf, -inf, -inf],
            [0., 0., 0., 0., -inf, -inf],
            [0., 0., 0., 0., 0., -inf],
            [0., 0., 0., 0., 0., 0.]])
    """
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if isinstance(device, str):
        device = torch.device(device)

    mask = (torch.triu(torch.ones(size, size, device=device)) == 1).transpose(0, 1)
    mask = (
        mask.float()
        .masked_fill(mask == 0, float("-inf"))
        .masked_fill(mask == 1, float(0.0))
    )
    return mask


def generate_shifted_sq_mask(
    size: int,
    right_shift: int = 0,
    device: Union[str, torch.device, None] = None,
) -> Tensor:
    """
    Example 1
    ----------
    >>> generate_shifted_sq_mask(6, 2)
    tensor([[0., 0., 0., -inf, -inf, -inf],
            [0., 0., 0., 0., -inf, -inf],
            [0., 0., 0., 0., 0., -inf],
            [0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0.]])
    """
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if isinstance(device, str):
        device = torch.device(device)

    mask = (
        torch.triu(torch.ones(size, size, device=device), diagonal=-right_shift) == 1
    ).transpose(0, 1)
    mask = (
        mask.float()
        .masked_fill(mask == 0, float("-inf"))
        .masked_fill(mask == 1, float(0.0))
    )
    return mask


def tensor_to_lengths(
    tensor: Tensor,
    pad_value: Optional[float] = None,
    end_value: Optional[float] = None,
    dim: int = -1,
) -> Tensor:
    """
    You must provide a value for one of pad_value or end_value. If both values are provided, the end_value is ignored.

    Note: The end_value is not included in the length of the sentence.

    Example 1
    -----------
    ```
    >>> input = torch.as_tensor([1, 10, 20, 2, 0, 0])
    >>> tensor_to_lengths(input, end_value=2)
    tensor(3)
    ```

    Example 2
    -----------
    ```
    >>> input = torch.as_tensor([1, 10, 20, 2, 0, 0])
    >>> tensor_to_lengths(input, pad_value=0)
    tensor(4)
    ```

    :param tensor: Tensor of shape (N, *)
    :param pad_value: The pad value used in tensor. defaults to None.
    :param end_value: The end value used in tensor. defaults to None.
    :param dim: The dimension of the length. defaults to -1.
    :returns: The lengths as IntTensor of shape (N,)
    """
    if (pad_value is None) == (end_value is None):
        raise ValueError(
            "Invalid arguments. Please provide only one of the arguments: end_value, pad_value."
        )

    if pad_value is not None:
        non_pad_mask = tensor != pad_value
        lengths = non_pad_mask.sum(dim=dim)

    elif end_value is not None:
        contains_eos = (tensor == end_value).any(dim=dim)
        indexes_eos = (tensor == end_value).int().argmax(dim=dim)
        lengths = torch.where(contains_eos, indexes_eos, tensor.shape[dim])

    else:
        raise ValueError(
            "Invalid arguments. Please provide only one of the arguments : end_value, pad_value."
        )

    return lengths


def tensor_to_non_pad_mask(
    tensor: Tensor,
    pad_value: Optional[float] = None,
    end_value: Optional[float] = None,
    include_end: bool = False,
) -> Tensor:
    """Convert tensor to non-pad binary mask.
    You must provide a value for one of pad_value or end_value. If both values are provided, the end_value is ignored.

    :param tensor: A tensor of values. If end_value is given instead of pad_value, the number of dims must be <= 2.
    :param pad_value: The pad value used in tensor. defaults to None.
    :param end_value: The end value used in tensor. defaults to None.
    :param include_end: If True, the end value will be included in non_pad_mask.
        This parameter is ignored if end_value is None.
        defaults to False.
    :returns: A binary mask representing the non-padded values. Shape is the same than the input tensor.

    Example 1
    -----------
    ```
    >>> input = torch.as_tensor([1, 10, 20, 2, 0, 0])
    >>> tensor_to_pad_mask(input, end_value=2)
    tensor([True, True, True, False, False, False])
    ```
    """
    if (pad_value is None) == (end_value is None):
        raise ValueError(
            "Invalid arguments. Please provide only one of the arguments: end_value, pad_value."
        )

    if pad_value is not None:
        non_pad_mask = tensor.ne(pad_value)

    elif end_value is not None:
        if tensor.ndim > 2:
            raise ValueError(
                f"Cannot compute non_pad_mask for with more than 2 dimensions with {end_value=}. (found {tensor.ndim=})"
            )
        lengths = tensor_to_lengths(tensor, end_value=end_value, dim=-1)
        non_pad_mask = lengths_to_non_pad_mask(lengths, tensor.shape[-1], include_end)

    else:
        raise ValueError(
            "Invalid arguments. Please provide only one of the arguments : end_value, pad_value."
        )

    return non_pad_mask


def tensor_to_pad_mask(
    tensor: Tensor,
    pad_value: Optional[float] = None,
    end_value: Optional[float] = None,
    include_end: bool = True,
) -> Tensor:
    """Convert tensor to pad binary mask.
    You must provide a value for one of pad_value or end_value. If both values are provided, the end_value is ignored.

    :param tensor: A tensor of values. If end_value is given instead of pad_value, the number of dims must be <= 2.
    :param pad_value: The pad value used in tensor. defaults to None.
    :param end_value: The end value used in tensor. defaults to None.
    :param include_end: If True, the end value will be included in pad_mask. defaults to True.
    :returns: A binary mask representing the padded values. Shape is the same than the input tensor.

    Example 1
    -----------
    ```
    >>> input = torch.as_tensor([1, 10, 20, 2, 0, 0])
    >>> tensor_to_pad_mask(input, end_value=2)
    tensor([False, False, False, True, True, True])
    ```
    """
    non_pad_mask = tensor_to_non_pad_mask(tensor, pad_value, end_value, not include_end)
    return non_pad_mask.logical_not()


def non_pad_mask_to_lengths(mask: Tensor, dim: int = -1) -> Tensor:
    return mask.sum(dim=dim)


def pad_mask_to_lengths(mask: Tensor, dim: int = -1) -> Tensor:
    return mask.shape[dim] - non_pad_mask_to_lengths(mask, dim)


def lengths_to_non_pad_mask(
    lengths: Tensor,
    max_len: Optional[int],
    include_end: bool = False,
) -> Tensor:
    """Convert lengths to binary mask of non-padded values.

    :param lengths: (bsize,)
    :param max_len: Optional int for indicate the maximal length.
        If None, it will be set to lengths.max().
        defaults to None.
    :param include_end: If True, the value at index of len will be True in returned mask.
        defaults to False.
    :returns: (bsize, max_len)

    Example 1
    -----------
    ```
    >>> input = torch.as_tensor([4, 2, 0, 3, 0])
    >>> lengths_to_non_pad_mask(input, max_len=6, include_end=False)
    tensor([[True, True, True, True, False, False],
            [True, True, False, False, False, False],
            [False, False, False, False, False, False],
            [True, True, True, False, False, False],
            [False, False, False, False, False, False]])
    ```
    """
    dim = -1
    if max_len is None:
        max_len = int(lengths.max(dim=dim)[0].item())
    indexes = torch.arange(0, max_len, device=lengths.device)
    lengths = lengths.unsqueeze(dim=-1)
    non_pad_mask = indexes <= lengths if include_end else indexes < lengths
    return non_pad_mask


def lengths_to_pad_mask(
    lengths: Tensor,
    max_len: Optional[int],
    include_end: bool = True,
) -> Tensor:
    """Convert lengths to binary mask of padded values.

    :param lengths: (bsize,)
    :param max_len: Optional int for indicate the maximal length.
        If None, it will be set to lengths.max().
        defaults to None.
    :param include_end: If True, the last value of each size will be set to False. defaults to True.
    :returns: (bsize, max_len)

    Example 1
    -----------
    ```
    >>> input = torch.as_tensor([4, 2, 0, 3, 0])
    >>> lengths_to_non_pad_mask(input, max_len=None, include_end=True)
    tensor([[False, False, False, False],
            [False, False, True, True],
            [True, True, True, True],
            [False, False, False, True],
            [True, True, True, True]])
    ```
    """
    non_pad_mask = lengths_to_non_pad_mask(lengths, max_len, not include_end)
    return non_pad_mask.logical_not()


def tensors_list_to_lengths(tensors: list[Tensor], dim: int = -1) -> Tensor:
    """Return the size of the tensor at a specific dim.

    :param tensors: List of N tensors.
    :param dim: The dimension of the output sizes. defaults to -1.
    :returns: A tensor of size N with the sizes.
    """
    return torch.as_tensor([tensor.shape[dim] for tensor in tensors])


def tensor_to_tensors_list(
    tensor: Tensor,
    pad_value: Optional[float] = None,
    end_value: Optional[float] = None,
    non_pad_mask: Optional[Tensor] = None,
    lengths: Optional[Tensor] = None,
) -> list[Tensor]:
    """
    You must provide a value for one of pad_value, end_value non_pad_mask or lengths.
    If multiple values are provided, only one will be used and the priority order is [pad_value, end_value non_pad_mask, lengths].

    :param tensor: A tensor of values. If end_value is given instead of pad_value, the number of dims must be <= 2.
    :param pad_value: The pad value used in tensor. defaults to None.
    :param end_value: The end value used in tensor. defaults to None.
    :param non_pad_mask: Non-adding mask used in tensor. defaults to None.
    :param lengths: Lengths of used in tensor. defaults to None.
    :returns: A list of N tensors of shape (*)
    """

    if pad_value is not None:
        lengths = tensor_to_lengths(tensor, pad_value=pad_value)
        return tensor_to_tensors_list(tensor, lengths=lengths)

    elif end_value is not None:
        lengths = tensor_to_lengths(tensor, end_value=end_value)
        return tensor_to_tensors_list(tensor, lengths=lengths)

    elif non_pad_mask is not None:
        lengths = non_pad_mask_to_lengths(non_pad_mask)
        return tensor_to_tensors_list(tensor, lengths=lengths)

    elif lengths is not None:
        slices_lst = [
            [slice(None) for _ in range(tensor.ndim)] + [slice(0, len_)]
            for len_ in lengths
        ]
        tensors = [tensor[slices] for slices in slices_lst]

    else:
        raise ValueError(
            "Invalid arguments. Please provide only one of the arguments : end_value, pad_value, mask, lengths."
        )

    return tensors
