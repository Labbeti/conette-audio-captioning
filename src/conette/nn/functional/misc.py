#!/usr/bin/env python
# -*- coding: utf-8 -*-

import copy

from typing import (
    Any,
    Callable,
    Generator,
    Iterable,
    Mapping,
    Optional,
    Sequence,
    TypeVar,
    Union,
)

import torch

from torch import nn, Tensor
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence

from conette.nn.functional.crop import crop_dim
from conette.nn.functional.mask import tensor_to_pad_mask
from conette.nn.functional.pad import pad_dim


T = TypeVar("T")


def sharpen(x: Tensor, temperature: float, dim: int) -> Tensor:
    x = x ** (1.0 / temperature)
    x = x / x.norm(p=1, dim=dim, keepdim=True)  # type: ignore
    return x


def sort_batch_by_lengths(
    batch: dict[str, Tensor],
    key: str = "audio_lens",
) -> dict[str, Tensor]:
    indexes = torch.argsort(batch["audio_lens"])

    keys = list(batch.keys())
    result = {}
    for key in keys:
        value = batch[key]
        del batch[key]
        if isinstance(value, Tensor):
            result[key] = value[indexes]
        elif isinstance(value, Sequence):
            result[key] = [value[i] for i in indexes]
        else:
            raise ValueError(f"Unsupported value type. ({value=})")
    return result


def count_params(model: nn.Module, only_trainable: bool = False) -> int:
    return sum(
        param.numel()
        for param in model.parameters()
        if not only_trainable or param.requires_grad
    )


def module_eq(m1: nn.Module, m2: nn.Module) -> bool:
    n_params1 = sum(1 for _ in m1.parameters())
    n_params2 = sum(1 for _ in m2.parameters())
    return n_params1 == n_params2 and all(
        p1.shape == p2.shape and p1.eq(p2).all()
        for p1, p2 in zip(m1.parameters(), m2.parameters())
    )


def module_mean(modules: Iterable[nn.Module], with_buffers: bool = True) -> nn.Module:
    modules = list(modules)
    assert len(modules) > 0

    output = copy.deepcopy(modules[0])

    all_params = [output.parameters()] + [module.parameters() for module in modules]
    for params in zip(*all_params):
        params[0][:] = torch.stack(params[1:]).mean(dim=0)

    if with_buffers:
        all_buffers = [output.buffers()] + [module.buffers() for module in modules]
        for buffers in zip(*all_buffers):
            if buffers[0].is_floating_point():
                buffers[0][:] = torch.stack(buffers[1:]).mean(dim=0)

    return output


def tensor_eq(t1: Tensor, t2: Tensor) -> bool:
    return t1.shape == t2.shape and bool(t1.eq(t2).all().item())


def tensor_close(
    t1: Tensor,
    t2: Tensor,
    rtol: float = 0.00001,
    atol: float = 1e-8,
    equal_nan: bool = False,
) -> bool:
    return t1.shape == t2.shape and torch.allclose(
        t1, t2, rtol=rtol, atol=atol, equal_nan=equal_nan
    )


def tensors_list_to_tensor(
    tensors: list[Tensor],
    pad_value: float,
    batch_first: bool = True,
) -> Tensor:
    """Pad a list of tensors to a tensor.

    :param tensors: List of N tensors with the same number of dims and with a different size at the first dim.
    :param pad_value: The value used for fill the tensors.
    :returns: (N, *)
    """
    return pad_sequence(tensors, batch_first=batch_first, padding_value=pad_value)


def reduce_mask(
    mask: Tensor,
    target_size: int,
    dim: int = -1,
    round_fn: Union[str, Callable[[Tensor], Tensor]] = "round",
) -> Tensor:
    if isinstance(round_fn, str):
        if round_fn == "round":
            round_fn = torch.round
        elif round_fn == "floor":
            round_fn = torch.floor
        elif round_fn == "ceil":
            round_fn = torch.ceil
        else:
            raise ValueError(f"Invalid argument {round_fn=}.")

    orig_size = mask.shape[dim]
    factor = target_size / orig_size
    indexes = round_fn(torch.arange(0, orig_size, step=factor))
    slices: list[Any] = [slice(None) for _ in range(mask.ndim)]
    slices[dim] = indexes
    return mask[slices]


def stack_tensors_rec(
    sequence: Union[Tensor, int, float, tuple, list],
    relaxed: bool = False,
    dtype: Union[None, torch.dtype] = None,
    device: Union[str, torch.device, None] = "auto",
) -> Union[Tensor, list]:
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if isinstance(device, str):
        device = torch.device(device)

    def stack_tensors_rec_impl(
        seq: Union[Tensor, int, float, tuple, list]
    ) -> Union[Tensor, list]:
        if isinstance(seq, Tensor):
            return seq.to(dtype=dtype, device=device)
        elif isinstance(seq, (int, float)):
            return torch.as_tensor(seq, dtype=dtype, device=device)  # type: ignore
        elif isinstance(seq, (list, tuple)):
            if all(isinstance(elt, (int, float)) for elt in seq):  # type: ignore
                return torch.as_tensor(seq, dtype=dtype, device=device)  # type: ignore

            seq = [stack_tensors_rec_impl(elt) for elt in seq]  # type: ignore
            if all(isinstance(elt, Tensor) for elt in seq):
                shapes = [elt.shape for elt in seq]  # type: ignore
                if len(seq) == 0 or all(shape == shapes[0] for shape in shapes):
                    return torch.stack(seq)  # type: ignore
                elif relaxed:
                    return seq
                else:
                    raise ValueError(
                        f"Cannot stack tensors of different shapes. (found {shapes=})"
                    )
            elif relaxed:
                return seq
            else:
                raise ValueError("Cannot stack tensors of different shape or types.")
        else:
            raise TypeError(
                f"Invalid type {type(seq)}. (expected Tensor, int, float, list or tuple)"
            )

    sequence = stack_tensors_rec_impl(sequence)
    return sequence


def batch_conv2d_naive(x: Tensor, weight: Tensor) -> Tensor:
    """
    Conv2d with a batch of distincts weights. (slow version using Conv2d multiple times)

    :param x: (bsize, in_channels, x_width, x_height)
    :param weight: (bsize, out_channels, in_channels, weight_width, weight_height)
    :returns: (bsize, out_channels, x_width, x_height)
    """
    if (
        x.ndim != 4
        or weight.ndim != 5
        or x.shape[0] != weight.shape[0]
        or x.shape[1] != weight.shape[2]
    ):
        raise ValueError(
            f"Invalid arguments for batch_conv2d_naive. ({x.shape=}; {weight.shape=})"
        )

    x = torch.stack(
        [
            F.conv2d(x_i.unsqueeze(dim=0), weight=w_i, bias=None, padding="same")
            for x_i, w_i in zip(x, weight)
        ]
    )
    x = x.squeeze(dim=1)
    return x.contiguous()


def batch_conv2d(x: Tensor, weight: Tensor) -> Tensor:
    """
    Conv2d with a batch of distincts weights. (faster version using only 1 Conv2d with groups)

    :param x: (bsize, in_channels, x_width, x_height)
    :param weight: (bsize, out_channels, in_channels, weight_width, weight_height)
    :returns: (bsize, out_channels, x_width, x_height)
    """
    if (
        x.ndim != 4
        or weight.ndim != 5
        or x.shape[0] != weight.shape[0]
        or x.shape[1] != weight.shape[2]
    ):
        raise ValueError(
            f"Invalid arguments for batch_conv2d. ({x.shape=}; {weight.shape=})"
        )

    x_width, x_height = x.shape[2:]
    bsize, out_channels, in_channels, weight_width, weight_height = weight.shape
    x = x.view(1, bsize * in_channels, x_width, x_height).contiguous()
    weight = weight.view(
        bsize * out_channels, in_channels, weight_width, weight_height
    ).contiguous()
    x = F.conv2d(x, weight=weight, bias=None, padding="same", groups=bsize)
    x = x.view(bsize, out_channels, x_width, x_height)
    return x.contiguous()


def move_to_rec(
    x: Any,
    *args,
    predicate: Optional[Callable[[Any], bool]] = None,
    **kwargs,
) -> Any:
    if isinstance(x, (Tensor, nn.Module)):
        if predicate is None or predicate(x):
            return x.to(*args, **kwargs)
        else:
            return x
    elif isinstance(x, (str, float, int)):
        return x
    elif isinstance(x, Mapping):
        return {
            k: move_to_rec(v, predicate=predicate, *args, **kwargs)
            for k, v in x.items()
        }
    elif isinstance(x, Iterable):
        generator = (move_to_rec(xi, predicate=predicate, *args, **kwargs) for xi in x)
        if isinstance(x, Generator):
            return generator
        elif isinstance(x, tuple):
            return tuple(generator)
        elif isinstance(x, list):
            return list(generator)
        else:
            return list(generator)
    else:
        return x


def pad_crop_dim(
    x: Tensor,
    target_length: int,
    align: str = "left",
    fill_value: float = 0.0,
    dim: int = -1,
    mode: str = "constant",
) -> Tensor:
    if x.shape[dim] == target_length:
        return x
    elif x.shape[dim] > target_length:
        return crop_dim(x, target_length, align, dim)
    else:
        return pad_dim(x, target_length, align, fill_value, dim, mode)


def pad_and_cat(
    tensors: Iterable[Tensor],
    dim_pad: int,
    dim_cat: int,
    fill_value: float,
    align: str = "left",
) -> Tensor:
    target_len = max(tensor.shape[dim_pad] for tensor in tensors)
    tensors = [
        pad_dim(tensor, target_len, align, fill_value, dim_pad) for tensor in tensors
    ]
    tensors = torch.cat(tensors, dim=dim_cat)
    return tensors


def pad_and_stack(
    tensors: Iterable[Tensor],
    dim_pad: int,
    fill_value: float,
    align: str = "left",
) -> Tensor:
    target_len = max(tensor.shape[dim_pad] for tensor in tensors)
    tensors = [
        pad_dim(tensor, target_len, align, fill_value, dim_pad) for tensor in tensors
    ]
    tensors = torch.stack(tensors)
    return tensors


def can_be_stacked(tensors: list[Tensor]) -> bool:
    """Returns True if the list contains tensor that can be stacked with :func:`~torch.stack`."""
    return len(tensors) == 0 or all(
        tensor.shape == tensors[0].shape for tensor in tensors
    )


def can_be_stacked_v2(tensors: list[Tensor]) -> bool:
    """Returns True if the list contains tensor that can be stacked with :func:`~torch.stack`."""
    if len(tensors) == 0:
        return False
    else:
        shapes = torch.as_tensor([tensor.shape for tensor in tensors])
        return shapes.eq(shapes[0]).all().item()  # type: ignore


def can_be_padded(tensors: list[Tensor]) -> bool:
    """Returns True if the list contains tensor that can be stacked with :func:`~torch.nn.utils.rnn.pad_sequence`."""
    return len(tensors) == 0 or all(
        tensor.shape[1:] == tensors[0].shape[1:] for tensor in tensors
    )


def check_pred(
    pred: Tensor,
    pad_id: int = 0,
    bos_id: int = 1,
    eos_id: int = 2,
    unk_id: int = 3,
) -> tuple[bool, bool, bool, bool]:
    """Check if a prediction tensor is valid.

    :param pred: (bsize, pred_len)
    :returns: (sos_at_start, eos_at_end, no_unk, pad_at_end)
    """
    assert pred.ndim == 2
    dim = 1
    sos_at_start = pred[:, 0].eq(bos_id).all().item()
    contains_eos = (pred == eos_id).any(dim=dim)
    eos_at_end = contains_eos.all().item()
    no_unk = pred.ne(unk_id).all().item()

    indexes_eos = (pred == eos_id).int().argmax(dim=dim)
    lengths = torch.where(contains_eos, indexes_eos, pred.shape[dim])
    pad_at_end = True
    for pred_i, len_i in zip(pred, lengths):
        pad_at_end = pad_at_end and pred_i[len_i + 1 :].eq(pad_id).all().item()

    return sos_at_start, eos_at_end, no_unk, pad_at_end  # type: ignore


def find(
    tensor: Tensor,
    value: Any,
    default: Union[None, Tensor, int, float] = None,
    dim: int = -1,
) -> Tensor:
    """Return the index of the first occurrence of value in a tensor."""
    assert tensor.ndim > 0
    mask = tensor.eq(value)
    contains_eos = mask.any(dim=dim)
    indexes_eos = mask.int().argmax(dim=dim)

    if default is None:
        if not contains_eos.all():
            raise RuntimeError(f"Cannot find {value=} in tensor.")
        return indexes_eos
    else:
        output = torch.where(contains_eos, indexes_eos, default)
        return output


def pad_after_eos(pred: Tensor, eos_id: int, pad_id: int) -> Tensor:
    """
    :param pred: (bsize, pred_size)
    :returns: (bsize, pred_size)
    """
    pad_mask = tensor_to_pad_mask(
        pred,
        end_value=eos_id,
        include_end=False,
    )
    pred[pad_mask] = pad_id
    return pred


def prepend_value(pred: Tensor, value: Union[int, float, bool], dim: int = 1) -> Tensor:
    """
    :param pred: (bsize, pred_size)
    :returns: (bsize, pred_size+1)
    """
    shape = list(pred.shape)
    shape[dim] = 1
    values = torch.full(shape, value, dtype=pred.dtype, device=pred.device)
    pred = torch.cat((values, pred), dim=dim)
    return pred


def cat_padded_batch(
    x1: Tensor,
    x1_lens: Tensor,
    x2: Tensor,
    x2_lens: Tensor,
    seq_dim: int,
    batch_dim: int = 0,
) -> tuple[Tensor, Tensor]:
    assert x1.ndim == x2.ndim
    assert x1_lens.ndim == x2_lens.ndim == 1
    assert (
        x1.shape[batch_dim]
        == x2.shape[batch_dim]
        == x1_lens.shape[0]
        == x2_lens.shape[0]
    )

    x12_lens = x1_lens + x2_lens
    sum_size_12 = x1.shape[seq_dim] + x2.shape[seq_dim]

    x12 = pad_dim(x1, sum_size_12, dim=seq_dim)
    kwd: dict[str, Any] = dict(device=x1.device, dtype=torch.long)
    indexes = torch.arange(x2_lens.max().item(), **kwd)

    unsq_x1_lens = x1_lens
    ndim = x1.ndim
    for i in range(ndim):
        if i != (seq_dim % ndim):
            indexes = indexes.unsqueeze(dim=i)
        if i != (batch_dim % ndim):
            unsq_x1_lens = unsq_x1_lens.unsqueeze(dim=i)

    expand_size = list(x2.shape)
    expand_size[seq_dim] = -1
    indexes = indexes.expand(*expand_size)
    indexes = indexes + unsq_x1_lens
    x12.scatter_(seq_dim, indexes, x2)

    max_size_12 = int(x12_lens.max().item())
    if max_size_12 < sum_size_12:
        slices = [slice(None) for _ in range(ndim)]
        slices[seq_dim] = slice(max_size_12)
        x12 = x12[slices]
    x12 = x12.contiguous()
    return x12, x12_lens


def detect_time_dim(shapes: Tensor, default_if_all_eq: int = -1) -> int:
    assert shapes.ndim == 2
    dim = default_if_all_eq
    for i in range(shapes.shape[1]):
        lens_i = shapes[:, i]
        if lens_i.ne(lens_i[0]).any():
            dim = i
            break
    return dim
