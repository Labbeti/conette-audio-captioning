#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Any, Sequence

import h5py
import torch

from torch import Tensor


# Force this encoding
HDF_ENCODING = "utf-8"
# Type for strings
HDF_STRING_DTYPE = h5py.string_dtype(HDF_ENCODING, None)
# Type for empty lists
HDF_VOID_DTYPE = h5py.opaque_dtype("V1")
# Key suffix to store tensor shapes (because they are padded in hdf file)
SHAPE_SUFFIX = "_shape"


def all_eq(seq: Sequence[Any]) -> bool:
    """Returns True if all element in list are the same."""
    if len(seq) == 0:
        return True
    else:
        first = seq[0]
        return all(first == elt for elt in seq[1:])


def get_inverse_perm(indexes: Tensor, dim: int = -1) -> Tensor:
    """Return inverse permutation indexes.

    :param indexes: Original permutation indexes as tensor of shape (..., N).
    :param dim: Dimension of indexes. defaults to -1.
    :returns: Inverse permutation indexes of shape (..., N).
    """
    arange = torch.arange(
        indexes.shape[dim],
        dtype=indexes.dtype,
        device=indexes.device,
    )
    arange = arange.expand(*indexes.shape)
    indexes_inv = torch.empty_like(indexes)
    indexes_inv = indexes_inv.scatter(dim, indexes, arange)
    return indexes_inv
