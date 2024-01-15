#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Utilities for conversion between classes indices, multihot, names and probabilities for multilabel classification.
"""

from typing import Dict, TypeVar, Union

import torch

from torch import Tensor


T = TypeVar("T")


def indices_to_multihot(
    indices: Union[list[list[int]], list[Tensor]],
    num_classes: int,
) -> Tensor:
    bsize = len(indices)
    multihot = torch.full((bsize, num_classes), False, dtype=torch.bool)
    for i, indices_i in enumerate(indices):
        if isinstance(indices_i, list):
            indices_i = torch.as_tensor(indices_i, dtype=torch.long)
        multihot[i].scatter_(0, indices_i, True)
    return multihot


def indices_to_names(
    indices: Union[list[list[int]], list[Tensor]],
    idx_to_name: Dict[int, T],
) -> list[list[T]]:
    names = []
    for indices_i in indices:
        names_i = [idx_to_name[idx] for idx in indices_i]  # type: ignore
        names.append(names_i)
    return names


def multihot_to_indices(
    multihot: Tensor,
) -> list[list[int]]:
    preds = []
    for multihot_i in multihot:
        preds_i = torch.where(multihot_i)[0].tolist()
        preds.append(preds_i)
    return preds


def multihot_to_names(
    multihot: Tensor,
    idx_to_name: Dict[int, T],
) -> list[list[T]]:
    indices = multihot_to_indices(multihot)
    names = indices_to_names(indices, idx_to_name)
    return names


def names_to_indices(
    names: list[list[T]],
    idx_to_name: Dict[int, T],
) -> list[list[int]]:
    name_to_idx = {name: idx for idx, name in idx_to_name.items()}
    indices = []
    for names_i in names:
        indices_i = [name_to_idx[name] for name in names_i]
        indices.append(indices_i)
    return indices


def names_to_multihot(
    names: list[list[T]],
    idx_to_name: Dict[int, T],
) -> Tensor:
    indices = names_to_indices(names, idx_to_name)
    multihot = indices_to_multihot(indices, len(idx_to_name))
    return multihot


def probs_to_indices(
    probs: Tensor,
    threshold: Union[float, Tensor],
) -> list[list[int]]:
    multihot = probs_to_multihot(probs, threshold)
    preds = multihot_to_indices(multihot)
    return preds


def probs_to_multihot(
    probs: Tensor,
    threshold: Union[float, Tensor],
) -> Tensor:
    if probs.ndim != 2:
        raise ValueError(
            "Invalid argument probs. (expected a batch of probabilities of shape (N, n_classes))."
        )
    nb_classes = probs.shape[1]

    if isinstance(threshold, Tensor) and threshold.ndim == 1:
        threshold = threshold.item()

    if isinstance(threshold, (float, int)):
        threshold = torch.full(
            (nb_classes,), threshold, dtype=torch.float, device=probs.device
        )
    else:
        if threshold.shape[1] != nb_classes:
            raise ValueError("Invalid argument threshold.")
        threshold = threshold.to(device=probs.device)

    multihot = probs >= threshold
    return multihot


def probs_to_names(
    probs: Tensor,
    threshold: Union[float, Tensor],
    idx_to_name: Dict[int, T],
) -> list[list[T]]:
    indices = probs_to_indices(probs, threshold)
    names = indices_to_names(indices, idx_to_name)
    return names
