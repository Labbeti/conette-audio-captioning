#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging

from typing import Any, Iterable, Optional

import torch

from torch import Tensor

from conette.datasets.hdf.common import SHAPE_SUFFIX
from conette.nn.functional.misc import can_be_stacked
from conette.nn.functional.pad import pad_sequence_rec
from conette.utils.collections import all_eq


pylog = logging.getLogger(__name__)


class CollateDict:
    """Collate list of dict into a dict of list WITHOUT auto-padding."""

    def __call__(self, items_lst: list[dict[str, Any]]) -> dict[str, list[Any]]:
        common_keys = items_lst[0].keys()
        for i in range(1, len(items_lst)):
            common_keys = [key for key in common_keys if key in items_lst[i].keys()]
        return {key: [item[key] for item in items_lst] for key in common_keys}


class AdvancedCollateDict:
    def __init__(
        self,
        pad_values: Optional[dict[str, Any]] = None,
        crop_keys: Iterable[str] = (),
        batch_keys: Optional[Iterable[str]] = None,
    ) -> None:
        """Collate list of dict into a dict of list WITH auto-padding for given keys.

        :param pad_values: The dictionnary of key with pad value.
        :param crop_keys: Depreciated crop keys.
        :param batch_keys: The expected batch keys.
        """

        if pad_values is None:
            pad_values = {}
        crop_keys = list(dict.fromkeys(crop_keys))
        if batch_keys is not None:
            batch_keys = list(batch_keys)

        super().__init__()
        self._pad_values = pad_values
        self._crop_keys = crop_keys
        self._batch_keys = batch_keys

    def __call__(self, batch_lst: list[dict[str, Any]]) -> dict[str, Any]:
        if self._batch_keys is None:
            # Intersection of keys and keep the same order
            batch_keys = list(batch_lst[0].keys())
            for item in batch_lst[1:]:
                batch_keys = [key for key in batch_keys if key in item.keys()]
        else:
            batch_keys = self._batch_keys

        batch_dic: dict[str, Any] = {
            key: [item[key] for item in batch_lst] for key in batch_keys
        }
        batch_dic = {
            key: (torch.stack(items) if key.endswith(SHAPE_SUFFIX) else items)
            for key, items in batch_dic.items()
        }

        for key in batch_keys:
            items = batch_dic[key]
            key_shape = f"{key}{SHAPE_SUFFIX}"

            if key in self._crop_keys:
                shapes = batch_dic[key_shape]
                max_shape = shapes.max(dim=0).values

                slices = [slice(shape_i) for shape_i in max_shape]
                for i in range(len(items)):
                    items[i] = items[i][slices]
                items = torch.stack(items)

            elif key in self._pad_values.keys():
                if key_shape not in batch_dic.keys():
                    try:
                        shapes = [item.shape for item in items]
                    except AttributeError as err:
                        raise err
                    if not all_eq(map(len, shapes)):
                        pylog.error(
                            f"Cannot collate list of tensors with a different number of dims. ({shapes=})"
                        )
                        continue

                    shapes = torch.as_tensor(shapes)
                    batch_dic[key_shape] = shapes

                pad_value = self._pad_values[key]
                items = pad_sequence_rec(items, pad_value=pad_value)

            elif (
                not key.endswith(SHAPE_SUFFIX)
                and all(isinstance(item, Tensor) for item in items)
                and can_be_stacked(items)
            ):
                items = torch.stack(items)

            batch_dic[key] = items

        return batch_dic


def detect_scalar_type(item: Any) -> type:
    types = set()
    queue = [item]
    while len(queue) > 0:
        item = queue.pop()
        if isinstance(item, (list, tuple)) and len(item) > 0:
            queue += item
        else:
            types.add(type(item))

    if len(types) == 1:
        return list(types)[0]
    else:
        raise RuntimeError(f"Multiple types detected: {types=}.")


def detect_shape(item: Any) -> Tensor:
    if isinstance(item, (int, float, str)):
        return torch.as_tensor((), dtype=torch.long)
    elif isinstance(item, Tensor) and item.ndim in (0, 1):
        return torch.as_tensor(item.shape, dtype=torch.long)
    elif isinstance(item, (Tensor, list, tuple)):
        if len(item) == 0 or isinstance(item[0], (int, float, str)):
            return torch.as_tensor((len(item),), dtype=torch.long)
        else:
            subshapes = [detect_shape(subitem) for subitem in item]
            subdims = list(map(len, subshapes))
            if not all_eq(subdims):
                pylog.error(f"Function detech_shape: found {subshapes=}")
                raise RuntimeError(
                    f"Invalid number of dims with {subdims=} in function 'detect_shape'."
                )
            return torch.stack([torch.as_tensor(subshape) for subshape in subshapes])
    else:
        raise RuntimeError(f"Unknown subtype {item.__class__.__name__}.")
