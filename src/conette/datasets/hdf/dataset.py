#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import os
import os.path as osp

from typing import Any, Callable, Iterable, Optional, Union, overload

import h5py
import numpy as np
import pickle
import torch
import yaml

from h5py import Dataset as HDFRawDataset
from torch import Tensor
from torch.utils.data.dataset import Dataset

from .common import (
    HDF_ENCODING,
    HDF_STRING_DTYPE,
    HDF_VOID_DTYPE,
    SHAPE_SUFFIX,
    all_eq,
    get_inverse_perm,
)


pylog = logging.getLogger(__name__)


class HDFDataset(Dataset):
    # Initialization
    def __init__(
        self,
        hdf_fpath: str,
        transform: Optional[Callable] = None,
        keep_padding: Iterable[str] = (),
        open_hdf: bool = True,
    ) -> None:
        """
        :param hdf_fpath: The path to the HDF file.
        :param transforms: The transform to apply values (Tensor). default to None.
        :param keep_padding: Keys to keep padding values. defaults to ().
        :param open_hdf: If True, open the HDF file at start. defaults to True.
        """
        if not osp.isfile(hdf_fpath):
            names = os.listdir(osp.dirname(hdf_fpath))
            names = [name for name in names if name.endswith(".hdf")]
            names = list(sorted(names))
            raise FileNotFoundError(
                f"Cannot find HDF file in path {hdf_fpath=}. Possible HDF files are:\n{yaml.dump(names, sort_keys=False)}"
            )
        keep_padding = list(keep_padding)

        super().__init__()
        self._hdf_fpath = hdf_fpath
        self._transform = transform
        self._keep_padding = keep_padding

        self._hdf_file: Any = None

        if open_hdf:
            self.open()

    # Properties
    @property
    def column_names(self) -> list[str]:
        """The name of each column of the dataset."""
        return list(self.get_hdf_keys())

    @property
    def shape(self) -> tuple[int, ...]:
        """The shape of the Clotho dataset."""
        return len(self), len(self.column_names)

    @property
    def info(self) -> dict[str, Any]:
        """Return the global dataset info."""
        return eval(self._hdf_file.attrs.get("info", "{}"))

    # Public methods
    @overload
    def at(self, idx: int) -> dict[str, Any]:
        ...

    @overload
    def at(self, idx: Union[Iterable[int], slice, None]) -> dict[str, list]:
        ...

    @overload
    def at(self, idx: Any, column: Any) -> Any:
        ...

    def at(
        self,
        idx: Union[int, Iterable[int], slice, None] = None,
        column: Union[str, Iterable[str], None] = None,
        raw: bool = False,
    ) -> Any:
        if not self.is_open():
            raise RuntimeError(
                f"Cannot get_raw value with closed HDF file. ({self._hdf_file is not None=} and {bool(self._hdf_file)=})"
            )

        if idx is None:
            idx = slice(None)
        elif isinstance(idx, Tensor):
            idx = idx.tolist()
        if column is None:
            column = self.column_names

        if not isinstance(column, str) and isinstance(column, Iterable):
            return {column_i: self.at(idx, column_i) for column_i in column}

        if column not in self.column_names:
            raise ValueError(
                f"Invalid argument {column=}. (expected one of {tuple(self.column_names)})"
            )

        if isinstance(idx, slice):
            is_mult = True
        elif isinstance(idx, Iterable):
            if not all(isinstance(idx_i, int) for idx_i in idx):
                raise TypeError(f"Invalid argument {idx=}.")
            is_mult = True
        elif isinstance(idx, int):
            if not (-len(self) <= idx < len(self)):
                raise IndexError(
                    f"Invalid argument {idx=}. (expected int in range [{-len(self)}, {len(self)-1}])"
                )
            is_mult = False
        else:
            raise TypeError(f"Invalid argument type {type(idx)=}.")

        hdf_value = self._raw_at(idx, column)
        if raw:
            return hdf_value

        if is_mult:
            hdf_values = hdf_value
        else:
            hdf_values = [hdf_value]
        del hdf_value

        shape_name = f"{column}{SHAPE_SUFFIX}"
        must_remove_padding = (
            shape_name in self._hdf_file.keys() and column not in self._keep_padding
        )
        hdf_ds: HDFRawDataset = self._hdf_file[column]
        hdf_dtype = hdf_ds.dtype

        if must_remove_padding:
            shapes = self._raw_at(idx, shape_name)
            if not is_mult:
                shapes = [shapes]
            slices_lst = [
                tuple(slice(shape_i) for shape_i in shape) for shape in shapes
            ]
        else:
            slices_lst = [None] * int(hdf_ds.shape[0])

        outputs = []

        for hdf_value, slices in zip(hdf_values, slices_lst):
            # Remove the padding part
            if slices is not None:
                hdf_value = hdf_value[slices]

            # Decode all bytes to string
            if hdf_dtype == HDF_STRING_DTYPE:
                hdf_value = _decode_rec(hdf_value, HDF_ENCODING)
            # Convert numpy.array to torch.Tensor
            elif isinstance(hdf_value, np.ndarray):
                if hdf_dtype != HDF_VOID_DTYPE:
                    hdf_value = torch.from_numpy(hdf_value)
                else:
                    hdf_value = hdf_value.tolist()
            # Convert numpy scalars
            elif np.isscalar(hdf_value) and hasattr(hdf_value, "item"):
                hdf_value = hdf_value.item()  # type: ignore

            outputs.append(hdf_value)

        if not is_mult:
            outputs = outputs[0]
        return outputs

    def close(self) -> None:
        if not self.is_open():
            raise RuntimeError("Cannot close the HDF file twice.")
        self._hdf_file.close()
        self._hdf_file = None

    def get_attrs(self) -> dict[str, Any]:
        return self._hdf_file.attrs

    def get_hdf_fpath(self) -> str:
        return self._hdf_fpath

    def get_hdf_keys(self) -> tuple[str, ...]:
        if self.is_open():
            return tuple(self._hdf_file.keys())
        else:
            raise RuntimeError("Cannot get keys from a closed HDF file.")

    def get_column_shape(self, column_name: str) -> tuple[int, ...]:
        if not self.is_open():
            raise RuntimeError(
                f"Cannot get max_shape with a closed HDF file. ({self._hdf_file is not None=} and {bool(self._hdf_file)=})"
            )
        return tuple(self._hdf_file[column_name].shape)

    def is_open(self) -> bool:
        return self._hdf_file is not None and bool(self._hdf_file)

    def open(self) -> None:
        if self.is_open():
            raise RuntimeError("Cannot open the HDF file twice.")
        self._hdf_file = h5py.File(self._hdf_fpath, "r")
        self._sanity_check()

    # Magic methods
    def __eq__(self, __o: object) -> bool:
        return isinstance(__o, HDFDataset) and pickle.dumps(self) == pickle.dumps(__o)

    def __exit__(self) -> None:
        if self.is_open():
            self.close()

    @overload
    def __getitem__(self, idx: int) -> dict[str, Any]:
        ...

    @overload
    def __getitem__(self, idx: Union[Iterable[int], slice, None]) -> dict[str, list]:
        ...

    @overload
    def __getitem__(self, idx: Any) -> Any:
        ...

    def __getitem__(
        self,
        idx: Union[int, Iterable[int], None, slice, tuple[Any, Any]],
    ) -> Any:
        if (
            isinstance(idx, tuple)
            and len(idx) == 2
            and (isinstance(idx[1], (str, Iterable)) or idx[1] is None)
        ):
            idx, column = idx
        else:
            column = None

        item = self.at(idx, column)  # type: ignore
        if isinstance(idx, int) and column is None and self._transform is not None:
            item = self._transform(item)
        return item

    def __getstate__(self) -> dict[str, Any]:
        return {
            "hdf_fpath": self._hdf_fpath,
            "transform": self._transform,
            "keep_padding": self._keep_padding,
        }

    def __hash__(self) -> int:
        hash_value = 0
        if self.is_open():
            hash_value += self._hdf_file.attrs["global_hash_value"]
        if self._transform is not None:
            hash_value += hash(self._transform)
        hash_value += sum(map(hash, self._keep_padding))
        return hash_value

    def __len__(self) -> int:
        return self._hdf_file.attrs["length"]

    def __repr__(self) -> str:
        return (
            f"HDFDataset(size={len(self)}, hdf_fname={osp.basename(self._hdf_fpath)})"
        )

    def __setstate__(self, data: dict[str, Any]) -> None:
        is_init = hasattr(self, "_hdf_fpath") and hasattr(self, "_hdf_file")
        files_are_different = is_init and self._hdf_fpath != data["hdf_fpath"]
        is_open = is_init and self.is_open()

        if is_init and files_are_different and is_open:
            self.close()

        self._hdf_fpath = data["hdf_fpath"]
        self._transform = data["transform"]
        self._keep_padding = data["keep_padding"]
        self._hdf_file = None

        if not is_init or (files_are_different and is_open):
            self.open()

    # Private methods
    def _sanity_check(self) -> None:
        lens = [dset.shape[0] for dset in self._hdf_file.values()]
        if not all_eq(lens) or lens[0] != len(self):
            pylog.error(
                f"Incorrect length stored in HDF file. (found {lens=} and {len(self)=})"
            )

    def _raw_at(self, idx: Union[int, Iterable[int], slice], column: str) -> Any:
        if isinstance(idx, Iterable):
            sorted_idxs, local_idxs = torch.as_tensor(idx).sort(dim=-1)
            sorted_idxs = sorted_idxs.numpy()
            hdf_value: Any = self._hdf_file[column][sorted_idxs]
            inv_local_idxs = get_inverse_perm(local_idxs)
            hdf_value = [hdf_value[local_idx] for local_idx in inv_local_idxs]
        else:
            hdf_value: Any = self._hdf_file[column][idx]
        return hdf_value


def _decode_rec(value: Union[bytes, list], encoding: str) -> Union[str, list]:
    """Decode bytes to str with the specified encoding. Works recursively on list of str, list of list of str, etc."""
    if isinstance(value, bytes):
        return value.decode(encoding=encoding)
    elif isinstance(value, Iterable):
        return [_decode_rec(elt, encoding) for elt in value]
    else:
        raise TypeError(
            f"Invalid argument type {type(value)}. (expected bytes or Iterable)"
        )
