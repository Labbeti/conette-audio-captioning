#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import math
import os.path as osp
import time

from functools import cache
from typing import (
    Any,
    Callable,
    Generic,
    Iterable,
    Mapping,
    MutableMapping,
    Optional,
    Sequence,
    Sized,
    TypeVar,
    Union,
)

import torch
import torchaudio
import tqdm

from torch import Tensor
from torch.utils.data.dataset import Dataset
from torchaudio.backend.common import AudioMetaData

from conette.datasets.typing import AACDatasetLike, SizedDatasetLike
from conette.utils.disk_cache import disk_cache
from conette.utils.log_utils import warn_once
from conette.utils.misc import pass_filter


pylog = logging.getLogger(__name__)
T = TypeVar("T")


def _process_idx(
    idx: Union[int, Iterable[int], Tensor, slice, None],
) -> Union[int, list[int], slice]:
    if isinstance(idx, (int, slice)):
        return idx
    elif idx is None:
        return slice(None)
    elif isinstance(idx, Tensor):
        if idx.dtype == torch.bool:
            idx = torch.where(idx)[0]
        elif idx.is_floating_point():
            raise ValueError(
                f"Invalid argument dtype {idx.dtype=}. (expected int or bool)"
            )
        return idx.tolist()
    elif isinstance(idx, Iterable):
        return list(idx)
    else:
        raise TypeError(f"Invalid argument type {type(idx)=}.")


class EmptyDataset(Generic[T], Dataset[T]):
    def __getitem__(self, *args, **kwargs) -> None:
        raise NotImplementedError(
            f"Invalid call of getitem for {self.__class__.__name__}."
        )

    def __len__(self) -> int:
        return 0


class LambdaDataset(Generic[T], Dataset[T]):
    def __init__(
        self,
        fn: Callable[[int], Any],
        length: int,
        fn_kws: Optional[dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        self._fn = fn
        self._length = length
        self._fn_kws = fn_kws if fn_kws is not None else {}

    def __getitem__(self, *args, **kwargs) -> Any:
        return self._fn(*args, **kwargs, **self._fn_kws)

    def __len__(self) -> int:
        return self._length


class Wrapper(Generic[T], Dataset[T]):
    """
    Base class for dataset wrappers.

    :param source: The source dataset to wrap.
    """

    def __init__(self, source: T) -> None:
        super().__init__()
        self._source = source

    # Properties
    @property
    def source(self) -> T:
        return self._source

    # Public methods
    def unwrap(self, recursive: bool = True) -> Any:
        if not recursive:
            return self._source
        else:
            dset = self._source
            while isinstance(dset, Wrapper):
                dset = dset.unwrap()
            return dset

    # Magic methods
    def __getitem__(self, idx: int) -> Any:
        return self._source.__getitem__(idx)  # type: ignore

    def __len__(self) -> int:
        if isinstance(self._source, Sized):
            return len(self._source)
        else:
            raise NotImplementedError(
                f"Wrapped dataset {self._source.__class__.__name__} is not Sized."
            )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({repr(self._source)})"


class AACSubset(Wrapper[AACDatasetLike]):
    """Similar to torch.utils.data.Subset but for AACDataset classes."""

    def __init__(
        self,
        dataset: AACDatasetLike,
        indexes: Iterable[int],
        overwrite_index: bool = False,
    ) -> None:
        super().__init__(dataset)
        self._indexes = list(indexes)
        self._overwrite_index = overwrite_index

    # Public properties
    @property
    def column_names(self) -> list[str]:
        return self._source.column_names

    @property
    def indexes(self) -> list[int]:
        return self._indexes

    # Public methods
    def at(
        self,
        idx: Union[int, slice, Iterable[int], Tensor, None],
        column: Union[str, Iterable[str], None],
    ) -> Any:
        if idx is None:
            idx = slice(None)
        if isinstance(idx, Tensor):
            idx = idx.tolist()

        if isinstance(idx, Iterable):
            idx = list(idx)
            local_idx = [self._indexes[idx_i] for idx_i in idx]
        else:  # int or slice
            local_idx = self._indexes[idx]
        del idx
        item = self._source.at(local_idx, column)

        if not self._overwrite_index:
            return item

        if isinstance(item, dict):
            index = item.get("index", None)
            if isinstance(index, int) or (
                isinstance(index, Iterable)
                and all(isinstance(index_i, int) for index_i in index)
            ):
                item["index"] = local_idx

        elif column == "index":
            item = local_idx

        return item

    # Magic methods
    def __getitem__(self, idx: Any) -> Any:
        if (
            isinstance(idx, tuple)
            and len(idx) == 2
            and (isinstance(idx[1], (str, Iterable)) or idx[1] is None)
        ):
            idx, column = idx
        else:
            column = None
        return self.at(idx, column)

    def __len__(self) -> int:
        return len(self._indexes)


class AACConcat(Wrapper[tuple[AACDatasetLike, ...]]):
    """Similar to torch.utils.data.ConcatDataset but for AACDataset classes."""

    def __init__(self, *datasets: AACDatasetLike) -> None:
        super().__init__(datasets)
        cumsum = []
        prev_size = 0
        for dset in datasets:
            dset_size = len(dset)
            cumsum.append(dset_size + prev_size)
            prev_size += dset_size
        assert len(cumsum) == 0 or cumsum[-1] == len(
            self
        ), f"Found {cumsum[-1]=} != {len(self)=}."

        self._cumsum = cumsum

    @property
    def column_names(self) -> list[str]:
        column_names_lst = [dset.column_names for dset in self._source]
        column_names = intersect_lists(column_names_lst)
        return column_names

    @cache
    def _index_to_dset_and_local_indexes(self, idx: int) -> tuple[int, int]:
        if not isinstance(idx, int) or idx < 0 or idx >= self._cumsum[-1]:
            raise IndexError(f"Invalid index {idx} for {self.__class__.__name__}.")

        local_index = None
        dset_idx = None
        prevsum = 0
        for i, sum_ in enumerate(self._cumsum):
            if idx < sum_:
                dset_idx = i
                local_index = idx - prevsum
                break
            prevsum = sum_

        if local_index is None or dset_idx is None:
            raise IndexError(
                f"Invalid index {idx} for {self.__class__.__name__}. (found {local_index=} and {dset_idx=})"
            )

        return dset_idx, local_index

    def at(
        self,
        idx: Union[int, Iterable[int], Tensor, slice, None],
        column: Union[str, Iterable[str], None] = None,
    ) -> Any:
        if idx is None:
            idx = slice(None)
        if isinstance(idx, slice):
            idx = range(len(self))[idx]
        if isinstance(idx, Tensor):
            idx = idx.tolist()
        if column is None:
            column = self.column_names
        if not isinstance(column, str) and isinstance(column, Iterable):
            return {column_i: self.at(idx, column_i) for column_i in column}

        assert isinstance(column, str)

        if isinstance(idx, Iterable):
            item = []
            for idx_i in idx:
                dset_idx, local_index = self._index_to_dset_and_local_indexes(idx_i)
                item_i = self._source[dset_idx].at(local_index, column)
                item.append(item_i)
            return item

        elif isinstance(idx, int):
            dset_idx, local_index = self._index_to_dset_and_local_indexes(idx)
            return self._source[dset_idx].at(local_index, column)

        else:
            raise TypeError(f"Invalid argument type {idx=}.")

    def __getitem__(
        self,
        idx: Any,
    ) -> Any:
        if (
            isinstance(idx, tuple)
            and len(idx) == 2
            and (isinstance(idx[1], (str, Iterable)) or idx[1] is None)
        ):
            idx, column = idx
        else:
            column = None
        return self.at(idx, column)  # type: ignore

    def __len__(self) -> int:
        return sum(map(len, self._source))


class TransformWrapper(Wrapper):
    def __init__(
        self,
        dataset: SizedDatasetLike,
        transforms: Union[Callable, Iterable[Callable], None],
        index: Union[None, int, str] = None,
        default_kwargs: Optional[dict[str, Any]] = None,
    ) -> None:
        """Wrap a dataset method `getitem` with a transform."""
        if transforms is None:
            transforms = []
        elif isinstance(transforms, Callable):
            transforms = [transforms]
        else:
            transforms = list(transforms)

        if default_kwargs is None:
            default_kwargs = {}
        super().__init__(dataset)
        self._transforms = transforms
        self._index = index
        self._default_kwargs = default_kwargs

    def apply_transform(self, item: Any) -> Any:
        for tfm in self._transforms:
            item = tfm(item, **self._default_kwargs)
        return item

    def __getitem__(self, idx: Any) -> Any:
        item = self._source.__getitem__(idx)
        if self._index is None:
            return self.apply_transform(item)

        elif isinstance(item, MutableMapping):
            item[self._index] = self.apply_transform(
                item[self._index], **self._default_kwargs
            )
            return item

        elif isinstance(item, Iterable):
            return tuple(
                (
                    self.apply_transform(sub_item, **self._default_kwargs)
                    if i == self._index
                    else sub_item
                )
                for i, sub_item in enumerate(item)
            )

        else:
            raise TypeError(
                f"Invalid item type {type(item)}. (expected dict or iterable)"
            )


class CacheWrap(Wrapper):
    def __init__(self, dataset: Any) -> None:
        super().__init__(dataset)

    @cache
    def __getitem__(self, idx: int) -> tuple:
        return self._source.__getitem__(idx)

    @cache
    def __len__(self) -> int:
        return len(self._source)

    def load_items(
        self, verbose: bool = False, desc: str = "Loading dataset..."
    ) -> None:
        for i in tqdm.trange(len(self), disable=not verbose, desc=desc):
            self[i]


class DatasetCycle(Wrapper):
    def __init__(self, dataset: SizedDatasetLike, target_size: int) -> None:
        assert isinstance(dataset, Sized)
        assert len(dataset) <= target_size
        super().__init__(dataset)
        self._target_size = target_size

    def __getitem__(self, idx: int) -> Any:
        local_index = idx % len(self._source)
        return self._source[local_index]

    def __len__(self) -> int:
        return self._target_size


class WrapperSampler(Wrapper[AACDatasetLike]):
    """Randomly sample each element from source."""

    def __init__(
        self,
        source: AACDatasetLike,
        size: int,
        generator: Union[int, torch.Generator, None] = None,
    ) -> None:
        assert len(source) >= size
        if isinstance(generator, int):
            generator = torch.Generator().manual_seed(generator)
        super().__init__(source)
        self.size = size
        self.generator = generator
        self.indexes = torch.arange(size)
        self.reset_indexes()

    def reset_indexes(self) -> None:
        self.indexes = torch.randperm(len(self.source), generator=self.generator)[
            : self.size
        ]

    @property
    def column_names(self) -> list[str]:
        return self._source.column_names

    def at(self, idx: Any, column: Any) -> Any:
        assert isinstance(
            idx, int
        ), f"WrapperSampler does not support non-integer indexes. (found {idx=})"
        idx = self.indexes[idx]
        return self._source.at(idx, column)

    def __getitem__(self, idx: Any) -> Any:
        if (
            isinstance(idx, tuple)
            and len(idx) == 2
            and (isinstance(idx[1], (str, Iterable)) or idx[1] is None)
        ):
            idx, column = idx
        else:
            column = None
        return self.at(idx, column)

    def __len__(self) -> int:
        return self.size


class Duplicate(Wrapper[SizedDatasetLike]):
    def __init__(self, source: SizedDatasetLike, maxsize: int) -> None:
        super().__init__(source)
        self.maxsize = maxsize

    def __getitem__(self, idx: int) -> Any:
        idx = idx % len(self._source)
        return super().__getitem__(idx)

    def __len__(self) -> int:
        return self.maxsize


class AACDuplicate(Wrapper[AACDatasetLike]):
    def __init__(self, source: AACDatasetLike, maxsize: int) -> None:
        super().__init__(source)
        self.maxsize = maxsize

    @property
    def column_names(self) -> list[str]:
        return self._source.column_names

    def at(self, idx: Union[int, Iterable[int], None], column: Any = None) -> Any:
        idx = self._map_index(idx)
        return self._source.at(idx, column)

    def __getitem__(self, idx: Any) -> Any:
        if (
            isinstance(idx, tuple)
            and len(idx) == 2
            and (isinstance(idx[1], (str, Iterable)) or idx[1] is None)
        ):
            idx, column = idx
        else:
            column = None
        return self.at(idx, column)

    def __len__(self) -> int:
        return self.maxsize

    def _map_index(self, idx: Union[int, Iterable[int], None]) -> Any:
        if isinstance(idx, int):
            idx = idx % len(self._source)
        elif isinstance(idx, Iterable) and all(isinstance(idx_i, int) for idx_i in idx):
            idx = [self._map_index(idx_i) for idx_i in idx]
        elif idx is None:
            idx = self._map_index(range(len(self)))
        else:
            raise TypeError(f"Invalid argument type {idx=}.")
        return idx


class DsetTestSample(Dataset):
    def __init__(self) -> None:
        super().__init__()
        self._all_captions = [
            (
                "Cars travel past in the distance as a clock ticks",
                "A clock is ticking with traffic in the background",
                "An old clock with a pendulum that is swinging back and forth is ticking.",
                "An old clock with a pendulum is ticking.",
                "The industrial time card clock emits a thick, hallow ticking.",
            ),
            (
                "Chicks are chirping when a rooster is crowing.",
                "Chicks are chirping while a rooster is crowing.",
                "Seagulls squawk, then hens and chicks chirp and a rooster crows thrice as waves break against the shore.",
                "Waves breaking on a shore and seagulls squawking followed by hens and chicks chirping and a rooster crowing three times",
                "Many varieties of bird sing their songs, including a crowing cock.",
            ),
            (
                "A liquid is completely squeezed out of a tube.",
                "A liquid is squeezed out of a tube until it is empty.",
                "An air bladder being emptied into a jelly like material.",
                "Something is being squeezed out of a bottle with difficulty.",
                "The last of the liquid soap is being squeezed out of the bottle.",
            ),
        ]

    @property
    def column_names(self) -> list[str]:
        return ["audio", "captions", "index"]

    def at(self, idx: Union[int, slice, Iterable[int]], column: str) -> Any:
        if isinstance(idx, slice):
            idx = range(len(self))[idx]
        if isinstance(idx, Iterable):
            return [self.at(i, column) for i in idx]

        if column == "audio":
            return torch.full((3,), idx)
        elif column == "captions":
            return self._all_captions[idx]
        else:
            raise ValueError(f"Invalid index {idx=}.")

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return {
            "index": idx,
            "audio": self.at(idx, "audio"),
            "captions": self.at(idx, "captions"),
        }

    def __len__(self) -> int:
        return len(self._all_captions)


class ZipDataset(Dataset):
    def __init__(
        self,
        *datasets: SizedDatasetLike,
        transform: Optional[Callable] = None,
        mode: str = "equal",
    ) -> None:
        if len(datasets) == 0:
            raise ValueError(f"Cannot zip without datasets. (found {len(datasets)=})")
        if any(len(dset) == 0 for dset in datasets):
            raise ValueError(
                f"Cannot zip an empty dataset. (found sizes {tuple(len(dset) for dset in datasets)})"
            )

        if mode == "equal":
            lens = list(map(len, datasets))
            if any(lens[0] != len_ for len_ in lens):
                raise ValueError(
                    f"Invalid datasets lengths for ZipDatasets. (found {lens=})"
                )

            length = len(datasets[0])

        elif mode == "min":
            length = min(map(len, datasets))

        elif mode == "max":
            length = max(map(len, datasets))

        else:
            MODES = ("equal", "min", "max")
            raise ValueError(f"Invalid argument {mode=}. (expected one of {MODES})")

        super().__init__()
        self._datasets = datasets
        self._transform = transform
        self._mode = mode
        self._length = length

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = {}
        for dset in self._datasets:
            item |= dset[idx % len(dset)]
        if self._transform is not None:
            item = self._transform(item)
        return item

    def __len__(self) -> int:
        return self._length


class AACDatasetFromRaw(AACDatasetLike, Generic[T]):
    def __init__(self, all_items: Mapping[str, Iterable[T]]) -> None:
        all_items = {k: list(v) for k, v in all_items.items()}
        super().__init__()
        self._all_items: dict[str, list[T]] = all_items  # type: ignore

    @classmethod
    def from_iter(cls, all_items: Iterable[Mapping[str, T]]) -> "AACDatasetFromRaw[T]":
        all_items = list(all_items)
        if len(all_items) == 0:
            col_names = {}
        else:
            col_names = set(all_items[0].keys())

        if not all(set(col.keys()) == col_names for col in all_items):
            raise ValueError("Invalid column names keys.")

        all_items = {k: [col[k] for col in all_items] for k in col_names}  # type: ignore
        return AACDatasetFromRaw(all_items)  # type: ignore

    @property
    def column_names(self) -> list[str]:
        return list(self._all_items.keys())

    def at(self, idx: Any, column: Any) -> Any:
        if idx is None:
            idx = slice(None)
        if column is None:
            column = self.column_names

        if not isinstance(column, str) and isinstance(column, Iterable):
            return {column_i: self.at(idx, column_i) for column_i in column}

        if isinstance(idx, (int, slice)) and column in self._all_items.keys():
            return self._all_items[column][idx]

        if isinstance(idx, slice):
            idx = range(len(self))[idx]

        if isinstance(idx, Iterable):
            idx = list(idx)
            if not all(isinstance(idx_i, int) for idx_i in idx):
                raise TypeError(
                    f"Invalid input type for idx={idx}. (expected Iterable[int], not Iterable[{idx.__class__.__name__}])"
                )
            return [self.at(idx_i, column) for idx_i in idx]

    def __getitem__(
        self,
        idx: Any,
    ) -> dict[str, Any]:
        if (
            isinstance(idx, tuple)
            and len(idx) == 2
            and (isinstance(idx[1], (str, Iterable)) or idx[1] is None)
        ):
            idx, column = idx
        else:
            column = None

        item = self.at(idx, column)
        return item

    def __len__(self) -> int:
        if len(self._all_items) > 0:
            return len(next(iter(self._all_items.values())))
        else:
            return 0


def filter_audio_sizes(
    dset: AACDatasetLike,
    min_audio_size: float = 0.0,
    max_audio_size: float = math.inf,
    cache_path: Optional[str] = osp.join("~", ".cache"),
    verbose: int = 0,
    previous_indexes: Optional[Iterable[int]] = None,
    use_duration_column: bool = False,
) -> list[int]:
    if verbose >= 2:
        len_ = len(dset if previous_indexes is None else list(previous_indexes))
        pylog.debug(
            f"Loading durations from {len_} audio files... (with {cache_path=} and {use_duration_column=})"
        )

    if use_duration_column and "duration" in dset.column_names:
        durations = dset.at(previous_indexes, "duration")
    else:
        fpaths = dset.at(previous_indexes, "fpath")
        if cache_path is not None:
            infos = disk_cache(load_audio_metadata, fpaths, cache_path=cache_path)
        else:
            infos = load_audio_metadata(fpaths)
        durations = [(info.num_frames / info.sample_rate) for info in infos.values()]

    indexes = [
        i
        for i, duration in enumerate(durations)
        if min_audio_size <= duration <= max_audio_size
    ]
    n_excluded = len(dset) - len(indexes)
    if verbose >= 1:
        pylog.info(
            f"Exclude {n_excluded}/{len(dset)} files with audio size not in [{min_audio_size}, {max_audio_size}] seconds."
        )
    if verbose >= 2:
        lim = 10
        excluded_indexes = list(sorted(set(range(len(dset))).difference(indexes)))
        pylog.debug(f"Show first {lim} excluded indexes: {excluded_indexes[:lim]}.")
    return indexes


def load_audio_metadata(
    fpaths: list[str],
) -> dict[str, AudioMetaData]:
    infos = {
        fpath: torchaudio.info(fpath)  # type: ignore
        for fpath in tqdm.tqdm(
            fpaths,
            desc=f"Loading audio metadata from {len(fpaths)} files...",
            disable=pylog.level >= logging.INFO,
        )
    }
    return infos


def intersect_lists(lst_of_lst: list[list[T]]) -> list[T]:
    if len(lst_of_lst) <= 0:
        return []
    out = lst_of_lst[0]
    for lst_i in lst_of_lst[1:]:
        out = [name for name in out if name in lst_i]
        if len(out) == 0:
            break
    return out


class Cacher(Wrapper):
    def __init__(
        self,
        source: AACDatasetLike,
        cache_keys: Optional[dict[str, str]] = None,
    ) -> None:
        super().__init__(source)

        if cache_keys is None:
            cache_keys = {}
        if not all(v in ("get_raw", "get") for v in cache_keys.values()):
            raise ValueError

        null_value = "NULL"

        self._cache_keys = cache_keys
        self._caches = {
            k: [null_value for _ in range(len(source))] for k in cache_keys.keys()
        }
        self._null_value = null_value

    def at(self, idx: int, column: str) -> Any:
        return self._get_cache(idx, column, "get_field")

    def _get_cache(self, idx: int, column: str, type_: str) -> Any:
        if self._cache_keys.get(column) == type_:
            cached_value = self._caches[column][idx]
            if cached_value != self._null_value:
                return cached_value
            else:
                value = self._source.at(idx, column)
                self._caches[column][idx] = value
                return value
        else:
            value = self._source.at(idx, column)
            return value


class DatasetList(Dataset[T]):
    def __init__(
        self,
        items: Iterable[T],
        transform: Optional[Callable[[T], Any]] = None,
    ) -> None:
        if not isinstance(items, Sequence):
            items = list(items)

        super().__init__()
        self._items = items
        self._transform = transform

    def __getitem__(self, idx: int) -> Any:
        item = self._items[idx]
        if self._transform is not None:
            item = self._transform(item)
        return item

    def __len__(self) -> int:
        return len(self._items)


class DebugTracker(Wrapper):
    def __init__(self, source: SizedDatasetLike) -> None:
        super().__init__(source)
        self._delta_sum = 0.0
        self._delta_count = 0

    def __getitem__(self, idx: int) -> Any:
        start = time.perf_counter()
        item = super().__getitem__(idx)
        end = time.perf_counter()
        self._delta_sum += end - start
        self._delta_count += 1
        return item

    def get_average(self) -> float:
        if self._delta_count == 0:
            return 0.0
        else:
            return self._delta_sum / self._delta_count


class AACSelectColumnsWrapper(Wrapper[AACDatasetLike]):
    """Wrapper to filter columns in AACDatasetLike.

    ```python
    >>> dset = ...
    >>> dset = AACSelectColumnsWrapper(dset, include=("captions",))
    >>> dset[0]
    ... {"captions": ...}
    ```
    """

    DEFAULT_VAL = None

    def __init__(
        self,
        source: AACDatasetLike,
        /,
        include: Optional[Iterable[str]] = None,
        exclude: Optional[Iterable[str]] = None,
        use_default: bool = True,
    ) -> None:
        if include is None:
            not_found = []
        else:
            not_found = [name for name in include if name not in source.column_names]

        if len(not_found) > 0:
            warn_once(
                f"Cannot find {len(not_found)} column(s) {not_found} in {source} dataset. (found only {source.column_names})",
                pylog,
            )

        column_names = [
            name for name in source.column_names if pass_filter(name, include, exclude)
        ]
        if use_default:
            column_names += not_found
        super().__init__(source)
        self._column_names = column_names
        self._use_default = use_default
        self._not_found = not_found

    @property
    def column_names(self) -> list[str]:
        return self._column_names

    def at(self, idx: Any, column: Union[str, Iterable[str], None]) -> Any:
        if column is None:
            column = self.column_names

        if isinstance(column, str):
            if column not in self.column_names:
                raise ValueError(
                    f"Invalid argument {column=}. (expected one of {tuple(self.column_names)})"
                )
            if self._use_default and column in self._not_found:
                if isinstance(idx, Tensor):
                    if idx.dtype == torch.bool:
                        idx = torch.where(idx)
                    elif idx.is_floating_point():
                        raise ValueError(
                            f"Invalid argument {idx=}. (expected int or bool tensor)"
                        )
                    idx = idx.tolist()

                if isinstance(idx, int):
                    return self.DEFAULT_VAL
                elif isinstance(idx, Iterable):
                    idx = list(idx)
                    return [self.DEFAULT_VAL] * len(idx)
                else:
                    raise NotImplementedError(
                        f"Calling method at() with {self._use_default=} on a column with {type(idx)=} is currently not supported. (with {column=} and {self=})"
                    )
            else:
                return self._source.at(idx, column)

        elif isinstance(column, Iterable):
            if not all(column_i in self.column_names for column_i in column):
                raise ValueError(
                    f"Invalid argument {column=}. (expected one of {tuple(self.column_names)})"
                )

            return {column_i: self.at(idx, column_i) for column_i in column}
        else:
            raise TypeError(
                f"Invalid argument type {column=}. (expected str or Iterable[str])"
            )

    def __getitem__(self, idx: Any) -> dict[str, Any]:
        if (
            isinstance(idx, tuple)
            and len(idx) == 2
            and (isinstance(idx[1], (str, Iterable)) or idx[1] is None)
        ):
            idx, column = idx
        else:
            column = None
        item = self.at(idx, column)
        return item


class AACReplaceColumnWrapper(Wrapper[AACDatasetLike]):
    def __init__(
        self, source: AACDatasetLike, target_column: str, values: Iterable[Any]
    ) -> None:
        if hasattr(source, "_transform"):
            raise ValueError
        values = list(values)
        super().__init__(source)
        self._target_column = target_column
        self._values = values

    @property
    def column_names(self) -> list[str]:
        return self.source.column_names

    def at(
        self,
        idx: Union[int, Iterable[int], slice, None],
        column: Union[str, Iterable[str], None],
    ) -> Any:
        if idx is None:
            idx = slice(None)
        if column is None:
            column = self.column_names

        if not isinstance(column, str) and isinstance(column, Iterable):
            return {column_i: self.at(idx, column_i) for column_i in column}

        if isinstance(idx, (int, slice)) and column == self._target_column:
            return self._values[idx]

        if isinstance(idx, slice):
            idx = range(len(self))[idx]

        if isinstance(idx, Iterable):
            idx = list(idx)
            if not all(isinstance(idx_i, int) for idx_i in idx):
                raise TypeError(
                    f"Invalid input type for idx={idx}. (expected Iterable[int], not Iterable[{idx.__class__.__name__}])"
                )
            return [self.at(idx_i, column) for idx_i in idx]

        assert column != self._target_column
        return self.source.at(idx, column)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        if (
            isinstance(idx, tuple)
            and len(idx) == 2
            and (isinstance(idx[1], (str, Iterable)) or idx[1] is None)
        ):
            idx, column = idx
        else:
            column = None
        return self.at(idx, column)


class PostSelectColumnsWrapper(Wrapper[SizedDatasetLike]):
    def __init__(
        self,
        source: SizedDatasetLike,
        column_names: Iterable[str],
        /,
        include: Optional[Iterable[str]] = None,
        exclude: Optional[Iterable[str]] = None,
    ) -> None:
        column_names = [
            name for name in column_names if pass_filter(name, include, exclude)
        ]
        super().__init__(source)
        self._column_names = column_names

    @property
    def column_names(self) -> list[str]:
        return self._column_names

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = self._source[idx]
        item = {column: item[column] for column in self._column_names}
        return item


class AACTransformWrapper(Wrapper[AACDatasetLike]):
    def __init__(
        self,
        source: AACDatasetLike,
        transforms: dict[str, Callable[[Any, int], Any]],
        verbose: int = 0,
    ) -> None:
        super().__init__(source)
        self._transforms = transforms
        self._verbose = verbose

    @property
    def column_names(self) -> list[str]:
        return self._source.column_names

    def at(self, idx: Any, column: Any) -> Any:
        if idx is None:
            idx = slice(None)
        elif isinstance(idx, Tensor):
            idx = idx.tolist()
        if column is None:
            column = self.column_names

        if not isinstance(column, str) and isinstance(column, Iterable):
            return {column_i: self.at(idx, column_i) for column_i in column}
        assert isinstance(column, str)

        transform = self._transforms.get(column)
        if transform is None:
            return self._source.at(idx, column)

        if isinstance(idx, slice):
            idx = range(len(self))[idx]

        if isinstance(idx, Iterable):
            idx = list(idx)
            if not all(isinstance(idx_i, int) for idx_i in idx):
                raise TypeError(
                    f"Invalid input type for idx={idx}. (expected Iterable[int], not Iterable[{idx.__class__.__name__}])"
                )

            values = self._source.at(idx, column)
            values = [transform(value, idx_i) for value, idx_i in zip(values, idx)]
            return values

        elif isinstance(idx, int):
            value = self._source.at(idx, column)
            return transform(value, idx)

        else:
            raise TypeError(f"Invalid argument type {type(idx)=}.")

    def __getitem__(self, idx: Any) -> Any:
        if (
            isinstance(idx, tuple)
            and len(idx) == 2
            and (isinstance(idx[1], (str, Iterable)) or idx[1] is None)
        ):
            idx, column = idx
        else:
            column = None
        return self.at(idx, column)

    def __len__(self) -> int:
        return len(self._source)


class DummyAACDataset(AACDatasetLike):
    def __init__(self, size: int = 10) -> None:
        super().__init__()
        self.size = size

    @property
    def column_names(self) -> list[str]:
        return ["index", "value"]

    def at(self, idx, column) -> Any:
        if idx is None:
            idx = slice(None)
        if isinstance(idx, slice):
            idx = range(len(self))[idx]

        if column is None:
            column = self.column_names
        if not isinstance(column, str) and isinstance(column, Iterable):
            return {col_i: self.at(idx, col_i) for col_i in column}

        if isinstance(idx, Iterable):
            return [self.at(idx_i, column) for idx_i in idx]

        if column == "value":
            return f"value_{idx}"
        elif column == "index":
            return idx
        else:
            raise ValueError(f"Invalid argument {column=}.")

    def __getitem__(self, idx: Any) -> Any:
        if (
            isinstance(idx, tuple)
            and len(idx) == 2
            and (isinstance(idx[1], (str, Iterable)) or idx[1] is None)
        ):
            idx, column = idx
        else:
            column = None
        return self.at(idx, column)

    def __len__(self) -> int:
        return self.size
