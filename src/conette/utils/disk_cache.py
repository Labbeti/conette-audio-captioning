#!/usr/bin/env python
# -*- coding: utf-8 -*-

import inspect
import json
import logging
import os
import os.path as osp
import pickle
import shutil
import tempfile
import time

from pathlib import Path
from typing import Any, Callable, Optional, TypeVar, Union

import torch

from conette.utils.csum import csum_any


T = TypeVar("T")
pylog = logging.getLogger(__name__)


class DiskCache:
    CACHE_DNAME = "disk_cache"

    __global_instance: Optional["DiskCache"] = None

    def __init__(
        self,
        cache_path: Union[Path, str] = tempfile.gettempdir(),
        force: bool = False,
        filetype: str = "pickle",
        verbose: int = 0,
        fn: Optional[Callable] = None,
        use_pylog: bool = True,
    ) -> None:
        cache_path = osp.expandvars(cache_path)
        if not osp.isdir(cache_path):
            raise RuntimeError(
                f"Invalid cache directory {cache_path} for {self.__class__.__name__}."
            )

        if filetype not in ("json", "pickle"):
            raise ValueError(
                f"Invalid argument {filetype=}. (expected 'json' or 'pickle')"
            )

        super().__init__()
        self._cache_path = cache_path
        self._force = force
        self._filetype = filetype
        self._verbose = verbose
        self._fn = fn
        self._use_pylog = use_pylog
        self._print = pylog.info if use_pylog else print
        self._disable = False

        self._in_mode = "r" if filetype == "json" else "rb"
        self._out_mode = "w" if filetype == "json" else "wb"

        self._n_hits = 0
        self._n_calls = 0

    @classmethod
    def get(cls, *args, **kwargs) -> "DiskCache":
        if cls.__global_instance is None:
            cls.__global_instance = DiskCache(*args, **kwargs)
        return cls.__global_instance

    def clean(self, fn_or_name: Union[Callable, str]) -> int:
        if isinstance(fn_or_name, str):
            fn_name = fn_or_name
        else:
            fn_name = _get_callable_name(fn_or_name)

        target_path = osp.join(self._cache_path, self.CACHE_DNAME, fn_name)

        if not osp.exists(target_path):
            pylog.warning(f"Target path does not exists. ({target_path})")
            return 0

        if not osp.isdir(target_path):
            pylog.error(
                f"Target path exists but it is not a directory. ({target_path})"
            )
            return 0

        n_items_removed = len(os.listdir(target_path))
        shutil.rmtree(target_path)
        return n_items_removed

    def reset_stats(self) -> None:
        self._n_hits = 0
        self._n_calls = 0

    def wrap(self, fn: Callable) -> "DiskCache":
        return DiskCache(
            self._cache_path,
            self._force,
            self._filetype,
            self._verbose,
            fn,
            self._use_pylog,
        )

    def unwrap(self) -> Optional[Callable]:
        fn = self._fn
        self._fn = None
        return fn

    def __call__(self, *args, **kwargs) -> Any:
        if self._fn is None:
            raise RuntimeError(
                f"Cannot call {self.__class__.__name__} without wrapping a callable object."
            )
        return self.cache(self._fn, *args, **kwargs)

    def cache(
        self,
        fn: Callable[..., T],
        *args,
        force: Optional[bool] = None,
        dc_verbose: Optional[int] = None,
        csum_kwargs: Optional[dict[str, Any]] = None,
        ignore_fn_csum: bool = False,
        allow_compute: bool = True,
        filetype: Optional[str] = None,
        in_mode: Optional[str] = None,
        out_mode: Optional[str] = None,
        **kwargs,
    ) -> T:
        if dc_verbose is None:
            dc_verbose = self._verbose
        if force is None:
            force = self._force

        fpath = self.get_fpath(
            fn,
            *args,
            ignore_fn_csum=ignore_fn_csum,
            csum_kwargs=csum_kwargs,
            **kwargs,
        )
        self._n_calls += 1

        if not force:
            outs, loaded = self.load(fpath, dc_verbose, filetype, in_mode)
        else:
            outs, loaded = None, False

        if loaded:
            self._n_hits += 1
        else:
            if not allow_compute:
                raise ValueError(
                    f"Cannot compute outs for {_get_callable_name(fn)} with {allow_compute=}."
                )

            outs, duration = self._compute_outs(
                fn, *args, dc_verbose=dc_verbose, **kwargs
            )
            if not self._disable:
                if dc_verbose >= 2:
                    self._print(f"Overwrite file {osp.basename(fpath)} with {force=}.")
                self.dump(outs, fpath, duration, dc_verbose, filetype, out_mode)

        return outs  # type: ignore

    @torch.no_grad()
    def get_fpath(
        self,
        fn: Callable,
        *args,
        ignore_fn_csum: bool = False,
        csum_kwargs: Optional[dict[str, Any]] = None,
        **kwargs,
    ) -> str:
        if ignore_fn_csum:
            values = (args, kwargs)
        else:
            values = (fn, args, kwargs)

        default_csum_kwargs: dict[str, Any] = dict(
            bytes_mode="adler32",
            tensor_mode="sum_order",
            iter_order=True,
            accumulator=0,
            unk_mode="pickle",
        )
        if csum_kwargs is not None:
            default_csum_kwargs |= csum_kwargs

        csum = csum_any(values, **default_csum_kwargs)

        fn_name = _get_callable_name(fn)
        fname = f"{csum}.{self._filetype}"

        fpath = osp.join(self._cache_path, self.CACHE_DNAME, fn_name, fname)
        return fpath

    def load(
        self,
        fpath: str,
        dc_verbose: int = 0,
        filetype: Optional[str] = None,
        in_mode: Optional[str] = None,
    ) -> tuple[Any, bool]:
        if filetype is None:
            filetype = self._filetype
        if in_mode is None:
            in_mode = self._in_mode

        try:
            with open(fpath, in_mode) as file:
                if filetype == "json":
                    outs = json.load(file)["data"]
                elif filetype == "pickle":
                    outs = pickle.load(file)["data"]
                else:
                    raise ValueError(f"Invalid value {filetype=}.")

                if dc_verbose >= 2:
                    self._print(
                        f"[HIT_] Outputs loaded from '{osp.basename(fpath)}'. (hits={self.get_n_hits()+1}/{self.get_n_calls()})"
                    )
                return outs, True
        except (FileNotFoundError, json.JSONDecodeError, KeyError, EOFError):
            return None, False

    def dump(
        self,
        outs: Any,
        fpath: str,
        duration: float = -1.0,
        dc_verbose: int = 0,
        filetype: Optional[str] = None,
        out_mode: Optional[str] = None,
    ) -> None:
        if osp.isfile(fpath) and osp.getsize(fpath) == 0:
            os.remove(fpath)
        else:
            parent = osp.dirname(fpath)
            os.makedirs(parent, exist_ok=True)

        if filetype is None:
            filetype = self._filetype
        if out_mode is None:
            out_mode = self._out_mode

        with open(fpath, out_mode) as file:
            data = {"data": outs, "duration": duration}
            if filetype == "json":
                json.dump(data, file)
            elif filetype == "pickle":
                pickle.dump(data, file)  # type: ignore
            else:
                raise ValueError(f"Invalid value {filetype=}.")

        if dc_verbose >= 2:
            self._print(f"[MISS] Outputs dumped into '{osp.basename(fpath)}'.")

    def disable(self) -> None:
        self._disable = True

    def is_disabled(self) -> bool:
        return self._disable

    def set_forcing(self, force: bool) -> None:
        self._force = force

    def is_forcing(self) -> bool:
        return self._force

    def get_cache_path(self) -> str:
        return self._cache_path

    def get_n_hits(self) -> int:
        return self._n_hits

    def get_n_misses(self) -> int:
        return self.get_n_calls() - self.get_n_hits()

    def get_n_calls(self) -> int:
        return self._n_calls

    @property
    def force(self) -> bool:
        return self._force

    @force.setter
    def force(self, force_: bool) -> None:
        self._force = force_

    def _compute_outs(
        self,
        fn: Callable,
        *args,
        dc_verbose: int = 0,
        **kwargs,
    ) -> tuple[Any, float]:
        fn_name = _get_callable_name(fn)
        if dc_verbose >= 1:
            self._print(f"[MISS] Computing outs for fn '{fn_name}'...\r")

        start = time.perf_counter()
        outs = fn(*args, **kwargs)
        duration = time.perf_counter() - start

        if dc_verbose >= 1:
            self._print(
                f'[MISS] Outputs computed in {duration:.2f}s for file "{fn_name}".'
            )
        return outs, duration


def disk_cache(
    fn: Callable[..., T],
    *args,
    cache_path: str = "~/.cache",
    force: bool = False,
    dc_verbose: Optional[int] = None,
    csum_kwargs: Optional[dict[str, Any]] = None,
    ignore_fn_csum: bool = False,
    allow_compute: bool = True,
    **kwargs,
) -> T:
    cache_path = osp.expandvars(cache_path)
    cache_path = osp.expanduser(cache_path)

    global_cacher = DiskCache.get()
    if global_cacher.get_cache_path() == cache_path:
        cacher = global_cacher
    else:
        cacher = DiskCache(cache_path=cache_path)

    outs = cacher.cache(
        fn,
        *args,
        force=force,
        dc_verbose=dc_verbose,
        csum_kwargs=csum_kwargs,
        ignore_fn_csum=ignore_fn_csum,
        allow_compute=allow_compute,
        **kwargs,
    )
    return outs


def _get_callable_name(fn: Callable) -> str:
    if isinstance(fn, type) or inspect.isfunction(fn) or inspect.ismethod(fn):
        fn_name = fn.__qualname__
    else:
        fn_name = fn.__class__.__name__
    return fn_name
