#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import os
import os.path as osp
import random
import re
from typing import Any, Callable, Iterable, Optional, Union

import yaml
from torch import Tensor

from conette.tokenization.aac_tokenizer import AACTokenizer

pylog = logging.getLogger(__name__)


def get_hdf_fpaths(
    dataname: str,
    subsets: Iterable[str],
    hdf_root: str,
    hdf_suffix: Optional[str],
    hdf_dname: str = "HDF",
) -> dict[str, str]:
    """Returns the dictionary of HDF datasets filepaths for each subset :
    ```
    {
        {subset_1}: {hdf_root}/{hdf_dname}/{dataname}_{subset_1}_{hdf_suffix}.hdf
        {subset_2}: {hdf_root}/{hdf_dname}/{dataname}_{subset_2}_{hdf_suffix}.hdf
        ...
    }
    ```
    If hdf_suffix is None, returns an empty dict.
    """
    if hdf_suffix is None:
        return {}

    dataname = dataname.lower()
    subsets = list(map(str.lower, subsets))
    pattern = re.compile(
        r"(?P<dataname>[a-z]+)_(?P<subset>[a-z]+)_(?P<hdf_suffix>.+)\.hdf"
    )
    hdf_root = osp.expandvars(hdf_root)

    if not osp.isdir(osp.join(hdf_root, hdf_dname)):
        raise FileNotFoundError(f"Cannot find {hdf_dname} directory in {hdf_root=}.")

    hdf_fpaths = {}

    for subset in subsets:
        hdf_fname = f"{dataname}_{subset}_{hdf_suffix}.hdf"
        hdf_fpath = osp.join(hdf_root, hdf_dname, hdf_fname)

        if not osp.isfile(hdf_fpath):
            names = os.listdir(osp.join(hdf_root, hdf_dname))
            matches = [re.match(pattern, name) for name in names]
            availables_hdf_suffix = [
                match["hdf_suffix"]
                for match in matches
                if match is not None
                and match["dataname"] == dataname
                and match["subset"] == subset
            ]

            pylog.error(
                f"Cannot find HDF file '{hdf_fpath}' with {hdf_suffix=}.\n"
                f"Maybe run conette-prepare before and use another hdf_suffix for {dataname}.\n"
                f"Available hdf_suffix for '{dataname}_{subset}' are:\n{yaml.dump(availables_hdf_suffix, sort_keys=False)}"
            )
        hdf_fpaths[subset] = hdf_fpath

    return hdf_fpaths


class OnlineEncodeCaptionsTransform:
    def __init__(
        self,
        audio_tfm: Optional[Callable[[Tensor], Tensor]],
        ref_selection: Union[str, int, slice],
        add_raw_refs: bool,
        tokenizer: AACTokenizer,
        encode_kwargs: dict[str, Any],
        mrefs_src_key: Optional[str] = "captions",
        audio_time_dim: int = -2,
        ref_tfm: Optional[Callable[[str], str]] = None,
    ) -> None:
        super().__init__()
        self.audio_tfm = audio_tfm
        self.ref_selection = ref_selection
        self.add_raw_refs = add_raw_refs
        self.tokenizer = tokenizer
        self.encode_kwargs = encode_kwargs
        self.mrefs_src_key = mrefs_src_key
        self.audio_time_dim = audio_time_dim
        self.ref_tfm = ref_tfm

    def __call__(self, item: dict[str, Any]) -> dict[str, Any]:
        if self.audio_tfm is not None:
            audio = item["audio"]
            audio_shape = item["audio_shape"]
            audio_len = audio_shape[self.audio_time_dim]
            if audio_len < audio.shape[self.audio_time_dim]:
                mask = [slice(None) for _ in range(audio.ndim)]
                mask[self.audio_time_dim] = slice(audio_len)
                audio[mask] = self.audio_tfm(audio[mask])
            else:
                audio = self.audio_tfm(audio)
            item["audio"] = audio.contiguous()

        if self.mrefs_src_key is not None:
            refs = item[self.mrefs_src_key]

            if isinstance(self.ref_selection, str):
                if self.ref_selection == "random":
                    idxs = random.randint(0, len(refs) - 1)
                else:
                    raise ValueError(f"Invalid argument {self.ref_selection=}.")
            else:
                idxs = self.ref_selection

            if isinstance(idxs, int):
                ref = refs[idxs]
                if self.ref_tfm is not None:
                    ref = self.ref_tfm(ref)

                cap = self.tokenizer.encode_single(
                    ref,
                    **self.encode_kwargs,
                )

                item["captions"] = cap
                if self.add_raw_refs:
                    item["references"] = ref

            elif idxs == slice(None):
                item.pop("captions")

                if self.ref_tfm is not None:
                    refs = [self.ref_tfm(ref) for ref in refs]

                mcaps = self.tokenizer.encode_batch(
                    refs,
                    **self.encode_kwargs,
                )
                item["mult_captions"] = mcaps

                if self.add_raw_refs:
                    item["mult_references"] = refs

            else:
                raise ValueError(
                    f"Invalid argument {idxs=} with {self.ref_selection=}."
                )

        return item


class OnlineEncodeCaptionsTransformWithEmbs:
    def __init__(
        self,
        audio_tfm: Optional[Callable],
        ref_selection: Union[str, int, slice],
        add_raw_refs: bool,
        tokenizer: AACTokenizer,
        encode_kwargs: dict[str, Any],
        mrefs_src_key: str = "captions",
        mrefs_embs_src_key: str = "captions_embs",
    ) -> None:
        super().__init__()
        self.audio_tfm = audio_tfm
        self.ref_selection = ref_selection
        self.add_raw_refs = add_raw_refs
        self.tokenizer = tokenizer
        self.encode_kwargs = encode_kwargs
        self.mrefs_src_key = mrefs_src_key
        self.mrefs_embs_src_key = mrefs_embs_src_key

    def __call__(self, item: dict[str, Any]) -> dict[str, Any]:
        references = item[self.mrefs_src_key]
        references_embs = item[self.mrefs_embs_src_key]

        if self.audio_tfm is not None:
            item["audio"] = self.audio_tfm(item["audio"])

        if isinstance(self.ref_selection, str):
            if self.ref_selection == "random":
                idxs = random.randint(0, len(references) - 1)
            else:
                raise ValueError(f"Invalid argument {self.ref_selection=}.")
        else:
            idxs = self.ref_selection

        if isinstance(idxs, int):
            reference = references[idxs]
            caption = self.tokenizer.encode_single(
                reference,
                **self.encode_kwargs,
            )

            item["captions"] = caption
            item[self.mrefs_embs_src_key] = references_embs[idxs]
            if self.add_raw_refs:
                item["references"] = reference

        elif idxs == slice(None):
            item.pop("captions")
            mult_captions = self.tokenizer.encode_batch(
                references,
                **self.encode_kwargs,
            )

            item["mult_captions"] = mult_captions
            item[f"mult_{self.mrefs_embs_src_key}"] = references_embs[idxs]
            if self.add_raw_refs:
                item["mult_references"] = references

        else:
            raise ValueError(f"Invalid argument {idxs=} with {self.ref_selection=}.")

        return item
