#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import math
import random
import re
import os
import os.path as osp

from collections import Counter
from typing import Any, Callable, Iterable, Optional, Union

import torch
import yaml

from nltk.util import ngrams
from torch import Generator, Tensor
from torch.utils.data.dataset import ConcatDataset

from aac_datasets.datasets.audiocaps import AudioCaps

from conette.tokenization.aac_tokenizer import AACTokenizer
from conette.datasets.hdf import HDFDataset
from conette.datasets.typing import SizedDatasetLike
from conette.datasets.utils import (
    TransformWrapper,
    ZipDataset,
)


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


class PreEncodedCaptionsTransform:
    def __init__(
        self,
        audio_tfm: Optional[Callable],
        ref_selection: Union[str, int, slice],
        add_raw_refs: bool,
        mult_captions: Union[list, Tensor],
        mrefs_src_key: str = "captions",
    ) -> None:
        super().__init__()
        self.audio_tfm = audio_tfm
        self.ref_selection = ref_selection
        self.mult_captions = mult_captions
        self.add_raw_refs = add_raw_refs
        self.mrefs_src_key = mrefs_src_key

    def __call__(self, item: dict[str, Any]) -> dict[str, Any]:
        item_idx = item["index"]
        captions = self.mult_captions[item_idx]
        references = item[self.mrefs_src_key]

        if self.audio_tfm is not None:
            item["audio"] = self.audio_tfm(item["audio"])

        if isinstance(self.ref_selection, str):
            if self.ref_selection == "random":
                idxs = random.randint(0, len(captions) - 1)
            else:
                raise ValueError(f"Invalid argument {self.ref_selection=}.")
        else:
            idxs = self.ref_selection

        if isinstance(idxs, int):
            caption = captions[idxs]
            reference = references[idxs]

            item["captions"] = caption
            if self.add_raw_refs:
                item["references"] = reference

        elif idxs == slice(None):
            item.pop("captions")
            item["mult_captions"] = captions
            if self.add_raw_refs:
                item["mult_references"] = references

        else:
            raise ValueError(f"Invalid argument {idxs=} with {self.ref_selection=}.")

        return item


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


def split_indexes(
    indexes: Iterable[int],
    ratios: Iterable[float],
) -> list[list[int]]:
    assert 0 <= sum(ratios) <= 1.0 + 1e-20, f"Found {sum(ratios)=} not in [0, 1]."
    indexes = list(indexes)
    ratio_cumsum = 0.0
    outs = []
    for ratio in ratios:
        start = math.floor(ratio_cumsum * len(indexes))
        end = math.floor((ratio_cumsum + ratio) * len(indexes))
        sub_indexes = indexes[start:end]
        outs.append(sub_indexes)
        ratio_cumsum += ratio
    return outs


def generate_random_split(
    size: int, ratios: Iterable[float], seed: Union[int, None, Generator]
) -> list[list[int]]:
    if isinstance(seed, int):
        generator = Generator().manual_seed(seed)
    else:
        generator = seed

    indexes = torch.randperm(size, generator=generator).tolist()
    splitted_indexes = split_indexes(indexes, ratios)
    return splitted_indexes


def get_auto_num_cpus() -> int:
    return len(os.sched_getaffinity(0))


def auto_n_workers(n_workers: Optional[int]) -> int:
    if n_workers is None:
        num_cpus = get_auto_num_cpus()
        return num_cpus
    else:
        return n_workers


def build_mult_task_train_dataset(
    train_dset: SizedDatasetLike,
    train_tokenizer: AACTokenizer,
    root: str,
    audio_padding: str,
    task_hdfs: list[str],
    task_tag_types: list[str],
    idx_to_name_dicts: list[dict[int, str]],
    task_data_add: str,
) -> SizedDatasetLike:
    def tfm_task_0(item: dict) -> dict:
        item["task"] = torch.as_tensor(0)
        return item

    def get_tfm_task_1(
        task_tag_type: str,
        idx_to_name: dict[int, str],
    ) -> Callable:
        assert task_tag_type in ("audioset", "fsd50k")

        def tfm_task_1(item: dict) -> dict:
            tags = item["tags"].tolist()

            indexes = torch.randperm(len(tags))
            tags = [tags[idx] for idx in indexes]
            joined_tags_names = ", ".join(idx_to_name[tag] for tag in tags)
            encoded_tags = train_tokenizer.encode_single(
                joined_tags_names, default=False
            )
            item["captions"] = encoded_tags
            item["task"] = torch.as_tensor(1)

            return item

        return tfm_task_1

    train_dset_task_0 = TransformWrapper(train_dset, tfm_task_0)

    train_dsets = [train_dset_task_0]
    keep_padding = ("audio",) if audio_padding in ("crop", "longest") else ()

    for task_hdf, task_tag_type, idx_to_name in zip(task_hdfs, task_tag_types, idx_to_name_dicts):  # type: ignore
        hdf_fpath = osp.join(root, "HDF", task_hdf)  # type: ignore
        hdf_dset = HDFDataset(
            hdf_fpath,
            get_tfm_task_1(task_tag_type, idx_to_name),
            keep_padding=keep_padding,
        )
        train_dsets.append(hdf_dset)  # type: ignore

    if task_data_add == "cat":
        train_dset = ConcatDataset(train_dsets)
    elif task_data_add == "cat_lim":
        # train_dsets = [train_dset_task_0] + [
        #     WrapperSampler(dset, len(train_dset_task_0)) for dset in train_dsets[1:]
        # ]
        raise NotImplementedError
    elif task_data_add == "zip_max":
        train_dset = ZipDataset(*train_dsets, mode="max")
    elif task_data_add == "zip_min":
        train_dset = ZipDataset(*train_dsets, mode="min")
    else:
        raise ValueError(f"Invalid argument {task_data_add=}.")

    return train_dset


def get_counter(sents: list[list[str]], nmax: int) -> Counter[tuple[str, ...]]:
    assert nmax > 0
    counter = Counter()
    for n in range(1, nmax + 1):
        for sent in sents:
            for ngram in ngrams(sent, n):
                assert len(ngram) == n
                counter[ngram] += 1
    return counter


def build_caps_complexity_scores(
    sents: list[list[str]], nmax: int, mode: str = "ign_ngram_sup"
) -> Tensor:
    assert mode in ("ign_ngram_sup", "zero_ngram_sup")
    counter = dict(get_counter(sents, nmax))
    count_per_ngram = [
        sum(count for ngram, count in counter.items() if len(ngram) == n)
        for n in range(1, nmax + 1)
    ]
    scores = torch.stack(
        [
            torch.as_tensor(
                [
                    (
                        (
                            torch.as_tensor(
                                [
                                    count_per_ngram[n - 1] / counter[ngram]
                                    for ngram in ngrams(sent, n)
                                ]
                            ).mean()
                        )
                        if mode != "zero_ngram_sup" or len(sent) >= n
                        else 0.0
                    )
                    for n in range(1, nmax + 1)
                    if mode != "ign_ngram_sup" or len(sent) >= n
                ]
            ).mean()
            for sent in sents
        ]
    )
    max_score = scores.max().item()
    scores = scores / max_score
    scores = scores.numpy()
    return scores


def _get_unscaled_score(
    tok_sent: list[str],
    counter: dict[tuple[str, ...], int],
    count_per_ngram: list[int],
    nmax: int,
    mode: str,
) -> Tensor:
    ngram_scores = []
    for n in range(1, nmax + 1):
        if len(tok_sent) < n:
            if mode == "ign_ngram_sup":
                continue
            elif mode == "zero_ngram_sup":
                ngram_scores.append(0.0)
            else:
                raise ValueError(f"Invalid argument {mode=}.")
        else:
            ngram_score = torch.as_tensor(
                [
                    count_per_ngram[n - 1] / counter[ngram]
                    for ngram in ngrams(tok_sent, n)
                ]
            ).mean()
            ngram_scores.append(ngram_score)
    score = torch.as_tensor(ngram_scores).mean()
    return score


class CapsComplexity:
    def __init__(self, nmax: int, mode: str = "ign_ngram_sup") -> None:
        super().__init__()
        self._nmax = nmax
        self._mode = mode

        self._counter = {}
        self._count_per_ngram = []
        self._max_score = 1.0

    def fit(self, tok_sents: list[list[str]]) -> Tensor:
        counter = dict(get_counter(tok_sents, self._nmax))
        count_per_ngram = [
            sum(count for ngram, count in counter.items() if len(ngram) == n)
            for n in range(1, self._nmax + 1)
        ]

        unscaled_scores = [
            _get_unscaled_score(sent, counter, count_per_ngram, self._nmax, self._mode)
            for sent in tok_sents
        ]
        unscaled_scores = torch.as_tensor(unscaled_scores)
        max_score = unscaled_scores.max().item()

        self._counter = counter
        self._count_per_ngram = count_per_ngram
        self._max_score = max_score

        return self.get_scores(tok_sents)

    def get_scores(self, tok_sents: list[list[str]]) -> Tensor:
        scores = torch.as_tensor([self.get_score(tok_sent) for tok_sent in tok_sents])
        return scores

    def get_score(self, tok_sent: list[str]) -> Tensor:
        score = _get_unscaled_score(
            tok_sent, self._counter, self._count_per_ngram, self._nmax, self._mode
        )
        score = score / self._max_score
        return score
