#!/usr/bin/env python
# -*- coding: utf-8 -*-

import csv
import os
import os.path as osp

from pathlib import Path
from typing import Union

import torch

from torch import Tensor
from torch.hub import download_url_to_file


AUDIOSET_INFOS = {
    "class_labels_indices": {
        "fname": "class_labels_indices.csv",
        "url": "http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/class_labels_indices.csv",
    },
}


def probs_to_binarized(
    probs: Tensor,
    threshold: Union[float, Tensor],
) -> Tensor:
    """Perform thresholding to binarize probabilities."""
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

    binarized = probs >= threshold
    return binarized


def binarized_to_indices(
    binarized: Tensor,
) -> list[list[int]]:
    """Convert binarized probs to list of indexes."""
    preds = []
    for binarized_i in binarized:
        preds_i = torch.where(binarized_i)[0].tolist()
        preds.append(preds_i)
    return preds


def probs_to_indices(
    probs: Tensor,
    threshold: Union[float, Tensor],
) -> list[list[int]]:
    """Convert probs to list of indexes."""
    binarized = probs_to_binarized(probs, threshold)
    preds = binarized_to_indices(binarized)
    return preds


def probs_to_labels(
    probs: Tensor,
    threshold: Union[float, Tensor],
    offline: bool = False,
    verbose: int = 0,
) -> list[list[str]]:
    """Convert probs to list of labels."""
    indices = probs_to_indices(probs, threshold)
    labels = indices_to_labels(indices, offline, verbose)
    return labels


def indices_to_labels(
    indices: Union[list[list[int]], list[Tensor]],
    offline: bool = False,
    verbose: int = 0,
) -> list[list[str]]:
    """Convert indices to list of labels."""
    name_to_idx = load_audioset_mapping(offline, verbose)
    idx_to_name = {idx: name for name, idx in name_to_idx.items()}

    labels = []
    for indices_i in indices:
        names = [idx_to_name[idx] for idx in indices_i]  # type: ignore
        labels.append(names)
    return labels


def get_audioset_mapping_dir_path() -> Path:
    dpath = Path.home().joinpath(".cache", "conette")
    return dpath


def load_audioset_mapping(offline: bool = False, verbose: int = 0) -> dict[str, int]:
    info = AUDIOSET_INFOS["class_labels_indices"]
    dpath = get_audioset_mapping_dir_path()

    map_fname = info["fname"]
    map_fpath = dpath.joinpath(map_fname)

    if not osp.isfile(map_fpath):
        if offline:
            raise FileNotFoundError(
                f"Cannot find or download audioset mapping file in '{map_fpath}' with mode {offline=}."
            )

        download_audioset_mapping(verbose)

    with open(map_fpath, "r") as file:
        reader = csv.DictReader(file, skipinitialspace=True, strict=True)
        data = list(reader)

    name_to_index = {info["display_name"]: int(info["index"]) for info in data}
    return name_to_index


def download_audioset_mapping(verbose: int = 0) -> None:
    info = AUDIOSET_INFOS["class_labels_indices"]
    dpath = get_audioset_mapping_dir_path()
    map_fname = info["fname"]
    map_fpath = dpath.joinpath(map_fname)

    url = info["url"]
    os.makedirs(dpath, exist_ok=True)
    download_url_to_file(url, str(map_fpath), progress=verbose >= 1)
