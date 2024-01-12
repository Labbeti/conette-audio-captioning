#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import os.path as osp

from typing import Union

import torch

from torch import Tensor

from conette.transforms.audioset_labels import load_audioset_mapping


pylog = logging.getLogger(__name__)


# Zenodo link : https://zenodo.org/record/8020843/
# Hash type : md5
CNEXT_PRETRAINED_URLS = {
    "cnext_nobl": {
        "model": "ConvNeXt",
        "url": "https://zenodo.org/record/8020843/files/convnext_tiny_471mAP.pth?download=1",
        "hash": "e069ecd1c7b880268331119521c549f2",
        "fname": "convnext_tiny_471mAP.pth",
    },
    "cnext_bl": {
        "model": "ConvNeXt",
        "url": "https://zenodo.org/record/8020843/files/convnext_tiny_465mAP_BL_AC_70kit.pth?download=1",
        "hash": "0688ae503f5893be0b6b71cb92f8b428",
        "fname": "convnext_tiny_465mAP_BL_AC_70kit.pth",
    },
}


def cnext_get_ckpt_dir_path() -> str:
    """Return the path to the directory containing CNEXT checkpoints files."""
    return osp.join(torch.hub.get_dir(), "checkpoints")


def cnext_get_ckpt_path(model_name: str) -> str:
    """Return the path to the CNEXT checkpoint file."""
    if model_name not in CNEXT_PRETRAINED_URLS:
        raise ValueError(
            f"Invalid argument {model_name=}. (expected one of {tuple(CNEXT_PRETRAINED_URLS.keys())})"
        )

    fname = CNEXT_PRETRAINED_URLS[model_name]["fname"]
    fpath = osp.join(cnext_get_ckpt_dir_path(), fname)
    return fpath


def cnext_load_state_dict(
    model_name_or_path: str,
    device: Union[str, torch.device, None] = None,
    offline: bool = False,
    verbose: int = 0,
) -> dict[str, Tensor]:
    """Load CNEXT state_dict weights.

    :param model_name_or_path: Model name (case sensitive) or path to CNEXT checkpoint file.
    :param device: Device of checkpoint weights. defaults to None.
    :param offline: If False, the checkpoint from a model name will be automatically downloaded.
        defaults to False.
    :param verbose: Verbose level. defaults to 0.
    :returns: State dict of model weights.
    """
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if isinstance(device, str):
        device = torch.device(device)

    if osp.isfile(model_name_or_path):
        model_path = model_name_or_path
    else:
        try:
            model_path = cnext_get_ckpt_path(model_name_or_path)
        except ValueError:
            raise ValueError(
                f"Invalid argument {model_name_or_path=}. (expected a path to a checkpoint file or a model name in {tuple(CNEXT_PRETRAINED_URLS.keys())})"
            )

        if not osp.isfile(model_path):
            if offline:
                raise FileNotFoundError(
                    f"Cannot find checkpoint model file in '{model_path}' with mode {offline=}."
                )
            else:
                cnext_download_ckpt(model_name_or_path, verbose)

    del model_name_or_path

    data = torch.load(model_path, map_location=device)
    state_dict = data["model"]

    if verbose >= 1:
        test_map = data.get("test_mAP", "unknown")
        pylog.info(
            f"Loading encoder weights from '{model_path}'... (with test_mAP={test_map})"
        )

    return state_dict


def cnext_download_ckpt(model_name: str, verbose: int = 0) -> None:
    """Download CNEXT checkpoint file."""
    fpath = cnext_get_ckpt_path(model_name)
    url = CNEXT_PRETRAINED_URLS[model_name]["url"]
    torch.hub.download_url_to_file(url, fpath, progress=verbose >= 1)


def probs_to_binarized(
    probs: Tensor,
    threshold: Union[float, Tensor],
) -> Tensor:
    if probs.ndim != 2:
        raise ValueError(
            f"Invalid argument probs. (expected a batch of probabilities of shape (N, n_classes))."
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
    preds = []
    for binarized_i in binarized:
        preds_i = torch.where(binarized_i)[0].tolist()
        preds.append(preds_i)
    return preds


def probs_to_indices(
    probs: Tensor,
    threshold: Union[float, Tensor],
) -> list[list[int]]:
    binarized = probs_to_binarized(probs, threshold)
    preds = binarized_to_indices(binarized)
    return preds


def probs_to_labels(
    probs: Tensor,
    threshold: Union[float, Tensor],
    audioset_indices_fpath: str,
) -> list[list[str]]:
    indices = probs_to_indices(probs, threshold)
    labels = indices_to_labels(indices, audioset_indices_fpath)
    return labels


def indices_to_labels(
    indices: Union[list[list[int]], list[Tensor]],
    audioset_indices_fpath: str,
) -> list[list[str]]:
    name_to_idx = load_audioset_mapping()
    idx_to_name = {idx: name for name, idx in name_to_idx.items()}

    labels = []
    for indices_i in indices:
        names = [idx_to_name[idx] for idx in indices_i]  # type: ignore
        labels.append(names)
    return labels
