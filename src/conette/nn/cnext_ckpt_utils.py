#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import os.path as osp

from typing import Union

import torch

from torch import Tensor


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
