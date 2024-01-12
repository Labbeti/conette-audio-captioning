#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import os.path as osp

from typing import Union

import torch

from torch import Tensor


pylog = logging.getLogger(__name__)


# Zenodo link : https://zenodo.org/record/3987831
# Hash type : md5
PANN_PRETRAINED_URLS = {
    "Cnn10": {
        "model": "Cnn10",
        "url": "https://zenodo.org/record/3987831/files/Cnn10_mAP%3D0.380.pth?download=1",
        "hash": "bfb1f1f9968938fa8ef4012b8471f5f6",
        "fname": "Cnn10_mAP_0.380.pth",
    },
    "Cnn14_DecisionLevelAtt": {
        "model": "Cnn14_DecisionLevelAtt",
        "url": "https://zenodo.org/record/3987831/files/Cnn14_DecisionLevelAtt_mAP%3D0.425.pth?download=1",
        "hash": "c8281ca2b9967244b91d557aa941e8ca",
        "fname": "Cnn14_DecisionLevelAtt_mAP_0.425.pth",
    },
    "Cnn14": {
        "model": "Cnn14",
        "url": "https://zenodo.org/record/3987831/files/Cnn14_mAP%3D0.431.pth?download=1",
        "hash": "541141fa2ee191a88f24a3219fff024e",
        "fname": "Cnn14_mAP_0.431.pth",
    },
    "Cnn6": {
        "model": "Cnn6",
        "url": "https://zenodo.org/record/3987831/files/Cnn6_mAP%3D0.343.pth?download=1",
        "hash": "e25e26b84585b14c7754c91e48efc9be",
        "fname": "Cnn6_mAP_0.343.pth",
    },
    "ResNet22": {
        "model": "ResNet22",
        "url": "https://zenodo.org/record/3987831/files/ResNet22_mAP%3D0.430.pth?download=1",
        "hash": "cf36d413096793c4e15dc752a3abd599",
        "fname": "ResNet22_mAP_0.430.pth",
    },
    "ResNet38": {
        "model": "ResNet38",
        "url": "https://zenodo.org/record/3987831/files/ResNet38_mAP%3D0.434.pth?download=1",
        "hash": "bf12f36aaabac4e0855e22d3c3239c1b",
        "fname": "ResNet38_mAP_0.434.pth",
    },
    "ResNet54": {
        "model": "ResNet54",
        "url": "https://zenodo.org/record/3987831/files/ResNet54_mAP%3D0.429.pth?download=1",
        "hash": "4f1f1406d37a29e2379916885e18c5f3",
        "fname": "ResNet54_mAP_0.429.pth",
    },
    "Wavegram_Cnn14": {
        "model": "Wavegram_Cnn14",
        "url": "https://zenodo.org/record/3987831/files/Wavegram_Cnn14_mAP%3D0.389.pth?download=1",
        "hash": "1e3506ab640371e0b5a417b15fd66d21",
        "fname": "Wavegram_Cnn14_mAP_0.389.pth",
    },
    "Wavegram_Logmel_Cnn14": {
        "model": "Wavegram_Logmel_Cnn14",
        "url": "https://zenodo.org/record/3987831/files/Wavegram_Logmel_Cnn14_mAP%3D0.439.pth?download=1",
        "hash": "17fa9ab65af3c0eb5ffbc5f65552c4e1",
        "fname": "Wavegram_Logmel_Cnn14_mAP_0.439.pth",
    },
}


def pann_get_ckpt_dir_path() -> str:
    """Return the path to the directory containing PANN checkpoints files."""
    return osp.join(torch.hub.get_dir(), "checkpoints")


def pann_get_ckpt_path(model_name: str) -> str:
    """Return the path to the PANN checkpoint file."""
    if model_name not in PANN_PRETRAINED_URLS:
        raise ValueError(
            f"Invalid argument {model_name=}. (expected one of {tuple(PANN_PRETRAINED_URLS.keys())})"
        )

    fname = PANN_PRETRAINED_URLS[model_name]["fname"]
    fpath = osp.join(pann_get_ckpt_dir_path(), fname)
    return fpath


def pann_load_state_dict(
    model_name_or_path: str,
    device: Union[str, torch.device, None] = None,
    offline: bool = False,
    verbose: int = 0,
) -> dict[str, Tensor]:
    """Load PANN state_dict weights.

    :param model_name_or_path: Model name (case sensitive) or path to PANN checkpoint file.
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
            model_path = pann_get_ckpt_path(model_name_or_path)
        except ValueError:
            raise ValueError(
                f"Invalid argument {model_name_or_path=}. (expected a path to a checkpoint file or a model name in {tuple(PANN_PRETRAINED_URLS.keys())})"
            )

        if not osp.isfile(model_path):
            if offline:
                raise FileNotFoundError(
                    f"Cannot find checkpoint model file in '{model_path}' with mode {offline=}."
                )
            else:
                pann_download_ckpt(model_name_or_path, verbose)

    del model_name_or_path

    data = torch.load(model_path, map_location=device)
    state_dict = data["model"]

    if verbose >= 1:
        test_map = data.get("test_mAP", "unknown")
        pylog.info(
            f"Loading encoder weights from '{model_path}'... (with test_mAP={test_map})"
        )

    return state_dict


def pann_download_ckpt(model_name: str, verbose: int = 0) -> None:
    """Download PANN checkpoint file."""
    fpath = pann_get_ckpt_path(model_name)
    url = PANN_PRETRAINED_URLS[model_name]["url"]
    torch.hub.download_url_to_file(url, fpath, progress=verbose >= 1)
