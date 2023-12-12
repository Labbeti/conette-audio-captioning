#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Any, Optional, Union

import torch

from torch import nn

from conette.nn.pann_utils import models
from conette.nn.pann_utils.ckpt import (
    PANN_PRETRAINED_URLS,
    pann_load_state_dict,
)


def build_pann_model(
    model_name: str,
    pretrained: bool = True,
    model_kwargs: Optional[dict[str, Any]] = None,
    device: Union[str, torch.device, None] = "auto",
    offline: bool = False,
    verbose: int = 0,
    strict_load: bool = True,
) -> nn.Module:
    """Build pretrained PANN model from name.

    :param model_name: PANN model name. (case sensitive)
    :param pretrained: If True, load pretrained weights. defaults to True.
    :param model_kwargs: Optional keywords arguments passed to PANN model initializer. defaults to None.
    :param device: Output device of the model. defaults to "auto".
    :param offline: If True, disable automatic checkpoint downloading. defaults to False.
    :param verbose: Verbose level during model build. defaults to 0.
    :param strict_load: If True, check if checkpoint entirely corresponds to the initialized model. defaults to True.
    :returns: The PANN model built as nn.Module.
    """
    if model_name not in PANN_PRETRAINED_URLS:
        raise ValueError(
            f"Invalid argument {model_name=}. (expected one of {tuple(PANN_PRETRAINED_URLS.keys())})"
        )

    if model_kwargs is None:
        model_kwargs = {}

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if isinstance(device, str):
        device = torch.device(device)

    classpath = f"{models.__name__}.{model_name}"
    classtype = eval(classpath)
    model: nn.Module = classtype(**model_kwargs)

    if pretrained:
        state_dict = pann_load_state_dict(
            model_name_or_path=model_name,
            offline=offline,
            verbose=verbose,
        )
        model.load_state_dict(state_dict, strict=strict_load)

    model = model.to(device=device)
    return model
