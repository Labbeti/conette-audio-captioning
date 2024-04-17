#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Any, Optional, Union

import torch
from torch import nn
from torchoutil.nn.functional import get_device

from conette.nn.ckpt import PANN_REGISTRY
from conette.nn.pann_utils import models


def build_pann_model(
    model_name: str,
    pretrained: bool = True,
    model_kwargs: Optional[dict[str, Any]] = None,
    device: Union[str, torch.device, None] = "cuda_if_available",
    offline: bool = False,
    verbose: int = 0,
    strict_load: bool = True,
) -> nn.Module:
    """Build pretrained PANN model from name.

    :param model_name: PANN model name. (case sensitive)
    :param pretrained: If True, load pretrained weights. defaults to True.
    :param model_kwargs: Optional keywords arguments passed to PANN model initializer. defaults to None.
    :param device: Output device of the model. defaults to "cuda_if_available".
    :param offline: If True, disable automatic checkpoint downloading. defaults to False.
    :param verbose: Verbose level during model build. defaults to 0.
    :param strict_load: If True, check if checkpoint entirely corresponds to the initialized model. defaults to True.
    :returns: The PANN model built as nn.Module.
    """
    if model_name not in PANN_REGISTRY.names:
        raise ValueError(
            f"Invalid argument {model_name=}. (expected one of {tuple(PANN_REGISTRY.names)})"
        )

    if model_kwargs is None:
        model_kwargs = {}

    device = get_device(device)

    classpath = f"{models.__name__}.{model_name}"
    classtype = eval(classpath)
    model: nn.Module = classtype(**model_kwargs)

    if pretrained:
        state_dict = PANN_REGISTRY.load_state_dict(
            model_name, offline=offline, verbose=verbose
        )
        model.load_state_dict(state_dict, strict=strict_load)

    model = model.to(device=device)
    return model
