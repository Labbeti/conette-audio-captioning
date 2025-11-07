#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
from typing import Any, Iterable, Iterator, Optional, Union

from pythonwrench.functools import filter_and_call
from torch import Tensor, nn
from torch.optim import SGD, Adam, AdamW, Optimizer

from conette.utils.func_utils import filter_kwargs

pylog = logging.getLogger(__name__)


def get_optimizer(
    optim_name: str,
    parameters: Union[nn.Module, Iterator[Tensor], list],
    **kwargs,
) -> Optimizer:
    use_custom_wd = kwargs.get("use_custom_wd", False)
    if isinstance(parameters, nn.Module):
        if not use_custom_wd:
            parameters = parameters.parameters()
        else:
            parameters = _custom_weight_decay(
                parameters, kwargs.get("weight_decay", 0.01), ()
            )

    elif use_custom_wd:
        msg = (
            f"Invalid argument {parameters.__class__.__name__=} with {use_custom_wd=}."
        )
        raise ValueError(msg)

    classes = (
        Adam,
        AdamW,
        SGD,
    )
    optimizer = None
    for class_ in classes:
        if optim_name.lower() != class_.__name__.lower():
            continue

        actual_kwargs = filter_kwargs(class_, kwargs)
        optimizer = class_(parameters, **actual_kwargs)
        msg = f"Build optimizer {class_.__name__}({', '.join(f'{k}={v}' for k, v in actual_kwargs.items())})."
        pylog.info(msg)
        break

    if optimizer is not None:
        return optimizer
    else:
        raise RuntimeError(f"Unknown optimizer {optim_name=}.")


def _custom_weight_decay(
    model: nn.Module,
    weight_decay: float,
    skip_list: Optional[Iterable[str]] = (),
) -> list[dict[str, Any]]:
    decay = []
    no_decay = []
    if skip_list is None:
        skip_list = {}
    else:
        skip_list = set(skip_list)

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            no_decay.append(param)
            pylog.debug(f"No wd for {name}")
        else:
            decay.append(param)

    return [
        {"params": no_decay, "weight_decay": 0.0},
        {"params": decay, "weight_decay": weight_decay},
    ]
