#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import math

from typing import Any

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR, MultiStepLR, ReduceLROnPlateau
from torch.optim.swa_utils import SWALR

from conette.optim.cyclic_cos_decay import CyclicCosineDecayLR


pylog = logging.getLogger(__name__)


def get_scheduler_list(
    sched_name: str,
    optimizer: Optimizer,
    interval: str = "epoch",
    frequency: int = 1,
    monitor: str = "val/loss",
    **kwargs,
) -> list[dict[str, Any]]:
    """Returns the scheduler object corresponding to sched_name with the optimizer. Can also return None."""
    sched_name = str(sched_name).lower()

    if sched_name in ("cos_decay", "cosdecayrule"):
        n_steps = kwargs["sched_n_steps"]
        n_steps = max(n_steps, 1)
        scheduler = LambdaLR(optimizer, CosDecayRule(n_steps))

    elif sched_name in ("trf", "trfrule", "transformer_scheduler"):
        d_model = kwargs["d_model"]
        warmup_steps = kwargs["warmup_steps"]
        warmup_steps = max(warmup_steps, 1)
        scheduler = LambdaLR(optimizer, TrfRule(d_model, warmup_steps))

    elif sched_name == "multisteplr":
        milestones = kwargs["milestones"]
        gamma = kwargs["gamma"]

        scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=gamma)

    elif sched_name == "swalr":
        anneal_strategy = kwargs["anneal_strategy"]  # "linear"
        anneal_epochs = kwargs["anneal_epochs"]  # 20
        swa_lr = kwargs["swa_lr"]  # 0.05
        scheduler = SWALR(
            optimizer,
            anneal_strategy=anneal_strategy,
            anneal_epochs=anneal_epochs,
            swa_lr=swa_lr,
        )

    elif sched_name == (
        "cyclic_cos_decay",
        "CyclicCosineDecayLR".lower(),
    ):
        sched_kwargs = {
            k: v
            for k, v in kwargs.items()
            if k in CyclicCosineDecayLR.__init__.__code__.co_varnames
        }
        sched_kwargs["optimizer"] = optimizer
        scheduler = CyclicCosineDecayLR(**sched_kwargs)

    elif sched_name == "ReduceLROnPlateau".lower():
        scheduler = ReduceLROnPlateau(
            optimizer,
            **kwargs,
        )

    elif sched_name == "none":
        scheduler = None

    else:
        raise RuntimeError(
            f'Unknown scheduler "{sched_name}". Must be one of ("cos_decay", "trf", "multisteplr", "none").'
        )

    scheduler_dict = {
        "scheduler": scheduler,
        "interval": interval,
        "frequency": frequency,
        "strict": True,
        "monitor": monitor,
    }

    return [scheduler_dict]


class CosDecayRule:
    # Note : use class instead of function for scheduler rules for being pickable for multiple-GPU with Lightning
    def __init__(self, n_steps: int) -> None:
        if n_steps < 0:
            raise ValueError(
                f"Invalid argument {n_steps=} < 0 in {self.__class__.__name__}."
            )
        elif n_steps == 0:
            pylog.warning(
                f"Replacing {n_steps=} by n_steps=1 in {self.__class__.__name__}."
            )
            n_steps = max(n_steps, 1)
        super().__init__()
        self.n_steps = n_steps

    def __call__(self, step: int) -> float:
        step = min(step, self.n_steps - 1)
        return 0.5 * (1.0 + math.cos(math.pi * step / self.n_steps))


class TrfRule:
    # Note : use class instead of function for scheduler rules for being pickable for multiple-GPU with Lightning
    def __init__(self, d_model: int, warmup_steps: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.warmup_steps = warmup_steps

    def __call__(self, step: int) -> float:
        return self.d_model ** (-0.5) * min(
            step ** (-0.5), step * self.warmup_steps ** (-1.5)
        )
