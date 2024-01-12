#!/usr/bin/env python
# -*- coding: utf-8 -*-

import gc

from typing import Optional

import torch

from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks.callback import Callback
from torch.optim import Optimizer
from torch.random import get_rng_state


class LogLRCallback(Callback):
    """Log the learning rate (lr) in the pylog and each iteration."""

    def __init__(
        self,
        prefix: str = "train/",
        on_epoch: bool = False,
        on_step: bool = True,
        bsize: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.prefix = prefix
        self.on_epoch = on_epoch
        self.on_step = on_step
        self.bsize = bsize

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        *args,
        **kwargs,
    ) -> None:
        optimizers = pl_module.optimizers()

        if isinstance(optimizers, Optimizer):
            optimizers = [optimizers]
        elif isinstance(optimizers, (tuple, list)):
            pass
        else:
            raise TypeError(
                f"Unsupported optimizers type {type(optimizers)}. (expected Optimizer, tuple[Optimizer, ...] or list[Optimizer])"
            )

        for i, optimizer in enumerate(optimizers):
            if not isinstance(optimizer, Optimizer):
                raise TypeError(
                    f"Unsupported optimizers type {type(optimizers)}. (expected Optimizer)"
                )

            learning_rates = [
                param_group["lr"] for param_group in optimizer.param_groups
            ]

            for j, lr in enumerate(learning_rates):
                if len(optimizers) == 1:
                    if len(learning_rates) == 1:
                        name = f"{self.prefix}lr"
                    else:
                        name = f"{self.prefix}lr{j}"
                else:
                    if len(learning_rates) == 1:
                        name = f"{self.prefix}optim{i}_lr"
                    else:
                        name = f"{self.prefix}optim{i}_lr{j}"

                pl_module.log(
                    name,
                    lr,
                    on_epoch=self.on_epoch,
                    on_step=self.on_step,
                    batch_size=self.bsize,
                    sync_dist=not trainer.move_metrics_to_cpu,  # type: ignore
                )


class LogGCCallback(Callback):
    def __init__(self, prefix: str = "train/", bsize: Optional[int] = None) -> None:
        super().__init__()
        self.prefix = prefix
        self.bsize = bsize

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        *args,
        **kwargs,
    ) -> None:
        counts = gc.get_count()
        thresholds = gc.get_threshold()

        for i, (count, threshold) in enumerate(zip(counts, thresholds)):
            name = f"{self.prefix}debug_gc_gen{i}"
            prop = count / threshold
            pl_module.log(
                name,
                prop,
                on_epoch=False,
                on_step=True,
                batch_size=self.bsize,
                sync_dist=not trainer.move_metrics_to_cpu,  # type: ignore
            )


class LogGradNorm(Callback):
    def __init__(
        self,
        name: str = "train/grad_norm2",
        p_norm: int = 2,
        bsize: Optional[int] = None,
        on_epoch: bool = True,
        on_step: bool = False,
    ) -> None:
        super().__init__()
        self.name = name
        self.p_norm = p_norm
        self.bsize = bsize
        self.on_epoch = on_epoch
        self.on_step = on_step

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        *args,
        **kwargs,
    ) -> None:
        parameters = [
            param.grad.norm(p=self.p_norm)  # type: ignore
            for param in pl_module.parameters()
            if param.grad is not None
        ]
        grad_norm = torch.as_tensor(parameters, dtype=torch.float64).sum()

        pl_module.log(
            self.name,
            grad_norm,
            on_epoch=self.on_epoch,
            on_step=self.on_step,
            batch_size=self.bsize,
            sync_dist=not trainer.move_metrics_to_cpu,  # type: ignore
        )


class LogRngState(Callback):
    def __init__(self, prefix: str = "train/", bsize: Optional[int] = None) -> None:
        super().__init__()
        self.prefix = prefix
        self.bsize = bsize

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        *args,
        **kwargs,
    ) -> None:
        rng_state = get_rng_state().sum().float()
        pl_module.log(
            f"{self.prefix}rng_state",
            rng_state,
            on_epoch=True,
            on_step=False,
            batch_size=self.bsize,
            sync_dist=not trainer.move_metrics_to_cpu,  # type: ignore
        )
