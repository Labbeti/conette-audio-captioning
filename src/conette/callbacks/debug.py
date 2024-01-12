#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging

import torch

from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks.callback import Callback

from conette.utils.csum import csum_module


pylog = logging.getLogger(__name__)


class PrintDebug(Callback):
    def __init__(self, verbose: int = 2) -> None:
        super().__init__()
        self.verbose = verbose

    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        _print_csum(pl_module, "on_fit_start", self.verbose)

    def on_train_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        _print_csum(pl_module, "on_train_start", self.verbose)

    def on_validation_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        _print_csum(pl_module, "on_validation_start", self.verbose)

    def on_test_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        _print_csum(pl_module, "on_test_start", self.verbose)

    def on_fit_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        _print_csum(pl_module, "on_fit_end", self.verbose)

    def on_train_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        _print_csum(pl_module, "on_train_end", self.verbose)

    def on_validation_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        _print_csum(pl_module, "on_validation_end", self.verbose)

    def on_test_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        _print_csum(pl_module, "on_test_end", self.verbose)


def _print_csum(pl_module: LightningModule, fn_name: str, verbose: int) -> None:
    if verbose < 2:
        return None

    with torch.inference_mode():
        training = pl_module.training
        pl_module.train(False)
        csum = csum_module(pl_module)
        pl_module.train(training)

    pylog.debug(
        f"Model checksum for '{fn_name}': {csum} ({len(list(pl_module.named_parameters()))} tensors)"
    )
