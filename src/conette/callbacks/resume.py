#!/usr/bin/env python
# -*- coding: utf-8 -*-

import glob
import logging
import os.path as osp
import re

from typing import Iterable, Optional, Union

import torch

from pytorch_lightning import LightningModule
from pytorch_lightning.callbacks.callback import Callback
from torch import Tensor

from conette.utils.csum import csum_module


pylog = logging.getLogger(__name__)


class ResumeCallback(Callback):
    def __init__(
        self,
        resume: Optional[str],
        strict: bool = True,
        ign_weights: Union[str, Iterable[str]] = (),
        use_glob: bool = False,
        verbose: int = 1,
    ) -> None:
        """
        :param pl_resume_path: The path to the checkpoint file containing the weights or to the logdir path containing the weight file in '{pl_resume_path}/checkpoints/best.ckpt'.
        :param strict: If True, the loading will crash if all keys weights does not match with the pl_module. defaults to False.
        :param verbose: The verbose level. defaults to 1.
        """
        super().__init__()
        self._resume = resume
        self._strict = strict
        self._ign_weights = ign_weights
        self._use_glob = use_glob
        self._verbose = verbose

        self._loaded = False

    def on_fit_start(self, trainer, pl_module: LightningModule) -> None:
        self.load_checkpoint(pl_module)

    def on_validation_start(self, trainer, pl_module: LightningModule) -> None:
        self.load_checkpoint(pl_module)

    def on_test_start(self, trainer, pl_module: LightningModule) -> None:
        self.load_checkpoint(pl_module)

    def load_checkpoint(self, pl_module: LightningModule) -> None:
        if self._loaded:
            return None

        load_checkpoint(
            pl_module=pl_module,
            resume=self._resume,
            strict=self._strict,
            ign_weights=self._ign_weights,
            use_glob=self._use_glob,
            verbose=self._verbose,
        )
        self._loaded = True


def load_checkpoint(
    pl_module: LightningModule,
    resume: Optional[str],
    strict: bool = True,
    ign_weights: Union[str, Iterable[str]] = (),
    use_glob: bool = False,
    verbose: int = 0,
) -> None:
    if resume is None:
        return None

    if isinstance(ign_weights, str):
        ign_weights = [ign_weights]
    else:
        ign_weights = list(ign_weights)

    if use_glob:
        matchs = glob.glob(resume)
        if len(matchs) == 0:
            raise ValueError(f"Cannot find ckpt file with glob pattern '{resume}'.")
        elif len(matchs) > 1:
            raise ValueError(
                f"Found multiple ckpt files with glob pattern '{resume}'. (found {len(matchs)} matchs)"
            )
        resume = matchs[0]

    if not isinstance(resume, str) or not osp.exists(resume):
        raise ValueError(
            f"Invalid resume checkpoint fpath {resume=}. (path does not exists)"
        )

    if osp.isfile(resume):
        ckpt_fpath = resume
    elif osp.isdir(resume):
        ckpt_fpath = osp.join(resume, "checkpoints", "best.ckpt")
        if not osp.isfile(ckpt_fpath):
            raise FileNotFoundError(
                f"Cannot find checkpoint in {resume=} (expected in {{resume}}/checkpoints/best.ckpt)."
            )
    else:
        raise ValueError(f"Invalid path type {resume=}.")

    if verbose >= 1:
        pylog.info(f"Loading pl_module from checkpoint {ckpt_fpath=}.")
        pylog.debug(f"pl_module csum before resume weights = {csum_module(pl_module)}")

    # Load best model before training
    checkpoint_data = torch.load(ckpt_fpath, map_location=pl_module.device)
    state_dict: dict[str, Tensor] = checkpoint_data["state_dict"]
    state_dict = {
        k: v
        for k, v in state_dict.items()
        if all((re.match(pattern, k) is None) for pattern in ign_weights)
    }

    try:
        incompatible_keys = pl_module.load_state_dict(state_dict, strict=strict)

        if verbose >= 2:
            pylog.debug(f"Found incompatible keys: {incompatible_keys}")

    except RuntimeError as err:
        pylog.error(
            f"Cannot load weights from ckpt file '{ckpt_fpath}'. (with strict={strict})"
        )
        raise err

    if verbose >= 1:
        pylog.debug(f"pl_module csum after resume weights = {csum_module(pl_module)}")
