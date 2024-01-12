#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import math
import os
import os.path as osp
import re

from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, Optional, Union

from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from lightning_utilities.core.rank_zero import rank_zero_info


pylog = logging.getLogger(__name__)


class CustomModelCheckpoint(ModelCheckpoint):
    """Custom Model Checkpoint class.

    Changes:
    - checkpoint filenames use '-' instead of '=' for separate name and values in checkpoint names
        It help for avoiding errors with hydra which also uses the character '=' between arguments and values
    - replace "/" from metrics names by "_" in chekcpoint filenames to avoid errors with metrics like "val/loss" in checkpoint filename
    - create a symlink "best.ckpt" to the best model path
    - track the best monitor candidates. (method 'get_best_monitor_candidates()')
    - option to save checkpoint after only a certain epoch (arg 'save_after_epoch')

    Example :
            With ModelCheckpoint :
                    epoch=0-step=479-val/loss=3.3178.ckpt
            With CustomModelCheckpoint :
                    epoch_0-step_479-val_loss_3.3178.ckpt
    """

    CHECKPOINT_JOIN_CHAR = "-"
    CHECKPOINT_SEP_CHAR = "_"

    def __init__(
        self,
        # Herited args
        dirpath: Optional[Any] = None,
        filename: Optional[str] = None,
        monitor: Optional[str] = None,
        verbose: bool = False,
        save_last: Optional[bool] = None,
        save_top_k: int = 1,
        save_weights_only: bool = False,
        mode: str = "min",
        auto_insert_metric_name: bool = True,
        every_n_train_steps: Optional[int] = None,
        train_time_interval: Optional[timedelta] = None,
        every_n_epochs: Optional[int] = None,
        save_on_train_epoch_end: Optional[bool] = None,
        # New args
        log_best_score: bool = True,
        save_after_epoch: Union[None, int, float] = None,
        create_symlink: bool = True,
    ) -> None:
        if isinstance(dirpath, (str, Path)):
            dirpath = osp.expandvars(dirpath)
            dirpath = osp.expanduser(dirpath)

        super().__init__(
            dirpath=dirpath,
            filename=filename,
            monitor=monitor,
            verbose=verbose,
            save_last=save_last,
            save_top_k=save_top_k,
            save_weights_only=save_weights_only,
            mode=mode,
            auto_insert_metric_name=auto_insert_metric_name,
            every_n_train_steps=every_n_train_steps,
            train_time_interval=train_time_interval,
            every_n_epochs=every_n_epochs,
            save_on_train_epoch_end=save_on_train_epoch_end,
        )
        self.log_best_score = log_best_score
        self.save_after_epoch = save_after_epoch
        self.create_symlink = create_symlink

        self._best_monitor_candidates = {}

    @classmethod
    def _format_checkpoint_name(
        cls,
        filename: Optional[str],
        metrics: Dict[str, Any],
        prefix: str = "",
        auto_insert_metric_name: bool = True,
    ) -> str:
        if not filename:
            # filename is not set, use default name
            filename = "{epoch}" + cls.CHECKPOINT_JOIN_CHAR + "{step}"

        # check and parse user passed keys in the string
        groups = re.findall(r"(\{.*?)[:\}]", filename)
        if len(groups) >= 0:
            for group in groups:
                name = group[1:]

                if auto_insert_metric_name:
                    # Note source:
                    # filename = filename.replace(group, name + "={" + name)
                    # Change LABBETI: replace slash in metrics name by underscore
                    name_filt = name.replace("/", "_")
                    # Change LABBETI: SEP char will be "CHECKPOINT_SEP_CHAR" instead of "="
                    filename = filename.replace(
                        group,
                        name_filt + cls.CHECKPOINT_SEP_CHAR + "{" + name,
                    )

                if name not in metrics:
                    metrics[name] = 0
            filename = filename.format(**metrics)

        if prefix:
            filename = cls.CHECKPOINT_JOIN_CHAR.join([prefix, filename])

        return filename

    def _save_topk_checkpoint(
        self,
        trainer: Trainer,
        monitor_candidates: dict[str, Any],
    ) -> None:
        if self.monitor is None or self.save_top_k == 0:
            return

        current = monitor_candidates.get(self.monitor)
        epoch = monitor_candidates.get("epoch", -1)
        step = monitor_candidates.get("step", -1)

        if self.save_after_epoch is None:
            min_epoch = -1
        elif isinstance(self.save_after_epoch, int):
            min_epoch = self.save_after_epoch
        elif isinstance(self.save_after_epoch, float):
            if trainer.max_epochs is None:
                raise RuntimeError(
                    f"Cannot use float {self.save_after_epoch=} with {trainer.max_epochs=}."
                )
            min_epoch = math.floor(self.save_after_epoch * trainer.max_epochs)
        else:
            raise TypeError(
                f"Invalid argument {self.save_after_epoch=}. (expected None, int or float)"
            )

        if self.check_monitor_top_k(trainer, current) and (
            epoch is None or epoch >= min_epoch
        ):
            self._update_best_and_save(current, trainer, monitor_candidates)  # type: ignore

            # Track best epoch and best step
            self._best_monitor_candidates = monitor_candidates

        # Log current monitor value
        if self.log_best_score and self.best_model_score is not None:
            monitor_best_name = f"{self.monitor}_{self.mode}"
            trainer.lightning_module.log(
                monitor_best_name,
                self.best_model_score,
                on_epoch=True,
                on_step=False,
                sync_dist=not trainer.move_metrics_to_cpu,  # type: ignore
            )
            self._best_monitor_candidates[monitor_best_name] = self.best_model_score

        elif self.verbose:
            message_prefix = f"Epoch {epoch:d}, global step {step:d}: {self.monitor} reached {current:.2e},"

            # Change LABBETI: Modify info message
            if epoch is None or epoch >= min_epoch:
                current_best = self._best_monitor_candidates.get(self.monitor, None)
                current_best = (
                    f"{current_best:.2e}" if current_best is not None else "None"
                )
                message_suffix = (
                    f"but was not in top {self.save_top_k} (best {current_best})"
                )
            else:
                message_suffix = f"but found {epoch=} < {min_epoch}"

            rank_zero_info(f"{message_prefix} {message_suffix}")

    def on_fit_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if self.dirpath is not None:
            os.makedirs(str(self.dirpath), exist_ok=True)
            self.to_yaml()

        if (
            not self.create_symlink
            or not trainer.is_global_zero
            or not osp.isfile(self.best_model_path)
        ):
            return None

        ckpt_dpath = osp.dirname(self.best_model_path)
        ckpt_fname = osp.basename(self.best_model_path)
        lpath = osp.join(ckpt_dpath, "best.ckpt")

        if osp.exists(lpath):
            pylog.warning(f"Link {osp.basename(lpath)} already exists.")
            return None

        os.symlink(ckpt_fname, lpath)

        if not osp.isfile(lpath):
            pylog.error(f"Invalid symlink file {lpath=}.")
        elif self.verbose:
            pylog.debug(
                f"Create relative symlink for best model checkpoint '{lpath}'. (from='{self.best_model_path}')"
            )

    def get_best_monitor_candidates(self) -> dict[str, Any]:
        return self._best_monitor_candidates
