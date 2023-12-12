#!/usr/bin/env python
# -*- coding: utf-8 -*-

import csv
import logging
import os
import os.path as osp

from argparse import Namespace
from typing import Any, Iterable, Optional, Union

import yaml

from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.callbacks.callback import Callback
from pytorch_lightning.callbacks.checkpoint import Checkpoint
from pytorch_lightning.core.saving import save_hparams_to_yaml
from torch import Tensor

from conette.callbacks.time import TimeTrackerCallback
from conette.info import get_install_info
from conette.nn.functional.misc import count_params
from conette.tokenization.aac_tokenizer import AACTokenizer
from conette.utils.csum import csum_module
from conette.utils.custom_logger import CustomTensorboardLogger
from conette.utils.misc import get_current_git_hash, save_conda_env, save_micromamba_env


pylog = logging.getLogger(__name__)


class StatsSaver(Callback):
    """Callback for saving some stats about the training in the pylog."""

    def __init__(
        self,
        subrun_path: Optional[str],
        tokenizers: Optional[dict[str, Optional[AACTokenizer]]] = None,
        on_end: str = "none",
        close_logger_on_end: bool = True,
        git_hash: Optional[str] = None,
        cfg: Optional[DictConfig] = None,
        verbose: int = 1,
    ) -> None:
        if subrun_path is not None:
            subrun_path = osp.expandvars(subrun_path)

        if tokenizers is None:
            tokenizers = {}
        else:
            tokenizers = {
                name: tokenizer
                for name, tokenizer in tokenizers.items()
                if tokenizer is not None
            }

        if on_end not in ("fit", "test", "none"):
            raise ValueError(f"Invalid argument {on_end=}.")

        if git_hash is None:
            git_hash = get_current_git_hash(default=None)

        super().__init__()
        self._subrun_dir = subrun_path
        self._tokenizers = tokenizers
        self._on_end = on_end
        self._close_logger_on_end = close_logger_on_end
        self._git_hash = git_hash
        self._cfg = cfg
        self._verbose = verbose

        self._time_tracker = TimeTrackerCallback()
        self._start_csum = 0
        self._end_csum = 0

    # Callback methods
    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self._time_tracker.on_fit_start(trainer, pl_module)
        self._start_csum = csum_module(pl_module)
        self._end_csum = self._start_csum

    def on_fit_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self._time_tracker.on_fit_end(trainer, pl_module)
        self._end_csum = csum_module(pl_module)

        if self._on_end == "fit":
            self.save_metrics_stats(trainer, pl_module)

    def on_train_epoch_start(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        self._time_tracker.on_train_epoch_start(trainer, pl_module)

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self._time_tracker.on_train_epoch_end(trainer, pl_module)

    def on_test_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self._time_tracker.on_test_start(trainer, pl_module)

    def on_test_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self._time_tracker.on_test_end(trainer, pl_module)
        if self._on_end == "test":
            self.save_metrics_stats(trainer, pl_module)

    # Other methods
    def save_metrics_stats(
        self,
        trainer: Trainer,
        pl_module: Optional[LightningModule],
        datamodule: Optional[LightningDataModule] = None,
        add_params: Optional[dict[str, Any]] = None,
        add_metrics: Optional[dict[str, Any]] = None,
    ) -> None:
        if datamodule is None:
            datamodule = trainer.datamodule  # type: ignore
        ckpts = trainer.checkpoint_callbacks if trainer is not None else None

        save_to_dir(
            subrun_path=self._subrun_dir,
            datamodule=datamodule,
            pl_module=pl_module,
            tokenizers=self._tokenizers,  # type: ignore
            time_tracker=self._time_tracker,
            checkpoint=ckpts,
            close_logger_on_end=self._close_logger_on_end,
            git_hash=self._git_hash,
            cfg=self._cfg,
            start_csum=self._start_csum,
            end_csum=self._end_csum,
            verbose=self._verbose,
            add_params=add_params,
            add_metrics=add_metrics,
        )


def save_to_dir(
    subrun_path: Optional[str],
    datamodule: Optional[LightningDataModule] = None,
    pl_module: Optional[LightningModule] = None,
    tokenizers: Optional[dict[str, Optional[AACTokenizer]]] = None,
    time_tracker: Optional[TimeTrackerCallback] = None,
    checkpoint: Union[Checkpoint, Iterable[Checkpoint], None] = None,
    close_logger_on_end: bool = True,
    git_hash: Optional[str] = None,
    cfg: Optional[DictConfig] = None,
    start_csum: Optional[int] = None,
    end_csum: Optional[int] = None,
    verbose: int = 0,
    add_slurm_vars_to_params: bool = False,
    add_version_info_to_params: bool = False,
    add_params: Optional[dict[str, Any]] = None,
    add_metrics: Optional[dict[str, Any]] = None,
    save_conda: bool = False,
    save_micromamba: bool = True,
) -> None:
    """Save callbacks and miscellaneous information in subrun_path directory."""
    if subrun_path is None:
        return None

    subrun_path = osp.expandvars(subrun_path)
    if not osp.isdir(subrun_path):
        return None

    if add_params is None:
        params = {}
    else:
        params = add_params

    if add_metrics is None:
        other_metrics = {}
    else:
        other_metrics = add_metrics

    if git_hash is None:
        git_hash = get_current_git_hash(default=None)

    params |= {
        "git_hash": git_hash,
        "start_csum": start_csum,
        "end_csum": end_csum,
    }

    if add_slurm_vars_to_params:
        params |= {
            key.lower(): value
            for key, value in os.environ.items()
            if key.startswith("SLURM_")
        }

    if add_version_info_to_params:
        versions = get_install_info()
        versions = {f"{name}_version": version for name, version in versions.items()}
        params |= versions

    hp_dpath = osp.join(subrun_path, "hparams")
    os.makedirs(hp_dpath, exist_ok=True)

    # Note: do not use save_hparams_to_yaml for os.environ to avoid interpolation errors
    with open(osp.join(hp_dpath, "os_env.yaml"), "w") as file:
        yaml.dump(dict(os.environ), file, sort_keys=False)

    if save_conda:
        if cfg is not None:
            conda_path = cfg.get("path", {}).get("conda", "conda")
        else:
            conda_path = "conda"
        save_conda_env(osp.join(hp_dpath, "conda_env.yaml"), conda_path)

    if save_micromamba:
        if cfg is not None:
            micromamba_path = cfg.get("path", {}).get("micromamba", "micromamba")
        else:
            micromamba_path = "micromamba"
        save_micromamba_env(osp.join(hp_dpath, "micromamba_env.yaml"), micromamba_path)

    hydra_cfg = HydraConfig.get()
    hydra_cfg = {"hydra": hydra_cfg}
    save_hparams_to_yaml(osp.join(hp_dpath, "resolved_hydra.yaml"), hydra_cfg)

    if cfg is not None:
        save_hparams_to_yaml(osp.join(hp_dpath, "resolved_config.yaml"), cfg)  # type: ignore

    if pl_module is not None:
        save_hparams_to_yaml(
            osp.join(hp_dpath, "pl_module.yaml"),
            pl_module.hparams_initial,
        )
        other_metrics |= {
            "total_params": count_params(pl_module, only_trainable=False),
            "train_params": count_params(pl_module, only_trainable=True),
        }

    if datamodule is not None:
        save_hparams_to_yaml(
            osp.join(hp_dpath, "datamodule.yaml"),
            datamodule.hparams_initial,
        )

    if time_tracker is not None:
        params |= {
            "fit_duration": time_tracker.get_fit_duration_formatted(),
            "test_duration": time_tracker.get_test_duration_formatted(),
        }
        other_metrics |= {
            "fit_duration_h": time_tracker.get_fit_duration_in_hours(),
            "test_duration_h": time_tracker.get_test_duration_in_hours(),
            "epoch_mean_duration_min": time_tracker.get_epoch_mean_duration_in_min(),
        }

    if checkpoint is None:
        ckpts = []
    elif not isinstance(checkpoint, Iterable):
        ckpts = [checkpoint]
    else:
        ckpts = list(checkpoint)
    del checkpoint

    for ckpt in ckpts:
        if not all(
            hasattr(ckpt, attr) for attr in ("get_best_monitor_candidates", "monitor")
        ):
            pylog.warning(
                f"Cannot save best epoch values for checkpoint type {ckpt.__class__.__name__}."
            )
            continue

        best_monitor_candidates: dict[str, Any] = ckpt.get_best_monitor_candidates()  # type: ignore

        monitor = ckpt.monitor  # type: ignore
        # note : no need to handle case where / is not found because:
        # example: s = "abcabc"; s.rfind("d") gives -1, so s[s.rfind("d")+1:] == s[0:] == s
        monitor = monitor[monitor.rfind("/") + 1 :]

        best_monitor_candidates = {
            f"best_{monitor}_{name}": _clean_value(value)
            for name, value in best_monitor_candidates.items()
        }
        other_metrics |= best_monitor_candidates

    if tokenizers is None:
        tokenizers = {}

    for name, tokenizer in tokenizers.items():
        if tokenizer is None:
            continue

        # Save tokenizer to pickle file
        tokenizer_fname = f"{name}.pickle"
        tokenizer_fpath = osp.join(subrun_path, tokenizer_fname)
        tokenizer.save_file(tokenizer_fpath)

        # Save tokenizer hparams to yaml file
        hparams_fpath = osp.join(hp_dpath, f"{name}.yaml")
        hparams = tokenizer.get_hparams()
        with open(hparams_fpath, "w") as file:
            yaml.dump(hparams, file)

        if tokenizer.is_fit():
            # Save vocabulary to csv file
            vocab_fname = f"vocabulary_{name}.csv"
            vocab_fpath = osp.join(subrun_path, vocab_fname)

            fieldnames = ("token", "occurrence", "index")
            data = [
                {
                    "token": token,
                    "occurrence": occurrence,
                    "index": tokenizer.token_to_id(token),
                }
                for token, occurrence in tokenizer.get_vocab().items()
            ]

            with open(vocab_fpath, "w") as file:
                writer = csv.DictWriter(file, fieldnames)
                writer.writeheader()
                writer.writerows(data)  # type: ignore

            other_metrics[f"{name}_vocab_size"] = tokenizer.get_vocab_size()
            other_metrics[
                f"{name}_min_sentence_size"
            ] = tokenizer.get_min_sentence_size()
            other_metrics[
                f"{name}_max_sentence_size"
            ] = tokenizer.get_max_sentence_size()

    # Remove optional None values
    params = {k: v for k, v in params.items() if v is not None}
    other_metrics = {k: v for k, v in other_metrics.items() if v is not None}

    other_metrics = {f"other/{name}": value for name, value in other_metrics.items()}

    if verbose >= 2:
        pylog.debug(
            f"Adding {len(params)} params :\n{yaml.dump(params, sort_keys=False)}"
        )
        pylog.debug(
            f"Adding {len(other_metrics)} metrics :\n{yaml.dump(other_metrics, sort_keys=False)}"
        )

    # Store params and metrics
    if pl_module is not None:
        for pl_logger in pl_module.loggers:
            if isinstance(pl_logger, CustomTensorboardLogger):
                pl_logger.log_hyperparams(params=params, metrics=other_metrics)

                if close_logger_on_end:
                    pl_logger.save_and_close()
            else:
                ns_params = Namespace()
                ns_params.__dict__.update(params)
                pl_logger.log_hyperparams(ns_params)
                pl_logger.log_metrics(other_metrics)


def _clean_value(value) -> Any:
    if isinstance(value, Tensor):
        if value.ndim == 0:
            return value.item()
        else:
            return value.tolist()
    else:
        return value
