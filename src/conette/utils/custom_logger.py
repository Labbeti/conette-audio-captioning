#!/usr/bin/env python
# -*- coding: utf-8 -*-

import copy
import datetime
import logging
import os.path as osp

from argparse import Namespace
from typing import Any, Optional, Union

from omegaconf import DictConfig
from pytorch_lightning.core.saving import save_hparams_to_yaml
from pytorch_lightning.loggers import TensorBoardLogger
from torch import Tensor


pylog = logging.getLogger(__name__)


class CustomTensorboardLogger(TensorBoardLogger):
    """Custom Tensorboard Logger for saving hparams and metrics in tensorboard because we cannot save hparams and metrics several times in SummaryWriter.

    Note : hparams and metrics are saved only when 'save_and_close' is called.
    """

    FNAME_HPARAMS = "hparams.yaml"
    FNAME_METRICS = "metrics.yaml"
    FNAME_ENDFILE = "endfile.txt"

    def __init__(
        self,
        save_dir: str,
        name: Optional[str] = "default",
        version: Union[None, int, str] = None,
        log_graph: bool = False,
        default_hp_metric: bool = True,
        prefix: str = "",
        params: Union[dict[str, Any], DictConfig, None] = None,
        verbose: bool = False,
        log_to_text: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(
            save_dir=save_dir,
            name=name,
            version=version,
            log_graph=log_graph,
            default_hp_metric=default_hp_metric,
            prefix=prefix,
            **kwargs,
        )

        params = _convert_dict_like_to_dict(params)
        if default_hp_metric:
            metrics = {"hp_metric": -1}
        else:
            metrics = {}

        self._all_hparams = params
        self._all_metrics = metrics
        self._verbose = verbose
        self._log_to_text = log_to_text

        self._closed = False

    def __exit__(self) -> None:
        if not self.is_closed():
            self.save_and_close()

    def log_hyperparams(
        self,
        params: Union[dict[str, Any], Namespace, None] = None,
        metrics: Union[dict[str, Any], Namespace, None] = None,
    ) -> None:
        params = _convert_dict_like_to_dict(params)
        metrics = _convert_dict_like_to_dict(metrics)

        none_metrics = {k: v for k, v in metrics.items() if v is None}
        if len(none_metrics) > 0:
            raise ValueError(f"Found None in metrics. (keys={none_metrics.keys()})")

        self._all_hparams.update(params)
        self._all_metrics.update(metrics)

    def finalize(self, status: str) -> None:
        # Called at the end of the training (after trainer.fit())
        self.experiment.flush()

    def update_files(self) -> None:
        self._all_hparams = {k: _convert_value(v) for k, v in self._all_hparams.items()}
        self._all_metrics = {k: _convert_value(v) for k, v in self._all_metrics.items()}

        self._all_hparams = dict(sorted(self._all_hparams.items()))

        fpath_hparams = osp.join(self.log_dir, self.FNAME_HPARAMS)
        save_hparams_to_yaml(fpath_hparams, self._all_hparams)

        fpath_metrics = osp.join(self.log_dir, self.FNAME_METRICS)
        save_hparams_to_yaml(fpath_metrics, self._all_metrics)

    def save_and_close(self) -> None:
        if self.is_closed():
            raise RuntimeError("CustomTensorboardLogger cannot be closed twice.")

        self.update_files()

        if self._log_to_text:
            prefix = f"{self.name}_{self.version}"
            self.experiment.add_text(f"{prefix}/all_hparams", str(self._all_hparams))
            self.experiment.add_text(f"{prefix}/all_metrics", str(self._all_metrics))

            for dic in (self._all_hparams, self._all_metrics):
                for name, value in dic.items():
                    self.experiment.add_text(f"{prefix}/{name}", str(value))

        super().log_hyperparams(self._all_hparams, self._all_metrics)
        self.experiment.flush()

        fpath_endfile = osp.join(self.log_dir, self.FNAME_ENDFILE)
        with open(fpath_endfile, "w") as file:
            now = datetime.datetime.now()
            now = now.strftime("%Y:%m:%d_%H:%M:%S")
            file.write(f"Process finished at {now}.\n")

        self._close()

    def _close(self) -> None:
        if self._verbose:
            pylog.debug(
                f"Closing {self.__class__.__name__}... ({self.is_closed()=}; {self.expt_is_closed()=})"
            )
        self.experiment.flush()
        super().finalize("test")
        self._closed = True

    def is_closed(self) -> bool:
        return self._closed or self.expt_is_closed()

    def expt_is_closed(self) -> bool:
        return self.experiment.all_writers is None

    @property
    def hparams(self) -> dict:
        return self._all_hparams

    @hparams.setter
    def hparams(self, other: dict) -> None:
        self._all_hparams = copy.deepcopy(other)

    @property
    def metrics(self) -> dict:
        return self._all_metrics


def _convert_value(v: Any) -> Any:
    if isinstance(v, Tensor):
        if v.nelement() == 1:
            return v.item()
        else:
            return v.tolist()
    elif isinstance(v, bool):
        return str(v)
    else:
        return v


def _convert_dict_like_to_dict(dic: Union[dict, Namespace, DictConfig, None]) -> dict:
    if dic is None:
        return {}
    elif isinstance(dic, Namespace):
        return dic.__dict__
    elif isinstance(dic, DictConfig):
        return dict(dic)
    else:
        return dic
