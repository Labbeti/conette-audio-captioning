#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Any, Iterable, Optional, Union

import torch

from pytorch_lightning import LightningModule
from pytorch_lightning.callbacks.callback import Callback
from torch import nn

from aac_metrics.classes.cider_d import CIDErD
from aac_metrics.classes.fense import FENSE

from conette.metrics.classes.diversity import Diversity
from conette.metrics.classes.text_stats import TextStats
from conette.nn.functional.get import get_device


class AACValidator(Callback):
    def __init__(
        self,
        monitors: Union[str, Iterable[str]],
        metrics_keys: Union[str, Iterable[str]] = (),
        computation_device: Union[str, torch.device, None] = "auto",
        other_device: Union[str, torch.device, None] = "cpu",
        build_on_start: bool = False,
    ) -> None:
        if isinstance(metrics_keys, str):
            metrics_keys = [metrics_keys]
        else:
            metrics_keys = list(metrics_keys)

        computation_device = get_device(computation_device)
        other_device = get_device(other_device)

        if isinstance(monitors, str):
            monitors = [monitors]
        else:
            monitors = list(monitors)

        super().__init__()
        self._monitors = monitors
        self._metrics_keys = metrics_keys
        self._computation_device = computation_device
        self._other_device = other_device

        self._cands_dic: dict[str, list[str]] = {}
        self._mrefs_lst: list[list[str]] = []
        self._metrics = {}

        if build_on_start:
            self.__build_metrics(computation_device)

    # Callback methods
    def on_fit_start(self, trainer, pl_module) -> None:
        if len(self._metrics) == 0:
            self.__build_metrics(pl_module.device)

    def on_fit_end(self, trainer, pl_module) -> None:
        del self._metrics
        self._metrics = {}

    def on_train_batch_end(
        self,
        trainer,
        pl_module,
        outputs: Optional[dict[str, Any]],
        batch,
        batch_idx,
        unused=0,
    ) -> None:
        self.__on_batch_end(outputs)

    def on_validation_batch_end(
        self,
        trainer,
        pl_module,
        outputs: Optional[dict[str, Any]],
        batch,
        batch_idx,
        dataloader_idx,
    ) -> None:
        self.__on_batch_end(outputs)

    def on_train_epoch_end(self, trainer, pl_module: LightningModule) -> None:
        self.__on_epoch_end(pl_module, "train/")
        self._cands_dic = {}
        self._mrefs_lst = []

    def on_validation_epoch_end(self, trainer, pl_module: LightningModule) -> None:
        self.__on_epoch_end(pl_module, "val/")
        self._cands_dic = {}
        self._mrefs_lst = []

    # Other methods
    def __build_metrics(
        self,
        computation_device: Union[str, torch.device, None],
    ) -> None:
        metrics: dict[str, nn.Module] = {
            "cider_d": CIDErD(return_all_scores=True),
            "div1": Diversity(return_all_scores=True),
            "stats": TextStats(return_all_scores=True),
        }

        if computation_device is None:
            computation_device = self._computation_device
        else:
            self._computation_device = get_device(computation_device)

        if (
            any("fense" in monitor for monitor in self._monitors)
            or "fense" in self._metrics_keys
        ):
            fense = FENSE(
                return_all_scores=True,
                device=computation_device,
            )
            metrics["fense"] = fense

        self._metrics = metrics
        self._metrics = {
            name: metric.to(device=self._other_device)
            for name, metric in self._metrics.items()
        }

    def __on_batch_end(self, outputs: Optional[dict[str, Any]]) -> None:
        if outputs is None or not isinstance(outputs, dict):
            return None

        cands_dic: dict[str, list[str]] = {
            name: values for name, values in outputs.items() if name.startswith("cands")
        }
        refs: Optional[list[str]] = outputs.get("refs")
        mrefs: Optional[list[list[str]]] = outputs.get("mrefs")

        if len(cands_dic) == 0:
            return None
        if (refs is None) == (mrefs is None):
            raise RuntimeError(
                f"Invalid batch output with ({refs is None=}, {mrefs is None=}). (expected (None, [...]) or ([...], None))"
            )

        if mrefs is None:
            if refs is None:
                raise RuntimeError(
                    f"Found candidates but no references. ({cands_dic=})"
                )
            mrefs = [[ref] for ref in refs]

        for key, cands_lst in cands_dic.items():
            if key in self._cands_dic:
                self._cands_dic[key] += cands_lst
            else:
                self._cands_dic[key] = cands_lst

        self._mrefs_lst += mrefs

    def __on_epoch_end(
        self,
        pl_module: LightningModule,
        prefix: str,
    ) -> None:
        if any(
            len(cands_lst) != len(self._mrefs_lst)
            for cands_lst in self._cands_dic.values()
        ):
            cands_lens = list(map(len, self._cands_dic.values()))
            mrefs_lens = [len(self._mrefs_lst)] * len(cands_lens)
            raise ValueError(
                f"Invalid number of candidates and references. (found {cands_lens=} != {mrefs_lens=})"
            )

        if len(self._cands_dic) <= 0:
            return None

        self._metrics = {
            name: metric.to(device=self._computation_device)
            for name, metric in self._metrics.items()
        }

        if not hasattr(pl_module, "tokenizer"):
            raise RuntimeError("Cannot find tokenizer in pl_module.")
        tokenizer: Any = pl_module.tokenizer  # type: ignore
        mrefs_lst = tokenizer.detokenize_rec(tokenizer.tokenize_rec(self._mrefs_lst))

        if len(self._cands_dic) == 1:
            cands_lst = next(iter(self._cands_dic.values()))

            scores = {}
            for metric in self._metrics.values():
                scores |= metric(cands_lst, mrefs_lst)[0]

            scores_dic = {"": scores}
            # scores_dic ex: {"": {"fer": tensor(0.1), "fense": tensor(0.3)}}
        else:
            scores_dic = {}
            for cand_name, cands_lst in self._cands_dic.items():
                scores = {}
                for metric in self._metrics.values():
                    scores |= metric(cands_lst, mrefs_lst)[0]
                scores_dic[cand_name] = scores
            # scores_dic ex: {"cands.": {"fer": tensor(0.1), "fense": tensor(0.3)}, ...}

        scores = {
            f"{prefix}{k}{metric_name}": score
            for k, corpus_scores in scores_dic.items()
            for metric_name, score in corpus_scores.items()
        }

        monitors_scores = {}
        for monitor in self._monitors:
            score = scores.pop(monitor, None)
            if score is not None:
                monitors_scores[monitor] = score

        bar_scores = monitors_scores
        non_bar_scores = {
            name: score for name, score in scores.items() if name not in bar_scores
        }

        log_kwargs: dict[str, Any] = dict(on_step=False, on_epoch=True, sync_dist=True)
        pl_module.log_dict(bar_scores, prog_bar=True, **log_kwargs)
        pl_module.log_dict(non_bar_scores, prog_bar=False, **log_kwargs)

        self._metrics = {
            name: metric.to(device=self._other_device)
            for name, metric in self._metrics.items()
        }
