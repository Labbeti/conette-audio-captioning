#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import tempfile
import time

from typing import Iterable, Optional, Union

import torch

from torch import Tensor
from torch.nn.parameter import Parameter

from aac_metrics.classes.bert_score_mrefs import BERTScoreMRefs
from aac_metrics.classes.bleu import BLEU
from aac_metrics.classes.meteor import METEOR
from aac_metrics.classes.rouge_l import ROUGEL
from aac_metrics.classes.evaluate import Evaluate
from aac_metrics.classes.fense import FENSE
from aac_metrics.classes.spider import SPIDEr
from aac_metrics.functional.spider_fl import _spider_fl_from_outputs

from conette.metrics.classes.diversity import Diversity
from conette.metrics.classes.new_words import NewWords
from conette.metrics.classes.text_stats import TextStats
from conette.nn.functional.get import get_device


pylog = logging.getLogger(__name__)


class AllMetrics(Evaluate):
    def __init__(
        self,
        preprocess: bool = True,
        device: Union[str, torch.device, None] = "auto",
        cache_path: str = "~/.cache",
        java_path: str = "java",
        tmp_path: str = tempfile.gettempdir(),
        meteor_java_max_memory: str = "2G",
        spice_n_threads: Optional[int] = None,
        spice_java_max_memory: str = "8G",
        spice_timeout: Union[None, int, Iterable[int]] = None,
        train_vocab: Optional[Iterable[str]] = None,
        verbose: int = 0,
        metrics_names: Union[None, Iterable[str], str] = None,
    ) -> None:
        device = get_device(device)

        if verbose >= 2:
            pylog.debug(f"Use {device=} for metrics.")

        if metrics_names is None:
            metrics_names = [
                "bert_score",
                "diversity",
                "text_stats",
                "new_words",
                "bleu_1",
                "bleu_2",
                "bleu_3",
                "bleu_4",
                "meteor",
                "rouge_l",
                "spider",
                "fense",
                "spider_fl",
            ]
        elif isinstance(metrics_names, str):
            metrics_names = [metrics_names]
        else:
            metrics_names = list(metrics_names)

        return_all_scores: bool = True
        metrics = []

        if "bert_score" in metrics_names:
            metric = BERTScoreMRefs(return_all_scores, device=device, verbose=verbose)
            metrics.append(metric)

        if "diversity" in metrics_names:
            metric = Diversity(return_all_scores, n_max=3)
            metrics.append(metric)

        if "text_stats" in metrics_names:
            metric = TextStats(return_all_scores)
            metrics.append(metric)

        if "new_words" in metrics_names and train_vocab is not None:
            metric = NewWords(return_all_scores, train_vocab)
            metrics.append(metric)

        if "bleu_1" in metrics_names:
            metric = BLEU(return_all_scores, n=1)
            metrics.append(metric)

        if "bleu_2" in metrics_names:
            metric = BLEU(return_all_scores, n=2)
            metrics.append(metric)

        if "bleu_3" in metrics_names:
            metric = BLEU(return_all_scores, n=3)
            metrics.append(metric)

        if "bleu_4" in metrics_names:
            metric = BLEU(return_all_scores, n=4)
            metrics.append(metric)

        if "meteor" in metrics_names:
            metric = METEOR(
                return_all_scores,
                cache_path=cache_path,
                java_path=java_path,
                java_max_memory=meteor_java_max_memory,
                verbose=verbose,
            )
            metrics.append(metric)

        if "rouge_l" in metrics_names:
            metric = ROUGEL(return_all_scores)
            metrics.append(metric)

        if "spider" in metrics_names:
            metric = SPIDEr(
                return_all_scores,
                cache_path=cache_path,
                java_path=java_path,
                tmp_path=tmp_path,
                n_threads=spice_n_threads,
                java_max_memory=spice_java_max_memory,
                timeout=spice_timeout,
                verbose=verbose,
            )
            metrics.append(metric)

        if "fense" in metrics_names:
            metric = FENSE(
                return_all_scores, device=device, return_probs=True, verbose=verbose
            )
            metrics.append(metric)

        super().__init__(
            preprocess=preprocess,
            cache_path=cache_path,
            java_path=java_path,
            tmp_path=tmp_path,
            verbose=verbose,
            metrics=metrics,
        )
        self._verbose = verbose
        self._metrics_names = metrics_names

        self.register_parameter(
            "placeholder", Parameter(torch.empty((0,), device=device))
        )
        self.placeholder: Parameter

    def compute(self) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
        outs = super().compute()

        if "spider_fl" in self._metrics_names:
            name = "SPIDEr-FL"
            if self._verbose >= 1:
                pylog.info(f"Computing {name} to outputs...")

            start = time.perf_counter()
            outs = _spider_fl_from_outputs(outs, outs)
            end = time.perf_counter()

            if self._verbose >= 1:
                duration = end - start
                pylog.info(f"Metric {name} computed in {duration:.2f}s.")

        return outs

    def extra_repr(self) -> str:
        return f"len={len(self)}, device={self.device}"

    @property
    def device(self) -> torch.device:
        return self.placeholder.device
