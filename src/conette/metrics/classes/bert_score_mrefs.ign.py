#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Union

import torch

from torch import nn, Tensor
from torchmetrics.text.bert import _DEFAULT_MODEL

from aac_metrics.classes.base import AACMetric

from conette.metrics.functional.bert_score_mrefs import (
    bert_score_mrefs,
    _load_model_and_tokenizer,
)


class BERTScoreMRefs(AACMetric):
    """BERTScore metric which supports multiple references.

    The implementation is based on the bert_score implementation of torchmetrics.

    - Paper: https://arxiv.org/pdf/1904.09675.pdf

    For more information, see :func:`~aac_metrics.functional.bert_score.bert_score_mrefs`.
    """

    full_state_update = False
    higher_is_better = True
    is_differentiable = False

    min_value = 0.0
    max_value = 1.0

    def __init__(
        self,
        return_all_scores: bool = True,
        model: Union[str, nn.Module] = _DEFAULT_MODEL,
        device: Union[str, torch.device, None] = "auto",
        batch_size: int = 32,
        num_threads: int = 0,
        max_length: int = 64,
        reset_state: bool = True,
        verbose: int = 0,
    ) -> None:
        model, tokenizer = _load_model_and_tokenizer(
            model, None, device, reset_state, verbose
        )

        super().__init__()
        self._return_all_scores = return_all_scores
        self._model = model
        self._tokenizer = tokenizer
        self._device = device
        self._batch_size = batch_size
        self._num_threads = num_threads
        self._max_length = max_length
        self._reset_state = reset_state
        self._verbose = verbose

        self._candidates = []
        self._mult_references = []

    def compute(self) -> Union[tuple[dict[str, Tensor], dict[str, Tensor]], Tensor]:
        return bert_score_mrefs(
            self._candidates,
            self._mult_references,
            self._return_all_scores,
            self._model,
            self._tokenizer,
            self._device,
            self._batch_size,
            self._num_threads,
            self._max_length,
            self._reset_state,
            self._verbose,
        )

    def get_output_names(self) -> tuple[str, ...]:
        return (
            "bert_score.precision",
            "bert_score.recalll",
            "bert_score.f1",
        )

    def reset(self) -> None:
        self._candidates = []
        self._mult_references = []
        return super().reset()

    def update(
        self,
        candidates: list[str],
        mult_references: list[list[str]],
    ) -> None:
        self._candidates += candidates
        self._mult_references += mult_references
