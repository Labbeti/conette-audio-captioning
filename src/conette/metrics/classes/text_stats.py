#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Callable, Union

import torch

from torch import Tensor

from aac_metrics.classes.base import AACMetric

from conette.metrics.functional.text_stats import text_stats


class TextStats(AACMetric):
    full_state_update = False
    higher_is_better = True
    is_differentiable = False

    def __init__(
        self,
        return_all_scores: bool = True,
        seed: Union[None, int, torch.Generator] = 123,
        tokenizer: Callable[[str], list[str]] = str.split,
    ) -> None:
        super().__init__()
        self._return_all_scores = return_all_scores
        self._seed = seed
        self._tokenizer = tokenizer

        self._candidates = []
        self._mult_references = []

    # Metric methods
    def compute(self) -> Union[tuple[dict[str, Tensor], dict[str, Tensor]], Tensor]:
        return text_stats(
            self._candidates,
            self._mult_references,
            self._return_all_scores,
            self._seed,
            self._tokenizer,
        )

    def get_output_names(self) -> tuple[str, ...]:
        return (
            "sent_len.cands",
            "sent_len.mrefs",
            "sent_len.ratio",
            "vocab_len.cands",
            "vocab_len.mrefs_full",
            "vocab_len.ratio_full",
            "vocab_len.mrefs_avg",
            "vocab_len.ratio_avg",
            "vocab_coverage",
            "vocab_in_ref_len",
            "vocab_in_ref_ratio",
            "empty_sents",
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
