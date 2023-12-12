#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math

from typing import Callable, Union

import torch

from torch import Tensor

from aac_metrics.classes.base import AACMetric

from conette.metrics.functional.diversity import (
    _diversity_compute,
    _diversity_update,
)


class Diversity(AACMetric):
    full_state_update = False
    higher_is_better = True
    is_differentiable = False

    min_value = 0.0
    max_value = math.inf

    def __init__(
        self,
        return_all_scores: bool = True,
        n_max: int = 1,
        cumulative: bool = False,
        use_ngram_count: bool = True,
        seed: Union[None, int, torch.Generator] = 123,
        tokenizer: Callable[[str], list[str]] = str.split,
    ) -> None:
        super().__init__()
        self._return_all_scores = return_all_scores
        self._n_max = n_max
        self._cumulative = cumulative
        self._use_ngram_count = use_ngram_count
        self._seed = seed
        self._tokenizer = tokenizer

        self._tok_cands = []
        self._tok_mrefs = []

    # Metric methods
    def compute(self) -> Union[tuple[dict[str, Tensor], dict[str, Tensor]], Tensor]:
        return _diversity_compute(
            tok_cands=self._tok_cands,
            tok_mrefs=self._tok_mrefs,
            return_all_scores=self._return_all_scores,
            n_max=self._n_max,
            cumulative=self._cumulative,
            use_ngram_count=self._use_ngram_count,
            seed=self._seed,
        )

    def get_output_names(self) -> tuple[str, ...]:
        return (
            f"sents_div{self._n_max}.cands",
            f"sents_div{self._n_max}.mrefs",
            f"sents_div{self._n_max}.ratio",
            f"corpus_div{self._n_max}.cands",
            f"corpus_div{self._n_max}.mrefs",
            f"corpus_div{self._n_max}.ratio",
        )

    def reset(self) -> None:
        self._tok_cands = []
        self._tok_mrefs = []
        return super().reset()

    def update(
        self,
        candidates: list[str],
        mult_references: list[list[str]],
    ) -> None:
        self._tok_cands, self._tok_mrefs = _diversity_update(
            candidates,
            mult_references,
            self._tokenizer,
            self._tok_cands,
            self._tok_mrefs,
        )
