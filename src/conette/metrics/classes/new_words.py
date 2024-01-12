#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math

from typing import Iterable, Union

from torch import Tensor
from torchmetrics import Metric

from conette.metrics.functional.new_words import new_words


class NewWords(Metric):
    """Jaccard similarity, also known as "intersection over union"."""

    is_differentiable = False
    higher_is_better = True
    full_state_update = False

    min_value = 0.0
    max_value = math.inf

    def __init__(
        self,
        return_all_scores: bool = True,
        train_vocab: Iterable[str] = (),
    ) -> None:
        train_vocab = dict.fromkeys(train_vocab)

        super().__init__()
        self.return_all_scores = return_all_scores
        self.train_vocab = train_vocab

        self.candidates = []
        self.mult_references = []

    # Metric methods
    def compute(self) -> Union[Tensor, tuple[dict[str, Tensor], dict[str, Tensor]]]:
        return new_words(
            self.candidates,
            self.mult_references,
            self.return_all_scores,
            self.train_vocab,
        )

    def get_output_names(self) -> tuple[str, ...]:
        return ("new_words",)

    def reset(self) -> None:
        self.candidates = []
        self.mult_references = []
        return super().reset()

    def update(
        self,
        candidates: list[str],
        mult_references: list[list[str]],
    ) -> None:
        self.candidates += candidates
        self.mult_references += mult_references
