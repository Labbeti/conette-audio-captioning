#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Any, Optional, Union

from nltk.stem import SnowballStemmer
from torch import Tensor
from torchmetrics import Metric

from conette.metrics.functional.jaccard import jaccard


class Jaccard(Metric):
    """Jaccard similarity, also known as "intersection over union"."""

    is_differentiable = False
    higher_is_better = True
    full_state_update = False

    min_value = 0.0
    max_value = 1.0

    def __init__(
        self,
        return_all_scores: bool = True,
        stemmer_lang: Optional[str] = "english",
    ) -> None:
        if stemmer_lang is not None:
            stemmer = SnowballStemmer(stemmer_lang)
        else:
            stemmer = None

        super().__init__()
        self.return_all_scores = return_all_scores
        self.stemmer_lang = stemmer_lang
        self.stemmer = stemmer

        self.candidates = []
        self.mult_references = []

    # Metric methods
    def compute(self) -> Union[Tensor, tuple[dict[str, Tensor], dict[str, Tensor]]]:
        return jaccard(
            self.candidates,
            self.mult_references,
            self.return_all_scores,
            self.stemmer,
        )

    def get_output_names(self) -> tuple[str, ...]:
        return ("jaccard",)

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

    # Magic methods
    def __getstate__(self) -> dict[str, Any]:
        return {
            "return_all_scores": self.return_all_scores,
            "stemmer_lang": self.stemmer_lang,
            "candidates": self.candidates,
            "mult_references": self.mult_references,
        }

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.return_all_scores = state["return_all_scores"]
        self.stemmer_lang = state["stemmer_lang"]
        self.candidates = state["candidates"]
        self.mult_references = state["mult_references"]

        if self.stemmer_lang is not None:
            stemmer = SnowballStemmer(self.stemmer_lang)
        else:
            stemmer = None
        self.stemmer = stemmer
