#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import math

from typing import Any, Callable, Union

from gensim import downloader
from gensim.downloader import load
from torch import Tensor
from torchmetrics import Metric

from conette.metrics.functional.wmd import wmdistance


pylog = logging.getLogger(__name__)


class WMDistance(Metric):
    """Word Mover Distance.

    Output is in range [0, +inf[.
    """

    is_differentiable = False
    higher_is_better = False
    full_state_update = False

    min_value = 0.0
    max_value = math.inf

    def __init__(
        self,
        return_all_scores: bool = True,
        tokenizer: Callable[[str], list[str]] = str.split,
        model_name: str = "word2vec-google-news-300",
        verbose: int = 0,
    ) -> None:
        if verbose >= 2:
            pylog.debug(f"Gensim data base dir: {downloader.BASE_DIR=}.")

        super().__init__()
        self._return_all_scores = return_all_scores
        self._tokenizer = tokenizer
        self._model_name = model_name
        self._model = load(model_name, return_path=False)
        self._verbose = verbose

        if verbose >= 2:
            path: str = load(model_name, return_path=True)  # type: ignore
            pylog.debug(f"Load gensim model {model_name=} from {path=}.")

        self._candidates = []
        self._mult_references = []

    # Metric methods
    def compute(self) -> Union[tuple[dict[str, Tensor], dict[str, Tensor]], Tensor]:
        return wmdistance(
            self._candidates,
            self._mult_references,
            self._return_all_scores,
            self._tokenizer,
            self._model,
        )

    def get_output_names(self) -> tuple[str, ...]:
        return ("wmd",)

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

    def __getstate__(self) -> dict[str, Any]:
        return {
            "tokenizer": self._tokenizer,
            "model_name": self._model_name,
            "candidates": self._candidates,
            "mult_references": self._mult_references,
        }

    def __setstate__(self, state: dict[str, Any]) -> None:
        self._tokenizer = state["tokenizer"]
        self._model_name = state["model_name"]
        self._candidates = state["candidates"]
        self._mult_references = state["mult_references"]

        self._model = load(self._model_name)
