#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging

from typing import Callable, Union

import torch

from nltk.util import ngrams
from torch import Tensor


pylog = logging.getLogger(__name__)


def div_n(
    candidates: list[str],
    mult_references: list[list[str]],
    return_all_scores: bool = True,
    n: int = 1,
    tokenizer: Callable[[str], list[str]] = str.split,
) -> Union[tuple[dict[str, Tensor], dict[str, Tensor]], Tensor]:
    tok_cands = list(map(tokenizer, candidates))
    del candidates

    dtype = torch.float64
    diversities = _compute_div_n(tok_cands, n, dtype)
    diversity = diversities.mean()

    if return_all_scores:
        corpus_scores = {
            "div": diversity,
        }
        sents_scores = {
            "div": diversities,
        }
        return corpus_scores, sents_scores
    else:
        return diversity


def _compute_div_n(
    sentences: list[list[str]],
    n: int,
    dtype: torch.dtype,
) -> Tensor:
    diversities = [len(set(ngrams(sent, n))) / len(sent) for sent in sentences]
    diversities = torch.as_tensor(diversities, dtype=dtype)
    return diversities
