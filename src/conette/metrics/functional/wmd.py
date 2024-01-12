#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging

from typing import Callable, Union

import torch

from gensim.downloader import load
from gensim.models.keyedvectors import KeyedVectors
from torch import Tensor


pylog = logging.getLogger(__name__)


def wmdistance(
    candidates: list[str],
    mult_references: list[list[str]],
    return_all_scores: bool = True,
    tokenizer: Callable[[str], list[str]] = str.split,
    model: Union[str, KeyedVectors] = "word2vec-google-news-300",  # type: ignore
    verbose: int = 0,
) -> Union[tuple[dict[str, Tensor], dict[str, Tensor]], Tensor]:
    if isinstance(model, str):
        model: KeyedVectors = load(model, return_path=False)  # type: ignore
        if verbose >= 2:
            path: str = load(model, return_path=True)  # type: ignore
            pylog.debug(f"Load gensim model from {path=}.")

    dtype = torch.float64
    tok_cands = list(map(tokenizer, candidates))
    tok_mrefs = [list(map(tokenizer, refs)) for refs in mult_references]

    distances = torch.zeros((len(tok_cands),), dtype=torch.float64)

    for i, (tok_cand, tok_refs) in enumerate(zip(tok_cands, tok_mrefs)):
        distances_i = [model.wmdistance(tok_cand, tok_ref) for tok_ref in tok_refs]
        distances_i = torch.as_tensor(distances_i, dtype=dtype)
        distances[i] = distances_i.mean()

    distance = distances.mean()

    if return_all_scores:
        corpus_scores = {
            "wmd": distance,
        }
        sents_scores = {
            "wmd": distances,
        }
        return corpus_scores, sents_scores
    else:
        return distance
