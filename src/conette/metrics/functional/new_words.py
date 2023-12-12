#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging

from typing import Callable, Iterable, Union

import torch

from torch import Tensor


pylog = logging.getLogger(__name__)


def new_words(
    candidates: list[str],
    mult_references: list[list[str]],
    return_all_scores: bool = True,
    train_vocab: Iterable[str] = (),
    tokenizer: Callable[[str], list[str]] = str.split,
) -> Union[tuple[dict[str, Tensor], dict[str, Tensor]], Tensor]:
    tok_cands = list(map(tokenizer, candidates))
    del candidates, mult_references

    train_vocab = dict.fromkeys(train_vocab)

    dtype = torch.float64
    new_words_lst = [set(tokens).difference(train_vocab) for tokens in tok_cands]
    new_words_counts = torch.as_tensor(list(map(len, new_words_lst)), dtype=dtype)
    new_words_total = new_words_counts.mean()

    if return_all_scores:
        corpus_scores = {
            "new_words": new_words_total,
        }
        sents_scores = {
            "new_words": new_words_counts,
        }
        return corpus_scores, sents_scores
    else:
        return new_words_total
