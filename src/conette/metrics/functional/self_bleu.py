#!/usr/bin/env python
# -*- coding: utf-8 -*-

import copy

from typing import Callable, Optional, Union

import torch

from torch import Tensor

from aac_metrics.functional.bleu import bleu


def self_bleu(
    sentences: list[str],
    n: int = 4,
    max_refs: Optional[int] = None,
    generator: Union[None, int, torch.Generator] = None,
    tokenizer: Callable[[str], list[str]] = str.split,
) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
    if isinstance(generator, int):
        generator = torch.Generator().manual_seed(generator)
    if max_refs is not None and max_refs >= len(sentences) - 1:
        raise ValueError(
            f"Invalid argument {max_refs=}. (found {max_refs=} >= {len(sentences)-1})"
        )

    self_bleu_scores = []
    for i, sentence in enumerate(sentences):
        if max_refs is None:
            other_candidates = copy.deepcopy(sentences)
            other_candidates.pop(i)
        else:
            continue_ = True
            indexes = []
            while continue_:
                indexes = torch.randperm(len(sentences), generator=generator)[
                    :max_refs
                ].tolist()
                continue_ = i in indexes
            other_candidates = [sentences[idx] for idx in indexes]

        score = bleu(
            [sentence],
            [other_candidates],
            n=n,
            tokenizer=tokenizer,
        )
        self_bleu_scores.append(score)

    dtype = torch.float64
    self_bleu_scores = torch.as_tensor(self_bleu_scores, dtype=dtype)
    self_bleu_score = self_bleu_scores.mean()

    corpus_scores = {
        f"self_bleu_{n}": self_bleu_score,
    }
    sents_scores = {
        f"self_bleu_{n}": self_bleu_scores,
    }
    return corpus_scores, sents_scores
