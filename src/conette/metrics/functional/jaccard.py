#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Callable, Optional, Union

import torch

from nltk.stem import StemmerI
from torch import Tensor


def jaccard(
    candidates: list[str],
    mult_references: list[list[str]],
    return_all_scores: bool = True,
    stemmer: Optional[StemmerI] = None,
    tokenizer: Callable[[str], list[str]] = str.split,
) -> Union[tuple[dict[str, Tensor], dict[str, Tensor]], Tensor]:
    """Compute Jaccard score."""

    tok_cands = list(map(tokenizer, candidates))
    tok_mrefs = [list(map(tokenizer, refs)) for refs in mult_references]
    del candidates, mult_references

    jaccard_scores = torch.empty((len(tok_cands),), dtype=torch.float64)

    for i, (tok_cand, tok_refs) in enumerate(zip(tok_cands, tok_mrefs)):
        if stemmer is not None:
            tok_cand = [stemmer.stem(token) for token in tok_cand]
            tok_refs = [[stemmer.stem(token) for token in ref] for ref in tok_refs]

        tok_cand = set(tok_cand)
        tok_refs = [set(ref) for ref in tok_refs]

        similarities = []
        for tok_ref in tok_refs:
            similarity = len(tok_cand.intersection(tok_ref)) / len(
                tok_cand.union(tok_ref)
            )
            similarities.append(similarity)

        if len(similarities) > 0:
            sim = sum(similarities) / len(similarities)
        else:
            sim = 0.0
        jaccard_scores[i] = sim

    jaccard_score = jaccard_scores.mean()

    if return_all_scores:
        corpus_scores = {
            "jaccard": jaccard_score,
        }
        sents_scores = {
            "jaccard": jaccard_scores,
        }
        return corpus_scores, sents_scores
    else:
        return jaccard_score
