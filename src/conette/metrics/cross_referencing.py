#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import sys

from typing import Union

import torch
import tqdm

from torch import Tensor

from aac_metrics.classes.base import AACMetric
from aac_metrics.utils.tokenization import preprocess_mult_sents
from conette.metrics.classes.all_metrics import AllMetrics


pylog = logging.getLogger(__name__)
MODES = ("random", "columns")


def compute_cross_referencing(
    msents: list[list[str]],
    mode: str = "random",
    seed: Union[int, torch.Generator, None] = 1234,
    preprocess: bool = True,
    max_cross_refs: int = sys.maxsize,
    metrics: Union[None, AACMetric] = None,
    verbose: int = 1,
) -> tuple[list[dict[str, Tensor]], list[dict[str, Tensor]]]:
    """Compute cross-referencing 'Human' scores for all metrics.

    Works only when all multiple sentences have the same number of sentences individually.
    """
    if len(msents) == 0:
        raise ValueError(
            "Invalid number of mult sentences. (expected at least 1 set of sentences)"
        )

    n_sents_per_item_lst = list(map(len, msents))
    if not all(n_sents == n_sents_per_item_lst[0] for n_sents in n_sents_per_item_lst):
        n_sents_set = list(set(n_sents_per_item_lst))
        raise ValueError(
            f"Invalid n_sents list. (found different number of sentences per item with {n_sents_set=})"
        )

    n_sents_per_item = n_sents_per_item_lst[0]
    del n_sents_per_item_lst
    if n_sents_per_item <= 1:
        raise ValueError(f"Cannot compute cross-referencing with {n_sents_per_item=}")
    elif verbose >= 2:
        pylog.debug(f"Found {n_sents_per_item=}.")

    if isinstance(seed, int):
        gen = torch.Generator().manual_seed(seed)
    else:
        gen = seed

    if preprocess:
        msents = preprocess_mult_sents(msents)

    if metrics is None:
        metrics = AllMetrics(preprocess=False)

    max_cross_refs = min(n_sents_per_item, max_cross_refs)
    all_outs_corpus = []
    all_outs_sents = []

    for i in tqdm.trange(max_cross_refs, disable=verbose < 1):
        if mode == "columns":
            cands_i = [sents[i] for sents in msents]
            mrefs_not_i = [
                [sent for j, sent in enumerate(sents) if j != i] for sents in msents
            ]

        elif mode == "random":
            indexes = torch.randint(0, n_sents_per_item, (len(msents),), generator=gen)
            indexes = indexes.tolist()
            cands_i = [sents[idx] for sents, idx in zip(msents, indexes)]
            mrefs_not_i = [
                [sent for j, sent in enumerate(sents) if j != idx]
                for sents, idx in zip(msents, indexes)
            ]

        else:
            raise ValueError(f"Invalid argument {mode=}. (expected one of {MODES})")

        outs_corpus, outs_sents = metrics(
            cands_i,
            mrefs_not_i,
        )
        all_outs_corpus.append(outs_corpus)
        all_outs_sents.append(outs_sents)

    return all_outs_corpus, all_outs_sents
