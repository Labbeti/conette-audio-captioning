#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging

from collections import Counter
from typing import Callable, Union

import torch

from torch import Tensor


pylog = logging.getLogger(__name__)


def text_stats(
    candidates: list[str],
    mult_references: list[list[str]],
    return_all_scores: bool = True,
    seed: Union[None, int, torch.Generator] = 123,
    tokenizer: Callable[[str], list[str]] = str.split,
) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
    """Compute text statistics about sentences lengths and vocab sizes."""

    if len(mult_references) <= 0:
        raise ValueError(
            f"Invalid number of references. (found {len(mult_references)} references)"
        )

    tok_cands = list(map(tokenizer, candidates))
    tok_mrefs = [list(map(tokenizer, refs)) for refs in mult_references]
    del candidates, mult_references

    sent_lens_cands = list(map(len, tok_cands))
    sent_lens_mrefs = [sum(map(len, refs)) / len(refs) for refs in tok_mrefs]

    dtype = torch.float64
    sent_lens_cands = torch.as_tensor(sent_lens_cands, dtype=dtype)
    sent_lens_mrefs = torch.as_tensor(sent_lens_mrefs, dtype=dtype)
    sent_lens_ratios = sent_lens_cands / sent_lens_mrefs

    global_cands_counter = Counter(token for cand in tok_cands for token in cand)
    global_mrefs_counter = Counter(
        token for refs in tok_mrefs for ref in refs for token in ref
    )

    total_mrefs_tokens = max(sum(global_mrefs_counter.values()), 1)
    vocab_coverage = sum(
        global_mrefs_counter[token] / total_mrefs_tokens
        for token in global_cands_counter.keys()
    )

    cands_vocab_in_ref = [
        token
        for token in global_cands_counter.keys()
        if token in global_mrefs_counter.keys()
    ]
    vocab_in_ref_len = torch.as_tensor(len(cands_vocab_in_ref), dtype=dtype)
    vocab_in_ref_ratio = vocab_in_ref_len / len(global_cands_counter)

    vocab_len_cands = torch.as_tensor(len(global_cands_counter), dtype=dtype)
    vocab_len_mrefs_full = torch.as_tensor(len(global_mrefs_counter), dtype=dtype)
    vocab_len_ratio_full = vocab_len_cands / vocab_len_mrefs_full
    vocab_coverage = torch.as_tensor(vocab_coverage, dtype=dtype)

    if isinstance(seed, int):
        generator = torch.Generator().manual_seed(seed)
    else:
        generator = seed

    max_n_refs_per_audio = max(len(refs) for refs in tok_mrefs)
    vocab_len_lst = torch.empty((max_n_refs_per_audio,), dtype=dtype)

    for i in range(max_n_refs_per_audio):
        indexes = [
            int(torch.randint(0, len(refs), (), generator=generator).item())
            for refs in tok_mrefs
        ]
        popped_refs = [refs[idx] for idx, refs in zip(indexes, tok_mrefs)]
        vocab_len = len(set(token for ref in popped_refs for token in ref))
        vocab_len_lst[i] = vocab_len

    vocab_len_mrefs_avg = vocab_len_lst.mean()
    vocab_len_ratio_avg = vocab_len_cands / vocab_len_mrefs_avg

    empty_sents = torch.as_tensor(
        [(1 if len(cand) == 0 else 0) for cand in tok_cands], dtype=dtype
    )
    empty_sents_rate = empty_sents.mean()

    if return_all_scores:
        sents_scores = {
            "sent_len_cands": sent_lens_cands,
            "sent_len.mrefs": sent_lens_mrefs,
            "sent_len.ratio": sent_lens_ratios,
            "empty_sents": empty_sents,
        }
        corpus_scores = {
            "sent_len.cands": sent_lens_cands.mean(),
            "sent_len.mrefs": sent_lens_mrefs.mean(),
            "sent_len.ratio": sent_lens_ratios.mean(),
            "vocab_len.cands": vocab_len_cands,
            "vocab_len.mrefs_full": vocab_len_mrefs_full,
            "vocab_len.ratio_full": vocab_len_ratio_full,
            "vocab_len.mrefs_avg": vocab_len_mrefs_avg,
            "vocab_len.ratio_avg": vocab_len_ratio_avg,
            "vocab_coverage": vocab_coverage,
            "vocab_in_ref_len": vocab_in_ref_len,
            "vocab_in_ref_ratio": vocab_in_ref_ratio,
            "empty_sents": empty_sents_rate,
            "sent_len.cands.min": sent_lens_cands.min(),
            "sent_len.cands.max": sent_lens_cands.max(),
        }

        return corpus_scores, sents_scores
    else:
        raise ValueError(
            f"Cannot compute text_stats() function with {return_all_scores=}."
        )
