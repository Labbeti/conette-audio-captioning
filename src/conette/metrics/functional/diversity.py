#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging

from typing import Callable, Union

import torch

from nltk.util import ngrams
from torch import Tensor


pylog = logging.getLogger(__name__)


def vocab_size(
    candidates: list[str],
    mult_references: list[list[str]],
    return_all_scores: bool = True,
    seed: Union[None, int, torch.Generator] = 123,
    tokenizer: Callable[[str], list[str]] = str.split,
) -> Union[tuple[dict[str, Tensor], dict[str, Tensor]], Tensor]:
    outs: tuple[dict[str, Tensor], dict[str, Tensor]] = diversity(  # type: ignore
        candidates=candidates,
        mult_references=mult_references,
        return_all_scores=True,
        n=1,
        cumulative=False,
        use_ngram_count=True,
        seed=seed,
        tokenizer=tokenizer,
    )
    corpus_outs, sents_outs = outs

    if return_all_scores:
        corpus_outs = {
            (
                k.replace("sents_div1.", "sents_vocab.").replace(
                    "corpus_div1.", "corpus_vocab."
                )
            ): v
            for k, v in corpus_outs.items()
        }
        sents_outs = {
            (k.replace("sents_div1.", "sents_vocab.")): v for k, v in sents_outs.items()
        }
        return corpus_outs, sents_outs
    else:
        return corpus_outs["corpus_div1.cands"]


def diversity(
    candidates: list[str],
    mult_references: list[list[str]],
    return_all_scores: bool = True,
    n: int = 1,
    cumulative: bool = False,
    use_ngram_count: bool = True,
    seed: Union[None, int, torch.Generator] = 123,
    tokenizer: Callable[[str], list[str]] = str.split,
) -> Union[tuple[dict[str, Tensor], dict[str, Tensor]], Tensor]:
    """Compute sentences and corpus n-grams diversities ratios from candidates and references with n-grams from 1 to n.

    :param candidates: The list of sentences to evaluate.
    :param mult_references: The list of list of sentences used as target.
    :param return_all_scores: If True, returns a tuple containing the globals and locals scores.
        Otherwise returns a scalar tensor containing the main global score.
        defaults to True.
    """
    tok_cands, tok_mrefs = _diversity_update(
        candidates,
        mult_references,
        tokenizer,
        [],
        [],
    )
    return _diversity_compute(
        tok_cands, tok_mrefs, return_all_scores, n, cumulative, use_ngram_count, seed
    )


def _diversity_compute(
    tok_cands: list[list[str]],
    tok_mrefs: list[list[list[str]]],
    return_all_scores: bool,
    n_max: int,
    cumulative: bool,
    use_ngram_count: bool,
    seed: Union[None, int, torch.Generator],
) -> Union[tuple[dict[str, Tensor], dict[str, Tensor]], Tensor]:
    if len(tok_mrefs) <= 0:
        raise ValueError(
            f"Invalid number of references. (found {len(tok_mrefs)} references)"
        )

    dtype = torch.float64
    sents_divs_cands = torch.empty((len(tok_cands), n_max), dtype=dtype)
    sents_divs_mrefs = torch.empty((len(tok_mrefs), n_max), dtype=dtype)

    for i, (cand, refs) in enumerate(zip(tok_cands, tok_mrefs)):
        div_cand = _sent_diversities(cand, n_max, cumulative, use_ngram_count, dtype)
        refs_divs = [
            _sent_diversities(ref, n_max, cumulative, use_ngram_count, dtype)
            for ref in refs
        ]
        if len(refs_divs) > 0:
            div_refs = sum(refs_divs) / len(refs_divs)
        else:
            div_refs = torch.zeros((n_max,), dtype=dtype)

        sents_divs_cands[i] = div_cand
        sents_divs_mrefs[i] = div_refs

    sents_divs_ratios = torch.where(
        sents_divs_mrefs != 0.0, sents_divs_cands / sents_divs_mrefs, 0.0
    )
    corpus_div_cands = _corpus_diversities(
        tok_cands, n_max, cumulative, use_ngram_count, dtype
    )

    if isinstance(seed, int):
        generator = torch.Generator().manual_seed(seed)
    else:
        generator = seed

    max_n_refs_per_audio = max(len(refs) for refs in tok_mrefs)
    corpus_div_mrefs_all = torch.empty((max_n_refs_per_audio, n_max), dtype=dtype)

    for i in range(max_n_refs_per_audio):
        indexes = [
            int(torch.randint(0, len(refs), (), generator=generator).item())
            for refs in tok_mrefs
        ]
        popped_refs = [refs[idx] for idx, refs in zip(indexes, tok_mrefs)]
        corpus_div_mrefs_i = _corpus_diversities(
            popped_refs, n_max, cumulative, use_ngram_count, dtype
        )
        corpus_div_mrefs_all[i] = corpus_div_mrefs_i

    # corpus_div_mrefs_all: (n_refs_per_audio, n_max)
    corpus_div_mrefs = corpus_div_mrefs_all.mean(dim=0)
    corpus_div_ratio = torch.where(
        corpus_div_mrefs != 0.0,
        corpus_div_cands / corpus_div_mrefs,
        0.0,
    )

    sents_div_cands = sents_divs_cands.mean(dim=0)
    sents_div_mrefs = sents_divs_mrefs.mean(dim=0)
    sents_div_ratio = sents_divs_ratios.mean(dim=0)

    if return_all_scores:
        corpus_outs = {}
        sents_outs = {}
        for n in range(1, n_max + 1):
            corpus_outs |= {
                f"sents_div{n}.cands": sents_div_cands[n - 1],
                f"sents_div{n}.mrefs": sents_div_mrefs[n - 1],
                f"sents_div{n}.ratio": sents_div_ratio[n - 1],
                f"corpus_div{n}.cands": corpus_div_cands[n - 1],
                f"corpus_div{n}.mrefs": corpus_div_mrefs[n - 1],
                f"corpus_div{n}.ratio": corpus_div_ratio[n - 1],
            }
            sents_outs |= {
                f"sents_div{n}.cands": sents_divs_cands[:, n - 1],
                f"sents_div{n}.mrefs": sents_divs_mrefs[:, n - 1],
                f"sents_div{n}.ratio": sents_divs_ratios[:, n - 1],
            }

        return corpus_outs, sents_outs
    else:
        return sents_div_ratio[-1]


def _diversity_update(
    candidates: list[str],
    mult_references: list[list[str]],
    tokenizer: Callable[[str], list[str]],
    prev_tok_cands: list[list[str]],
    prev_tok_mrefs: list[list[list[str]]],
) -> tuple[list[list[str]], list[list[list[str]]]]:
    new_tok_cands = list(map(tokenizer, candidates))
    new_tok_mrefs = [list(map(tokenizer, refs)) for refs in mult_references]
    prev_tok_cands += new_tok_cands
    prev_tok_mrefs += new_tok_mrefs
    return prev_tok_cands, prev_tok_mrefs


def _sent_diversities(
    sent: list[str],
    n_max: int,
    cumulative: bool,
    use_ngram_count: bool,
    dtype: torch.dtype,
) -> Tensor:
    """
    :returns: tensor shape: (n_max,)
    """
    diversities = torch.zeros((n_max,), dtype=dtype)

    if len(sent) == 0:
        return diversities

    deno_count = torch.zeros((n_max,), dtype=dtype)
    uniq_ngrams_count = torch.zeros((n_max,), dtype=dtype)

    for n in range(1, min(n_max, len(sent)) + 1):
        ngrams_lst = list(ngrams(sent, n))
        ngrams_set = set(ngrams_lst)

        if use_ngram_count:
            deno_count[n - 1] += len(ngrams_lst)
        else:
            deno_count[n - 1] += len(sent)
        uniq_ngrams_count[n - 1] = len(ngrams_set)

    if cumulative:
        uniq_ngrams_count = uniq_ngrams_count.cumsum(0)
        deno_count = deno_count.cumsum(0)

        diversities = uniq_ngrams_count / deno_count.clamp(min=1.0)
        arange = torch.arange(1, n_max + 1, dtype=dtype)
        diversities = diversities / arange

    else:
        diversities = uniq_ngrams_count / deno_count.clamp(min=1.0)

    return diversities


def _corpus_diversities(
    sents: list[list[str]],
    n_max: int,
    cumulative: bool,
    use_ngram_count: bool,
    dtype: torch.dtype,
) -> Tensor:
    """
    :returns: tensor shape: (n_max,)
    """
    deno_count = torch.zeros((n_max,), dtype=dtype)
    uniq_ngrams_sets = [set() for _ in range(n_max)]

    for sent in sents:
        for n in range(1, min(n_max, len(sent)) + 1):
            ngrams_lst = list(ngrams(sent, n))
            ngrams_set = set(ngrams_lst)

            if use_ngram_count:
                deno_count[n - 1] += len(ngrams_lst)
            else:
                deno_count[n - 1] += len(sent)
            uniq_ngrams_sets[n - 1] |= ngrams_set

    uniq_ngrams_count = torch.as_tensor([len(s) for s in uniq_ngrams_sets], dtype=dtype)

    if cumulative:
        uniq_ngrams_count = uniq_ngrams_count.cumsum(0)
        deno_count = deno_count.cumsum(0)
        diversities = uniq_ngrams_count / deno_count.clamp(min=1.0)
        arange = torch.arange(1, n_max + 1, dtype=dtype)
        diversities = diversities / arange

    else:
        diversities = uniq_ngrams_count / deno_count.clamp(min=1.0)

    return diversities
