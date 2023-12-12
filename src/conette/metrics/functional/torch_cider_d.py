#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math

from collections import defaultdict, Counter
from typing import Any, Mapping, Union

import torch

from torch import Tensor


class FrozenHashableTensor(Tensor):
    def __init__(self, x: Tensor) -> None:
        super().__init__()
        self.set_(x.storage())
        self._hash = self.hash()

    def hash(self) -> int:
        arange = torch.arange(self.nelement(), dtype=self.dtype, device=self.device)
        x = self.flatten() * arange
        return int(x.sum().item())

    def __eq__(self, other: Any) -> bool:
        return (
            isinstance(other, FrozenHashableTensor)
            and self.shape == other.shape
            and bool(torch.eq(self, other).all().item())
        )

    def __hash__(self) -> int:
        return self._hash


def torch_cider_d(
    candidates: Tensor,
    mult_references: Tensor,
    return_all_scores: bool = True,
    n: int = 4,
    bos_id: int = 1,
    eos_id: int = 2,
    sigma: float = 6.0,
    return_tfidf: bool = False,
) -> Union[Tensor, tuple[dict[str, Tensor], dict[str, Any]]]:
    """
    :param n: set cider to sum over 1 to 4-grams
    :param sigma: set the standard deviation parameter for gaussian penalty
    """
    cooked_cands, cooked_mrefs = _torch_cider_d_update(
        candidates,
        mult_references,
        n,
        bos_id,
        eos_id,
        [],
        [],
    )
    return _torch_cider_d_compute(
        cooked_cands,
        cooked_mrefs,
        return_all_scores,
        n,
        sigma,
        return_tfidf,
    )


def _torch_cider_d_update(
    candidates: Tensor,
    mult_references: Tensor,
    n: int,
    bos_id: int,
    eos_id: int,
    prev_cooked_cands: list,
    prev_cooked_mrefs: list,
) -> tuple[list, list]:
    if len(candidates) != len(mult_references):
        raise ValueError(
            f"Invalid number of candidates and references. (found {len(candidates)=} != {len(mult_references)=})"
        )
    new_cooked_mrefs = [
        [_cook_sentence(ref, n, bos_id, eos_id) for ref in refs]
        for refs in mult_references
    ]
    new_cooked_cands = [_cook_sentence(cand, n, bos_id, eos_id) for cand in candidates]
    prev_cooked_cands += new_cooked_cands
    prev_cooked_mrefs += new_cooked_mrefs
    return prev_cooked_cands, prev_cooked_mrefs


def _torch_cider_d_compute(
    cooked_cands: list[Counter[FrozenHashableTensor]],
    cooked_mrefs: list[list[Counter[FrozenHashableTensor]]],
    return_all_scores: bool,
    n: int,
    sigma: float,
    return_tfidf: bool,
) -> Union[Tensor, tuple[dict[str, Tensor], dict[str, Any]]]:
    if len(cooked_cands) <= 1:
        raise ValueError(
            f"CIDEr metric does not support less than 2 candidates with 2 references. (found {len(cooked_cands)} candidates, but expected > 1)"
        )
    # compute idf
    document_frequency = _compute_doc_freq(cooked_mrefs)
    # compute log reference length
    log_ref_len = math.log(float(len(cooked_mrefs)))
    # sanity check: assert to check document frequency
    assert len(cooked_cands) >= max(document_frequency.values())
    # compute cider score
    cider_d_scores, tfidf_lst = _compute_cider(
        cooked_cands,
        cooked_mrefs,
        document_frequency,
        log_ref_len,
        n,
        sigma,
    )
    cider_d_score = cider_d_scores.mean()

    if return_all_scores:
        cider_d_global_outs = {
            "cider_d": cider_d_score,
        }
        cider_d_local_outs = {
            "cider_d": cider_d_scores,
        }
        if return_tfidf:
            cider_d_local_outs["tfidf_lst"] = tfidf_lst  # type: ignore

        return cider_d_global_outs, cider_d_local_outs
    else:
        return cider_d_score


def _cook_sentence(
    sentence: Tensor,
    n: int,
    bos_id: int,
    eos_id: int,
) -> Counter[FrozenHashableTensor]:
    if sentence[0] == bos_id:
        sentence = sentence[1:]

    if eos_id in sentence:
        eos_pos = sentence.eq(eos_id).int().argmax()
        sentence = sentence[:eos_pos]

    sentence = FrozenHashableTensor(sentence)

    counter = Counter()
    for k in range(1, n + 1):
        for i in range(len(sentence) - k + 1):
            ngram = sentence[i : i + k]
            counter[ngram] += 1

    return counter


def _compute_doc_freq(
    cooked_mrefs: list[list[Counter[FrozenHashableTensor]]],
) -> Counter[FrozenHashableTensor]:
    """
    Compute term frequency for reference data.
    This will be used to compute idf (inverse document frequency later)
    The term frequency is stored in the object
    :return: None
    """
    document_frequency = Counter()
    for cooked_refs in cooked_mrefs:
        # refs, k ref captions of one image
        for ngram in set(
            ngram for cooked_ref in cooked_refs for ngram in cooked_ref.keys()
        ):
            document_frequency[ngram] += 1
    # maxcounts[ngram] = max(maxcounts.get(ngram,0), count)
    return document_frequency


def _counter_to_vec(
    counters: Mapping[FrozenHashableTensor, int],
    log_ref_len: float,
    n: int,
    document_frequency: Counter[FrozenHashableTensor],
) -> tuple[list[defaultdict], Tensor, int]:
    """
    Function maps counts of ngram to vector of tfidf weights.
    The function returns vec, an array of dictionary that store mapping of n-gram and tf-idf weights.
    The n-th entry of array denotes length of n-grams.
    :param cnts:
    :return: vec (array of dict), norm (array of float), length (int)
    """
    vec = [defaultdict(float) for _ in range(n)]
    length = 0
    norm = torch.zeros((n,), dtype=torch.float64)

    for ngram, term_freq in counters.items():
        # give word count 1 if it doesn't appear in reference corpus
        log_df = math.log(max(1.0, document_frequency[ngram]))

        # ngram index
        ni = len(ngram) - 1

        # tf (term_freq) * idf (precomputed idf) for n-grams
        vec[ni][ngram] = float(term_freq) * (log_ref_len - log_df)

        # compute norm for the vector.  the norm will be used for computing similarity
        norm[ni] += pow(vec[ni][ngram], 2)

        if ni == 1:
            length += term_freq

    norm = torch.sqrt(norm)
    return vec, norm, length


def _similarity(
    cand_vec: list[defaultdict],
    ref_vec: list[defaultdict],
    cand_norm: Tensor,
    ref_norm: Tensor,
    cand_len: int,
    ref_len: int,
    n: int,
    sigma: float,
) -> Tensor:
    """
    Compute the cosine similarity of two vectors.
    :param vec_hyp: array of dictionary for vector corresponding to hypothesis
    :param vec_ref: array of dictionary for vector corresponding to reference
    :param norm_hyp: array of float for vector corresponding to hypothesis
    :param norm_ref: array of float for vector corresponding to reference
    :param length_hyp: int containing length of hypothesis
    :param length_ref: int containing length of reference
    :return: array of score for each n-grams cosine similarity
    """
    # measure consine similarity
    val = torch.zeros((n,), dtype=torch.float64)

    for ni in range(n):
        # ngram
        for ngram, count in cand_vec[ni].items():
            # vrama91 : added clipping
            val[ni] += min(count, ref_vec[ni][ngram]) * ref_vec[ni][ngram]

    norms = cand_norm * ref_norm
    norms[norms == 0.0] = 1.0
    val = val / norms

    # vrama91: added a length based gaussian penalty
    delta = float(cand_len - ref_len)
    val = val * math.e ** (-(delta**2) / (2 * sigma**2))

    return val


def _compute_cider(
    cooked_cands: list[Counter[FrozenHashableTensor]],
    cooked_mrefs: list[list[Counter[FrozenHashableTensor]]],
    document_frequency: Counter,
    log_ref_len: float,
    n: int,
    sigma: float,
    scale: float = 10.0,
) -> tuple[Tensor, list[tuple]]:
    scores = torch.empty((len(cooked_cands),), dtype=torch.float64)
    tfidf_lst = []

    for i, (cooked_cand, cooked_refs) in enumerate(zip(cooked_cands, cooked_mrefs)):
        # compute vector for test captions
        vec, norm, length = _counter_to_vec(
            cooked_cand, log_ref_len, n, document_frequency
        )
        # compute vector for ref captions
        ngrams_scores = torch.zeros((n,), dtype=torch.float64)
        vec_refs = []
        for ref in cooked_refs:
            vec_ref, norm_ref, length_ref = _counter_to_vec(
                ref, log_ref_len, n, document_frequency
            )
            vec_refs.append(vec_ref)
            ngrams_scores += _similarity(
                vec, vec_ref, norm, norm_ref, length, length_ref, n, sigma
            )
        # change by vrama91 - mean of ngram scores, instead of sum
        # divide by number of mult_references
        agg_ngrams_scores = ngrams_scores.mean() / len(cooked_refs)
        # multiply score by 10
        agg_ngrams_scores *= scale
        # append score of an image to the score list
        scores[i] = agg_ngrams_scores
        tfidf_lst.append((vec, vec_refs))

    return scores, tfidf_lst
