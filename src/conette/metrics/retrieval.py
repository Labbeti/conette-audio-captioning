#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Any, Callable, Union

import numpy as np
import torch
import tqdm

from torch import Tensor

from conette.nn.functional.misc import can_be_stacked


def retrieval_metrics(
    scores: Tensor,
    is_matching: Union[Callable[[int, int], bool], np.ndarray, Tensor],
    return_all_scores: bool = True,
    return_retrieved_indexes: bool = False,
    limit_relevant_with_k: bool = False,
    consider_only_best: bool = True,
    verbose: int = 0,
) -> Union[Tensor, tuple[dict[str, Tensor], dict[str, Tensor]]]:
    row_names = list(map(str, range(scores.shape[0])))
    col_names = list(map(str, range(scores.shape[1])))

    if isinstance(is_matching, Callable):
        matching_matrix = _matching_fn_to_matrix(is_matching, scores.shape)
    else:
        matching_matrix = is_matching
        assert (
            matching_matrix.shape == scores.shape
        ), f"{matching_matrix.shape=} != {scores.shape=}"

    qid2items = _matrix_to_qid2items(scores, row_names, col_names, matching_matrix)
    qid_mAP10s, qid_R1s, qid_R5s, qid_R10s, retrieved_indexes = _measure(
        qid2items, limit_relevant_with_k, consider_only_best, verbose
    )

    dtype = torch.float64
    qid_mAP10s = torch.as_tensor(qid_mAP10s, dtype=dtype)
    qid_R1s = torch.as_tensor(qid_R1s, dtype=dtype)
    qid_R5s = torch.as_tensor(qid_R5s, dtype=dtype)
    qid_R10s = torch.as_tensor(qid_R10s, dtype=dtype)

    retrieval_outs_sents: dict[str, Any] = {
        "mAP10": qid_mAP10s,
        "R1": qid_R1s,
        "R5": qid_R5s,
        "R10": qid_R10s,
    }
    retrieval_outs_corpus = {
        name: c_scores.mean() for name, c_scores in retrieval_outs_sents.items()
    }

    for n in (1, 5):
        uniq_top_n_count = torch.as_tensor(
            len(set(scores.argsort(dim=1, descending=True)[:, :n].flatten().tolist())),
            dtype=dtype,
        )
        uniq_top_n_max = torch.as_tensor(scores.shape, dtype=dtype).min()
        uniq_top_n_rate = uniq_top_n_count / uniq_top_n_max
        retrieval_outs_corpus[f"uniq_top{n}_count"] = uniq_top_n_count
        retrieval_outs_corpus[f"uniq_top{n}_max"] = uniq_top_n_max
        retrieval_outs_corpus[f"uniq_top{n}_rate"] = uniq_top_n_rate

    sorted_indexes = scores.argsort(dim=1, descending=True)
    mean_ranks = torch.empty((len(sorted_indexes),), dtype=dtype)
    ranks = []

    for i, indexes in enumerate(sorted_indexes):
        ranks_i = torch.where(matching_matrix[i][indexes])[0]
        # ranks of shape (n,)
        mean_rank_i = (ranks_i / matching_matrix.shape[1]).mean()
        mean_ranks[i] = mean_rank_i

        ranks.append(ranks_i)

    mean_rank = mean_ranks.mean()
    med_rank = mean_ranks.median()

    retrieval_outs_sents["mean_rank"] = mean_ranks
    retrieval_outs_corpus["mean_rank"] = mean_rank
    retrieval_outs_corpus["med_rank"] = med_rank

    if can_be_stacked(ranks):
        ranks = torch.stack(ranks)
    retrieval_outs_sents["rank"] = ranks

    if return_retrieved_indexes:
        retrieval_outs_sents["retrieved_indexes"] = torch.as_tensor(
            retrieved_indexes, dtype=torch.long
        )

    if return_all_scores:
        return retrieval_outs_corpus, retrieval_outs_sents
    else:
        return retrieval_outs_sents["mAP10"]


def _matching_fn_to_matrix(
    is_matching: Callable[[int, int], bool], size: tuple[int, int]
) -> Tensor:
    matching_matrix = torch.full(size, False, dtype=torch.bool)
    for i in range(size[0]):
        for j in range(size[1]):
            matching_matrix[i, j] = is_matching(i, j)
    return matching_matrix


def _matrix_to_qid2items(
    scores: Tensor,
    row_names: list[str],
    col_names: list[str],
    is_matching: Union[np.ndarray, Tensor],
) -> dict[str, list[tuple[str, float, bool]]]:
    assert tuple(scores.shape) == (len(row_names), len(col_names))
    qid2items = {}
    for i, name_i in enumerate(row_names):
        qid2items[name_i] = [
            (name_j, scores[i, j].item(), is_matching[i, j])
            for j, name_j in enumerate(col_names)
        ]
    return qid2items


def _measure(
    qid2items: dict[Any, list[tuple[Any, float, bool]]],
    limit_relevant_with_k: bool,
    consider_only_best: bool,
    verbose: int,
) -> tuple[list, list, list, list, list]:
    """Retrieval metrics over sample queries

    i.e., recall@{1, 5, 10}, mAP@10.
    BASED on https://github.com/xieh97/dcase2023-audio-retrieval/blob/master/postprocessing/xmodal_retrieval.py#L32
    """
    mAP_top = 10

    qid_R1s = []
    qid_R5s = []
    qid_R10s = []
    qid_mAP10s = []
    retrieved_indexes = []

    for items in tqdm.tqdm(qid2items.values(), disable=verbose < 2):
        scores = np.array([i[1] for i in items])
        targets = np.array([i[2] for i in items])

        # assert (
        #     targets.sum() == 1
        # )  # DEBUG: for text-to-audio, we expect only 1 audio per query

        desc_indices = np.argsort(scores, axis=-1)[::-1]
        targets = np.take_along_axis(arr=targets, indices=desc_indices, axis=-1)

        retrieved_indexes.append(desc_indices.tolist())

        # Recall at cutoff K
        targets_sum = np.sum(targets, dtype=float)

        top1_sum = np.sum(targets[:1], dtype=float)
        top5_sum = np.sum(targets[:5], dtype=float)
        top10_sum = np.sum(targets[:10], dtype=float)

        if limit_relevant_with_k and consider_only_best:
            raise ValueError(
                f"Incompatible arguments values {limit_relevant_with_k=} and {consider_only_best=}. (one or both must be False)"
            )

        elif limit_relevant_with_k:
            recall_at_1 = top1_sum / min(targets_sum, 1)
            recall_at_5 = top5_sum / min(targets_sum, 5)
            recall_at_10 = top10_sum / min(targets_sum, 10)

        elif consider_only_best:
            recall_at_1 = min(top1_sum, 1)
            recall_at_5 = min(top5_sum, 1)
            recall_at_10 = min(top10_sum, 1)

        else:  # default behaviour
            recall_at_1 = top1_sum / targets_sum
            recall_at_5 = top5_sum / targets_sum
            recall_at_10 = top10_sum / targets_sum

        qid_R1s.append(recall_at_1)
        qid_R5s.append(recall_at_5)
        qid_R10s.append(recall_at_10)

        # Mean average precision
        positions = np.arange(1, mAP_top + 1, dtype=float)[targets[:mAP_top] > 0]
        if len(positions) > 0:
            precisions = np.divide(
                np.arange(1, len(positions) + 1, dtype=float), positions
            )
            avg_precision = np.sum(precisions, dtype=float) / targets_sum
            qid_mAP10s.append(avg_precision)
        else:
            qid_mAP10s.append(0.0)

    return qid_mAP10s, qid_R1s, qid_R5s, qid_R10s, retrieved_indexes
