#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Iterable, Optional, Union

import torch

from torch import nn, Tensor
from torchmetrics.metric import Metric

from conette.nn.functional.mask import tensor_to_lengths, tensor_to_non_pad_mask


class MeanPredLen(Metric):
    is_differentiable: Optional[bool] = False
    higher_is_better: Optional[bool] = None
    full_state_update: bool = True

    def __init__(self, eos_id: int) -> None:
        super().__init__()
        self.eos_id = eos_id

        self.sum_lens: Tensor
        self.total: Tensor
        self.add_state("sum_lens", default=torch.zeros(()), dist_reduce_fx="sum")
        self.add_state("total", default=torch.zeros(()), dist_reduce_fx="sum")

    # Metric methods
    def update(
        self,
        preds: Tensor,
    ) -> None:
        assert preds.ndim == 2
        lengths = tensor_to_lengths(preds, end_value=self.eos_id)

        self.sum_lens += lengths.sum()
        self.total += lengths.shape[0]

    def compute(self) -> Tensor:
        return self.sum_lens / self.total


class TensorDiversity1(Metric):
    """Compute Diversity on 1-gram (also called Type-Token Ratio) on encoded sentences tensors."""

    is_differentiable: Optional[bool] = False
    higher_is_better: Optional[bool] = None
    full_state_update: bool = True

    def __init__(
        self,
        eos: int,
        excluded: Union[Iterable[int], Tensor] = (),
        return_sents_scores: bool = False,
    ) -> None:
        if isinstance(excluded, Tensor):
            excluded = excluded.flatten().tolist()
        super().__init__()
        self._eos = eos
        self._excluded = list(excluded)
        self._return_sents_scores = return_sents_scores

        self.scores: list[Tensor]
        self.add_state("scores", default=[], dist_reduce_fx=None)

    # Metric methods
    def update(
        self,
        preds: Tensor,
    ) -> None:
        """
        :param preds: Tensor of shape (bsize, N)
        """
        assert preds.ndim == 2
        preds_mask = tensor_to_non_pad_mask(preds, end_value=self._eos)

        scores = torch.empty((preds.shape[0],), dtype=torch.float, device=preds.device)
        for i, (pred, mask) in enumerate(zip(preds, preds_mask)):
            for value in self._excluded:
                mask &= pred.ne(value)
            vocab = pred[mask].unique()
            scores[i] = vocab.shape[0] / max(mask.sum().item(), 1)

        self.scores.append(scores)

    def compute(self) -> Tensor:
        if len(self.scores) > 0:
            scores = torch.cat(self.scores)
            if self._return_sents_scores:
                return scores
            else:
                return scores.mean()
        else:
            return torch.zeros((), dtype=torch.float, device=self.device)


class GlobalTensorVocabUsage(nn.Module):
    r"""Global Vocab Usage.

    Returns \frac{|hyp\_vocab|}{|ref\_vocab|}
    """

    def __init__(self, ignored_indexes: Union[Iterable[int], Tensor] = ()) -> None:
        if isinstance(ignored_indexes, Tensor):
            ignored_indexes = ignored_indexes.flatten().tolist()
        super().__init__()
        self._ignored_indexes = list(ignored_indexes)
        self._preds_vocab = None
        self._captions_vocab = None

    # Metric methods
    def reset(self) -> None:
        self._preds_vocab = None
        self._captions_vocab = None

    def forward(self, preds: Tensor, captions: Tensor) -> float:
        """
        :param preds: (bsize, pred_len) tensor
        :param captions: (bsize, capt_len) tensor
        """
        self.update(preds, captions)
        return self.compute()

    def update(self, preds: Tensor, captions: Tensor) -> None:
        preds = preds[preds == self._ignored_indexes]
        captions = captions[captions == self._ignored_indexes]

        preds_vocab = torch.unique(preds)
        captions_vocab = torch.unique(captions)

        if self._preds_vocab is None:
            self._preds_vocab = preds_vocab
        else:
            self._preds_vocab = torch.unique(torch.cat(self._preds_vocab, preds_vocab))

        if self._captions_vocab is None:
            self._captions_vocab = captions_vocab
        else:
            self._captions_vocab = torch.unique(
                torch.cat(self._captions_vocab, captions_vocab)
            )

    def compute(self) -> float:
        if (
            self._preds_vocab is not None
            and self._captions_vocab is not None
            and len(self._captions_vocab) > 0
        ):
            return len(self._preds_vocab) / len(self._captions_vocab)
        else:
            return 0.0
