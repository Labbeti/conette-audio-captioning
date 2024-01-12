#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Optional, Union

import torch

from torch import Tensor
from torchmetrics import Metric

from conette.metrics.functional.self_bleu import self_bleu


class SelfBleuCands(Metric):
    is_differentiable = False
    higher_is_better = False
    full_state_update = False

    min_value = 0.0
    max_value = 1.0

    def __init__(
        self,
        max_ngram_sizes: int = 4,
        max_refs: Optional[int] = None,
        generator: Union[None, int, torch.Generator] = 1234,
    ) -> None:
        super().__init__()
        self.max_ngram_sizes = max_ngram_sizes
        self.max_refs = max_refs
        self.generator = generator
        self.candidates = []

    # Metric methods
    def compute(self) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
        return self_bleu(
            self.candidates,
            self.max_ngram_sizes,
            self.max_refs,
            self.generator,
        )

    def reset(self) -> None:
        self.candidates = []
        return super().reset()

    def update(
        self,
        candidates: list[list[str]],
        mult_references: list[list[list[str]]],
    ) -> None:
        self.candidates += candidates


class SelfBleuMRefs(Metric):
    is_differentiable = False
    higher_is_better = False
    full_state_update = False

    def __init__(
        self,
        max_ngram_sizes: int = 4,
        max_refs: Optional[int] = None,
        generator: Union[None, int, torch.Generator] = 1234,
    ) -> None:
        super().__init__()
        self.max_ngram_sizes = max_ngram_sizes
        self.max_refs = max_refs
        self.generator = generator
        self.references_flat = []

    # Metric methods
    def compute(self) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
        return self_bleu(
            self.references_flat,
            self.max_ngram_sizes,
            self.max_refs,
            self.generator,
        )

    def reset(self) -> None:
        self.references_flat = []
        return super().reset()

    def update(
        self,
        candidates: list[list[str]],
        mult_references: list[list[list[str]]],
    ) -> None:
        references_flat = [ref for refs in mult_references for ref in refs]
        self.references_flat += references_flat
