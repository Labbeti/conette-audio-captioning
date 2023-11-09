#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Iterable, Optional, Union

from torch import nn, Tensor

from conette.nn.functional.mask import masked_mean


class CrossEntropyLossMean(nn.CrossEntropyLoss):
    def __init__(
        self,
        weight: Optional[Tensor] = None,
        ignore_index: int = -100,
        label_smoothing: float = 0.0,
        dim: Union[int, Iterable[int], None] = None,
    ) -> None:
        """CrossEntropyLoss with dim option to average specific dimension(s) in output."""
        if isinstance(dim, Iterable):
            dim = tuple(dim)
        super().__init__(
            weight=weight,
            ignore_index=ignore_index,
            reduction="none",
            label_smoothing=label_smoothing,
        )
        self.dim = dim

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        losses = super().forward(input, target)
        non_pad_mask = target != self.ignore_index
        losses = masked_mean(losses, non_pad_mask, self.dim)
        return losses

    def extra_repr(self) -> str:
        return f"dim={self.dim}"
