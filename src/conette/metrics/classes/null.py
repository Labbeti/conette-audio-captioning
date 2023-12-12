#!/usr/bin/env python
# -*- coding: utf-8 -*-

from torchmetrics import Metric


class NullMetric(Metric):
    """Placeholder Metric. Method `compute` always returns 0."""

    is_differentiable = False
    higher_is_better = True
    full_state_update = False

    min_value = 0.0
    max_value = 0.0

    # Metric methods
    def update(self, *args, **kwargs) -> None:
        pass

    def compute(self) -> float:
        return 0.0
