#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time

from typing import Optional, Union

from pytorch_lightning import LightningModule
from pytorch_lightning.callbacks.callback import Callback


class TimeTrackerCallback(Callback):
    def __init__(self, log_fit_duration: bool = False) -> None:
        super().__init__()
        self._log_fit_duration = log_fit_duration

        self._fit_start_time = 0.0
        self._fit_end_time = 0.0
        self._test_start_time = 0.0
        self._test_end_time = 0.0
        self._epoch_starts = []
        self._epoch_ends = []
        self._n_fit_ended = 0
        self._n_test_ended = 0

    def on_fit_start(self, trainer, pl_module) -> None:
        self._fit_start_time = time.perf_counter()

    def on_fit_end(self, trainer, pl_module: LightningModule) -> None:
        self._fit_end_time = time.perf_counter()

        if self._log_fit_duration:
            key = "fit_duration" + (
                "" if self._n_test_ended == 0 else str(self._n_test_ended)
            )
            pl_module.log(key, self.get_fit_duration(), on_step=False, on_epoch=True)
        self._n_fit_ended += 1

    def on_train_epoch_start(self, trainer, pl_module) -> None:
        self._epoch_starts.append(time.perf_counter())

    def on_train_epoch_end(self, trainer, pl_module) -> None:
        self._epoch_ends.append(time.perf_counter())

    def on_test_start(self, trainer, pl_module) -> None:
        self._test_start_time = time.perf_counter()

    def on_test_end(self, trainer, pl_module) -> None:
        self._test_end_time = time.perf_counter()

        if self._log_fit_duration:
            key = "test_duration" + (
                "" if self._n_test_ended == 0 else str(self._n_test_ended)
            )
            pl_module.log(key, self.get_fit_duration(), on_step=False, on_epoch=True)
        self._n_test_ended += 1

    def get_fit_duration(self) -> float:
        """Return the fit duration in seconds."""
        return self._fit_end_time - self._fit_start_time

    def get_test_duration(self) -> float:
        """Return the test duration in seconds."""
        return self._test_end_time - self._test_start_time

    def get_fit_duration_in_hours(self) -> float:
        return self.get_fit_duration() / 3600.0

    def get_test_duration_in_hours(self) -> float:
        return self.get_test_duration() / 3600.0

    def get_fit_duration_formatted(self) -> str:
        """Return the fit duration as ISO format ddTHH:mm:ss."""
        return format_duration(self.get_fit_duration())

    def get_test_duration_formatted(self) -> str:
        """Return the test duration as ISO format ddTHH:mm:ss."""
        return format_duration(self.get_test_duration())

    def get_epoch_mean_duration_in_min(self, epoch: Optional[int] = None) -> float:
        if len(self._epoch_ends) > 0:
            if epoch is None:
                if len(self._epoch_starts) == len(self._epoch_ends):
                    maxidx = None
                elif len(self._epoch_starts) - 1 == len(self._epoch_ends):
                    maxidx = -1
                else:
                    raise ValueError("Invalid epoch starts list.")

                return (
                    (sum(self._epoch_ends) - sum(self._epoch_starts[:maxidx]))
                    / len(self._epoch_ends)
                    / 60.0
                )
            else:
                return (self._epoch_ends[epoch] - self._epoch_starts[epoch]) / 60.0
        else:
            return -1.0


def format_duration(
    duration_sec: Union[int, float],
    days_hours_sep: str = "_",  # "T"
    other_sep: str = "-",  # ":"
    force_days: bool = False,
) -> str:
    """Get formatted duration as {dd}_{HH}-{mm}-{ss} or {HH}-{mm}-{ss}"""
    duration_sec = int(duration_sec)
    rest, seconds = divmod(duration_sec, 60)
    rest, minutes = divmod(rest, 60)
    if rest > 24 or force_days:
        days, hours = divmod(rest, 24)
        duration_str = f"{days:02d}{days_hours_sep}{hours:02d}{other_sep}{minutes:02d}{other_sep}{seconds:02d}"
    else:
        hours = rest
        duration_str = f"{hours:02d}{other_sep}{minutes:02d}{other_sep}{seconds:02d}"
    return duration_str
