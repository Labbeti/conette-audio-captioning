#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging

from argparse import Namespace
from typing import Any, Callable, Iterable, Optional

from pytorch_lightning import LightningDataModule
from torch.utils.data.dataloader import DataLoader


pylog = logging.getLogger(__name__)


class AACDataModule(LightningDataModule):
    DISABLE_TEARDOWN: bool = False
    _IGNORE_ARGS: tuple[str, ...] = ()

    def __init__(
        self,
        root: str = "data",
        bsize: int = 512,
        n_workers: Optional[int] = 0,
        pin_memory: bool = True,
        train_drop_last: bool = False,
        verbose: int = 1,
        train_cols: Iterable[str] = (),
        val_cols: Iterable[str] = (),
        test_cols: Iterable[str] = (),
    ) -> None:
        super().__init__()
        self._setup_fit_done = False
        self._setup_test_done = False
        self._setup_predict_done = False

        self._train_dset: Any = None
        self._val_dset: Any = None
        self._test_dsets: dict[str, Any] = {}
        self._predict_dsets: dict[str, Any] = {}

        self._train_collate: Optional[Callable] = None
        self._val_collate: Optional[Callable] = None
        self._test_collate: Optional[Callable] = None
        self._predict_collate: Optional[Callable] = None

        self.save_hyperparameters(ignore=self._IGNORE_ARGS)

    # Abstract methods
    def _setup_fit(self) -> None:
        raise NotImplementedError("Abstract method")

    def _setup_test(self) -> None:
        raise NotImplementedError("Abstract method")

    def _setup_predict(self) -> None:
        raise NotImplementedError("Abstract method")

    # LightningDataModule methods
    def prepare_data(self) -> None:
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        if stage in ("fit", "validate", None) and not self._setup_fit_done:
            if self.hp.verbose >= 1:
                pylog.info("Starting fit setup...")

            self._setup_fit()
            self._setup_fit_done = True

            if self.hp.verbose >= 1:
                dsets = {"train": self._train_dset, "val": self._val_dset}
                dsets = {name: ds for name, ds in dsets.items() if ds is not None}
                sizes = {name: len(ds) for name, ds in dsets.items()}
                pylog.info(f"Setup for train is done with {sizes}.")

        if stage in ("test", None) and not self._setup_test_done:
            if self.hp.verbose >= 1:
                pylog.info("Starting test setup...")

            self._setup_test()
            self._setup_test_done = True

            if self.hp.verbose >= 1:
                dsets = self._test_dsets
                dsets = {
                    name: ds for name, ds in self._test_dsets.items() if ds is not None
                }
                sizes = {name: len(ds) for name, ds in dsets.items()}
                pylog.info(f"Setup for test is done with {sizes}.")

        if stage in ("predict", None) and not self._setup_predict_done:
            if self.hp.verbose >= 1:
                pylog.info("Starting predict setup...")

            self._setup_predict()
            self._setup_predict_done = True

            if self.hp.verbose >= 1:
                sizes = {name: len(dset) for name, dset in self._predict_dsets.items()}
                pylog.info(f"Setup for predict is done with {sizes}.")

    def teardown(self, stage: Optional[str] = None) -> None:
        if self.DISABLE_TEARDOWN:
            if self.hp.verbose >= 0:
                pylog.warning(
                    f"Teardown has been called with {stage=}, but it has been disabled with global var DISABLE_TEARDOWN."
                )
            return None

        if self.hp.verbose >= 2:
            pylog.debug(f"Teardown stage {stage}.")

        # note: do not teardown when stage=="validate" to avoid re-build fit datasets twice
        if stage in ("fit", None):
            self._train_dset: Any = None
            self._val_dset: Any = None
            self._setup_fit_done = False

        if stage in ("test", None):
            self._test_dsets = {}
            self._predict_dsets = {}
            self._setup_test_done = False

        if stage in ("predict", None):
            self._predict_dsets = {}
            self._setup_predict_done = False

    def train_dataloader(self) -> DataLoader:
        if self.hp.verbose >= 2:
            pylog.debug("Build train dataloader(s)...")

        return DataLoader(
            dataset=self._train_dset,
            batch_size=self.hp.bsize,
            num_workers=self.hp.n_workers,
            shuffle=True,
            collate_fn=self._train_collate,
            pin_memory=self.hp.pin_memory,
            drop_last=self.hp.train_drop_last,
            sampler=None,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self._val_dset,
            batch_size=self.hp.bsize,
            num_workers=self.hp.n_workers,
            shuffle=False,
            collate_fn=self._val_collate,
            pin_memory=self.hp.pin_memory,
            drop_last=False,
        )

    def test_dataloader(self) -> list[DataLoader]:
        return [
            DataLoader(
                dataset=dset,
                batch_size=self.hp.bsize,
                num_workers=self.hp.n_workers,
                shuffle=False,
                collate_fn=self._test_collate,
                pin_memory=self.hp.pin_memory,
                drop_last=False,
            )
            for dset in self._test_dsets.values()
        ]

    def predict_dataloader(self) -> list[DataLoader]:
        return [
            DataLoader(
                dataset=dset,
                batch_size=self.hp.bsize,
                num_workers=self.hp.n_workers,
                shuffle=False,
                collate_fn=self._predict_collate,
                pin_memory=self.hp.pin_memory,
                drop_last=False,
            )
            for dset in self._predict_dsets.values()
        ]

    # Other methods
    @property
    def hp(self) -> Namespace:
        return Namespace(**self.hparams)

    @property
    def hp_init(self) -> Namespace:
        return Namespace(**self.hparams_initial)

    @property
    def root(self) -> str:
        return self.hparams["root"]

    @property
    def bsize(self) -> int:
        return self.hparams["bsize"]

    @property
    def n_workers(self) -> int:
        return self.hparams["n_workers"]

    @property
    def pin_memory(self) -> bool:
        return self.hparams["pin_memory"]

    @property
    def verbose(self) -> int:
        return self.hparams["verbose"]

    @property
    def train_cols(self) -> list[str]:
        return self.hparams["train_cols"]

    @property
    def val_cols(self) -> list[str]:
        return self.hparams["val_cols"]

    @property
    def test_cols(self) -> list[str]:
        return self.hparams["test_cols"]
