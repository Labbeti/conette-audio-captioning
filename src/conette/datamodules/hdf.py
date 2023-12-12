#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import os.path as osp

from typing import Iterable, Optional, Union

import tqdm

from torch import nn
from torch.utils.data.dataloader import DataLoader

from conette.datamodules.aac_dm import AACDataModule
from conette.datamodules.collate import AdvancedCollateDict
from conette.datamodules.common import (
    OnlineEncodeCaptionsTransform,
    get_auto_num_cpus,
)
from conette.datasets.hdf import HDFDataset
from conette.datasets.utils import (
    AACConcat,
    AACDuplicate,
    AACSelectColumnsWrapper,
    TransformWrapper,
    WrapperSampler,
)
from conette.tokenization.aac_tokenizer import AACTokenizer
from conette.utils.csum import csum_any


pylog = logging.getLogger(__name__)

DEFAULT_TRAIN_COLS = ("audio", "audio_shape", "captions")
DEFAULT_VAL_COLS = ("audio", "audio_shape", "captions")
DEFAULT_TEST_COLS = (
    "audio",
    "audio_shape",
    "captions",
    "dataset",
    "subset",
    "fname",
    "index",
)


class HDFDataModule(AACDataModule):
    TUNE_MODE = False
    _IGNORE_ARGS = (
        "train_audio_tfm",
        "val_audio_tfm",
        "test_audio_tfm",
        "train_tokenizer",
    )
    AUDIO_PADDINGS = ("batch", "longest", "crop")

    def __init__(
        self,
        # AACDataModule params
        root: str = "data",
        bsize: int = 512,
        n_workers: Optional[int] = 0,
        pin_memory: bool = True,
        train_drop_last: bool = False,
        verbose: int = 1,
        train_cols: Iterable[str] = DEFAULT_TRAIN_COLS,
        val_cols: Iterable[str] = DEFAULT_VAL_COLS,
        test_cols: Iterable[str] = DEFAULT_TEST_COLS,
        train_audio_tfm: Optional[nn.Module] = None,
        val_audio_tfm: Optional[nn.Module] = None,
        test_audio_tfm: Optional[nn.Module] = None,
        train_tokenizer: Optional[AACTokenizer] = None,
        # Other params
        train_hdfs: Union[str, Iterable[str]] = (),
        val_hdfs: Union[str, Iterable[str]] = (),
        test_hdfs: Union[str, Iterable[str]] = (),
        predict_hdfs: Union[str, Iterable[str]] = (),
        audio_padding: str = "batch",
        main_hdf_duplicate: Optional[str] = None,
        main_hdf_min: Optional[str] = None,
        main_hdf_balanced: Optional[Iterable[str]] = None,
        n_added_data: Optional[int] = None,
    ) -> None:
        """Initialize the AudioCaps datamodule for building dataloaders.

        :param root: The dataset parent directory. defaults to "data".
        :param bsize: The batch size of the dataloaders. defaults to 512.
        :param n_workers: The number of workers of the dataloaders. defaults to 0.
        :param pin_memory: If True, the dataloaders will pin memory of tensors. defaults to True.
        :param verbose: Verbose level. defaults to 1.
        :param train_cols: The columns to extract from the original HDF dataset source during training.
        :param val_cols: The columns to extract from the original HDF dataset source during validation.
        :param test_cols: The columns to extract from the original HDF dataset source during testing.
        :param train_audio_tfm: The train audio transform to apply to each item. defaults to None.
        :param val_audio_tfm: The val audio transform to apply to each item. defaults to None.
        :param test_audio_tfm: The test audio transform to apply to each item. defaults to None.
        :param train_tokenizer: The AACTokenizer for train captions. None will create a default AACTokenizer. defaults to None.
        :param train_hdfs: List of HDF filenames for training. defaults to ().
        :param val_hdfs: List of HDF filenames for validation. defaults to ().
        :param test_hdfs: List of HDF filenames for testing. defaults to ().
        :param predict_hdfs: List of HDF filenames for prediction. defaults to ().
        :param audio_padding: Audio batch padding mode. Can be one of ("batch", "crop", "longest"). defaults to "batch".
        :param main_hdf_duplicate: Duplicate the main train dataset to have the same length than the sum of other datasets added. defaults to None.
        :param main_hdf_min: Reduce other added per epoch to have the same length than the main train dataset. defaults to None.
        """
        # Process args
        root = osp.expanduser(osp.expandvars(root))

        if n_workers is None:
            n_workers = get_auto_num_cpus()
            if verbose >= 1:
                pylog.info(f"Found {n_workers} CPU that will be used for DataLoaders.")

        train_cols = list(train_cols)
        val_cols = list(val_cols)
        test_cols = list(test_cols)

        def process_hdfs_args(hdfs: Union[str, Iterable[str]]) -> list[str]:
            if isinstance(hdfs, str):
                return [hdfs]
            else:
                return list(hdfs)

        train_hdfs = process_hdfs_args(train_hdfs)
        val_hdfs = process_hdfs_args(val_hdfs)
        test_hdfs = process_hdfs_args(test_hdfs)
        predict_hdfs = process_hdfs_args(predict_hdfs)

        if train_tokenizer is None:
            train_tokenizer = AACTokenizer()

        # Check args
        if main_hdf_duplicate is not None and main_hdf_min is not None:
            raise ValueError(
                f"Cannot use arguments {main_hdf_duplicate=} and {main_hdf_min=} at the same time."
            )
        if main_hdf_duplicate is not None and main_hdf_duplicate not in train_hdfs:
            raise ValueError(
                f"Invalid argument {main_hdf_duplicate=}. (expected one of train hdf files {train_hdfs})"
            )

        if main_hdf_min is not None and main_hdf_min not in train_hdfs:
            raise ValueError(
                f"Invalid argument {main_hdf_min=}. (expected one of train hdf files {train_hdfs})"
            )

        if audio_padding not in self.AUDIO_PADDINGS:
            raise ValueError(
                f"Invalid argument {audio_padding=}. (expected one of {self.AUDIO_PADDINGS})"
            )

        if (
            main_hdf_min is None
            and main_hdf_balanced is None
            and n_added_data is not None
        ):
            raise ValueError(
                f"Invalid argument {n_added_data=} with {main_hdf_min=} and {main_hdf_balanced=}."
            )

        if main_hdf_balanced is not None and not all(
            hdf_name in train_hdfs for hdf_name in main_hdf_balanced
        ):
            raise ValueError(f"Invalid argument {main_hdf_balanced=}.")

        super().__init__(
            root=root,
            bsize=bsize,
            n_workers=n_workers,
            pin_memory=pin_memory,
            train_drop_last=train_drop_last,
            verbose=verbose,
            train_cols=train_cols,
            val_cols=val_cols,
            test_cols=test_cols,
        )
        self._train_audio_tfm = train_audio_tfm
        self._val_audio_tfm = val_audio_tfm
        self._test_audio_tfm = test_audio_tfm
        self._train_tokenizer = train_tokenizer

        self._wrapper_samplers: list[WrapperSampler] = []

    def train_dataloader(self) -> DataLoader:
        for wrapper_sampler in self._wrapper_samplers:
            prev_csum = csum_any(wrapper_sampler.indexes)
            wrapper_sampler.reset_indexes()
            if self.hp.verbose >= 2:
                csum = csum_any(wrapper_sampler.indexes)
                pylog.debug(f"Indexes has been shuffled. ({prev_csum}, {csum})")
        return super().train_dataloader()

    # Other methods
    def _setup_fit(self) -> None:
        keep_padding = (
            ("audio",) if self.hp.audio_padding in ("crop", "longest") else ()
        )
        train_dsets_lst = [
            HDFDataset(
                osp.join(self.hp.root, "HDF", fname),
                keep_padding=keep_padding,
            )
            for fname in self.hp.train_hdfs
        ]
        val_dsets_lst = [
            HDFDataset(
                osp.join(self.hp.root, "HDF", fname),
                keep_padding=keep_padding,
            )
            for fname in self.hp.val_hdfs
        ]
        if self.hp.verbose >= 2:
            pylog.debug(
                f"HDF datasets loaded. (train={len(train_dsets_lst)}, val={len(val_dsets_lst)})"
            )

        train_dsets_lst = [
            AACSelectColumnsWrapper(dset, include=self.hp.train_cols)
            for dset in train_dsets_lst
        ]
        val_dsets_lst = [
            AACSelectColumnsWrapper(dset, include=self.hp.val_cols)
            for dset in val_dsets_lst
        ]

        train_mrefs: list[list[str]] = [
            refs
            for train_dset_i in tqdm.tqdm(
                train_dsets_lst,
                disable=self.hp.verbose < 1,
                desc="Loading captions for build id-to-token mappings...",
            )
            for refs in train_dset_i.at(None, "captions")
        ]

        if self.hp.main_hdf_duplicate is not None:
            tgt_idx = self.hp.train_hdfs.index(self.hp.main_hdf_duplicate)
            tgt_dset = train_dsets_lst[tgt_idx]
            other_sum = sum(
                len(dset) for i, dset in enumerate(train_dsets_lst) if i != tgt_idx
            )

            if self.hp.verbose >= 1:
                pylog.info(
                    f"Duplicate dataset {self.hp.main_hdf_duplicate} from {len(tgt_dset)} to {other_sum}."
                )

            if len(tgt_dset) < other_sum:
                train_dsets_lst[tgt_idx] = AACDuplicate(tgt_dset, other_sum)  # type: ignore

        elif self.hp.main_hdf_min is not None:
            tgt_idx = self.hp.train_hdfs.index(self.hp.main_hdf_min)
            tgt_dset = train_dsets_lst[tgt_idx]
            other_dsets = [
                dset for i, dset in enumerate(train_dsets_lst) if i != tgt_idx
            ]
            other_dsets = AACConcat(*other_dsets)

            if self.hp.n_added_data is not None:
                n_added_data = self.hp.n_added_data
            else:
                n_added_data = len(tgt_dset)

            if self.hp.verbose >= 1:
                pylog.info(
                    f"Minimize others datasets from {len(other_dsets)} to {n_added_data}."
                )

            other_dsets = WrapperSampler(other_dsets, n_added_data)
            self._wrapper_samplers = [other_dsets]
            train_dsets_lst = [tgt_dset, other_dsets]

        elif self.hp.main_hdf_balanced is not None:
            train_hdf_fnames: list[str] = list(self.hp.train_hdfs)
            main_hdf_balanced: list[str] = list(self.hp.main_hdf_balanced)

            tgt_idxs = [
                train_hdf_fnames.index(hdf_name) for hdf_name in main_hdf_balanced
            ]
            tgt_dsets = [train_dsets_lst[tgt_idx] for tgt_idx in tgt_idxs]
            other_dsets = [
                dset for i, dset in enumerate(train_dsets_lst) if i not in tgt_idxs
            ]
            other_dsets = AACConcat(*other_dsets)

            max_ds_size = max(map(len, tgt_dsets + [other_dsets]))

            train_dsets_lst = []
            wrapper_samplers = []

            if self.hp.n_added_data is not None:
                n_added_data = self.hp.n_added_data
            else:
                n_added_data = max_ds_size
            del max_ds_size

            if self.hp.verbose >= 1:
                pylog.info(
                    f"Minimize others datasets from {len(other_dsets)} to {n_added_data}."
                )

            for tgt_ds in tgt_dsets + [other_dsets]:
                if len(tgt_ds) == n_added_data:
                    train_dsets_lst.append(tgt_ds)
                elif len(tgt_ds) < n_added_data:
                    train_dsets_lst.append(AACDuplicate(tgt_ds, n_added_data))
                else:  # >
                    wrapped = WrapperSampler(tgt_ds, n_added_data)
                    train_dsets_lst.append(wrapped)
                    wrapper_samplers.append(wrapped)

            self._wrapper_samplers = wrapper_samplers

        else:
            if self.hp.verbose >= 1:
                pylog.info("No change applied to added datasets.")

        if len(train_dsets_lst) == 1:
            train_dset = train_dsets_lst[0]
        else:
            train_dset = AACConcat(*train_dsets_lst)

        if len(val_dsets_lst) == 1:
            val_dset = val_dsets_lst[0]
        else:
            val_dset = AACConcat(*val_dsets_lst)

        del train_dsets_lst, val_dsets_lst

        if not self._train_tokenizer.is_fit():
            train_mrefs_flat = [ref for refs in train_mrefs for ref in refs]
            self._train_tokenizer.fit(train_mrefs_flat)

        train_tfm = OnlineEncodeCaptionsTransform(
            self._train_audio_tfm,
            "random",
            False,
            self._train_tokenizer,
            dict(add_bos_eos=True, default=None, padding=None),
        )
        val_tfm = OnlineEncodeCaptionsTransform(
            self._val_audio_tfm,
            slice(None),
            True,
            self._train_tokenizer,
            dict(
                add_bos_eos=True,
                default=self._train_tokenizer.unk_token,
                padding="batch",
            ),
        )

        train_dset = TransformWrapper(train_dset, train_tfm)
        val_dset = TransformWrapper(val_dset, val_tfm)

        self._train_dset = train_dset
        self._val_dset = val_dset

        pad_values = {
            "captions": self._train_tokenizer.pad_token_id,
            "mult_captions": self._train_tokenizer.pad_token_id,
        }
        if self.hp.audio_padding == "batch":
            pad_values["audio"] = 0.0  # type: ignore

        crop_keys = ("audio",) if self.hp.audio_padding == "crop" else ()
        self._train_collate = AdvancedCollateDict(pad_values, crop_keys)
        self._val_collate = AdvancedCollateDict(pad_values, crop_keys)

        if self.hp.verbose >= 1:
            vocab_size = self._train_tokenizer.get_vocab_size()
            pylog.info(f"Train dataset size: {len(train_dset)}")
            pylog.info(f"Validation dataset size: {len(val_dset)}")
            pylog.info(f"Vocabulary size: {vocab_size}")

    def _setup_test(self) -> None:
        keep_padding = (
            ("audio",) if self.hp.audio_padding in ("crop", "longest") else ()
        )
        dsets = {
            fname: HDFDataset(
                osp.join(self.hp.root, "HDF", fname),
                keep_padding=keep_padding,
            )
            for fname in self.hp.test_hdfs
        }
        test_tfm = OnlineEncodeCaptionsTransform(
            self._test_audio_tfm,
            slice(None),
            True,
            self._train_tokenizer,
            dict(
                add_bos_eos=True,
                default=self._train_tokenizer.unk_token,
                padding="batch",
            ),
        )

        dsets = {
            fname: AACSelectColumnsWrapper(dset, include=self.hp.test_cols)
            for fname, dset in dsets.items()
        }
        dsets = {
            fname: TransformWrapper(dset, test_tfm) for fname, dset in dsets.items()
        }

        self._test_dsets = dsets

        pad_values = {
            "captions": self._train_tokenizer.pad_token_id,
            "mult_captions": self._train_tokenizer.pad_token_id,
        }
        if self.hp.audio_padding == "batch":
            pad_values["audio"] = 0.0  # type: ignore

        crop_keys = ("audio",) if self.hp.audio_padding == "crop" else ()
        self._test_collate = AdvancedCollateDict(pad_values, crop_keys)

    def _setup_predict(self) -> None:
        keep_padding = (
            ("audio",) if self.hp.audio_padding in ("crop", "longest") else ()
        )
        dsets = {
            fname: HDFDataset(
                osp.join(self.hp.root, "HDF", fname),
                keep_padding=keep_padding,
            )
            for fname in self.hp.predict_hdfs
        }
        test_tfm = OnlineEncodeCaptionsTransform(
            self._test_audio_tfm,
            slice(None),
            True,
            self._train_tokenizer,
            dict(),
            mrefs_src_key=None,
        )

        dsets = {
            fname: AACSelectColumnsWrapper(
                dset, include=self.hp.test_cols, exclude=("captions",)
            )
            for fname, dset in dsets.items()
        }
        dsets = {
            fname: TransformWrapper(dset, test_tfm) for fname, dset in dsets.items()
        }

        self._predict_dsets = dsets

        pad_values = {}
        if self.hp.audio_padding == "batch":
            pad_values["audio"] = 0.0  # type: ignore

        crop_keys = ("audio",) if self.hp.audio_padding == "crop" else ()
        self._predict_collate = AdvancedCollateDict(pad_values, crop_keys)
