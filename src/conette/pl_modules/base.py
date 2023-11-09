#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging

from abc import abstractmethod
from argparse import Namespace
from typing import Any, Callable, ClassVar, Iterable, Mapping, Optional, Union

import torch

from pytorch_lightning import LightningModule, LightningDataModule, Trainer
from pytorch_lightning.utilities.types import _METRIC_COLLECTION
from torch import nn, Tensor
from torch.nn.modules.module import _IncompatibleKeys

from conette.pl_modules.common import (
    count_params,
    default_get_example,
    default_configure_optimizers,
    has_datamodule,
    has_trainer,
    ON_EPOCH_KWARGS,
)
from conette.tokenization.aac_tokenizer import AACTokenizer
from conette.utils.csum import csum_module
from conette.utils.log_utils import warn_once


pylog = logging.getLogger(__name__)


class AACLightningModule(LightningModule):
    ON_EPOCH_KWARGS: ClassVar[dict[str, Any]] = ON_EPOCH_KWARGS

    def __init__(
        self,
        tokenizers: Union[
            None,
            AACTokenizer,
            Mapping[str, AACTokenizer],
            Iterable[AACTokenizer],
            nn.ModuleDict,
        ],
    ) -> None:
        if tokenizers is None:
            tokenizers = AACTokenizer()
        if isinstance(tokenizers, AACTokenizer):
            tokenizers = {"0": tokenizers}
        if isinstance(tokenizers, Mapping):
            tokenizers = nn.ModuleDict(tokenizers)  # type: ignore
        elif isinstance(tokenizers, Iterable):
            tokenizers = list(tokenizers)  # type: ignore
            tokenizers = dict(range(len(tokenizers)), tokenizers)  # type: ignore
            tokenizers = nn.ModuleDict(tokenizers)  # type: ignore

        # Sanity check
        assert isinstance(tokenizers, nn.ModuleDict)
        assert all(
            isinstance(tokenizer, AACTokenizer) for tokenizer in tokenizers.values()
        )

        super().__init__()
        self.tokenizers: nn.ModuleDict = tokenizers

    # --- Setup methods
    def configure_optimizers(self) -> tuple[list, list]:
        return default_configure_optimizers(self)

    def setup(self, stage: Optional[str] = None) -> None:
        if stage in ("fit", None):
            self.attach_example()

        if not self.is_built():
            self.build_model()

    def load_state_dict(
        self,
        state_dict: Mapping[str, Any],
        strict: bool = True,
    ) -> _IncompatibleKeys:
        if not self.is_built():
            all_fit = True
            for name, tokenizer in self.tokenizers.items():
                if not isinstance(tokenizer, AACTokenizer):
                    continue

                tok_prefix = f"tokenizers.{name}."
                tok_data = {
                    k: v for k, v in state_dict.items() if k.startswith(tok_prefix)
                }
                if len(tok_data) > 0:
                    tok_data = {k[len(tok_prefix) :]: v for k, v in tok_data.items()}
                    tokenizer.load_state_dict(tok_data, strict)

                all_fit &= tokenizer.is_fit()

            if all_fit:
                self.build_model()
            else:
                pylog.error(
                    "Cannot build the model from state_dict. (tokenizer is not fit)"
                )

        incompatibles_keys = super().load_state_dict(state_dict, strict)
        return incompatibles_keys

    # Abstract methods
    @abstractmethod
    def build_model(self) -> None:
        raise NotImplementedError("Abstract method.")

    @abstractmethod
    def is_built(self) -> bool:
        raise NotImplementedError("Abstract method.")

    @abstractmethod
    def encode_audio(
        self, audio: Tensor, audio_shape: Tensor, *args, **kwargs
    ) -> dict[str, Tensor]:
        raise NotImplementedError("Abstract method.")

    @abstractmethod
    def decode_audio(
        self,
        encoder_outs: dict[str, Tensor],
        decode_method: str,
        **kwargs,
    ) -> Any:
        raise NotImplementedError("Abstract method.")

    # Properties
    @property
    def hp(self) -> Namespace:
        # note: use Namespace instead of AttributeDict because linter does not like it
        return Namespace(**self.hparams)

    @property
    def hp_init(self) -> Namespace:
        # note: use Namespace instead of AttributeDict because linter does not like it
        return Namespace(**self.hparams_initial)

    @property
    def bos_id(self) -> int:
        return self.tokenizer.bos_token_id

    @property
    def eos_id(self) -> int:
        return self.tokenizer.eos_token_id

    @property
    def pad_id(self) -> int:
        return self.tokenizer.pad_token_id

    @property
    def unk_id(self) -> int:
        return self.tokenizer.unk_token_id

    @property
    def special_ids(self) -> list[int]:
        return [self.bos_id, self.eos_id, self.pad_id, self.unk_id]

    @property
    def tch_device(self) -> torch.device:
        return torch.device(self.device)

    @property
    def tch_dtype(self) -> torch.dtype:
        return self.dtype  # type: ignore

    @property
    def tokenizer(self) -> AACTokenizer:
        if len(self.tokenizers) == 0:
            raise ValueError("This module does not have any tokenizers.")
        elif len(self.tokenizers) > 1:
            warn_once(
                f"You are using property '.tokenizer' but this module has {len(self.tokenizers)} tokenizers."
                f"Please use method .get_tokenizer(.) to get a specific tokenizer by index or name."
                f"This property will return the first tokenizer by default.",
                pylog,
            )
        tokenizer = next(iter(self.tokenizers.values()))
        return tokenizer  # type: ignore

    # LightningModule methods
    def log(
        self,
        name: str,
        value: _METRIC_COLLECTION,
        prog_bar: bool = False,
        logger: bool = True,
        on_step: Optional[bool] = False,  # from None
        on_epoch: Optional[bool] = True,  # from None
        reduce_fx: Union[str, Callable] = "mean",
        enable_graph: bool = False,
        sync_dist: bool = True,  # from False
        sync_dist_group: Optional[Any] = None,
        add_dataloader_idx: bool = True,
        batch_size: Optional[int] = None,
        metric_attribute: Optional[str] = None,
        rank_zero_only: bool = False,
    ) -> None:
        return super().log(
            name,
            value,
            prog_bar,
            logger,
            on_step,
            on_epoch,
            reduce_fx,
            enable_graph,
            sync_dist,
            sync_dist_group,
            add_dataloader_idx,
            batch_size,
            metric_attribute,
            rank_zero_only,
        )

    def log_dict(
        self,
        dictionary: Mapping[str, _METRIC_COLLECTION],
        prog_bar: bool = False,
        logger: bool = True,
        on_step: Optional[bool] = False,  # from None
        on_epoch: Optional[bool] = True,  # from None
        reduce_fx: Union[str, Callable] = "mean",
        enable_graph: bool = False,
        sync_dist: bool = True,  # from False
        sync_dist_group: Optional[Any] = None,
        add_dataloader_idx: bool = True,
        batch_size: Optional[int] = None,
        rank_zero_only: bool = False,
    ) -> None:
        return super().log_dict(
            dictionary,
            prog_bar,
            logger,
            on_step,
            on_epoch,
            reduce_fx,
            enable_graph,
            sync_dist,
            sync_dist_group,
            add_dataloader_idx,
            batch_size,
            rank_zero_only,
        )

    # --- Other methods
    def get_example(self) -> Optional[dict[str, Any]]:
        return default_get_example(self, self.hparams.get("verbose", 1))

    def attach_example(self) -> None:
        self.example_input_array = self.get_example()

    def encode_text(self, *args, **kwargs) -> Any:
        assert isinstance(self.tokenizer, AACTokenizer)
        return self.tokenizer.encode_rec(*args, **kwargs)

    def tokenize_text(self, *args, **kwargs) -> Any:
        assert isinstance(self.tokenizer, AACTokenizer)
        return self.tokenizer.tokenize_rec(*args, **kwargs)

    def decode_text(self, *args, **kwargs) -> Any:
        assert isinstance(self.tokenizer, AACTokenizer)
        return self.tokenizer.decode_rec(*args, **kwargs)

    def detokenize_text(self, *args, **kwargs) -> Any:
        assert isinstance(self.tokenizer, AACTokenizer)
        return self.tokenizer.detokenize_rec(*args, **kwargs)

    def csum_module(self, only_trainable: bool = False) -> int:
        return csum_module(self, only_trainable=only_trainable)

    def count_params(self, only_trainable: bool = False) -> int:
        return count_params(self, only_trainable)

    def has_datamodule(self) -> bool:
        return has_datamodule(self)

    def has_trainer(self) -> bool:
        return has_trainer(self)

    def get_datamodule(self) -> LightningDataModule:
        assert (
            self.has_datamodule()
        ), f"PLM {self.__class__.__name__} does not have datamodule."
        return self._trainer.datamodule  # type: ignore

    def get_trainer(self) -> Trainer:
        assert (
            self.has_trainer()
        ), f"PLM {self.__class__.__name__} does not have trainer."
        return self._trainer  # type: ignore

    def get_tokenizer(self, id_: Union[int, str] = 0) -> AACTokenizer:
        if isinstance(id_, int):
            id_ = list(self.tokenizers.keys())[id_]
            return self.get_tokenizer(id_)
        else:
            return self.tokenizers[id_]  # type: ignore

    def get_tokenizers_count(self) -> int:
        return len(self.tokenizers)
