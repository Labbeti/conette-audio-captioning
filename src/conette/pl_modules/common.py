#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Common functions and arguments used in my PyTorch Lightning Modules."""

import logging

from typing import Any, Iterator, Mapping, Optional, TypedDict, Union

import torch
import yaml

from nltk.corpus import stopwords
from pytorch_lightning import LightningModule
from torch import nn, Tensor
from torch.optim import Optimizer

from conette.nn.modules.tensor import Transpose
from conette.nn.functional.get import get_activation_module, get_device
from conette.optim.optimizers import get_optimizer
from conette.optim.schedulers import get_scheduler_list
from conette.tokenization.aac_tokenizer import AACTokenizer


pylog = logging.getLogger(__name__)

ON_EPOCH_KWARGS = {
    "on_step": False,
    "on_epoch": True,
    "sync_dist": True,
}


class TrainBatch(TypedDict):
    audio: Tensor
    audio_shape: Tensor
    captions: Tensor
    dataset: str
    source: Optional[str]


class ValBatch(TypedDict):
    audio: Tensor
    audio_shape: Tensor
    mult_captions: Tensor
    mult_references: list[list[str]]
    dataset: str
    source: Optional[str]


class TestBatch(TypedDict):
    audio: Tensor
    audio_shape: Tensor
    mult_captions: Tensor
    mult_references: list[list[str]]
    dataset: str
    subset: str
    fname: str
    source: Optional[str]


def count_params(model: nn.Module, only_trainable: bool = False) -> int:
    return sum(
        param.numel()
        for param in model.parameters()
        if not only_trainable or param.requires_grad
    )


def build_proj_lin(
    in_features: int,
    out_features: int,
    transpose_start: bool,
    dropout_p: float = 0.5,
    activation: str = "relu",
) -> nn.Module:
    """Build default projection used between encoder and decoder.

    Note: Dropout + Transpose + Linear + ReLU + Transpose + Dropout
    """
    # transpose (bsize, seq_size, emb_size) -> (bsize, emb_size, seq_size) after linear
    return nn.Sequential(
        nn.Dropout(p=dropout_p),
        Transpose(1, 2) if transpose_start else nn.Identity(),
        nn.Linear(in_features, out_features),
        get_activation_module(activation),
        Transpose(1, 2),
        nn.Dropout(p=dropout_p),
    )


def build_proj_mha_lin(
    in_features: int,
    out_features: int,
    transpose_start: bool = False,
    dropout_p: float = 0.5,
    nhead: int = 8,
) -> nn.Module:
    """Build default projection used between encoder and decoder.

    Note: Dropout + Transpose + Linear + ReLU + Transpose + Dropout
    """

    # transpose (bsize, seq_size, emb_size) -> (bsize, emb_size, seq_size) after linear
    return nn.Sequential(
        nn.Dropout(p=dropout_p),
        Transpose(1, 2) if transpose_start else nn.Identity(),
        nn.MultiheadAttention(in_features, nhead, dropout_p, batch_first=True),
        nn.GELU(),
        nn.Linear(in_features, out_features),
        nn.GELU(),
        Transpose(1, 2),
        nn.Dropout(p=dropout_p),
    )


def default_get_example(
    plm: LightningModule, verbose: int = 2
) -> Optional[dict[str, Any]]:
    """Default implementation to get example from train_dataloader to PLM."""
    if not has_datamodule(plm):
        return None

    train_dataloader = plm.trainer.datamodule.train_dataloader()  # type: ignore
    batch = next(iter(train_dataloader))

    if not isinstance(batch, Mapping):
        pylog.warning(
            f"Cannot attach example automatically. (found invalid batch type {batch.__class__.__name__})"
        )
        return None

    if verbose >= 2:
        shapes = {
            k: str(tuple(v.shape)) for k, v in batch.items() if isinstance(v, Tensor)
        }
        shapes = yaml.dump(shapes, sort_keys=False)
        shapes = str(shapes).strip().split("\n")

        pylog.debug(f"Batch keys: {tuple(batch.keys())}")
        if len(shapes) > 0:
            pylog.debug("Batch shapes:")
            for shape in shapes:
                pylog.debug(shape)

    audio: Tensor = batch["audio"]
    audio_shape: Tensor = batch["audio_shape"]
    captions: Tensor = batch["captions"]

    assert isinstance(audio, Tensor), f"{audio.__class__.__name__=}"
    assert isinstance(audio_shape, Tensor), f"{audio_shape.__class__.__name__=}"
    assert isinstance(captions, Tensor), f"{captions.__class__.__name__=}"

    example_input_array = dict(
        batch=batch,
        decode_method="forcing",
    )
    return example_input_array


def default_configure_optimizers(
    plm: Any,
    parameters: Optional[Iterator[Tensor]] = None,
) -> tuple[list[Optimizer], list[dict[str, Any]]]:
    """Default implementation to configure PLM optimizers."""
    hp = plm.hparams

    if parameters is None:
        parameters = plm
    elif hp.use_custom_wd:
        raise ValueError(
            f"Invalid arguments {hp.use_custom_wd=} with custom parameters list."
        )

    optimizer = get_optimizer(
        hp.optim_name,
        parameters,  # type: ignore
        lr=hp.lr,
        weight_decay=hp.weight_decay,
        betas=hp.betas,
        eps=hp.eps,
        use_custom_wd=hp.use_custom_wd,
    )

    if hp.sched_n_steps is not None:
        sched_n_steps = hp.sched_n_steps
    elif has_trainer(plm) and isinstance(plm.trainer.max_epochs, int):
        sched_n_steps = plm.trainer.max_epochs
    else:
        raise RuntimeError(
            f"Cannot get param 'sched_n_steps' from Trainer. ({hp.sched_n_steps=}, {plm.trainer=})"
        )

    scheduler_list = get_scheduler_list(
        hp.sched_name,
        optimizer,
        interval=hp.sched_interval,
        frequency=hp.sched_freq,
        sched_n_steps=sched_n_steps,
    )
    return [optimizer], scheduler_list


def default_load_state_dict(
    plm: LightningModule,
    state_dict: Mapping[str, Any],
    strict: bool = True,
) -> Any:
    def replace_key(key: str) -> str:
        return key.replace("model.encoder", "encoder").replace(
            "model.decoder", "decoder"
        )

    if all(hasattr(plm, name) for name in ("is_built", "tokenizer", "build_model")):
        ftok_key = "train_tokenizer._extra_state"
        if ftok_key in state_dict and not plm.is_built():  # type: ignore
            plm.tokenizer.load_state_dict(state_dict[ftok_key], strict)  # type: ignore
            plm.build_model()  # type: ignore

    state_dict = {replace_key(key): value for key, value in state_dict.items()}
    keys = LightningModule.load_state_dict(plm, state_dict, strict)
    return keys


def has_trainer(plm: LightningModule) -> bool:
    return plm._trainer is not None


def has_datamodule(plm: LightningModule) -> bool:
    return plm._trainer is not None and plm._trainer.datamodule is not None  # type: ignore


def get_forbid_rep_mask(
    forbid_rep_mode: str,
    tokenizer: AACTokenizer,
    device: Union[str, torch.device, None],
    verbose: int = 0,
    lang: str = "english",
) -> Optional[Tensor]:
    device = get_device(device)

    if forbid_rep_mode == "none":
        forbid_mask = None

    elif forbid_rep_mode == "all":
        forbid_mask = torch.full(
            (tokenizer.get_vocab_size(),), True, dtype=torch.bool, device=device
        )

    elif forbid_rep_mode == "content_words":
        if tokenizer.get_level() != "word":
            pylog.warning(
                f"Forbid same token {forbid_rep_mode} with tokenizer level {tokenizer.get_level()} is not officially supported."
            )
        forbid_mask = _get_forbid_rep_mask_content_words(
            tokenizer, device, verbose, lang
        )

    else:
        FORBID_MODES = (
            "none",
            "content_words",
            "content_words_not_bpe_part",
            "content_words_stem_lemm",
        )
        raise ValueError(
            f"Invalid argument {forbid_rep_mode=}. (expected one of {FORBID_MODES})"
        )

    return forbid_mask


def _get_forbid_rep_mask_content_words(
    tokenizer: AACTokenizer,
    device: Union[torch.device, None],
    verbose: int = 0,
    lang: str = "english",
) -> Tensor:
    forbid_mask = torch.ones(
        (tokenizer.get_vocab_size(),), dtype=torch.bool, device=device
    )
    stopwords_set = set(stopwords.words(lang))
    stopwords_in_vocab = {word for word in stopwords_set if tokenizer.has(word)}

    for token in stopwords_in_vocab:
        idx = tokenizer.token_to_id(token)
        forbid_mask[idx] = False

    if verbose >= 2:
        pylog.debug(
            f"{len(stopwords_in_vocab)}/{len(stopwords_set)} stopwords found in vocab:"
        )
        pylog.debug(f"{stopwords_in_vocab}")

        stopwords_not_in_vocab = {
            word for word in stopwords_set if not tokenizer.has(word)
        }
        pylog.debug(
            f"{len(stopwords_not_in_vocab)}/{len(stopwords_set)} stopwords NOT found in vocab:"
        )
        pylog.debug(f"{stopwords_not_in_vocab}")

    if verbose >= 1:
        pylog.info(f"Found {len(stopwords_in_vocab)}/{len(stopwords_set)} stopwords.")

    if verbose >= 1:
        pylog.info(
            f"Forbid mask up to {forbid_mask.sum().item()}/{tokenizer.get_vocab_size()} tokens during testing."
        )

    return forbid_mask
