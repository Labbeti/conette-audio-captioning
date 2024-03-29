#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Callable, Optional, TypeVar, Union

import torch

from torch import nn, Tensor
from torchmetrics.functional.text.bert import bert_score, _DEFAULT_MODEL
from transformers.models.auto.modeling_auto import AutoModel
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers import logging as tfmers_logging

from aac_metrics.utils.collections import flat_list, unflat_list, duplicate_list


T = TypeVar("T")


def bert_score_mrefs(
    candidates: list[str],
    mult_references: list[list[str]],
    return_all_scores: bool = True,
    model: Union[str, nn.Module] = _DEFAULT_MODEL,
    tokenizer: Optional[Callable] = None,
    device: Union[str, torch.device, None] = "auto",
    batch_size: int = 32,
    num_threads: int = 0,
    max_length: int = 64,
    reset_state: bool = True,
    idf: bool = False,
    verbose: int = 0,
) -> Union[tuple[dict[str, Tensor], dict[str, Tensor]], Tensor]:
    """BERTScore metric which supports multiple references.

    The implementation is based on the bert_score implementation of torchmetrics.

    - Paper: https://arxiv.org/pdf/1904.09675.pdf

    :param candidates: The list of sentences to evaluate.
    :param mult_references: The list of list of sentences used as target.
    :param return_all_scores: If True, returns a tuple containing the globals and locals scores.
        Otherwise returns a scalar tensor containing the main global score.
        defaults to True.
    :param model: The model name or the instantiated model to use to compute token embeddings.
        defaults to "roberta-large".
    :param tokenizer: The fast tokenizer used to split sentences into words.
        If None, use the tokenizer corresponding to the model argument.
        defaults to None.
    :param device: The PyTorch device used to run the BERT model. defaults to "auto".
    :param batch_size: The batch size used in the model forward.
    :param num_threads: A number of threads to use for a dataloader. defaults to 0.
    :param max_length: Max length when encoding sentences to tensor ids. defaults to 64.
    :param idf: Whether or not using Inverse document frequency to ponderate the BERTScores. defaults to False.
    :param verbose: The verbose level. defaults to 0.
    :returns: A tuple of globals and locals scores or a scalar tensor with the main global score.
    """

    if isinstance(model, str):
        if tokenizer is not None:
            raise ValueError(
                f"Invalid argument combinaison {model=} with {tokenizer=}."
            )
        model, tokenizer = _load_model_and_tokenizer(
            model, tokenizer, device, reset_state, verbose
        )

    elif isinstance(model, nn.Module):
        if tokenizer is None:
            raise ValueError(
                f"Invalid argument combinaison {model=} with {tokenizer=}."
            )

    else:
        raise ValueError(
            f"Invalid argument type {type(model)=}. (expected str or nn.Module)"
        )

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if isinstance(device, str):
        device = torch.device(device)

    flat_mrefs, sizes = flat_list(mult_references)
    duplicated_cands = duplicate_list(candidates, sizes)

    tfmers_verbosity = tfmers_logging.get_verbosity()
    if verbose <= 1:
        tfmers_logging.set_verbosity_error()

    sents_scores = bert_score(
        duplicated_cands,
        flat_mrefs,
        model_name_or_path=None,
        model=model,  # type: ignore
        user_tokenizer=tokenizer,
        device=device,
        batch_size=batch_size,
        num_threads=num_threads,
        verbose=verbose >= 3,
        max_length=max_length,
        idf=idf,
    )
    if verbose <= 1:
        # Restore previous verbosity level
        tfmers_logging.set_verbosity(tfmers_verbosity)

    # sents_scores keys: "precision", "recall", "f1"
    sents_scores = {k: unflat_list(v, sizes) for k, v in sents_scores.items()}  # type: ignore

    if not return_all_scores:
        sents_scores = {"f1": sents_scores["f1"]}

    dtype = torch.float32
    if len(sizes) > 0 and all(size == sizes[0] for size in sizes):
        sents_scores = {
            k: torch.as_tensor(v, dtype=dtype).mean(dim=1)
            for k, v in sents_scores.items()
        }
    else:
        sents_scores = {
            k: torch.stack([torch.as_tensor(vi, dtype=dtype).mean() for vi in v])
            for k, v in sents_scores.items()
        }

    sents_scores = {f"bert_score.{k}": v for k, v in sents_scores.items()}
    sents_scores = {k: v.masked_fill(v.isnan(), 0.0) for k, v in sents_scores.items()}

    corpus_scores = {k: v.mean() for k, v in sents_scores.items()}

    if return_all_scores:
        return corpus_scores, sents_scores
    else:
        return corpus_scores["bert_score.f1"]


def _load_model_and_tokenizer(
    model: Union[str, nn.Module],
    tokenizer: Optional[Callable],
    device: Union[str, torch.device, None],
    reset_state: bool,
    verbose: int,
) -> tuple[nn.Module, Optional[Callable]]:
    state = torch.random.get_rng_state()

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if isinstance(device, str):
        device = torch.device(device)

    if isinstance(model, str):
        tfmers_verbosity = tfmers_logging.get_verbosity()
        if verbose <= 1:
            tfmers_logging.set_verbosity_error()

        # WARNING: tokenizer must be initialized BEFORE model to avoid connection errors
        tokenizer = AutoTokenizer.from_pretrained(model)
        model = AutoModel.from_pretrained(model)  # type: ignore

        if verbose <= 1:
            # Restore previous verbosity level
            tfmers_logging.set_verbosity(tfmers_verbosity)

    model.eval()  # type: ignore
    model.to(device=device)  # type: ignore

    if reset_state:
        torch.random.set_rng_state(state)

    return model, tokenizer  # type: ignore
