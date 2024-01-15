#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging

from typing import Any, Iterable, Optional, TypedDict, Union
from typing_extensions import NotRequired

import pickle
import torch

from torch import Size, Tensor
from transformers import PreTrainedModel

from conette.huggingface.config import CoNeTTEConfig
from conette.huggingface.preprocessor import CoNeTTEPreprocessor
from conette.huggingface.setup import setup_other_models
from conette.nn.functional.get import get_device
from conette.pl_modules.base import AACLightningModule
from conette.pl_modules.conette import CoNeTTEPLM
from conette.tokenization.aac_tokenizer import AACTokenizer
from conette.transforms.audioset_labels import probs_to_labels


pylog = logging.getLogger(__name__)


class CoNeTTEOutput(TypedDict):
    cands: list[str]
    preds: Tensor
    lprobs: Tensor
    mult_cands: list[list[str]]
    mult_preds: Tensor
    mult_lprobs: Tensor
    tasks: list[str]
    tags: NotRequired[list[list[str]]]
    tags_probs: NotRequired[Tensor]


class CoNeTTEModel(PreTrainedModel):
    """CoNeTTE PreTrainedModel for inference."""

    def __init__(
        self,
        config: CoNeTTEConfig,
        device: Union[str, torch.device, None] = "auto",
        inference: bool = True,
        offline: bool = False,
        model_override: Optional[AACLightningModule] = None,
    ) -> None:
        setup_other_models(offline)

        preprocessor = CoNeTTEPreprocessor(verbose=config.verbose)
        if model_override is not None:
            model = model_override
        else:
            if config.tokenizer_state is None:
                tokenizer = AACTokenizer()
            else:
                tokenizer = AACTokenizer.from_txt_state(config.tokenizer_state)

            model = CoNeTTEPLM(
                task_mode=config.task_mode,
                task_names=config.task_names,
                gen_test_cands=config.gen_test_cands,
                label_smoothing=config.label_smoothing,
                gen_val_cands=config.gen_val_cands,
                mixup_alpha=config.mixup_alpha,
                proj_name=config.proj_name,
                min_pred_size=config.min_pred_size,
                max_pred_size=config.max_pred_size,
                beam_size=config.beam_size,
                nhead=config.nhead,
                d_model=config.d_model,
                num_decoder_layers=config.num_decoder_layers,
                decoder_dropout_p=config.decoder_dropout_p,
                dim_feedforward=config.dim_feedforward,
                acti_name=config.acti_name,
                optim_name=config.optim_name,
                lr=config.lr,
                weight_decay=config.weight_decay,
                betas=config.betas,
                eps=config.eps,
                use_custom_wd=config.use_custom_wd,
                sched_name=config.sched_name,
                sched_n_steps=config.sched_n_steps,
                sched_interval=config.sched_interval,
                sched_freq=config.sched_freq,
                train_tokenizer=tokenizer,
                verbose=config.verbose,
            )

        super().__init__(config)
        self.config: CoNeTTEConfig
        self.preprocessor = preprocessor
        self.model = model

        self._register_load_state_dict_pre_hook(self._pre_hook_load_state_dict)

        device = get_device(device)
        self.to(device=device)  # type: ignore

        if inference:
            self.eval_and_disable_grad()

    @property
    def default_task(self) -> str:
        return next(iter(self.config.task_names))

    @property
    def tasks(self) -> list[str]:
        return list(self.config.task_names)

    def train_and_enable_grad(self, mode: bool = True) -> "CoNeTTEModel":
        self.train(mode)
        for p in self.parameters():
            p.requires_grad_(mode)
        return self

    def eval_and_disable_grad(self, mode: bool = True) -> "CoNeTTEModel":
        return self.train_and_enable_grad(not mode)

    def _pre_hook_load_state_dict(
        self,
        state_dict: dict[str, Tensor],
        prefix,
        local_metadata,
        strict,
        *args,
        **kwargs,
    ) -> Any:
        if "_extra_state_" in state_dict:
            extra_state = state_dict.pop("_extra_state_")
            extra_state = bytes(extra_state.tolist())
            extra_state = pickle.loads(extra_state)
            state_dict |= extra_state

        if not self.model.is_built():
            all_fit = True
            for name, tokenizer in self.model.tokenizers.items():
                if not isinstance(tokenizer, AACTokenizer):
                    continue

                tok_prefix = f"model.tokenizers.{name}."
                tok_data = {
                    k: v for k, v in state_dict.items() if k.startswith(tok_prefix)
                }
                if len(tok_data) > 0:
                    tok_data = {k[len(tok_prefix) :]: v for k, v in tok_data.items()}
                    tokenizer.load_state_dict(tok_data, strict)

                all_fit &= tokenizer.is_fit()

            if all_fit:
                self.model.build_model()
                self.model = self.model.to(device=self.device)
            else:
                pylog.error(
                    "Cannot build the model from state_dict. (tokenizer is not fit)"
                )

    def state_dict(self) -> dict[str, Tensor]:
        states = super().state_dict()
        tensor_states = {k: v for k, v in states.items() if isinstance(v, Tensor)}
        non_tensor_states = {
            k: v for k, v in states.items() if not isinstance(v, Tensor)
        }
        del states

        if len(non_tensor_states) > 0:
            pylog.debug(
                f"Storing into bytes values {tuple(non_tensor_states.keys())}..."
            )
            non_tensor_states = bytearray(pickle.dumps(non_tensor_states))
            non_tensor_states = torch.frombuffer(non_tensor_states, dtype=torch.uint8)
            tensor_states["_extra_state_"] = non_tensor_states

        # Enforce contiguous tensors
        tensor_states = {k: v.contiguous() for k, v in tensor_states.items()}
        return tensor_states

    def forward(
        self,
        # Inputs
        x: Union[Tensor, str, Iterable[str], Iterable[Tensor]],
        sr: Union[None, int, Iterable[int]] = None,
        x_shapes: Union[Tensor, None, list[Size]] = None,
        preprocess: bool = True,
        threshold: Union[float, Tensor] = 0.3,
        # Beam search options
        task: Union[str, list[str], None] = None,
        beam_size: Optional[int] = None,
        min_pred_size: Optional[int] = None,
        max_pred_size: Optional[int] = None,
        forbid_rep_mode: Optional[str] = None,
    ) -> CoNeTTEOutput:
        # Preprocessing (load data + encode features)
        if preprocess:
            batch = self.preprocessor(x, sr, x_shapes)
            clip_probs = batch.pop("clip_probs")
            tags = probs_to_labels(clip_probs, threshold, True, self.config.verbose)
        else:
            assert isinstance(x, Tensor) and isinstance(x_shapes, Tensor)
            batch: dict[str, Any] = {
                "audio": x.to(self.device),
                "audio_shape": x_shapes.to(self.device),
            }
            clip_probs = None
            tags = None

        # Add task information to batch
        bsize = len(batch["audio"])
        if task is None:
            tasks = [self.default_task] * bsize
        elif isinstance(task, str):
            tasks = [task] * bsize
        elif len(task) != bsize:
            raise ValueError(
                f"Invalid number of tasks with input. (found {len(task)} tasks but {bsize} elements)"
            )
        else:
            tasks = task
        del task

        for task in tasks:
            if task not in self.config.task_names:
                raise ValueError(
                    f"Invalid argument {tasks=}. (task {task} is not in {self.config.task_names})"
                )

        dataset_lst = [self.default_task] * bsize
        source_lst: list[Optional[str]] = [None] * bsize

        for i, task in enumerate(tasks):
            task = task.split("_")
            dataset_lst[i] = task[0]
            if len(task) >= 2:
                source_lst[i] = "_".join(task[1:])

        batch["dataset"] = dataset_lst
        batch["source"] = source_lst

        # Call model forward
        kwds = dict(
            beam_size=beam_size,
            min_pred_size=min_pred_size,
            max_pred_size=max_pred_size,
            forbid_rep_mode=forbid_rep_mode,
        )
        kwds = {k: v for k, v in kwds.items() if v is not None}
        outs = self.model(batch, **kwds)
        outs["tasks"] = tasks

        if clip_probs is not None and tags is not None:
            outs["tags_probs"] = clip_probs
            outs["tags"] = tags

        return outs

    def __call__(
        self,
        # Inputs
        x: Union[Tensor, str, Iterable[str], Iterable[Tensor]],
        sr: Union[None, int, Iterable[int]] = None,
        x_shapes: Union[Tensor, None, list[Size]] = None,
        preprocess: bool = True,
        threshold: Union[float, Tensor] = 0.3,
        # Beam search options
        task: Union[str, list[str], None] = None,
        beam_size: Optional[int] = None,
        min_pred_size: Optional[int] = None,
        max_pred_size: Optional[int] = None,
        forbid_rep_mode: Optional[str] = None,
    ) -> CoNeTTEOutput:
        return super().__call__(
            x=x,
            sr=sr,
            x_shapes=x_shapes,
            preprocess=preprocess,
            threshold=threshold,
            task=task,
            beam_size=beam_size,
            min_pred_size=min_pred_size,
            max_pred_size=max_pred_size,
            forbid_rep_mode=forbid_rep_mode,
        )
