#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import os
import subprocess
import sys

from pathlib import Path
from subprocess import CalledProcessError
from typing import Any, Iterable, Optional, TypeGuard, Union

import pickle
import torch
import torchaudio

from torch import Size, Tensor, nn
from torchaudio.functional import resample
from transformers import PretrainedConfig, PreTrainedModel

from conette.nn.encoders.convnext import convnext_tiny
from conette.nn.functional.get import get_device
from conette.nn.functional.pad import pad_and_stack
from conette.pl_modules.conette import CoNeTTEPLM
from conette.tokenization.aac_tokenizer import AACTokenizer
from conette.utils.collections import unzip, all_eq


pylog = logging.getLogger(__name__)


def _is_iter_str(x: Any) -> TypeGuard[Iterable[str]]:
    return isinstance(x, Iterable) and all(isinstance(xi, str) for xi in x)


def _is_list_tensor(x: Any) -> TypeGuard[list[Tensor]]:
    return isinstance(x, list) and all(isinstance(xi, Tensor) for xi in x)


def _is_iter_tensor(x: Any) -> TypeGuard[Iterable[Tensor]]:
    return isinstance(x, Iterable) and all(isinstance(xi, Tensor) for xi in x)


class CoNeTTEPreprocessor(nn.Module):
    def __init__(self, verbose: int = 0) -> None:
        encoder = convnext_tiny(
            pretrained=False,
            strict=False,
            drop_path_rate=0.0,
            after_stem_dim=[252, 56],
            use_speed_perturb=False,
            waveform_input=True,
            use_specaug=False,
            return_clip_outputs=True,
            return_frame_outputs=True,
        )
        super().__init__()
        self.encoder = encoder
        self.verbose = verbose

    @property
    def device(self) -> torch.device:
        return next(iter(self.parameters())).device

    @property
    def target_sr(self) -> int:
        return 32_000  # Hz

    @property
    def feat_size(self) -> int:
        return 768

    def forward(
        self,
        x: Union[Tensor, str, Iterable[str], Iterable[Tensor]],
        sr: Union[None, int, Iterable[int]] = None,
        x_shapes: Union[Tensor, None, list[Size]] = None,
    ) -> dict[str, Any]:
        x, x_shapes = self._load_resample(x, sr, x_shapes)
        outs = self.encoder(x, x_shapes)
        # outs["frame_embs"]: (bsize, feat_size, n_frames=31)
        # outs["frame_embs_lens"]: (bsize,)

        frame_embs = outs["frame_embs"]
        frame_embs_lens = outs["frame_embs_lens"]

        # Transpose (bsize, feat_size, time) -> (bsize, time, features=768)
        frame_embs = frame_embs.transpose(1, 2)
        audio_shape = torch.as_tensor(
            [[self.feat_size, len_i] for len_i in frame_embs_lens], device=self.device
        )
        del frame_embs_lens

        batch = {"audio": frame_embs, "audio_shape": audio_shape}
        return batch

    def _load(self, path: str) -> tuple[Tensor, int]:
        return torchaudio.load(path)  # type: ignore

    def _load_resample(
        self,
        x: Union[Tensor, str, Iterable[str], Iterable[Tensor]],
        sr: Union[None, int, Iterable[int]] = None,
        x_shapes: Union[Tensor, None, list[Size]] = None,
    ) -> tuple[Tensor, Tensor]:
        # LOAD
        if _is_iter_str(x):
            if isinstance(x, str):
                x = [x]
            gen = (self._load(xi) for xi in x)
            x, sr = unzip(gen)

        else:
            if isinstance(x, Tensor):
                # expected (n_time,), (n_channel, n_time) or (bsize, n_channels, n_time)
                if x.ndim == 1:
                    x = x.unsqueeze(dim=0).unsqueeze(dim=1)
                elif x.ndim == 2:
                    x = x.unsqueeze(dim=0)
                elif x.ndim == 3:
                    pass
                else:
                    raise ValueError(f"Invalid argument shape {x.shape=}.")
            else:
                x = list(x)  # type: ignore

            if isinstance(sr, int):
                sr = [sr]
            elif sr is None:
                sr = [self.target_sr]
            else:
                sr = list(sr)

        assert _is_list_tensor(x) or isinstance(x, Tensor), f"{type(x)=}"

        if len(sr) == 1 and len(x) != len(sr):
            sr = sr * len(x)

        if self.verbose >= 2:
            pylog.debug(f"Found {sr=}.")

        assert len(x) == len(sr) and len(x) > 0
        assert _is_iter_tensor(x) or isinstance(x, Tensor)

        # MOVE TO DEVICE
        if isinstance(x, Tensor):
            x = x.to(device=self.device)
        elif _is_iter_tensor(x):
            x = [xi.to(device=self.device) for xi in x]

        # RESAMPLE + MEAN
        if any(sri != self.target_sr for sri in sr):
            if x_shapes is not None:
                raise ValueError(f"Invalid argument {x_shapes=}.")

            if all_eq(sr) and isinstance(x, Tensor):
                x = resample(x, sr[0], self.target_sr)
            else:
                x = [resample(xi, sri, self.target_sr) for xi, sri in zip(x, sr)]

        if isinstance(x, Tensor):
            x = x.mean(dim=1)
        else:
            x = [xi.mean(dim=0) for xi in x]

        # SHAPES + STACK
        if x_shapes is None:
            x_shapes = [xi.shape for xi in x]
        x_shapes = torch.as_tensor(x_shapes, device=self.device)
        x = pad_and_stack(x)

        return x, x_shapes


class CoNeTTEConfig(PretrainedConfig):
    def __init__(
        self,
        task_mode: str = "ds_src",
        task_names: Iterable[str] = (
            "clotho",
            "audiocaps",
            "macs",
            "wavcaps_audioset_sl",
            "wavcaps_bbc_sound_effects",
            "wavcaps_freesound",
            "wavcaps_soundbible",
        ),
        gen_test_cands: str = "generate",
        label_smoothing: float = 0.2,
        gen_val_cands: str = "generate",
        mixup_alpha: float = 0.4,
        proj_name: str = "lin768",
        min_pred_size: int = 3,
        max_pred_size: int = 20,
        beam_size: int = 3,
        nhead: int = 8,
        d_model: int = 256,
        num_decoder_layers: int = 6,
        decoder_dropout_p: float = 0.2,
        dim_feedforward: int = 2048,
        acti_name: str = "gelu",
        optim_name: str = "AdamW",
        lr: float = 5e-4,
        weight_decay: float = 2.0,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        use_custom_wd: bool = True,
        sched_name: str = "cos_decay",
        sched_n_steps: int = 400,
        sched_interval: str = "epoch",
        sched_freq: int = 1,
        verbose: int = 0,
        tokenizer_state: Optional[dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        betas = list(betas)  # type: ignore
        super().__init__()
        self.task_mode = task_mode
        self.task_names = task_names
        self.gen_test_cands = gen_test_cands
        self.label_smoothing = label_smoothing
        self.gen_val_cands = gen_val_cands
        self.mixup_alpha = mixup_alpha
        self.proj_name = proj_name
        self.min_pred_size = min_pred_size
        self.max_pred_size = max_pred_size
        self.beam_size = beam_size
        self.nhead = nhead
        self.d_model = d_model
        self.num_decoder_layers = num_decoder_layers
        self.decoder_dropout_p = decoder_dropout_p
        self.dim_feedforward = dim_feedforward
        self.acti_name = acti_name
        self.optim_name = optim_name
        self.lr = lr
        self.weight_decay = weight_decay
        self.betas = betas
        self.eps = eps
        self.use_custom_wd = use_custom_wd
        self.sched_name = sched_name
        self.sched_n_steps = sched_n_steps
        self.sched_interval = sched_interval
        self.sched_freq = sched_freq
        self.verbose = verbose
        self.tokenizer_state = tokenizer_state


class CoNeTTEModel(PreTrainedModel):
    def __init__(
        self,
        config: CoNeTTEConfig,
        device: Union[str, torch.device, None] = "auto",
        inference: bool = True,
    ) -> None:
        setup()

        if config.tokenizer_state is None:
            tokenizer = AACTokenizer()
        else:
            tokenizer = AACTokenizer.from_txt_state(config.tokenizer_state)

        preprocessor = CoNeTTEPreprocessor(verbose=config.verbose)
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
        self.to(device=device)

        if inference:
            self.eval_and_detach()

    @property
    def default_task(self) -> str:
        return next(iter(self.config.task_names))

    @property
    def tasks(self) -> list[str]:
        return list(self.config.task_names)

    def eval_and_detach(self) -> None:
        self.eval()
        for p in self.parameters():
            p.requires_grad_(False)

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
            non_tensor_states = pickle.dumps(non_tensor_states)
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
        # Beam search options
        task: Union[str, list[str], None] = None,
        beam_size: Optional[int] = None,
        min_pred_size: Optional[int] = None,
        max_pred_size: Optional[int] = None,
    ) -> dict[str, Any]:
        # Preprocessing (load data + encode features)
        if preprocess:
            batch = self.preprocessor(x, sr, x_shapes)
        else:
            assert isinstance(x, Tensor) and isinstance(x_shapes, Tensor)
            batch: dict[str, Any] = {
                "audio": x.to(self.device),
                "audio_shape": x_shapes.to(self.device),
            }

        # Add task information to batch
        bsize = len(batch["audio"])
        if task is None:
            task = [self.default_task] * bsize
        elif isinstance(task, str):
            task = [task] * bsize
        elif len(task) != bsize:
            raise ValueError(
                f"Invalid number of tasks with input. (found {len(task)} tasks but {bsize} elements)"
            )

        for task_i in task:
            if task_i not in self.config.task_names:
                raise ValueError(
                    f"Invalid argument {task=}. (task {task_i} is not in {self.config.task_names})"
                )

        dataset_lst = [self.default_task] * bsize
        source_lst: list[Optional[str]] = [None] * bsize

        for i, task_i in enumerate(task):
            if task is None:
                task_i = self.default_task
            task_i = task_i.split("_")
            dataset_lst[i] = task_i[0]
            if len(task_i) == 2:
                source_lst[i] = task_i[1]

        batch["dataset"] = dataset_lst
        batch["source"] = source_lst

        # Call model forward
        kwds = dict(
            beam_size=beam_size,
            min_pred_size=min_pred_size,
            max_pred_size=max_pred_size,
        )
        kwds = {k: v for k, v in kwds.items() if v is not None}
        outs = self.model(batch, **kwds)

        return outs

    def __call__(
        self,
        # Inputs
        x: Union[Tensor, str, Iterable[str], Iterable[Tensor]],
        sr: Union[None, int, Iterable[int]] = None,
        x_shapes: Union[Tensor, None, list[Size]] = None,
        # Beam search options
        task: Union[str, list[str], None] = None,
        beam_size: Optional[int] = None,
        min_pred_size: Optional[int] = None,
        max_pred_size: Optional[int] = None,
    ) -> dict[str, Any]:
        return super().__call__(
            x=x,
            sr=sr,
            x_shapes=x_shapes,
            task=task,
            beam_size=beam_size,
            min_pred_size=min_pred_size,
            max_pred_size=max_pred_size,
        )


def conette(
    pretrained_model_name_or_path: str = "Labbeti/conette",
    **kwargs,
) -> CoNeTTEModel:
    """Create pretrained CoNeTTEModel."""
    config = CoNeTTEConfig.from_pretrained(
        pretrained_model_name_or_path,
        **kwargs,
    )
    model = CoNeTTEModel.from_pretrained(
        pretrained_model_name_or_path,
        config=config,
        **kwargs,
    )
    return model  # type: ignore


def setup(offline: bool = False, verbose: int = 0) -> None:
    if offline:
        return None

    # Download spaCy model for AACTokenizer
    for model_name in ("en_core_web_sm",):
        command = f"{sys.executable} -m spacy download {model_name}".split(" ")
        try:
            subprocess.check_call(
                command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
            if verbose >= 1:
                pylog.info(f"Model '{model_name}' for spacy downloaded.")
        except (CalledProcessError, PermissionError) as err:  # type: ignore
            pylog.error(
                f"Cannot download spaCy model '{model_name}' for tokenizer. (command '{command}' with error={err})"
            )


def get_sample_path() -> str:
    path = Path(__file__).parent.parent.parent.parent.joinpath("data", "sample.wav")
    return str(path)
