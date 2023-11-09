#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Any, Iterable, Optional

from transformers import PretrainedConfig


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
