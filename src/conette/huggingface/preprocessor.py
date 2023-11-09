#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging

from typing import Any, Iterable, TypeGuard, Union

import torch
import torchaudio

from torch import Size, Tensor, nn
from torchaudio.functional import resample

from conette.nn.encoders.convnext import convnext_tiny
from conette.nn.functional.pad import pad_and_stack
from conette.utils.collections import all_eq, unzip


pylog = logging.getLogger(__name__)


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


def _is_iter_str(x: Any) -> TypeGuard[Iterable[str]]:
    return isinstance(x, Iterable) and all(isinstance(xi, str) for xi in x)


def _is_list_tensor(x: Any) -> TypeGuard[list[Tensor]]:
    return isinstance(x, list) and all(isinstance(xi, Tensor) for xi in x)


def _is_iter_tensor(x: Any) -> TypeGuard[Iterable[Tensor]]:
    return isinstance(x, Iterable) and all(isinstance(xi, Tensor) for xi in x)
