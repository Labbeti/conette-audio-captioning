#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging

from typing import Any, Mapping, Optional

import torch

from torch import nn, Tensor
from torch.nn import functional as F
from torch.nn.modules.module import _IncompatibleKeys
from torchlibrosa.stft import Spectrogram, LogmelFilterBank
from torchlibrosa.augmentation import SpecAugmentation

from conette.nn.pann_utils.models import (
    do_mixup,
    init_bn,
    init_layer,
    ConvBlock,
)


pylog = logging.getLogger(__name__)


class Cnn14(nn.Module):
    def __init__(
        self,
        sample_rate: int = 32000,
        window_size: int = 1024,
        hop_size: int = 320,
        mel_bins: int = 64,
        fmin: int = 50,
        fmax: int = 14000,
        classes_num: int = 527,
        waveform_input: bool = True,
        use_specaug: bool = True,
        return_clip_outputs: bool = True,
        return_frame_outputs: bool = False,
    ) -> None:
        super(Cnn14, self).__init__()
        self.waveform_input = waveform_input
        self.use_spec_aug = use_specaug
        self.return_clip_output = return_clip_outputs
        self.return_frame_output = return_frame_outputs

        window = "hann"
        center = True
        pad_mode = "reflect"
        ref = 1.0
        amin = 1e-10
        top_db = None

        if self.waveform_input:
            # Spectrogram extractor
            self.spectrogram_extractor = Spectrogram(
                n_fft=window_size,
                hop_length=hop_size,
                win_length=window_size,
                window=window,
                center=center,
                pad_mode=pad_mode,
                freeze_parameters=True,
            )
            # Logmel feature extractor
            self.logmel_extractor = LogmelFilterBank(
                sr=sample_rate,
                n_fft=window_size,
                n_mels=mel_bins,
                fmin=fmin,
                fmax=fmax,
                ref=ref,
                amin=amin,
                top_db=top_db,  # type: ignore
                freeze_parameters=True,
            )
        else:
            self.spectrogram_extractor = nn.Identity()
            self.logmel_extractor = nn.Identity()

        # Spec augmenter
        self.spec_augmenter = SpecAugmentation(
            time_drop_width=64,
            time_stripes_num=2,
            freq_drop_width=8,
            freq_stripes_num=2,
        )

        self.bn0 = nn.BatchNorm2d(64)

        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)
        self.conv_block5 = ConvBlock(in_channels=512, out_channels=1024)
        self.conv_block6 = ConvBlock(in_channels=1024, out_channels=2048)

        if self.return_clip_output:
            self.fc1 = nn.Linear(2048, 2048, bias=True)
            self.fc_audioset = nn.Linear(2048, classes_num, bias=True)
        else:
            self.fc1 = nn.Identity()
            self.fc_audioset = nn.Identity()

        self.init_weight()

    def init_weight(self) -> None:
        init_bn(self.bn0)
        if self.return_clip_output:
            init_layer(self.fc1)
            init_layer(self.fc_audioset)

    def load_state_dict(
        self,
        state_dict: Mapping[str, Any],
        strict: bool = True,
    ) -> _IncompatibleKeys:
        exclude_keys = []
        if not self.waveform_input:
            exclude_keys += [
                "spectrogram_extractor.stft.conv_real.weight",
                "spectrogram_extractor.stft.conv_imag.weight",
                "logmel_extractor.melW",
            ]
        if not self.return_clip_output:
            exclude_keys += [
                "fc1.weight",
                "fc1.bias",
                "fc_audioset.weight",
                "fc_audioset.bias",
            ]

        if len(exclude_keys) > 0:
            pylog.warning(f"Auto-exclude keys {tuple(exclude_keys)}.")

        state_dict = dict(state_dict)
        for key in exclude_keys:
            state_dict.pop(key, None)

        return super().load_state_dict(state_dict, strict)  # type: ignore

    def forward(
        self,
        input_: Tensor,
        input_shapes: Tensor,
        mixup_lambda: Optional[Tensor] = None,
    ) -> dict[str, Tensor]:
        """
        Input: (batch_size, data_length) if waveform_input=True else (batch_size, 1, time_steps, mel_bins)
        """

        if self.waveform_input:
            input_time_dim = -1
            x = self.spectrogram_extractor(
                input_
            )  # (batch_size, 1, time_steps, freq_bins)
            x = self.logmel_extractor(x)  # (batch_size, 1, time_steps, mel_bins)
        else:
            input_time_dim = -2
            x = input_

        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)

        if self.training and self.use_spec_aug:
            x = self.spec_augmenter(x)

        # Mixup on spectrogram
        if self.training and mixup_lambda is not None:
            x = do_mixup(x, mixup_lambda)

        x = self.conv_block1(x, pool_size=(2, 2), pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(2, 2), pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(2, 2), pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=(2, 2), pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block5(x, pool_size=(2, 2), pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block6(x, pool_size=(1, 1), pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)
        x = torch.mean(x, dim=3)

        output_dict = {}

        if self.return_frame_output:
            frame_embs = x

            # input_: (bsize, n_channels=1, time_steps=1001, mel_bins=64)
            # x: (bsize, emb_size=2048, time_steps=31)

            input_lens = input_shapes[:, input_time_dim]
            reduction_factor = input_.shape[input_time_dim] // frame_embs.shape[-1]
            frame_embs_lens = input_lens.div(reduction_factor, rounding_mode="trunc")

            output_dict |= {
                # (bsize, embed=2048, n_frames=31)
                "frame_embs": frame_embs,
                # (bsize,)
                "frame_embs_lens": frame_embs_lens,
            }

        if self.return_clip_output:
            (x1, _) = torch.max(x, dim=2)
            x2 = torch.mean(x, dim=2)
            x = x1 + x2
            x = F.dropout(x, p=0.5, training=self.training)
            x = F.relu_(self.fc1(x))
            embedding = F.dropout(x, p=0.5, training=self.training)
            clipwise_output = torch.sigmoid(self.fc_audioset(x))

            output_dict |= {"clipwise_output": clipwise_output, "embedding": embedding}

        return output_dict
