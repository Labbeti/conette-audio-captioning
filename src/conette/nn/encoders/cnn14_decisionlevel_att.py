#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Optional

import torch

from torch import nn, Tensor
from torch.nn import functional as F
from torchlibrosa.stft import Spectrogram, LogmelFilterBank
from torchlibrosa.augmentation import SpecAugmentation

from conette.nn.pann_utils.pytorch_utils import (
    do_mixup,
    interpolate,
    pad_framewise_output,
)
from conette.nn.pann_utils.models import AttBlock, ConvBlock, init_bn, init_layer
from conette.nn.pann_utils.ckpt import pann_load_state_dict
from conette.transforms.audio.cutoutspec import CutOutSpec
from conette.transforms.mixup import Mixup, sample_lambda


class Cnn14_DecisionLevelAtt(nn.Module):
    def __init__(
        self,
        sr: int = 32000,
        window_size: int = 1024,
        hop_size: int = 320,
        mel_bins: int = 64,
        fmin: int = 50,
        fmax: int = 14000,
        classes_num: int = 527,
        use_cutout: bool = False,
        use_pann_mixup: bool = False,
        use_spec_augment: bool = False,
        return_clip_outputs: bool = False,
        pretrained: bool = True,
        freeze_weight: str = "none",
        waveform_input: bool = True,
    ) -> None:
        if freeze_weight != "none" and pretrained:
            raise RuntimeError(
                f"Cannot freeze weights without using pre-trained weights. (found {freeze_weight=} && {pretrained=})"
            )

        super().__init__()
        self.use_cutout = use_cutout
        self.use_pann_mixup = use_pann_mixup
        self.use_specaug = use_spec_augment
        self.return_clip_outputs = return_clip_outputs
        self.waveform_input = waveform_input

        window = "hann"
        center = True
        pad_mode = "reflect"
        ref = 1.0
        amin = 1e-10
        top_db = None
        self.interpolate_ratio = 32  # Downsampled ratio

        if waveform_input:
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
                sr=sr,
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

        self.cutout = CutOutSpec(fill_value=float(fmin))
        self.mixup = Mixup(alpha=0.4, asymmetric=True)

        self.bn0 = nn.BatchNorm2d(64)

        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)
        self.conv_block5 = ConvBlock(in_channels=512, out_channels=1024)
        self.conv_block6 = ConvBlock(in_channels=1024, out_channels=2048)

        self.fc1 = nn.Linear(2048, 2048, bias=True)
        if self.return_clip_outputs:
            self.att_block = AttBlock(2048, classes_num, activation="sigmoid")
        else:
            self.att_block = None

        # Initialize weights
        self.init_weight()

        if pretrained:
            self.load_pretrained_weights()

        self.freeze_weights(freeze_weight)

    def load_pretrained_weights(self, strict: bool = False) -> None:
        device = self.fc1.weight.device
        state_dict = pann_load_state_dict("Cnn14_DecisionLevelAtt", device, True)
        self.load_state_dict(state_dict, strict=strict)

    def init_weight(self) -> None:
        init_bn(self.bn0)
        init_layer(self.fc1)

    def freeze_weights(self, freeze_mode: str) -> None:
        if freeze_mode == "none":
            pass
        else:
            if freeze_mode == "all":
                excluded_lst = []
            elif freeze_mode == "first1":
                excluded_lst = ["conv_block1"]
            elif freeze_mode == "first2":
                excluded_lst = ["conv_block1", "conv_block2"]
            elif freeze_mode == "first3":
                excluded_lst = ["conv_block1", "conv_block2", "conv_block3"]
            elif freeze_mode == "last1":
                excluded_lst = ["fc1"]
            elif freeze_mode == "last2":
                excluded_lst = ["fc1", "conv_block6"]
            elif freeze_mode == "last3":
                excluded_lst = ["fc1", "conv_block6", "conv_block5"]
            else:
                raise RuntimeError(f'Unknown freeze encoder mode "{freeze_mode=}".')

            for name, param in self.named_parameters():
                if all(not name.startswith(excluded) for excluded in excluded_lst):
                    param.requires_grad = False

    def forward(
        self,
        input_: Tensor,
        input_shapes: Tensor,
        mixup_params: Optional[dict[str, Tensor]] = None,
    ) -> dict[str, Tensor]:
        """
        :param input: (bsize, audio_len)
        :param input_lens: (bsize, ...)
        :param mixup_params: {'lambda1': float, 'lambda2': float, 'indexes': IntTensor of shape (bsize,)} or None
        """

        if self.waveform_input:
            if len(input_.shape) != 2:
                raise RuntimeError(
                    f'Model "{self.__class__.__name__}" expects raw audio batch tensor of shape (bsize, audio_len), but found shape {input_.shape}.'
                )

            x = self.spectrogram_extractor(
                input_
            )  # (batch_size, 1, time_steps, freq_bins)
            x = self.logmel_extractor(x)  # (batch_size, 1, time_steps, mel_bins)
        else:
            x = input_

        input_time_dim = -2
        frames_num = x.shape[2]

        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)

        if self.training and self.use_specaug:
            x = self.spec_augmenter(x)

        if self.training and self.use_cutout:
            x = self.cutout(x)

        if self.training and self.use_pann_mixup:
            mixup_lambda = sample_lambda(self.mixup.alpha, self.mixup.asymmetric)
            indexes = torch.randperm(len(x))
            x = x * mixup_lambda + x[indexes] * (1.0 - mixup_lambda)
            x = do_mixup(x, mixup_lambda)

        # Mixup on spectrogram
        if self.training and mixup_params is not None:
            lambda1 = mixup_params["lambda1"]
            lambda2 = mixup_params["lambda2"]
            indexes = mixup_params["indexes"]
            x = lambda1 * x + lambda2 * x[indexes]

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

        x1 = F.max_pool1d(x, kernel_size=3, stride=1, padding=1)
        x2 = F.avg_pool1d(x, kernel_size=3, stride=1, padding=1)
        x = x1 + x2
        x = F.dropout(x, p=0.5, training=self.training)
        x = x.transpose(1, 2)
        x = F.relu_(self.fc1(x))
        x = x.transpose(1, 2)
        x = F.dropout(x, p=0.5, training=self.training)

        frame_embs = x

        input_lens = input_shapes[:, input_time_dim]
        reduction_factor = input_.shape[input_time_dim] // frame_embs.shape[-1]
        frame_embs_lens = input_lens.div(reduction_factor, rounding_mode="trunc")

        output_dict = {
            # (bsize, embed=2048, n_frames)
            "frame_embs": frame_embs,
            # (bsize,)
            "frame_embs_lens": frame_embs_lens,
        }

        if self.return_clip_outputs:
            assert self.att_block is not None
            (clip_logits, _, segmentwise_output) = self.att_block(x)
            segmentwise_output = segmentwise_output.transpose(1, 2)

            # Get framewise output
            frame_logits = interpolate(segmentwise_output, self.interpolate_ratio)
            frame_logits = pad_framewise_output(frame_logits, frames_num)

            output_dict |= {
                # (bsize, n_frames, n_classes)
                "frame_logits": frame_logits,
                # (bsize, n_classes)
                "clip_logits": clip_logits,
            }

        return output_dict
