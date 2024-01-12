#!/usr/bin/env python
# -*- coding: utf-8 -*-

# BASED ON Cnn10 class from https://github.com/qiuqiangkong/audioset_tagging_cnn/blob/master/pytorch/models.py#L484

import logging

from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from torch import Tensor
from torchlibrosa.stft import Spectrogram, LogmelFilterBank
from torchlibrosa.augmentation import SpecAugmentation

from conette.nn.pann_utils.models import init_bn, init_layer, do_mixup
from conette.nn.pann_utils.models import ConvBlock
from conette.nn.pann_utils.ckpt import pann_load_state_dict


pylog = logging.getLogger(__name__)


class Cnn10(nn.Module):
    AUDIOSET_NUM_CLASSES = 527
    CONV_FEATURES_EMB_SIZE = 512

    def __init__(
        self,
        # Spectrogram extractor args
        sr: int = 32000,
        window_size: int = 1024,
        hop_size: int = 320,
        mel_bins: int = 64,
        fmin: int = 50,
        fmax: int = 14000,
        # Other args
        return_clip_outputs: bool = True,
        return_frame_outputs: bool = False,
        classes_num: int = 527,
        clip_emb_size: int = 512,
        frame_emb_size: int = 512,
        freeze_weight: str = "none",
        lens_rounding_mode: str = "trunc",
        pretrained: bool = False,
        use_specaug: bool = False,
        waveform_input: bool = True,
        convblock_dropout: float = 0.2,
        freeze_spectro_extractor: bool = True,
        freeze_logmel_extractor: bool = True,
        use_fc2_layer: bool = False,
    ) -> None:
        """
        Compute frame-embeddings of shape (bsize, embed_len, n_frames) from audio.

        :param sr: defaults to 32000.
        :param window_size: defaults to 1024.
        :param hop_size: defaults to 320.
        :param mel_bins: defaults to 64.
        :param fmin: defaults to 50.
        :param fmax: defaults to 14000.
        :param add_clip_linear: TODO
        :param add_frame_linear: TODO
        :param classes_num: TODO
        :param clip_emb_size: TODO
        :param frame_emb_size: TODO
        :param freeze_weight: TODO
        :param lens_rounding_mode: TODO
        :param pretrained: If True, use pretrained weights from PANN. defaults to True.
        :param use_spec_augment: TODO
        :param waveform_input: TODO
        :param convblock_dropout: Dropout used after ConvBlocks. defaults to 0.2.
        :param freeze_spectro_extractor: If true, freezes spectrogram extractor weights. defaults to True.
        :param freeze_logmel_extractor: If true, freezes logmel extrator weights. defaults to True.
        """
        if return_frame_outputs:
            pylog.warning(
                f"Deprecated argument value {return_frame_outputs=}. Please use projection in a separate module."
            )

        if not pretrained and freeze_weight != "none":
            raise ValueError(
                f"Cannot freeze weights without using pre-trained weights. (found {freeze_weight=} && {pretrained=})"
            )
        if (
            pretrained
            and return_clip_outputs
            and classes_num != self.AUDIOSET_NUM_CLASSES
        ):
            pylog.warning(
                f"Found argument {classes_num=} != {self.AUDIOSET_NUM_CLASSES} and {pretrained=}, so the layer 'fc_audioset' will not use pretrained weights."
            )
        if pretrained and return_clip_outputs and clip_emb_size != 512:
            raise ValueError(
                f"Invalid argument {clip_emb_size=} with {pretrained=} and {return_clip_outputs=}."
            )
        if not return_frame_outputs and frame_emb_size != 512:
            raise ValueError(
                f"Cannot remove Linear 'fc2' in CNN10 with {return_frame_outputs=} and {frame_emb_size=} != 512. Please use add_frame_linear=True or frame_emb_size=512."
            )
        if lens_rounding_mode not in ("trunc", "ceil"):
            raise ValueError(
                f"Invalid argument {lens_rounding_mode=} (expected 'trunc' or 'ceil')"
            )

        super().__init__()
        # Params
        self.return_clip_outputs = return_clip_outputs
        self.return_frame_outputs = return_frame_outputs
        self.classes_num = classes_num
        self.clip_emb_size = clip_emb_size
        self.frame_emb_size = frame_emb_size
        self.lens_rounding_mode = lens_rounding_mode
        self.pretrained = pretrained
        self.use_specaug = use_specaug
        self.waveform_input = waveform_input
        self.convblock_dropout = convblock_dropout
        self.use_fc2_layer = use_fc2_layer

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
                freeze_parameters=freeze_spectro_extractor,
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
                freeze_parameters=freeze_logmel_extractor,
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

        # For tag outputs
        self.fc1 = (
            nn.Linear(Cnn10.CONV_FEATURES_EMB_SIZE, clip_emb_size, bias=True)
            if return_clip_outputs
            else nn.Identity()
        )
        self.fc_audioset = (
            nn.Linear(clip_emb_size, classes_num, bias=True)
            if return_clip_outputs
            else nn.Identity()
        )
        # For frame outputs (not pretrained !)
        self.fc2 = (
            nn.Linear(Cnn10.CONV_FEATURES_EMB_SIZE, frame_emb_size, bias=True)
            if return_frame_outputs and use_fc2_layer
            else nn.Identity()
        )

        # Initialize weights
        self.init_weight()
        if pretrained:
            exclude_spectro_params = window_size != 1024 or hop_size != 320
            self.load_pretrained_weights(
                strict=False, exclude_spectro_params=exclude_spectro_params
            )
        self.freeze_weight(freeze_weight)

    def init_weight(self) -> None:
        init_bn(self.bn0)
        for layer in (self.fc1, self.fc_audioset, self.fc2):
            if hasattr(layer, "weight"):
                init_layer(layer)

    def load_pretrained_weights(
        self,
        strict: bool = False,
        exclude_spectro_params: bool = False,
    ) -> None:
        device = self.bn0.weight.device
        state_dict = pann_load_state_dict("Cnn10", device, offline=False)

        if exclude_spectro_params:
            state_dict = {
                key: weight
                for key, weight in state_dict.items()
                if all(
                    not key.startswith(prefix)
                    for prefix in ("spectrogram_extractor", "logmel_extractor")
                )
            }
        if self.pretrained and self.classes_num != self.AUDIOSET_NUM_CLASSES:
            state_dict = {
                key: weight
                for key, weight in state_dict.items()
                if all(not key.startswith(prefix) for prefix in ("fc_audioset",))
            }

        self.load_state_dict(state_dict, strict=strict)  # type: ignore

    def freeze_weight(self, freeze_weight: str) -> None:
        if freeze_weight == "none":
            return None

        if freeze_weight == "all":
            excluded_lst = ["fc2"]
        elif freeze_weight == "first1":
            excluded_lst = ["fc2", "conv_block1"]
        elif freeze_weight == "last1":
            excluded_lst = ["fc2", "conv_block4"]
        else:
            raise RuntimeError(f"Unknown freeze encoder mode {freeze_weight=}.")

        if self.pretrained and self.classes_num != self.AUDIOSET_NUM_CLASSES:
            excluded_lst.append("fc_audioset")

        pylog.debug(f"Freezing layers:\n{yaml.dump(excluded_lst, sort_keys=False)}.")

        for name, param in self.named_parameters():
            if all(not name.startswith(excluded) for excluded in excluded_lst):
                param.requires_grad = False
        self.eval()

    def _check_forward_input(
        self,
        x: Tensor,
        x_shape: Optional[Tensor],
        **kwargs,
    ) -> None:
        if self.waveform_input:
            if not (x.ndim == 2 or (x.ndim == 3 and x.shape[1] == 1)):
                raise ValueError(
                    f"Invalid input shape {x.shape=}. Expected (bsize, time_steps) or (bsize, 1, time_steps) tensor."
                )
        else:
            if not (x.ndim == 3 or (x.ndim == 4 and x.shape[1] == 1)):
                raise ValueError(
                    f"Invalid input shape {x.shape=}. Expected (bsize, time_steps, freq_bins) or (bsize, 1, time_steps, freq_bins) tensor."
                )

        if x_shape is not None:
            if x_shape.ndim != 2:
                raise ValueError(
                    f"Invalid number of dimensions for x_shape argument. (expected 2 dims with shape (bsize, x.ndim-1) tensor but found {x_shape.ndim=})"
                )
            if x.shape[0] != x_shape.shape[0]:
                raise ValueError(
                    f"Invalid batch dim 0 for arguments x and x_shape. (found {x.shape[0]=} != {x_shape.shape[0]=})"
                )
            if x_shape.shape[1] != x.ndim - 1:
                raise ValueError(
                    f"Invalid x_shape dim 1 {x_shape.shape[1]=}. (expected {x.ndim-1=})"
                )
            for x_shape_i in x_shape:
                for j, x_shape_ij in enumerate(x_shape_i):
                    if x_shape_ij > x.shape[j + 1]:
                        raise ValueError(
                            f"Found a shape greater than the dimension of the input. (found {x_shape_ij} > {x.shape[j+1]})"
                        )

    def forward(
        self,
        x: Tensor,
        x_shape: Optional[Tensor],
        mixup_params: Optional[dict] = None,
        mixup_lambda: Optional[Tensor] = None,
    ) -> dict[str, Any]:
        """
        :param x: Batch of audios tensors.
            Waveforms shapes (if self.waveform_input=True):
                (bsize, 1, time_steps) or (bsize, time_steps) tensor
            Spectrograms shapes (if self.waveform_input=False):
                (bsize, 1, time_steps, freq_bins) or (bsize, time_steps, freq_bins) tensor
        :param x_shape: Shape of non-padded audio of x. Has shape (bsize, x.ndim-1)
        :param mixup_params: Dictionary of MixUp parameters.
            'lambda1': coefficient of the first tensor x
            'lambda2': coefficient of the second tensor x2
            'indexes': tensor of indexes for shuffle x to x2, shape is (bsize,)
        :returns: A dictionary with embeddings and logits.
            'frame_embs': (bsize, embed_size, time_steps_reduced)
            'frame_embs_lens': (bsize, embed_size)
            "clip_embs": (bsize, 512),
            "clip_logits": (bsize, classes_num=527),
        """
        self._check_forward_input(x, x_shape)

        if self.waveform_input:
            # Convert and format to spectrogram and get x_lens
            if x.ndim == 3:
                x = x.squeeze_(dim=1)
            # x: (bsize, time_steps)
            source_len = x.shape[-1]
            x_lens = x_shape.squeeze(dim=1) if x_shape is not None else None

            x = self.spectrogram_extractor(x)  # (batch_size, 1, time_steps, freq_bins)
            x = self.logmel_extractor(x)  # (batch_size, 1, time_steps, mel_bins)

        else:
            # Format spectrogram and get x_lens
            if x.ndim == 3:
                x = x.unsqueeze_(dim=1)
                time_step_dim = 1
            else:  # x.ndim == 4
                time_step_dim = 2

            # x : (bsize, 1, time_steps, freq_bins)
            source_len = x.shape[-2]

            # x_shape : (bsize, N)
            x_lens = x_shape[:, time_step_dim - 1] if x_shape is not None else None

        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)

        if self.training and self.use_specaug:
            x = self.spec_augmenter(x)

        # Mixup on spectrogram
        if self.training and mixup_lambda is not None:
            x = do_mixup(x, mixup_lambda)

        # Mixup on spectrogram
        if self.training and mixup_params is not None:
            # x = do_mixup(x, mixup_lambda)
            lambda1 = mixup_params["lambda1"]
            lambda2 = mixup_params["lambda2"]
            indexes = mixup_params["indexes"]
            x = lambda1 * x + lambda2 * x[indexes]

        x = self.conv_block1(x, pool_size=(2, 2), pool_type="avg")
        x = F.dropout(x, p=self.convblock_dropout, training=self.training)

        x = self.conv_block2(x, pool_size=(2, 2), pool_type="avg")
        x = F.dropout(x, p=self.convblock_dropout, training=self.training)

        x = self.conv_block3(x, pool_size=(2, 2), pool_type="avg")
        x = F.dropout(x, p=self.convblock_dropout, training=self.training)

        x = self.conv_block4(x, pool_size=(2, 2), pool_type="avg")
        x = F.dropout(x, p=self.convblock_dropout, training=self.training)

        x = torch.mean(x, dim=3)
        conv_features = x
        # conv_features : (bsize, n_filters=512, time_steps, mel_bins)

        outs = {}

        if self.return_clip_outputs:
            x = conv_features
            (x1, _) = torch.max(x, dim=2)
            x2 = torch.mean(x, dim=2)
            x = x1 + x2
            del x1, x2
            x = F.dropout(x, p=0.5, training=self.training)
            x = F.relu_(self.fc1(x))
            clip_embs = F.dropout(x, p=0.5, training=self.training)
            clip_logits = self.fc_audioset(x)
            del x
            outs |= {
                "clip_embs": clip_embs,
                "clip_logits": clip_logits,
            }

        x = conv_features
        x1 = F.max_pool1d(x, kernel_size=3, stride=1, padding=1)
        x2 = F.avg_pool1d(x, kernel_size=3, stride=1, padding=1)
        x = x1 + x2

        if self.return_frame_outputs and self.use_fc2_layer:
            x = F.dropout(x, p=0.5, training=self.training)
            x = x.transpose(1, 2)
            x = self.fc2(x)
            x = F.relu_(x)
            x = x.transpose(1, 2)
            x = F.dropout(x, p=0.5, training=self.training)

        # x : (bsize, embed_size, time_steps_reduced)
        if x_lens is not None:
            if self.lens_rounding_mode == "trunc":
                reduction_factor = source_len // x.shape[-1]
                x_lens = x_lens.div(reduction_factor, rounding_mode="trunc")
            elif self.lens_rounding_mode == "ceil":
                reduction_factor = source_len / x.shape[-1]
                x_lens = x_lens.float().div(reduction_factor).ceil()
            elif self.lens_rounding_mode == "round":
                reduction_factor = source_len / x.shape[-1]
                x_lens = x_lens.float().div(reduction_factor).round()
            else:
                raise ValueError(f"Invalid parameter {self.lens_rounding_mode=}.")

        outs |= {
            "frame_embs": x,
            "frame_embs_lens": x_lens,
        }
        return outs
