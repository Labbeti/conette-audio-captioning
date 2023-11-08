#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Iterable, Optional

import torch

from torch import nn, Tensor
from torch.nn.parameter import Parameter
from torchlibrosa.stft import Spectrogram, LogmelFilterBank
from torchlibrosa.augmentation import SpecAugmentation

from conette.nn.functional.init import trunc_normal_
from conette.nn.modules.drop import DropPath
from conette.nn.modules.norm import LayerNorm
from conette.transforms.mixup import pann_mixup
from conette.transforms.speed_perturb import SpeedPerturbation


class ConvNeXtBlock(nn.Module):
    r"""ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(
        self, dim: int, drop_path: float = 0.0, layer_scale_init_value: float = 1e-6
    ) -> None:
        super().__init__()
        self.dwconv = nn.Conv2d(
            dim, dim, kernel_size=7, padding=3, groups=dim
        )  # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(
            dim, 4 * dim
        )  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.scale_layer = (
            Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            if layer_scale_init_value > 0
            else None
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        input_ = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.scale_layer is not None:
            x = self.scale_layer * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input_ + self.drop_path(x)
        return x


class ConvNeXt(nn.Module):
    r"""ConvNeXt
        A PyTorch impl of : `A ConvNet for the 2020s`  -
          https://arxiv.org/pdf/2201.03545.pdf

    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """

    def __init__(
        self,
        in_chans: int = 3,
        num_classes: int = 1000,
        depths: Iterable[int] = [3, 3, 9, 3],
        dims: Iterable[int] = [96, 192, 384, 768],
        drop_path_rate: float = 0.0,
        use_speed_perturb: bool = True,
        layer_scale_init_value: float = 1e-6,
        head_init_scale: float = 1.0,
        waveform_input: bool = True,
        return_clip_outputs: bool = True,
        return_frame_outputs: bool = False,
        use_specaug: bool = True,
    ) -> None:
        depths = list(depths)
        dims = list(dims)

        super().__init__()
        self.waveform_input = waveform_input
        self.return_clip_outputs = return_clip_outputs
        self.return_frame_outputs = return_frame_outputs
        self.use_specaug = use_specaug

        window = "hann"
        center = True
        pad_mode = "reflect"
        ref = 1.0
        amin = 1e-10
        top_db = None

        sample_rate = 32000
        window_size = 1024
        hop_size = 320
        mel_bins = 224
        fmin = 50
        fmax = 14000

        # note: build these layers even if waveform_input is False
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

        # Spec augmenter
        # freq_drop_width=8
        freq_drop_width = 28  # 28 = 8*224//64, in order to be the same as the nb of bins dropped in Cnn14
        if use_specaug:
            self.spec_augmenter = SpecAugmentation(
                time_drop_width=64,
                time_stripes_num=2,
                freq_drop_width=freq_drop_width,
                freq_stripes_num=2,
            )
        else:
            self.spec_augmenter = nn.Identity()

        self.use_speed_perturb = use_speed_perturb
        if self.use_speed_perturb:
            self.speed_perturb = SpeedPerturbation(rates=(0.5, 1.5), p=0.5)
        else:
            self.speed_perturb = nn.Identity()

        self.bn0 = nn.BatchNorm2d(224)

        self.downsample_layers = (
            nn.ModuleList()
        )  # stem and 3 intermediate downsampling conv layers

        stem = nn.Sequential(
            nn.Conv2d(3, dims[0], kernel_size=(4, 4), stride=(4, 4)),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first"),
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = (
            nn.ModuleList()
        )  # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[
                    ConvNeXtBlock(
                        dim=dims[i],
                        drop_path=dp_rates[cur + j],
                        layer_scale_init_value=layer_scale_init_value,
                    )
                    for j in range(depths[i])
                ]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)  # final norm layer
        self.head_audioset = nn.Linear(dims[-1], num_classes)

        self.apply(self._init_weights)
        self.head_audioset.weight.data.mul_(head_init_scale)
        self.head_audioset.bias.data.mul_(head_init_scale)

    def _init_weights(self, m) -> None:
        # pass
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=0.02)
            nn.init.constant_(m.bias, 0)  # type: ignore

    def forward_features(self, x: Tensor):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)

        x = torch.mean(x, dim=3)
        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2

        x = self.norm(x)  # global average+max pooling, (N, C, H, W) -> (N, C)
        return x

    def forward(
        self,
        input_: Tensor,
        input_shapes: Tensor,
        mixup_lambda: Optional[Tensor] = None,
    ) -> dict[str, Tensor]:
        if self.waveform_input:
            if input_.ndim != 2:
                raise ValueError(
                    f"Invalid argument shape {input_.shape=}. (expected 2 dims with {self.waveform_input=})"
                )
            input_time_dim = -1
            x = self.spectrogram_extractor(input_)
            # (batch_size, 1, time_steps, freq_bins)
            x = self.logmel_extractor(x)
        else:
            if input_.ndim != 4:
                raise ValueError(
                    f"Invalid argument shape {input_.shape=}. (expected 4 dims with {self.waveform_input=})"
                )
            x = input_
            input_time_dim = -2

        if self.training and self.use_speed_perturb:
            x = self.speed_perturb(x)

        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)

        if self.training:
            x = self.spec_augmenter(x)

        # Mixup on spectrogram
        if self.training and mixup_lambda is not None:
            x = pann_mixup(x, mixup_lambda)

        # forward features with frame_embs
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)

        x = torch.mean(x, dim=3)

        output_dict = {}
        if self.return_frame_outputs:
            frame_embs = x

            input_lens = input_shapes[:, input_time_dim]
            reduction_factor = input_.shape[input_time_dim] // frame_embs.shape[-1]

            frame_embs_lens = input_lens.div(reduction_factor).round().int()

            output_dict |= {
                # (bsize, embed=768, n_frames=31)
                "frame_embs": frame_embs,
                # (bsize,)
                "frame_embs_lens": frame_embs_lens,
            }

        if self.return_clip_outputs:
            (x1, _) = torch.max(x, dim=2)
            x2 = torch.mean(x, dim=2)
            x = x1 + x2

            x = self.norm(x)  # global average+max pooling, (N, C, H, W) -> (N, C)
            # end forward features

            x = self.head_audioset(x)
            clipwise_output = torch.sigmoid(x)
            output_dict |= {"clipwise_output": clipwise_output}

        return output_dict


CNEXT_IMAGENET_PRETRAINED_URLS = {
    "convnext_atto_1k": "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-rsb-weights/convnext_atto_d2-01bb0f51.pth",
    "convnext_femto_1k": "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-rsb-weights/convnext_femto_d1-d71d5b4c.pth",
    "convnext_pico_1k": "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-rsb-weights/convnext_pico_d1-10ad7f0d.pth",
    "convnext_nano_1k": "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-rsb-weights/convnext_nano_d1h-7eb4bdea.pth",
    "convnext_tiny_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_tiny_1k_224_ema.pth",
    "convnext_small_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_small_1k_224_ema.pth",
    "convnext_base_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_base_1k_224_ema.pth",
    "convnext_large_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_large_1k_224_ema.pth",
    "convnext_tiny_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_tiny_22k_224.pth",
    "convnext_small_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_small_22k_224.pth",
    "convnext_base_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_base_22k_224.pth",
    "convnext_large_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_large_22k_224.pth",
    "convnext_xlarge_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_xlarge_22k_224.pth",
}


def convnext_tiny(
    pretrained: bool = False,
    strict: bool = False,
    in_22k: bool = False,
    drop_path_rate: float = 0.1,
    after_stem_dim: Iterable[int] = (56,),
    use_speed_perturb: bool = True,
    **kwargs,
) -> ConvNeXt:
    after_stem_dim = list(after_stem_dim)

    model = ConvNeXt(
        in_chans=1,
        num_classes=527,
        depths=[3, 3, 9, 3],
        dims=[96, 192, 384, 768],
        drop_path_rate=drop_path_rate,
        use_speed_perturb=use_speed_perturb,
        **kwargs,
    )

    if pretrained:
        url = (
            CNEXT_IMAGENET_PRETRAINED_URLS["convnext_tiny_22k"]
            if in_22k
            else CNEXT_IMAGENET_PRETRAINED_URLS["convnext_tiny_1k"]
        )
        checkpoint = torch.hub.load_state_dict_from_url(
            url=url, map_location="cpu", check_hash=True  # type: ignore
        )
        model.load_state_dict(checkpoint["model"], strict=strict)

    stem_audioset = nn.Conv2d(
        1, 96, kernel_size=(18, 4), stride=(18, 4), padding=(9, 0)
    )
    if len(after_stem_dim) < 2:
        if after_stem_dim[0] == 56:
            stem_audioset = nn.Conv2d(
                1, 96, kernel_size=(18, 4), stride=(18, 4), padding=(9, 0)
            )
        elif after_stem_dim[0] == 112:
            stem_audioset = nn.Conv2d(
                1, 96, kernel_size=(9, 2), stride=(9, 2), padding=(4, 0)
            )
        else:
            raise ValueError(
                "ERROR: after_stem_dim can be set to 56 or 112 or [252,56]"
            )
    else:
        if after_stem_dim == [252, 56]:
            stem_audioset = nn.Conv2d(
                1, 96, kernel_size=(4, 4), stride=(4, 4), padding=(4, 0)
            )
        elif after_stem_dim == [504, 28]:
            stem_audioset = nn.Conv2d(
                1, 96, kernel_size=(4, 8), stride=(2, 8), padding=(5, 0)
            )
        elif after_stem_dim == [504, 56]:
            stem_audioset = nn.Conv2d(
                1, 96, kernel_size=(4, 4), stride=(2, 4), padding=(5, 0)
            )
        else:
            raise ValueError(
                "ERROR: after_stem_dim can be set to 56 or 112 or [252,56]"
            )

    trunc_normal_(stem_audioset.weight, std=0.02)
    nn.init.constant_(stem_audioset.bias, 0)  # type: ignore
    model.downsample_layers[0][0] = stem_audioset  # type: ignore

    return model
