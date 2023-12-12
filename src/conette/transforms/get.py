#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import os.path as osp
import pickle

from pathlib import Path
from typing import Any, Optional, Union

import torch
import yaml

from nnAudio.features import Gammatonegram
from torch import nn, Tensor
from torchaudio.transforms import Resample
from torchlibrosa.stft import Spectrogram, LogmelFilterBank

from conette.nn.encoders.convnext import convnext_tiny
from conette.nn.encoders.cnn10 import Cnn10
from conette.nn.encoders.cnn14_decisionlevel_att import Cnn14_DecisionLevelAtt
from conette.nn.encoders.cnn14 import Cnn14
from conette.nn.functional.get import get_device
from conette.nn.modules.misc import (
    Lambda,
    Standardize,
)
from conette.nn.modules.tensor import (
    Mean,
    Permute,
    Squeeze,
    TensorTo,
    Unsqueeze,
)
from conette.nn.pann_utils.ckpt import pann_load_state_dict
from conette.transforms.audio.spec_aug import SpecAugment


pylog = logging.getLogger(__name__)


def get_none() -> None:
    # Returns None. Can be used for hydra instantiations.
    return None


def get_pickle(
    fpath: Union[str, Path],
) -> nn.Module:
    if not isinstance(fpath, (str, Path)):
        raise TypeError(f"Invalid transform with pickle {fpath=}. (not a str or Path)")
    if not osp.isfile(fpath):
        raise FileNotFoundError(f"Invalid transform with pickle {fpath=}. (not a file)")

    with open(fpath, "rb") as file:
        transform = pickle.load(file)
    return transform


def get_resample_mean(
    src_sr: int,
    tgt_sr: int,
    mean_dim: Optional[int] = 0,
) -> nn.Sequential:
    if not isinstance(src_sr, int):
        error_message = _get_error_message(src_sr)
        pylog.error(error_message)
        raise ValueError(error_message)

    return nn.Sequential(
        Resample(src_sr, tgt_sr),
        Mean(dim=mean_dim),
    )


def get_resample_mean_cnn10(
    src_sr: int,
    tgt_sr: int,
    mean_dim: Optional[int] = 0,
    window_size: int = 1024,
    hop_size: int = 320,
    mel_bins: int = 64,
    device: Union[str, torch.device, None] = "auto",
    transpose_frame_embs: bool = True,
) -> nn.Sequential:
    if not isinstance(src_sr, int):
        error_message = _get_error_message(src_sr)
        pylog.error(error_message)
        raise ValueError(error_message)

    device = get_device(device)

    encoder = Cnn10(
        sr=tgt_sr,
        window_size=window_size,
        hop_size=hop_size,
        mel_bins=mel_bins,
        return_clip_outputs=True,
        return_frame_outputs=False,
        pretrained=False,
        waveform_input=True,
    )
    for p in encoder.parameters():
        p.requires_grad_(False)
    encoder.eval()

    state_dict = pann_load_state_dict("Cnn10", "cpu", offline=False)
    encoder.load_state_dict(state_dict)

    encoder = encoder.to(device)

    def get_cnn10_embs(wave: Tensor) -> dict[str, Tensor]:
        wave = wave.unsqueeze_(dim=0)
        wave_shape = torch.as_tensor(wave.shape[1:]).unsqueeze_(dim=0)
        out = encoder(wave, wave_shape)

        if transpose_frame_embs:
            # Transpose (n_channels=1, features=512, time) -> (n_channels=1, time, features=512)
            out["frame_embs"] = out["frame_embs"].transpose(1, 2)

        return out

    return nn.Sequential(
        Resample(src_sr, tgt_sr),
        Mean(dim=mean_dim),
        Unsqueeze(dim=0),
        TensorTo(device=device),
        Lambda(get_cnn10_embs),
    )


def get_resample_mean_cnn14_att(
    src_sr: int,
    tgt_sr: int,
    mean_dim: Optional[int] = 0,
    window_size: int = 1024,
    hop_size: int = 320,
    mel_bins: int = 64,
    device: Union[str, torch.device, None] = "auto",
    transpose_frame_embs: bool = True,
    only_frame_embs: bool = True,
) -> nn.Sequential:
    if not isinstance(src_sr, int):
        error_message = _get_error_message(src_sr)
        pylog.error(error_message)
        raise ValueError(error_message)

    device = get_device(device)

    encoder = Cnn14_DecisionLevelAtt(
        sr=tgt_sr,
        window_size=window_size,
        hop_size=hop_size,
        mel_bins=mel_bins,
        return_clip_outputs=True,
        pretrained=True,
    )
    for p in encoder.parameters():
        p.requires_grad_(False)
    encoder.eval()
    encoder = encoder.to(device)

    def get_cnn14_embs(wave: Tensor) -> Any:
        # Add batch dim
        wave = wave.unsqueeze_(dim=0)
        wave_shape = torch.as_tensor(wave.shape[1:]).unsqueeze_(dim=0)
        out = encoder(wave, wave_shape)
        frame_embs = out.pop("frame_embs")

        if transpose_frame_embs:
            # Transpose (n_channels, features=2048, time) -> (n_channels, time, features=2048)
            frame_embs = frame_embs.transpose(1, 2)

        if only_frame_embs:
            return frame_embs
        else:
            out["frame_embs"] = frame_embs
            return out

    return nn.Sequential(
        Resample(src_sr, tgt_sr),
        Mean(dim=mean_dim),
        TensorTo(device=device),
        Lambda(get_cnn14_embs),
    )


def get_resample_mean_cnn14(
    src_sr: int,
    tgt_sr: int,
    mean_dim: Optional[int] = 0,
    window_size: int = 1024,
    hop_size: int = 320,
    mel_bins: int = 64,
    device: Union[str, torch.device, None] = "auto",
    transpose_frame_embs: bool = True,
    only_frame_embs: bool = True,
    pretrain_path: Optional[str] = None,
) -> nn.Sequential:
    if not isinstance(src_sr, int):
        error_message = _get_error_message(src_sr)
        pylog.error(error_message)
        raise ValueError(error_message)

    device = get_device(device)
    encoder = Cnn14(
        sample_rate=tgt_sr,
        window_size=window_size,
        hop_size=hop_size,
        mel_bins=mel_bins,
        waveform_input=True,
        use_specaug=False,
        return_clip_outputs=True,
        return_frame_outputs=True,
    )

    if pretrain_path is None:
        state_dict = pann_load_state_dict("Cnn14", device, offline=False)
    else:
        state_dict = pann_load_state_dict(pretrain_path, device)
    encoder.load_state_dict(state_dict)

    for p in encoder.parameters():
        p.requires_grad_(False)
    encoder.eval()
    encoder = encoder.to(device=device)

    def get_cnn14_embs(wave: Tensor) -> Any:
        # Add batch dim
        wave = wave.unsqueeze_(dim=0)
        wave_shape = torch.as_tensor(wave.shape[1:]).unsqueeze_(dim=0)
        out = encoder(wave, wave_shape)
        frame_embs = out.pop("frame_embs")

        if transpose_frame_embs:
            # Transpose (n_channels, features=2048, time) -> (n_channels, time, features=2048)
            frame_embs = frame_embs.transpose(1, 2)

        if only_frame_embs:
            return frame_embs
        else:
            # note: empty string will use "audio" column in HDF instead of "audio.frame_embs".
            # see aac/datasets/hdf/pack.py source code for details.
            out[""] = frame_embs
            return out

    return nn.Sequential(
        Resample(src_sr, tgt_sr),
        Mean(dim=mean_dim),
        TensorTo(device=device),
        Lambda(get_cnn14_embs),
    )


def get_resample_mean_convnext(
    src_sr: int,
    tgt_sr: int,
    mean_dim: Optional[int] = 0,
    device: Union[str, torch.device, None] = "auto",
    transpose_frame_embs: bool = True,
    only_frame_embs: bool = True,
    pretrain_path: Optional[str] = None,
) -> nn.Sequential:
    if not isinstance(src_sr, int):
        error_message = _get_error_message(src_sr)
        pylog.error(error_message)
        raise ValueError(error_message)

    if not isinstance(pretrain_path, str):
        raise ValueError(
            f"Invalid argument type {type(pretrain_path)=}. (expected str)"
        )

    device = get_device(device)
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

    data = torch.load(pretrain_path, map_location=torch.device("cpu"))
    state_dict = data["model"]
    encoder.load_state_dict(state_dict, strict=False)

    for p in encoder.parameters():
        p.requires_grad_(False)
    encoder.eval()
    encoder = encoder.to(device=device)

    def get_model_outputs(wave: Tensor) -> Any:
        # Add batch dim
        wave = wave.unsqueeze_(dim=0)
        wave_shape = torch.as_tensor(wave.shape[1:]).unsqueeze_(dim=0)
        out = encoder(wave, wave_shape)
        frame_embs = out.pop("frame_embs")

        if transpose_frame_embs:
            # Transpose (n_channels, features=768, time) -> (n_channels, time, features=768)
            frame_embs = frame_embs.transpose(1, 2)

        if only_frame_embs:
            return frame_embs
        else:
            # note: empty string will use "audio" column in HDF instead of "audio.frame_embs".
            # see aac/datasets/hdf/pack.py source code for details.
            out[""] = frame_embs
            return out

    return nn.Sequential(
        Resample(src_sr, tgt_sr),
        Mean(dim=mean_dim),
        TensorTo(device=device),
        Lambda(get_model_outputs),
    )


def get_resample_mean_spec(
    src_sr: int,
    tgt_sr: int,
    window_size: int = 1024,
    hop_size: int = 320,
    mel_bins: int = 64,
    fmin: int = 50,
    fmax: int = 14000,
    window: str = "hann",
    center: bool = True,
    pad_mode: str = "reflect",
    ref: float = 1.0,
    amin: float = 1e-10,
    top_db: Optional[float] = None,
    freeze_parameters: bool = True,
    mean_dim: Optional[int] = 0,
    device: Union[str, torch.device, None] = "auto",
) -> nn.Sequential:
    if not isinstance(src_sr, int):
        error_message = _get_error_message(src_sr)
        pylog.error(error_message)
        raise ValueError(error_message)

    device = get_device(device)

    to_spectro = Spectrogram(
        n_fft=window_size,
        hop_length=hop_size,
        win_length=window_size,
        window=window,
        center=center,
        pad_mode=pad_mode,
        freeze_parameters=freeze_parameters,
    )
    to_logmel = LogmelFilterBank(
        sr=tgt_sr,
        n_fft=window_size,
        n_mels=mel_bins,
        fmin=fmin,
        fmax=fmax,
        ref=ref,
        amin=amin,
        top_db=top_db,  # type: ignore
        freeze_parameters=freeze_parameters,
    )

    to_spectro = to_spectro.to(device=device)
    to_logmel = to_logmel.to(device=device)

    transform = nn.Sequential(
        Resample(src_sr, tgt_sr),
        Mean(dim=mean_dim),
        Unsqueeze(dim=0),
        TensorTo(device=device),
        to_spectro,
        to_logmel,
        Squeeze(dim=0),
    )
    return transform


def get_resample_spec_mean_spec_aug(
    src_sr: int,
    tgt_sr: int,
    window_size: int = 1024,
    hop_size: int = 320,
    mel_bins: int = 64,
    fmin: int = 50,
    fmax: int = 14000,
    window: str = "hann",
    center: bool = True,
    pad_mode: str = "reflect",
    ref: float = 1.0,
    amin: float = 1e-10,
    top_db: Optional[float] = None,
    freeze_parameters: bool = True,
    mean_dim: Optional[int] = 0,
    time_drop_width: int = 64,
    time_stripes_num: int = 2,
    freq_drop_width: int = 2,
    freq_stripes_num: int = 1,
    spec_aug_p: float = 1.0,
) -> nn.Sequential:
    if not isinstance(src_sr, int):
        error_message = _get_error_message(src_sr)
        pylog.error(error_message)
        raise ValueError(error_message)

    return nn.Sequential(
        Resample(src_sr, tgt_sr),
        Spectrogram(
            n_fft=window_size,
            hop_length=hop_size,
            win_length=window_size,
            window=window,
            center=center,
            pad_mode=pad_mode,
            freeze_parameters=freeze_parameters,
        ),
        LogmelFilterBank(
            sr=tgt_sr,
            n_fft=window_size,
            n_mels=mel_bins,
            fmin=fmin,
            fmax=fmax,
            ref=ref,
            amin=amin,
            top_db=top_db,  # type: ignore
            freeze_parameters=freeze_parameters,
        ),
        Mean(dim=mean_dim),
        SpecAugment(
            time_max_width=time_drop_width,
            time_stripes_num=time_stripes_num,
            freq_max_width=freq_drop_width,
            freq_stripes_num=freq_stripes_num,
            p=spec_aug_p,
        ),
    )


def get_resample_spec_mean(
    src_sr: int,
    tgt_sr: int,
    window_size: int = 1024,
    hop_size: int = 320,
    mel_bins: int = 64,
    fmin: int = 50,
    fmax: int = 14000,
    window: str = "hann",
    center: bool = True,
    pad_mode: str = "reflect",
    ref: float = 1.0,
    amin: float = 1e-10,
    top_db: Optional[float] = None,
    freeze_parameters: bool = True,
    mean_dim: Optional[int] = 0,
    device: Union[str, torch.device, None] = "auto",
) -> nn.Sequential:
    if not isinstance(src_sr, int):
        error_message = _get_error_message(src_sr)
        pylog.error(error_message)
        raise ValueError(error_message)

    device = get_device(device)

    to_spectro = Spectrogram(
        n_fft=window_size,
        hop_length=hop_size,
        win_length=window_size,
        window=window,
        center=center,
        pad_mode=pad_mode,
        freeze_parameters=freeze_parameters,
    )
    to_logmel = LogmelFilterBank(
        sr=tgt_sr,
        n_fft=window_size,
        n_mels=mel_bins,
        fmin=fmin,
        fmax=fmax,
        ref=ref,
        amin=amin,
        top_db=top_db,  # type: ignore
        freeze_parameters=freeze_parameters,
    )

    to_spectro = to_spectro.to(device=device)
    to_logmel = to_logmel.to(device=device)

    transform = nn.Sequential(
        Resample(src_sr, tgt_sr),
        TensorTo(device=device),
        to_spectro,
        to_logmel,
        Mean(dim=mean_dim),
    )
    return transform


def get_resample_mean_gamma_perm(
    src_sr: int,
    tgt_sr: int,
    mean_dim: int = 0,
    n_fft: int = 1024,
    n_bins: int = 64,
    hop_length: int = 512,
    window: str = "hann",
    center: bool = True,
    pad_mode: str = "reflect",
    power: float = 2.0,
    htk: bool = False,
    fmin: float = 20.0,
    fmax: Optional[float] = None,
    norm: float = 1,
    trainable_bins: bool = False,
    trainable_STFT: bool = False,
) -> nn.Sequential:
    if not isinstance(src_sr, int):
        error_message = _get_error_message(src_sr)
        pylog.error(error_message)
        raise ValueError(error_message)

    return nn.Sequential(
        Resample(src_sr, tgt_sr),
        Mean(dim=mean_dim),
        Gammatonegram(
            sr=tgt_sr,
            n_fft=n_fft,
            n_bins=n_bins,
            hop_length=hop_length,
            window=window,
            center=center,
            pad_mode=pad_mode,
            power=power,
            htk=htk,
            fmin=fmin,
            fmax=fmax,
            norm=norm,  # type: ignore
            trainable_bins=trainable_bins,
            trainable_STFT=trainable_STFT,
            verbose=False,
        ),
        Permute(0, 2, 1),
    )


def get_stand_resample_spectro_mean_spec_aug(
    src_sr: int,
    tgt_sr: int,
    window_size: int = 1024,
    hop_size: int = 320,
    mel_bins: int = 64,
    fmin: int = 50,
    fmax: int = 14000,
    window: str = "hann",
    center: bool = True,
    pad_mode: str = "reflect",
    ref: float = 1.0,
    amin: float = 1e-10,
    top_db: Optional[float] = None,
    freeze_parameters: bool = True,
    mean_dim: Optional[int] = 0,
    time_drop_width: int = 64,
    time_stripes_num: int = 2,
    freq_drop_width: int = 2,
    freq_stripes_num: int = 1,
    spec_aug_p: float = 1.0,
) -> nn.Sequential:
    if not isinstance(src_sr, int):
        error_message = _get_error_message(src_sr)
        pylog.error(error_message)
        raise ValueError(error_message)

    return nn.Sequential(
        Standardize(),
        Resample(src_sr, tgt_sr),
        Spectrogram(
            n_fft=window_size,
            hop_length=hop_size,
            win_length=window_size,
            window=window,
            center=center,
            pad_mode=pad_mode,
            freeze_parameters=freeze_parameters,
        ),
        LogmelFilterBank(
            sr=tgt_sr,
            n_fft=window_size,
            n_mels=mel_bins,
            fmin=fmin,
            fmax=fmax,
            ref=ref,
            amin=amin,
            top_db=top_db,  # type: ignore
            freeze_parameters=freeze_parameters,
        ),
        Mean(dim=mean_dim),
        SpecAugment(
            time_max_width=time_drop_width,
            time_stripes_num=time_stripes_num,
            freq_max_width=freq_drop_width,
            freq_stripes_num=freq_stripes_num,
            p=spec_aug_p,
        ),
    )


def get_stand_resample_spectro_mean(
    src_sr: int,
    tgt_sr: int,
    window_size: int = 1024,
    hop_size: int = 320,
    mel_bins: int = 64,
    fmin: int = 50,
    fmax: int = 14000,
    window: str = "hann",
    center: bool = True,
    pad_mode: str = "reflect",
    ref: float = 1.0,
    amin: float = 1e-10,
    top_db: Optional[float] = None,
    freeze_parameters: bool = True,
    mean_dim: Optional[int] = 0,
) -> nn.Sequential:
    if not isinstance(src_sr, int):
        error_message = _get_error_message(src_sr)
        pylog.error(error_message)
        raise ValueError(error_message)

    return nn.Sequential(
        Standardize(),
        Resample(src_sr, tgt_sr),
        Spectrogram(
            n_fft=window_size,
            hop_length=hop_size,
            win_length=window_size,
            window=window,
            center=center,
            pad_mode=pad_mode,
            freeze_parameters=freeze_parameters,
        ),
        LogmelFilterBank(
            sr=tgt_sr,
            n_fft=window_size,
            n_mels=mel_bins,
            fmin=fmin,
            fmax=fmax,
            ref=ref,
            amin=amin,
            top_db=top_db,  # type: ignore
            freeze_parameters=freeze_parameters,
        ),
        Mean(dim=mean_dim),
    )


def _get_error_message(src_sr: Any) -> str:
    defaults_srs = {"clotho": 44100, "audiocaps": 32000, "macs": 48000}
    defaults_srs = yaml.dump(defaults_srs, sort_keys=False)
    message = (
        "\n"
        f"Invalid sr={src_sr} for get_resample_mean() function.\n"
        f"Please specify explicitely the source sample rate in Hz with audio_t.src_sr=SAMPLE_RATE.\n"
        f"BE CAREFUL, sample rate can be different if you use pre-processed HDF files.\n"
        f"Defaults sample rates are:\n{defaults_srs}"
    )
    return message
