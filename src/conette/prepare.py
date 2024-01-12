#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

os.environ["MKL_NUM_THREADS"] = "2"
os.environ["NUMEXPR_NUM_THREADS"] = "2"
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

os.environ["TRANSFORMERS_OFFLINE"] = "FALSE"
os.environ["HF_HUB_OFFLINE"] = "FALSE"

import logging
import math
import os.path as osp
import random
import subprocess
import sys
import time

from subprocess import CalledProcessError
from typing import Any

import hydra
import nltk
import spacy
import torch
import torchaudio
import yaml

from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from torch import nn
from torch.hub import download_url_to_file
from torchaudio.backend.common import AudioMetaData

from aac_datasets.datasets.audiocaps import AudioCaps, AudioCapsCard, _AUDIOCAPS_LINKS
from aac_datasets.datasets.clotho import Clotho, ClothoCard
from aac_datasets.datasets.macs import MACS, MACSCard
from aac_datasets.datasets.wavcaps import WavCaps
from aac_metrics.download import download_metrics as download_aac_metrics

from conette.callbacks.stats_saver import save_to_dir
from conette.datamodules.common import get_hdf_fpaths
from conette.datasets.hdf import HDFDataset, pack_to_hdf
from conette.datasets.typing import AACDatasetLike
from conette.datasets.utils import (
    AACSubset,
    AACSelectColumnsWrapper,
    TransformWrapper,
    load_audio_metadata,
)
from conette.nn.functional.misc import count_params
from conette.nn.cnext_ckpt_utils import CNEXT_PRETRAINED_URLS
from conette.nn.pann_utils.hub import PANN_PRETRAINED_URLS
from conette.transforms.utils import DictTransform
from conette.utils.collections import unzip
from conette.utils.csum import csum_any
from conette.utils.disk_cache import disk_cache
from conette.utils.hydra import setup_resolvers, get_subrun_path
from conette.train import setup_run, teardown_run


pylog = logging.getLogger(__name__)

# Note: this function must be called globally
setup_resolvers()


def download_models(cfg: DictConfig) -> None:
    if cfg.nltk:
        # Download wordnet and omw-1.4 NLTK model for nltk METEOR metric
        # Download punkt NLTK model for nltk tokenizer
        # Download stopwords for constrained beam seach generation
        for model_name in (
            "wordnet",
            "omw-1.4",
            "punkt",
            "averaged_perceptron_tagger",
            "stopwords",
        ):
            nltk.download(model_name)

    if cfg.spacy:
        # Download spaCy model for AACTokenizer
        SPACY_MODELS = ("en_core_web_sm", "fr_core_news_sm", "xx_ent_wiki_sm")
        for model_name in SPACY_MODELS:
            try:
                _model = spacy.load(model_name)
                pylog.info(f"Model '{model_name}' for spacy is already downloaded.")
            except OSError:
                command = [sys.executable, "-m", "spacy", "download", model_name]
                try:
                    subprocess.check_call(
                        command,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                    pylog.info(f"Model '{model_name}' for spacy has been downloaded.")
                except (CalledProcessError, PermissionError) as err:  # type: ignore
                    pylog.error(
                        f"Cannot download spaCy model '{model_name}' for tokenizer. (command '{command}' with error={err})"
                    )

    if str(cfg.pann).lower() != "none":
        ckpt_dir = osp.join(torch.hub.get_dir(), "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)

        def can_download(name: str, pattern: Any) -> bool:
            if pattern == "all":
                return True
            elif isinstance(pattern, str):
                return name.lower() == pattern.lower()
            elif isinstance(pattern, list):
                return name.lower() in [pann_name.lower() for pann_name in pattern]
            elif isinstance(pattern, (bool, int)):
                return can_download(name, "all" if pattern else "none")
            else:
                raise TypeError(
                    f"Invalid cfg.pann argument. Must be a string, a list of strings, a bool or an int, found {pattern.__class__.__name__}."
                )

        urls = {
            name: model_info
            for name, model_info in PANN_PRETRAINED_URLS.items()
            if can_download(name, cfg.pann)
        }

        for i, (name, model_info) in enumerate(urls.items()):
            fpath = osp.join(ckpt_dir, model_info["fname"])

            if osp.isfile(fpath):
                pylog.info(
                    f"Model '{name}' already downloaded in '{fpath}'. ({i+1}/{len(urls)})"
                )
            else:
                pylog.info(
                    f"Start downloading pre-trained PANN model '{name}' ({i+1}/{len(urls)})..."
                )
                download_url_to_file(
                    model_info["url"], fpath, progress=cfg.verbose >= 1
                )
                pylog.info(f"Model '{name}' downloaded in '{fpath}'.")

    if cfg.cnext:
        ckpt_dpath = osp.join(torch.hub.get_dir(), "checkpoints")
        urls = CNEXT_PRETRAINED_URLS

        for i, (name, info) in enumerate(urls.items()):
            url = info["url"]
            fname = info["fname"]
            fpath = osp.join(ckpt_dpath, fname)

            if osp.isfile(fpath):
                pylog.info(
                    f"Model '{name}' already downloaded in '{fpath}'. ({i+1}/{len(urls)})"
                )
            else:
                pylog.info(
                    f"Start downloading pre-trained CNext model '{name}' ({i+1}/{len(urls)})..."
                )
                download_url_to_file(url, fpath, progress=cfg.verbose >= 1)


def download_dataset(cfg: DictConfig) -> dict[str, AACDatasetLike]:
    # Download a dataset
    hydra_cfg = HydraConfig.get()
    dataname = hydra_cfg.runtime.choices["data"]

    dsets: dict[str, Any] = {}

    dataroot: str = cfg.data.root
    dataroot = osp.expandvars(dataroot)
    dataroot = osp.expanduser(dataroot)
    os.makedirs(dataroot, exist_ok=True)

    if dataname == "audiocaps":
        AudioCaps.FORCE_PREPARE_DATA = False

        if cfg.data.subsets is None:
            subsets = AudioCapsCard.SUBSETS
        else:
            subsets = cfg.data.subsets

        if cfg.data.audiocaps_caps_fix_fpath is not None:
            if "train" not in subsets:
                pylog.error(
                    f"Invalid combinaison of arguments {cfg.data.audiocaps_caps_fix_fpath=} with {subsets=}."
                )
            else:
                subsets = list(subsets)
                subsets.remove("train")
                new_subset = osp.basename(cfg.data.audiocaps_caps_fix_fpath)[:-4]
                subsets.append(new_subset)

                AudioCaps.SUBSETS = AudioCaps.SUBSETS + (new_subset,)  # type: ignore
                _AUDIOCAPS_LINKS.update(
                    {
                        new_subset: {
                            "captions": {
                                "url": None,
                                "fname": osp.basename(
                                    cfg.data.audiocaps_caps_fix_fpath
                                ),
                            },
                        },
                    }
                )

        for subset in subsets:
            dsets[subset] = AudioCaps(
                dataroot,
                subset,
                download=cfg.data.download,
                verbose=cfg.verbose,
                with_tags=cfg.data.with_tags,
                ffmpeg_path=cfg.path.ffmpeg,
                ytdl_path=cfg.path.ytdl,
            )

    elif dataname == "clotho":
        Clotho.FORCE_PREPARE_DATA = False
        Clotho.CLEAN_ARCHIVES = cfg.data.clean_archives

        if cfg.data.subsets is None:
            subsets = ClothoCard.SUBSETS
        else:
            subsets = cfg.data.subsets

        for subset in subsets:
            dsets[subset] = Clotho(
                dataroot,
                subset,
                download=cfg.data.download,
                verbose=cfg.verbose,
                version=cfg.data.version,
            )

    elif dataname == "macs":
        MACS.FORCE_PREPARE_DATA = False
        MACS.CLEAN_ARCHIVES = cfg.data.clean_archives

        if cfg.data.subsets is None:
            subsets = MACSCard.SUBSETS
        else:
            subsets = cfg.data.subsets

        for subset in subsets:
            dsets[subset] = MACS(
                dataroot,
                subset=subset,
                download=cfg.data.download,
                verbose=cfg.verbose,
            )

        if cfg.data.tags_to_str:
            dsets = {
                subset: TransformWrapper(dset, str, "tags")
                for subset, dset in dsets.items()
            }

    elif dataname == "hdf":
        hdf_fpaths = get_hdf_fpaths(
            cfg.data.name, cfg.data.subsets, dataroot, cfg.data.hdf_suffix
        )
        dsets = {}
        for subset, hdf_fpath in hdf_fpaths.items():
            ds = HDFDataset(hdf_fpath)
            ds = AACSelectColumnsWrapper(ds, include=cfg.data.include_columns)
            dsets[subset] = ds

    elif dataname == "wavcaps":
        if cfg.data.subsets is None:
            subsets = ("as_bbc_sb",)
        else:
            subsets = cfg.data.subsets

        dsets = {
            subset: WavCaps(
                dataroot,
                subset,
                download=cfg.data.download,
                hf_cache_dir=cfg.data.hf_cache_dir,
                verbose=cfg.verbose,
            )
            for subset in subsets
        }

    elif dataname in ("none",):
        dsets = {}

    else:
        accepted_datasets = (
            "audiocaps",
            "clotho",
            "hdf",
            "macs",
            "wavcaps",
            "none",
        )
        raise RuntimeError(
            f"Unknown dataset '{dataname}'. Expected one of {accepted_datasets}."
        )

    dsets = filter_dsets(cfg, dsets)

    if cfg.verbose >= 2 and len(dsets) > 0:
        rand_subset = random.choice(list(dsets.keys()))
        dset = dsets[rand_subset]
        if len(dset) > 0:
            rand_idx = random.randint(0, len(dset) - 1)
            meta_lst = dset.at(rand_idx, "audio_metadata")
            pylog.debug(f"Sample random metadata from subset '{rand_subset}':")
            pylog.debug(f"{meta_lst}")

    return dsets


def filter_dsets(
    cfg: DictConfig,
    dsets: dict[str, AACDatasetLike],
) -> dict[str, AACDatasetLike]:
    min_audio_size = float(cfg.datafilter.min_audio_size)
    max_audio_size = float(cfg.datafilter.max_audio_size)
    use_range_filt = cfg.datafilter.imin is not None or cfg.datafilter.imax is not None
    use_duration_filt = min_audio_size > 0.0 or not math.isinf(max_audio_size)
    use_sr_filt = cfg.datafilter.sr is not None

    if not any((use_range_filt, use_duration_filt, use_sr_filt)):
        return dsets

    indexes_dic: dict[str, list[int]] = {}
    for subset, ds in dsets.items():
        indexes_dic[subset] = list(range(len(ds)))

    meta_dic: dict[str, list[AudioMetaData]] = {}

    if use_duration_filt or use_sr_filt:
        for subset, ds in dsets.items():
            fpaths = ds[:, "fpath"]
            if cfg.verbose >= 2:
                pylog.debug(f"Loading durations from {len(ds)} audio files...")
            meta_lst = disk_cache(
                load_audio_metadata, fpaths, cache_path=cfg.path.cache
            )
            meta_dic[subset] = list(meta_lst.values())

    if use_range_filt:
        if cfg.verbose >= 1:
            pylog.info(
                f"Limit datasets in [{cfg.datafilter.imin}, {cfg.datafilter.imax}]."
            )

        imin = cfg.datafilter.imin
        imax = cfg.datafilter.imax
        indexes_dic = {
            subset: indexes[imin:imax] for subset, indexes in indexes_dic.items()
        }

    if use_duration_filt:
        for subset, indexes in indexes_dic.items():
            meta_lst = meta_dic[subset]
            meta_lst = [meta_lst[idx] for idx in indexes]
            durations = [(meta.num_frames / meta.sample_rate) for meta in meta_lst]
            prev_size = len(indexes)
            indexes_and_durations = [
                (idx, duration)
                for idx, duration in zip(indexes, durations, strict=True)
                if min_audio_size <= duration <= max_audio_size
            ]
            indexes, durations = unzip(indexes_and_durations)
            indexes_dic[subset] = indexes

            n_excluded = prev_size - len(indexes)
            if cfg.verbose >= 1:
                pylog.info(
                    f"Exclude {n_excluded}/{prev_size} files with audio size not in [{min_audio_size}, {max_audio_size}] seconds in {subset=}."
                )
                pylog.info(
                    f"Durations are now in range [{min(durations):.2f}, {max(durations):.2f}] s."
                )

    if use_sr_filt:
        for subset, indexes in indexes_dic.items():
            meta_lst = meta_dic[subset]
            meta_lst = [meta_lst[idx] for idx in indexes]
            sample_rates = [meta.sample_rate for meta in meta_lst]
            prev_size = len(indexes)
            indexes = [
                idx
                for idx, sr in zip(indexes, sample_rates, strict=True)
                if sr == cfg.datafilter.sr
            ]
            indexes_dic[subset] = indexes

            n_excluded = prev_size - len(indexes)
            if cfg.verbose >= 1:
                pylog.info(
                    f"Exclude {n_excluded}/{prev_size} files with sample_rate != {cfg.datafilter.sr} Hz in {subset=}."
                )

    dsets = {subset: AACSubset(ds, indexes_dic[subset]) for subset, ds in dsets.items()}
    return dsets


def pack_dsets_to_hdf(cfg: DictConfig, dsets: dict[str, Any]) -> None:
    if not cfg.pack_to_hdf:
        return None

    hydra_cfg = HydraConfig.get()
    dataname = hydra_cfg.runtime.choices["data"]
    audio_transform_name = hydra_cfg.runtime.choices["audio_t"]
    sentence_transform_name = hydra_cfg.runtime.choices["text_t"]

    if len(dsets) == 0:
        pylog.warning(
            f"Invalid value {dataname=} with pack_to_hdf=true. (found {len(dsets)} datasets)"
        )
        return None

    if hasattr(cfg.audio_t, "src_sr"):
        src_sr = cfg.audio_t.src_sr
        for name, dset in dsets.items():
            if (
                isinstance(dset, HDFDataset)
                or not isinstance(dset, AACDatasetLike)
                or "fpath" not in dset.column_names
                or len(dset) == 0
            ):
                continue
            fpath = dset[0, "fpath"]
            meta = torchaudio.info(fpath)  # type: ignore
            if src_sr != meta.sample_rate:
                raise ValueError(
                    f"Invalid input sr {src_sr} with audio sr {meta.sample_rate}. (with dataset '{name}')"
                )

    dataroot: str = cfg.path.data
    dataroot = osp.expandvars(dataroot)
    dataroot = osp.expanduser(dataroot)
    hdf_root = osp.join(dataroot, "HDF")
    os.makedirs(hdf_root, exist_ok=True)

    for subset, dset in dsets.items():
        audio_transform_params = dict(cfg.audio_t)
        sentence_transform_params = dict(cfg.text_t)

        audio_tfm = hydra.utils.instantiate(audio_transform_params)
        text_tfm = hydra.utils.instantiate(sentence_transform_params)

        if isinstance(audio_tfm, nn.Module) and cfg.verbose >= 1:
            n_params = count_params(audio_tfm, only_trainable=False)
            pylog.info(f"Nb params in audio transform: {n_params}")

        if isinstance(text_tfm, nn.Module) and cfg.verbose >= 1:
            n_params = count_params(text_tfm, only_trainable=False)
            pylog.info(f"Nb params in text transform: {n_params}")

        pre_save_transforms = {
            "audio": audio_tfm,
            "captions": text_tfm,
        }
        transforms_params = {
            "audio": audio_transform_params,
            "captions": sentence_transform_params,
        }
        if cfg.csum_in_hdf_name:
            csum = csum_any(transforms_params) % 1000
            csum_suffix = f"_{csum}"
        else:
            csum_suffix = ""

        hdf_fname = f"{dataname}_{subset}_{audio_transform_name}_{sentence_transform_name}{csum_suffix}.hdf"

        if cfg.datafilter.imin is not None or cfg.datafilter.imax is not None:
            hdf_fname = hdf_fname.replace(
                ".hdf", f"_lim_{cfg.datafilter.imin}_{cfg.datafilter.imax}.hdf"
            )
        if cfg.post_hdf_name is not None:
            hdf_fname = hdf_fname.replace(".hdf", f"_{cfg.post_hdf_name}.hdf")
        hdf_fpath = osp.join(hdf_root, hdf_fname)

        if not osp.isfile(hdf_fpath) or cfg.overwrite_hdf:
            if cfg.verbose >= 1:
                pylog.info(
                    f"Start packing the {dataname}_{subset} dataset to HDF file {hdf_fname}..."
                )

            metadata = {
                "transform_params": transforms_params,
            }
            if hasattr(cfg.audio_t, "tgt_sr"):
                metadata["sr"] = cfg.audio_t.tgt_sr

            if cfg.verbose >= 1:
                pylog.debug(yaml.dump({"Metadata": metadata}))

            pre_save_transform = DictTransform(pre_save_transforms)

            hdf_dset = pack_to_hdf(
                dset,
                hdf_fpath,
                pre_save_transform,  # type: ignore
                overwrite=cfg.overwrite_hdf,
                metadata=str(metadata),
                verbose=cfg.verbose,
                loader_bsize=cfg.data.bsize,
                loader_n_workers=cfg.data.n_workers,
            )
            hdf_dset.open()
        else:
            if cfg.verbose >= 1:
                pylog.info(
                    f"Dataset {dataname}_{subset} is already packed to hdf in {hdf_fpath=}."
                )

            hdf_dset = HDFDataset(hdf_fpath)

        if cfg.debug:
            # Sanity check
            idx = int(torch.randint(len(dset), ()).item())

            dset_item: dict[str, Any] = dict(dset[idx])
            for name, transform in pre_save_transforms.items():
                if name in dset_item.keys() and transform is not None:
                    dset_item[name] = transform(dset_item[name])
            hdf_item = hdf_dset[idx]

            dset_keys_in_hdf_keys = all(
                key in hdf_item.keys() for key in dset_item.keys()
            )
            same_dset_len = len(dset) == len(hdf_dset)

            pylog.debug(f"Check with item N°{idx=}")
            pylog.debug(
                f"Check {dset_keys_in_hdf_keys=} ({dset_item.keys()} in {hdf_item})"
            )
            pylog.debug(f"Check {same_dset_len=} ({len(dset)} == {len(hdf_dset)})")

            all_same = True

            if "audio" in dset_item.keys():
                rtol = 10**-3
                dset_audio, hdf_audio = dset_item["audio"], hdf_item["audio"]
                same_audio_shape = dset_audio.shape == hdf_audio.shape
                close_audio = same_audio_shape and torch.allclose(
                    dset_audio, hdf_audio, rtol=rtol
                )
                same_audio = same_audio_shape and dset_audio.eq(hdf_audio).all().item()
                all_same = all_same and close_audio and same_audio

                pylog.debug(
                    f"Check {same_audio_shape=} ({dset_audio.shape} == {hdf_audio.shape})"
                )
                pylog.debug(f"Check {close_audio=} ({rtol=})")
                pylog.debug(f"Check {same_audio=}")

            if "captions" in dset_item.keys():
                dset_captions, hdf_captions = (
                    dset_item["captions"],
                    hdf_item["captions"],
                )
                same_captions = len(dset_captions) == len(hdf_captions) and all(
                    c1 == c2 for c1, c2 in zip(dset_captions, hdf_captions)
                )
                captions_eq = (
                    f"(\n{dset_captions}\n == \n{hdf_captions}\n)"
                    if not same_captions
                    else ""
                )
                all_same = all_same and same_captions

                pylog.debug(f"Check {same_captions=} {captions_eq}")

            if not all_same:
                pylog.warning(
                    f"Check has failed after packing {dataname} to HDF. (dataset={dset.__class__.__name__}, {subset=})\n"
                    f"NOTE: if a transform is stochastic, you can ignore this warning."
                )


@hydra.main(
    version_base=None,
    config_path=osp.join("..", "conf"),
    config_name="prepare",
)
def main_prepare(cfg: DictConfig) -> None:
    """Download models and datasets."""
    run_start = time.perf_counter()
    setup_run(cfg)

    # Add JAVA to PATH for language tool usage on Osirim
    java_dir = osp.dirname(cfg.path.java)
    if java_dir not in ("", "."):
        os.environ["PATH"] += f"{os.pathsep}{java_dir}"

    download_models(cfg)
    dsets = download_dataset(cfg)
    pack_dsets_to_hdf(cfg, dsets)

    # Download AAC metrics
    download_aac_metrics(
        cache_path=cfg.path.cache,
        tmp_path=cfg.path.tmp,
        ptb_tokenizer=cfg.ptb_tokenizer,
        meteor=cfg.meteor,
        spice=cfg.spice,
        fense=cfg.fense,
        verbose=cfg.verbose,
    )

    subrun_path = get_subrun_path()
    save_to_dir(
        subrun_path=subrun_path,
        tokenizers=None,  # type: ignore
        git_hash=cfg.git_hash,
        cfg=cfg,
        verbose=cfg.verbose,
    )

    run_end = time.perf_counter()
    teardown_run(cfg, run_start, run_end)


if __name__ == "__main__":
    main_prepare()
