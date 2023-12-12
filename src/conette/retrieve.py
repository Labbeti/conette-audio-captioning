#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

os.environ["MKL_NUM_THREADS"] = "2"
os.environ["NUMEXPR_NUM_THREADS"] = "2"
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

os.environ["TRANSFORMERS_OFFLINE"] = "TRUE"
os.environ["HF_HUB_OFFLINE"] = "TRUE"

import glob
import logging
import os.path as osp
import pickle
import time

from typing import Any

import hydra
import torch
import tqdm
import yaml

from omegaconf import DictConfig
from torch import Tensor
from torch.nn import functional as F
from torch.utils.data.dataloader import DataLoader

from aac_metrics.utils.collections import flat_list
from aac_metrics.utils.tokenization import preprocess_mult_sents

from conette.callbacks.stats_saver import save_to_dir
from conette.datamodules.collate import AdvancedCollateDict
from conette.datasets.hdf import HDFDataset
from conette.datasets.utils import LambdaDataset
from conette.metrics.retrieval import retrieval_metrics
from conette.nn.functional.get import get_device
from conette.nn.functional.mask import masked_mean
from conette.nn.functional.misc import move_to_rec
from conette.utils.csv_utils import save_csv_list
from conette.utils.dcase import export_to_dcase_task6b_csv
from conette.utils.hydra import setup_resolvers, get_subrun_path
from conette.utils.yaml_utils import save_yaml
from conette.predict import load_pl_module
from conette.train import setup_run, teardown_run


# Note: this function must be called globally
setup_resolvers()

pylog = logging.getLogger(__name__)


def sizes_to_matrix(sizes: list[int]) -> Tensor:
    n_audios = len(sizes)
    n_caps = sum(sizes)
    matrix = torch.full((n_audios, n_caps), False, dtype=torch.bool)
    count = 0
    for idx, size in enumerate(sizes):
        matrix[idx, count : count + size] = True
        count += size
    return matrix


def scale_losses(losses: Tensor) -> Tensor:
    """(queries, targets), A2T is ok but T2A requires transpose before and after."""
    EPSILON = 1e-5
    mins = losses.min(dim=0).values
    maxs = losses.max(dim=0).values
    scaled_losses = (losses - mins) / (maxs - mins)

    n_queries = scaled_losses.shape[0]

    for query_idx in range(n_queries):
        scores = scaled_losses[query_idx]
        zero_mask = scores == scores.min()
        if zero_mask.sum() == 0:
            continue
        orig_losses = losses[query_idx, zero_mask]
        ranks = orig_losses.argsort(descending=False).to(dtype=scaled_losses.dtype)
        # if len(ranks) = 3, rank 0 -> -3, rank 1 -> -2 and rank 2 -> -1. (lower value is better for losses)
        scaled_losses[query_idx, zero_mask] = (
            scores.min() + (ranks - ranks.shape[0]) * EPSILON
        )

    return scaled_losses


@hydra.main(
    version_base=None,
    config_path=osp.join("..", "conf"),
    config_name="retrieve",
)
def run_retrieve(cfg: DictConfig) -> None:
    run_start = time.perf_counter()
    setup_run(cfg)

    resumes = cfg.resume
    if isinstance(resumes, str):
        resumes = [resumes]

    if isinstance(resumes, list):
        logdirs = [match for logdir in resumes for match in glob.glob(logdir)]
    else:
        raise TypeError(f"Invalid resume type {type(resumes)}.")

    hdf_root = osp.join(cfg.path.data, "HDF")
    hdf_fnames = cfg.hdf_fnames
    if isinstance(hdf_fnames, str):
        hdf_fnames = [hdf_fnames]

    logdirs = list(sorted(logdirs))
    if cfg.verbose >= 1:
        result_dnames = [
            osp.join(osp.basename(osp.dirname(logdir)), osp.basename(logdir))
            for logdir in logdirs
        ]
        print(f"Found {len(logdirs)} logdirs:")
        print(yaml.dump(result_dnames, sort_keys=False))

    if len(logdirs) <= 0:
        pylog.warning(f"No pre-trained model has been found in {cfg.resume}.")
        run_end = time.perf_counter()
        teardown_run(cfg, run_start, run_end)
        return None

    device = get_device(cfg.device)
    plms: dict[str, Any] = {  # type: ignore
        logdir: load_pl_module(logdir, device=device) for logdir in tqdm.tqdm(logdirs)
    }
    plm0 = next(iter(plms.values()))
    tokenizer0 = plm0.tokenizer  # assume all tokenizers are the same
    assert all(tokenizer0 == plm.tokenizer for plm in plms.values())

    dsets = {fname: HDFDataset(osp.join(hdf_root, fname)) for fname in hdf_fnames}
    mrefs_dic = {ds_name: dset[:, "captions"] for ds_name, dset in dsets.items()}
    mrefs_dic = {
        ds_name: preprocess_mult_sents(
            mrefs, cfg.path.cache, cfg.path.java, cfg.path.tmp, verbose=cfg.verbose
        )
        for ds_name, mrefs in mrefs_dic.items()
    }

    flat_mrefs_and_sizes_dic = {
        ds_name: flat_list(mrefs) for ds_name, mrefs in mrefs_dic.items()
    }
    flat_mrefs_dic = {
        ds_name: refs for ds_name, (refs, _sizes) in flat_mrefs_and_sizes_dic.items()
    }
    is_matching_matrices = {
        ds_name: sizes_to_matrix(sizes)
        for ds_name, (_refs, sizes) in flat_mrefs_and_sizes_dic.items()
    }
    del flat_mrefs_and_sizes_dic

    captions_dic: dict[str, Tensor] = {ds_name: tokenizer0.encode_batch(refs, padding="batch") for ds_name, refs in flat_mrefs_dic.items()}  # type: ignore
    captions_dic = {
        ds_name: queries.unsqueeze(dim=1).repeat(1, cfg.bsize, 1).to(device=device)
        for ds_name, queries in captions_dic.items()
    }

    caps_in_dic = {
        ds_name: queries[:, :, :-1].contiguous()
        for ds_name, queries in captions_dic.items()
    }
    caps_out_dic = {
        ds_name: queries[:, :, 1:].contiguous()
        for ds_name, queries in captions_dic.items()
    }
    n_caps_dic = {
        ds_name: len(queries_in) for ds_name, queries_in in caps_in_dic.items()
    }

    collator = AdvancedCollateDict(pad_values=dict(audio=0.0))

    def build_dset(dset) -> LambdaDataset:
        def get_item(idx: int) -> dict[str, Any]:
            return {
                "audio": dset[idx, "audio"],
                "audio_shape": dset[idx, "audio_shape"],
                "fname": dset[idx, "fname"],
                "index": torch.as_tensor(idx),
                "dataset": dset[idx, "dataset"],
            }

        lbd_dset = LambdaDataset(get_item, len(dset))
        return lbd_dset

    n_audios_dic = {ds_name: len(dset) for ds_name, dset in dsets.items()}
    lbd_dsets = {ds_name: build_dset(dset) for ds_name, dset in dsets.items()}
    loaders = {
        ds_name: DataLoader(
            lbd_dset,
            batch_size=cfg.bsize,
            shuffle=False,
            collate_fn=collator,
            num_workers=cfg.n_workers,
        )
        for ds_name, lbd_dset in lbd_dsets.items()
    }

    # Compute losses
    all_losses: dict[str, Tensor] = {}
    DEFAULT_SCORE = -999.0

    for ds_idx, (ds_fname, loader) in enumerate(loaders.items()):
        caps_in = caps_in_dic[ds_fname]
        caps_out = caps_out_dic[ds_fname]
        n_caps = n_caps_dic[ds_fname]
        n_audios = n_audios_dic[ds_fname]

        ds_losses = torch.full(
            (n_audios, len(plms), n_caps),
            DEFAULT_SCORE,
            dtype=torch.float,
            device=device,
        )

        for batch_idx, batch in enumerate(tqdm.tqdm(loader, disable=True)):
            batch = move_to_rec(batch, device=device)
            batch = plm0.on_after_batch_transfer(batch, ds_idx)

            audio = batch["audio"]
            audio_shape = batch["audio_shape"]
            indexes = batch["index"]

            cur_bsize = len(audio)
            batch_losses = torch.full(
                (cur_bsize, len(plms), n_caps),
                DEFAULT_SCORE,
                dtype=torch.float,
                device=device,
            )

            for plm_idx, (plm_name, plm) in enumerate(plms.items()):
                if cfg.verbose >= 1:
                    pylog.info(
                        f"{ds_idx=}/{len(loaders)}, {batch_idx=}/{len(loader)}, {plm_idx=}/{len(plms)}"
                    )

                batch_plm_losses = torch.full(
                    (cur_bsize, n_caps),
                    DEFAULT_SCORE,
                    dtype=torch.float,
                    device=device,
                )

                enc_outs = plm.encode_audio(audio, audio_shape)

                pbar = tqdm.tqdm(caps_in)
                for cap_idx, (cap_in, cap_out) in enumerate(zip(pbar, caps_out)):
                    if cur_bsize < cap_in.shape[0]:
                        cap_in = cap_in[:cur_bsize]
                        cap_out = cap_out[:cur_bsize]

                    logits = plm.decode_audio(enc_outs, "forcing", caps_in=cap_in)
                    losses = F.cross_entropy(
                        logits,
                        cap_out,
                        ignore_index=plm.pad_id,
                        reduction="none",
                        weight=None,
                    )
                    losses = masked_mean(losses, cap_out != plm.pad_id, dim=1)
                    # losses: (bsize,)
                    batch_plm_losses[:, cap_idx] = losses
                    if cfg.debug:
                        break

                batch_losses[:, plm_idx] = batch_plm_losses
                if cfg.debug:
                    break

            ds_losses[indexes] = batch_losses
            if cfg.debug:
                break

        all_losses[ds_fname] = ds_losses

    all_losses = {ds_name: ds_losses.cpu() for ds_name, ds_losses in all_losses.items()}

    # Save losses matrix
    subrun_path = get_subrun_path()
    os.makedirs(osp.join(subrun_path, "dcase"), exist_ok=True)
    os.makedirs(osp.join(subrun_path, "metrics"), exist_ok=True)

    all_retrieval_scores: dict[str, list[dict[str, Any]]] = {"t2a": [], "a2t": []}

    for ds_fname, ds_losses in all_losses.items():
        ds = dsets[ds_fname]
        ds_name = ds[0, "dataset"]
        ds_subset = ds[0, "subset"]
        captions = flat_mrefs_dic[ds_fname]

        # Dump losses
        # ds_losses: (n_audios, n_plms, n_queries)
        for plm_idx, plm_name in enumerate(plms.keys()):
            plm_losses = ds_losses[:, plm_idx].contiguous()

            fname = f"{ds_fname.replace('.hdf', '')}-plm_idx_{plm_idx}-losses.pickle"
            fpath = osp.join(subrun_path, fname)
            data = {
                "losses": plm_losses,
                "captions": captions,
                "audio_fnames": dsets[ds_fname][:, "fname"],
                "ds_name": ds_fname,
                "plm_idx": plm_idx,
                "plm_name": plm_name,
            }
            with open(fpath, "wb") as file:
                pickle.dump(data, file)
            del plm_losses

        # Compute retrieval results
        tasks_and_modes = [("t2a", cfg.t2a_modes), ("a2t", cfg.a2t_modes)]
        tasks_and_modes = [
            (task, [modes] if isinstance(modes, str) else list(modes))
            for task, modes in tasks_and_modes
        ]
        for task, modes in tasks_and_modes:
            for mode in modes:
                for plm_idx, plm_name in enumerate(plms.keys()):
                    plm_losses = ds_losses[:, plm_idx].contiguous()
                    if task == "t2a":
                        plm_losses = ds_losses[:, plm_idx].transpose(0, 1)
                        is_matching = is_matching_matrices[ds_fname].transpose(0, 1)
                    elif task == "a2t":
                        plm_losses = ds_losses[:, plm_idx]
                        is_matching = is_matching_matrices[ds_fname]
                    else:
                        raise RuntimeError(f"Invalid value {task=}.")
                    # plm_losses: (targets, queries)

                    if mode == "loss":
                        plm_scores = -plm_losses

                    elif mode == "scaled_loss":
                        scaled_losses = scale_losses(plm_losses)
                        plm_scores = -scaled_losses

                        n_max_per_audio = (
                            plm_scores == plm_scores.max(dim=1).values[:, None]
                        ).sum(dim=1)
                        assert (
                            n_max_per_audio.eq(1).all().item()
                        )  # check if there is always an unique top1

                    else:
                        raise ValueError(f"Invalid argument {mode=}.")

                    if is_matching is not None:
                        retrieval_outs_corpus, _retrieval_outs_sents = retrieval_metrics(plm_scores, is_matching)  # type: ignore
                        retrieval_outs_corpus: dict[str, Tensor]

                        retrieval_outs_corpus_lst = {
                            k: v.tolist() for k, v in retrieval_outs_corpus.items()
                        }

                        yaml_fname = f"{task}-{ds_name}_{ds_subset}-mode_{mode}-plm_idx_{plm_idx}.yaml"
                        yaml_fpath = osp.join(subrun_path, "metrics", yaml_fname)
                        save_yaml(retrieval_outs_corpus_lst, yaml_fpath)

                        if cfg.verbose >= 1:
                            pylog.info(
                                f"Audio-Text retrieval scores for {task.upper()} with {mode=}:\n{yaml.dump(retrieval_outs_corpus_lst, sort_keys=False)}"
                            )

                        all_retrieval_scores[task].append(
                            {
                                "ds_name": ds_name,
                                "ds_subset": ds_subset,
                                "ds_fname": ds_fname,
                                "mode": mode,
                                "plm_idx": plm_idx,
                                "plm_name": plm_name,
                            }
                            | retrieval_outs_corpus_lst
                        )

                    if task == "t2a":
                        csv_fname = f"{task}-{ds_name}_{ds_subset}-mode_{mode}-plm_idx_{plm_idx}.csv"
                        csv_fpath = osp.join(subrun_path, "dcase", csv_fname)
                        top_audio_indexes = plm_scores.argsort(dim=1, descending=True)[
                            :, :10
                        ].tolist()
                        predicted_fnames = [
                            ds[indexes, "fname"] for indexes in top_audio_indexes
                        ]
                        export_to_dcase_task6b_csv(
                            csv_fpath, captions, predicted_fnames
                        )

    for task, scores in all_retrieval_scores.items():
        csv_fname = f"metrics-{task}.csv"
        csv_fpath = osp.join(subrun_path, csv_fname)
        save_csv_list(scores, csv_fpath)

    flatten_scores = [
        {"task": task} | dic
        for task, scores in all_retrieval_scores.items()
        for dic in scores
    ]
    csv_fpath = osp.join(subrun_path, "metrics.csv")
    save_csv_list(flatten_scores, csv_fpath)

    run_end = time.perf_counter()
    save_to_dir(
        subrun_path,
        git_hash=cfg.git_hash,
        cfg=cfg,
        verbose=cfg.verbose,
    )
    teardown_run(cfg, run_start, run_end)


if __name__ == "__main__":
    run_retrieve()
