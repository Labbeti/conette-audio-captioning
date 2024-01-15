#!/usr/bin/env python
# -*- coding: utf-8 -*-

import csv
import logging
import os.path as osp

from argparse import ArgumentParser, Namespace
from typing import Optional, Union

import torch
import transformers
import yaml

from lightning_fabric.utilities.seed import seed_everything
from omegaconf import OmegaConf, DictConfig

from conette.nn.functional.get import get_device
from conette.huggingface.model import CoNeTTEConfig, CoNeTTEModel
from conette.pl_modules.baseline import BaselinePLM
from conette.pl_modules.conette import CoNeTTEPLM
from conette.utils.cmdline import _str_to_opt_str, _str_to_opt_int, _setup_logging
from conette.utils.csum import csum_module


pylog = logging.getLogger(__name__)


def get_predict_args() -> Namespace:
    """Return main_predict arguments."""
    parser = ArgumentParser(
        description="Download models and external code to evaluate captions."
    )
    parser.add_argument(
        "--audio",
        type=str,
        help="Path to an audio file.",
        default=(),
        required=True,
        nargs="+",
    )
    parser.add_argument(
        "--task",
        type=_str_to_opt_str,
        help="CoNeTTE task embedding input.",
        default=None,
        nargs="+",
    )
    parser.add_argument(
        "--model_name",
        type=_str_to_opt_str,
        help="Model name on huggingface.",
        default="Labbeti/conette",
    )
    parser.add_argument(
        "--model_path",
        type=_str_to_opt_str,
        help="Path to trained model directory.",
        default=None,
    )
    parser.add_argument(
        "--device",
        type=str,
        help="Torch device used to run the model.",
        default="auto",
    )
    parser.add_argument(
        "--token",
        type=_str_to_opt_str,
        help="Optional access token.",
        default=None,
    )
    parser.add_argument(
        "--seed",
        type=_str_to_opt_int,
        help="Random seed value.",
        default=1234,
    )
    parser.add_argument(
        "--csv_export",
        type=_str_to_opt_str,
        help="Path to CSV output file.",
        default=None,
    )
    parser.add_argument(
        "--verbose",
        type=int,
        help="Verbose level.",
        default=1,
    )
    args = parser.parse_args()
    return args


def _load_hf_model(
    model_name: str,
    token: Optional[str],
    device: Union[str, torch.device, None],
    verbose: int,
) -> CoNeTTEModel:
    if verbose >= 1:
        pylog.info(f"Initilizing '{model_name}' model...")

    # To support transformers < 4.35, which is required for aac-metrics dependancy
    major, minor, _patch = map(int, transformers.__version__.split("."))
    if major < 4 or (major == 4 and minor < 35):
        token_key = "use_auth_token"
    else:
        token_key = "token"

    common_args = {
        "pretrained_model_name_or_path": model_name,
        token_key: token,
    }
    config = CoNeTTEConfig.from_pretrained(**common_args)
    hf_model: CoNeTTEModel = CoNeTTEModel.from_pretrained(  # type: ignore
        config=config,
        device=device,
        **common_args,
    )
    if verbose >= 1:
        pylog.info(f"Model '{model_name}' is initialized.")
    return hf_model


def _check_model_path(
    model_path: str,
) -> None:
    cfg_fpath = osp.join(model_path, "hydra", "config.yaml")
    ckpt_fpath = osp.join(model_path, "checkpoints", "best.ckpt")

    if not osp.isdir(model_path):
        raise FileNotFoundError(
            f"Cannot find model_path directory. ({model_path} is not a directory)"
        )
    if not osp.isfile(cfg_fpath):
        raise FileNotFoundError(
            f"Cannot find config file in model_path directory. ({cfg_fpath} is not a file)"
        )
    if not osp.isfile(ckpt_fpath):
        raise FileNotFoundError(
            f"Cannot find checkpoint file in model_path directory. ({ckpt_fpath} is not a file)"
        )


def _load_model_from_path(
    model_path: str,
    device: Union[str, torch.device, None],
    verbose: int,
) -> CoNeTTEModel:
    _check_model_path(model_path)
    if verbose >= 1:
        pylog.info(f"Initilizing model from '{model_path}'...")

    cfg_fpath = osp.join(model_path, "hydra", "config.yaml")
    with open(cfg_fpath, "r") as file:
        raw_cfg = yaml.safe_load(file)
    cfg: DictConfig = OmegaConf.create(raw_cfg)  # type: ignore

    pl_cfg = cfg.get("pl", {})
    target = pl_cfg.pop("_target_", "unknown")
    if CoNeTTEPLM.__name__ in target:
        model = CoNeTTEPLM(**pl_cfg)
    elif BaselinePLM.__name__ in target:
        model = BaselinePLM(**pl_cfg)
    else:
        raise NotImplementedError(f"Unsupported pretrained model type '{target}'.")

    ckpt_fpath = osp.join(model_path, "checkpoints", "best.ckpt")
    ckpt_data = torch.load(ckpt_fpath, map_location=model.device)
    state_dict = ckpt_data["state_dict"]
    model.load_state_dict(state_dict, strict=True)

    device = get_device(device)
    config = CoNeTTEConfig(**pl_cfg)
    hf_model = CoNeTTEModel(config, device=device, model_override=model)

    if verbose >= 1:
        pylog.info(f"Model from '{model_path}' is initialized.")
    return hf_model


def main_predict() -> None:
    """Main entrypoint for CoNeTTE predict."""
    args = get_predict_args()
    _setup_logging("conette", verbose=args.verbose, set_format=False)
    seed_everything(args.seed)

    fpaths = list(args.audio)
    tasks = args.task

    if args.model_path is not None:
        hf_model = _load_model_from_path(args.model_path, args.device, args.verbose)
    elif args.model_name is not None:
        hf_model = _load_hf_model(
            args.model_name, args.token, args.device, args.verbose
        )
    else:
        raise ValueError(
            f"Invalid arguments {args.model_name=} and {args.model_path=}. (expected at one str value)"
        )

    hf_model.eval_and_disable_grad()

    if args.verbose >= 2:
        enc_csum = csum_module(hf_model.preprocessor.encoder, with_names=False)
        model_csum = csum_module(hf_model.model, with_names=False)
        pylog.debug(f"Enc checksum: '{enc_csum}'")
        pylog.debug(f"Model checksum: '{model_csum}'")

    outs = hf_model(fpaths, task=tasks)

    cands = outs["cands"]
    tasks = outs["tasks"]
    fnames = [osp.basename(fpath) for fpath in fpaths]

    results = [
        {"audio": fname, "task": task, "candidate": cand}
        for fname, task, cand in zip(fnames, tasks, cands)
    ]

    for result in results:
        fname = result["audio"]
        task = result["task"]
        cand = result["candidate"]
        pylog.info(f"File '{fname}' with task '{task}':\n - '{cand}'")

    csv_export = args.csv_export
    if csv_export is not None:
        with open(csv_export, "w") as file:
            fieldnames = ["audio", "task", "candidate"]
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)


if __name__ == "__main__":
    main_predict()
