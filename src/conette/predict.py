#!/usr/bin/env python
# -*- coding: utf-8 -*-

import csv
import logging
import os.path as osp

from argparse import ArgumentParser, Namespace

from lightning_fabric.utilities.seed import seed_everything

from conette.huggingface.model import CoNeTTEConfig, CoNeTTEModel
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
        type=str,
        help="Model name on huggingface.",
        default="Labbeti/conette",
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


def main_predict() -> None:
    """Main entrypoint for CoNeTTE predict."""
    args = get_predict_args()
    _setup_logging("conette", verbose=args.verbose, set_format=False)
    seed_everything(args.seed)

    fpaths = list(args.audio)
    tasks = args.task

    if args.verbose >= 1:
        pylog.info(f"Initilizing '{args.model_name}' model...")

    config = CoNeTTEConfig.from_pretrained(
        args.model_name,
        token=args.token,
    )
    hf_model: CoNeTTEModel = CoNeTTEModel.from_pretrained(  # type: ignore
        args.model_name,
        config=config,
        device=args.device,
        token=args.token,
    )
    hf_model.eval_and_detach()

    if args.verbose >= 1:
        pylog.info(f"Model '{args.model_name}' is initialized.")

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
