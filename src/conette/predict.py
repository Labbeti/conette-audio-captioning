#!/usr/bin/env python
# -*- coding: utf-8 -*-

from argparse import ArgumentParser, Namespace

import yaml

from conette.nn.huggingface import CoNeTTEConfig, CoNeTTEModel
from conette.utils.cmdline import _str_to_opt_str


def get_predict_args() -> Namespace:
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
    args = parser.parse_args()
    return args


def main_predict() -> None:
    args = get_predict_args()
    fpaths = list(args.audio)

    config = CoNeTTEConfig.from_pretrained("Labbeti/conette")
    model: CoNeTTEModel = CoNeTTEModel.from_pretrained("Labbeti/conette", config=config, device=args.device, token=args.token)  # type: ignore

    outputs = model(fpaths)
    cands = outputs["cands"]

    results = dict(zip(fpaths, cands))
    print(yaml.dump(results))


if __name__ == "__main__":
    main_predict()
