#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging

from typing import Any

from deepspeed.profiling.flops_profiler import get_model_profile
from pytorch_lightning import LightningModule
from pytorch_lightning.callbacks.callback import Callback
from torch import Tensor

from conette.nn.functional.misc import move_to_rec
from conette.utils.csum import csum_any


pylog = logging.getLogger(__name__)


class DeepSpeedCallback(Callback):
    def __init__(self, single_input: bool = False, verbose: int = 0) -> None:
        super().__init__()
        self._single_input = single_input
        self._verbose = verbose
        self._metrics = {}

    def state_dict(self) -> dict[str, Any]:
        return self._metrics

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self._metrics |= state_dict

    def on_fit_start(self, trainer, pl_module: LightningModule) -> None:
        return self.profile(pl_module)

    def profile(self, pl_module: LightningModule) -> None:
        example = pl_module.example_input_array

        if self._verbose >= 2:
            csum = csum_any(example)
            pylog.debug(f"Batch example csum: {csum}")

        if isinstance(example, dict):
            # Assuming that arguments are in the correct order

            if self._single_input:
                batch = example["batch"]
                single_batch = {}

                for k, v in batch.items():
                    if isinstance(v, Tensor):
                        v = v[0][None]
                    elif isinstance(v, (list,)):
                        v = [v[0]]
                    elif isinstance(v, (tuple,)):
                        v = (v[0],)
                    else:
                        raise TypeError(
                            f"Invalid item in batch. (found {type(v)} with {v=})"
                        )
                    single_batch[k] = v

                example["batch"] = single_batch
                bsize = 1

            else:
                audio = example.get("batch", {}).get("audio")
                if audio is None:
                    bsize = -1
                else:
                    bsize = len(audio)

            example = move_to_rec(example, device=pl_module.device)
            outputs: tuple[int, int, int] = get_model_profile(  # type: ignore
                pl_module,
                kwargs=example,
                print_profile=self._verbose >= 2,
                as_string=False,
            )
            flops, macs, params = outputs

            if bsize != -1:
                flops_per_sample = flops / bsize
                macs_per_sample = macs / bsize
            else:
                flops_per_sample = -1
                macs_per_sample = -1

            if self._verbose >= 1:
                pylog.info("According to deepspeed, model has:")
                pylog.info(f"- {params} parameters")

                if flops_per_sample == -1:
                    pylog.info(f"- {flops} FLOPs (with unknown bsize)")
                else:
                    pylog.info(
                        f"- {flops_per_sample} FLOPs (based on {flops=} with {bsize=})"
                    )

                if macs_per_sample == -1:
                    pylog.info(f"- {macs} MACs (with unknown bsize)")
                else:
                    pylog.info(
                        f"- {macs_per_sample} MACs (based on {macs=} with {bsize=})"
                    )

            metrics = {
                "other/dspeed_flops": flops,
                "other/dspeed_macs": macs,
                "other/dspeed_params": params,
                "other/dspeed_bsize": bsize,
                "other/dspeed_flops_per_sample": flops_per_sample,
                "other/dpseed_macs_per_sample": macs_per_sample,
            }
        else:
            metrics = {}
            if self._verbose >= 0:
                pylog.warning(
                    f"Unsupported example type {type(example)}. (expected dict)"
                )
            return None

        self._metrics |= metrics
        for pl_logger in pl_module.loggers:
            pl_logger.log_hyperparams({}, metrics=self._metrics)

    def get_metrics(self) -> dict[str, Any]:
        return self._metrics
