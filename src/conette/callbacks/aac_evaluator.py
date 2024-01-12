#!/usr/bin/env python
# -*- coding: utf-8 -*-

import csv
import logging
import os
import os.path as osp
import tempfile
import time

from typing import Any, Optional, Union

import torch
import yaml

from pytorch_lightning import LightningModule
from pytorch_lightning.callbacks.callback import Callback
from pytorch_lightning.loggers import TensorBoardLogger
from torch import Tensor
from torch.utils.data.dataloader import DataLoader

from aac_metrics.utils.checks import is_mono_sents, is_mult_sents
from aac_metrics.utils.collections import flat_list, unflat_list

from conette.metrics.classes.all_metrics import AllMetrics
from conette.nn.functional.misc import move_to_rec
from conette.tokenization.aac_tokenizer import AACTokenizer
from conette.utils.collections import all_eq
from conette.utils.custom_logger import CustomTensorboardLogger
from conette.utils.dcase import export_to_dcase_task6a_csv
from conette.utils.log_utils import warn_once


pylog = logging.getLogger(__name__)


class AACEvaluator(Callback):
    """Callback which stores candidates and references during testing to produce AAC scores.

    Include metrics : BLEU1, BLEU2, BLEU3, BLEU4, METEOR, ROUGE-L, CIDEr, SPICE, SPIDEr.
    """

    CANDS_PREFIX = "cands"
    MREFS_KEY = "mrefs"

    def __init__(
        self,
        subrun_path: Optional[str],
        test_tokenizer: AACTokenizer,
        cache_path: str = "~/.cache",
        java_path: str = "java",
        tmp_path: str = tempfile.gettempdir(),
        ckpt_name: str = "unk",
        verbose: int = 1,
        debug: bool = False,
        save_to_csv: bool = True,
        save_dcase_csv_file: bool = False,
        metric_device: Union[str, torch.device, None] = None,
        cpus: Optional[int] = None,
    ) -> None:
        if subrun_path is not None:
            subrun_path = osp.expandvars(subrun_path)

        super().__init__()
        self._subrun_dir = subrun_path
        self._test_tokenizer = test_tokenizer
        self._cache_path = cache_path
        self._java_path = java_path
        self._tmp_path = tmp_path
        self._model_name = ckpt_name
        self._verbose = verbose
        self._debug = debug
        self._save_to_csv = save_to_csv
        self._save_dcase_csv_file = save_dcase_csv_file
        self._metric_device = metric_device
        self._cpus = cpus

        self._all_outputs: dict[int, dict[str, Any]] = {}

        # Note : we avoid compute scores for
        # - AudioCaps/train because it is too large
        # - Clotho/test because it does not have any references
        # - Clotho/anasysis because it does not have any references
        self._excluded_datasubsets_metrics = (
            "audiocaps_train",
            "clotho_test",
            "clotho_analysis",
        )
        self._all_metrics = None

    # Callback methods
    def on_predict_epoch_start(self, trainer, pl_module) -> None:
        self._all_outputs = {}
        if self._verbose >= 1:
            pylog.debug(f"Starting PREDICT epoch with model_name='{self._model_name}'")

    def on_predict_batch_end(
        self,
        trainer,
        pl_module,
        outputs: dict[str, Any],
        batch: dict[str, Any],
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        return self.on_test_batch_end(
            trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
        )

    def on_predict_epoch_end(
        self,
        trainer,
        pl_module: LightningModule,
        outputs,
    ) -> None:
        if not self._save_to_csv:
            return None

        for outputs_ in self._all_outputs.values():
            datasubset = _get_datasubset_name(outputs_)

            if self._subrun_dir is not None and osp.isdir(self._subrun_dir):
                self._save_outputs_to_csv(
                    self._subrun_dir,
                    datasubset,
                    outputs_,
                    {},
                )
            else:
                pylog.error(
                    f"Cannot save outputs to CSV because logdir is not a valid directory. (logdir={self._subrun_dir}, {datasubset=})"
                )

    def on_test_start(self, trainer, pl_module: LightningModule) -> None:
        if self._all_metrics is None:
            if self._metric_device is not None:
                device = self._metric_device
            else:
                device = pl_module.device

            if hasattr(pl_module, "tokenizer") and isinstance(
                pl_module.tokenizer, AACTokenizer
            ):
                train_vocab = pl_module.tokenizer.get_vocab()
            else:
                train_vocab = None

            self._all_metrics = AllMetrics(
                preprocess=False,
                device=device,
                cache_path=self._cache_path,
                java_path=self._java_path,
                tmp_path=self._tmp_path,
                meteor_java_max_memory="2G",
                spice_java_max_memory="8G",
                spice_n_threads=self._cpus,
                spice_timeout=[3600],
                train_vocab=train_vocab,
                verbose=self._verbose,
            )

            if self._verbose >= 2:
                pylog.debug(f"{len(self._all_metrics)} metrics has been initialized.")

                datamodule = trainer.datamodule  # type: ignore
                if datamodule is not None:
                    test_loaders = datamodule.test_dataloader()
                else:
                    test_loaders = []

                assert isinstance(test_loaders, list) and all(
                    isinstance(loader, DataLoader) for loader in test_loaders
                )
                sizes = tuple(map(len, test_loaders))
                pylog.debug(f"Test loader sizes: {sizes}")

    def on_test_epoch_start(self, trainer, pl_module) -> None:
        self._all_outputs = {}
        if self._verbose >= 1:
            pylog.debug(f"Starting TEST epoch with model_name='{self._model_name}'")

    def on_test_batch_end(
        self,
        trainer,
        pl_module,
        outputs: dict[str, Any],
        batch: dict[str, Any],
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        if outputs is None:
            warn_once("Lightning module has returned None during test epoch.", pylog)
            return None

        outputs = move_to_rec(outputs, device=torch.device("cpu"))

        if dataloader_idx not in self._all_outputs.keys():
            self._all_outputs[dataloader_idx] = {}

        for key, batch_values in outputs.items():
            if key not in self._all_outputs[dataloader_idx].keys():
                self._all_outputs[dataloader_idx][key] = []
            self._all_outputs[dataloader_idx][key] += list(batch_values)

        for key in ("fname", "index", "dataset", "subset"):
            if key not in batch.keys():
                raise ValueError(f"Cannot find {key=} in batch.")
            if key not in self._all_outputs[dataloader_idx].keys():
                self._all_outputs[dataloader_idx][key] = []
            self._all_outputs[dataloader_idx][key] += batch[key]

    def on_test_epoch_end(self, trainer, pl_module: LightningModule) -> None:
        datasubsets = []
        for outputs in self._all_outputs.values():
            # Sanity check
            n_items = len(next(iter(outputs.values())))
            invalid_sizes_keys = [
                key for key, values in outputs.items() if len(values) != n_items
            ]
            if len(invalid_sizes_keys) > 0:
                sizes = [len(outputs[key]) for key in invalid_sizes_keys]
                raise RuntimeError(
                    f"Invalid number of values for keys={invalid_sizes_keys} (expected {n_items} but found {sizes=})."
                )

            datasubset = _get_datasubset_name(outputs)
            counter = datasubsets.count(datasubset)
            if counter > 0:
                old_datasubset = datasubset
                datasubset = f"{datasubset}_{counter+1}"
                pylog.error(
                    f"Found duplicated subset '{old_datasubset}'. Renaming to '{datasubset}'."
                )
                assert datasubset not in datasubsets
            datasubsets.append(datasubset)

            # Tokenize candidates and references
            sents_keys = [
                key
                for key in outputs.keys()
                if key.startswith(self.CANDS_PREFIX) or key == self.MREFS_KEY
            ]

            if self._verbose >= 2:
                pylog.debug(
                    f"Process sentences with tokenizer... ({tuple(sents_keys)=}"
                )

            for key in sents_keys:
                raw_sents = outputs[key]

                if is_mono_sents(raw_sents):
                    sents = self._test_tokenizer.tokenize_batch(raw_sents)
                    sents = self._test_tokenizer.detokenize_batch(sents)

                elif is_mult_sents(raw_sents):
                    flat_raw_sents, sizes = flat_list(raw_sents)
                    flat_sents = self._test_tokenizer.tokenize_batch(flat_raw_sents)
                    flat_sents = self._test_tokenizer.detokenize_batch(flat_sents)
                    sents = unflat_list(flat_sents, sizes)

                else:
                    raise TypeError(f"Cannot detect sentences type. (with {key=})")

                outputs[key] = sents

            if self._verbose >= 2:
                pylog.debug(f"Sentences processed. ({tuple(sents_keys)=})")

            sents_scores = {}
            if datasubset not in self._excluded_datasubsets_metrics:
                with torch.inference_mode():
                    corpus_scores, sents_scores = self._compute_metrics(
                        outputs, datasubset
                    )

                if self._verbose >= 1:
                    pylog.info(
                        f"Global scores for dataset {datasubset}:\n{yaml.dump(corpus_scores, sort_keys=False)}"
                    )
                self._log_global_scores(corpus_scores, datasubset, pl_module)
                if self._verbose >= 1:
                    self._print_example(outputs, datasubset, pl_module, sents_scores)
            else:
                pylog.debug(f"Skipping metrics for subset '{datasubset}'...")

            if self._save_to_csv:
                if self._subrun_dir is not None and osp.isdir(self._subrun_dir):
                    self._save_outputs_to_csv(
                        self._subrun_dir,
                        datasubset,
                        outputs,
                        sents_scores,
                    )
                else:
                    pylog.error(
                        f"Cannot save outputs to CSV because logdir is not a valid directory. (logdir={self._subrun_dir}, {datasubset=})"
                    )

        self._all_outputs = {}

    # AACEvaluator methods
    def set_model_name(self, model_name: str) -> None:
        self._model_name = model_name

    def _compute_metrics(
        self,
        outputs: dict[str, list],
        datasubset: str,
    ) -> tuple[dict[str, float], dict[str, list[float]]]:
        corpus_scores = {}
        sents_scores = {}

        if self._all_metrics is None:
            return corpus_scores, sents_scores

        start_time = time.perf_counter()

        pred_keys = [
            key
            for key, values in outputs.items()
            if key.startswith(self.CANDS_PREFIX)
            and isinstance(values, list)
            and all(isinstance(value, str) for value in values)
        ]
        all_mrefs = outputs[self.MREFS_KEY]

        if self._verbose >= 1:
            n_metrics = len(self._all_metrics)
            pylog.info(
                f"Start computing metrics... ({datasubset=}, n_preds={len(all_mrefs)}, n_preds_types={len(pred_keys)}, {n_metrics=})"
            )

        for pred_key in pred_keys:
            all_cands = outputs[pred_key]

            if self._verbose >= 1:
                pylog.debug(
                    f"Computing sentence level metrics... ({datasubset=}, {pred_key=})"
                )

            pred_global_scores, pred_sents_scores = self._all_metrics(
                all_cands,
                all_mrefs,
            )
            corpus_scores |= {
                f"{self._model_name}.{pred_key}.{metric_name}": score
                for metric_name, score in pred_global_scores.items()
            }
            sents_scores |= {
                f"{self._model_name}.{pred_key}.{metric_name}": scores
                for metric_name, scores in pred_sents_scores.items()
            }

        if self._verbose >= 1:
            end_time = time.perf_counter()
            duration_s = end_time - start_time
            pylog.info(
                f"Computing metrics finished in {duration_s:.2f}s. ({datasubset=})"
            )

        # Sanity check
        if __debug__:
            invalid_corpus_scores = tuple(
                [
                    name
                    for name, scores in corpus_scores.items()
                    if not isinstance(scores, Tensor) or scores.ndim != 0
                ]
            )
            invalid_sents_scores = tuple(
                [
                    name
                    for name, scores in sents_scores.items()
                    if not isinstance(scores, Tensor)
                    or scores.ndim != 1
                    or scores.shape[0] != len(all_mrefs)
                ]
            )
            if len(invalid_corpus_scores) > 0:
                raise ValueError(
                    f"Invalid global scores. (found {invalid_corpus_scores=})"
                )

            if len(invalid_sents_scores) > 0:
                raise ValueError(
                    f"Invalid local scores. (found {invalid_sents_scores=})"
                )

        corpus_scores = {name: score.item() for name, score in corpus_scores.items()}
        sents_scores = {name: scores.tolist() for name, scores in sents_scores.items()}

        return corpus_scores, sents_scores

    def _log_global_scores(
        self,
        corpus_scores: dict[str, float],
        datasubset: str,
        pl_module: LightningModule,
    ) -> None:
        global_scores_with_datasubset = {
            f"{datasubset}/{key}": score for key, score in corpus_scores.items()
        }
        for pl_logger in pl_module.loggers:
            if isinstance(pl_logger, CustomTensorboardLogger):
                pl_logger.log_hyperparams(
                    params={}, metrics=global_scores_with_datasubset
                )
                pl_logger.update_files()

    def _print_example(
        self,
        outputs: dict[str, list],
        datasubset: str,
        pl_module: LightningModule,
        sents_scores: dict[str, list[float]],
    ) -> None:
        assert self._test_tokenizer is not None
        n_outputs = len(outputs["fname"])
        indexes = torch.randint(0, n_outputs, (1,)).tolist()

        pylog.info(
            f"Show {len(indexes)} example(s) with model_name={self._model_name} : "
        )

        for idx in indexes:
            fname = outputs["fname"][idx]
            dset_index = outputs["index"][idx]
            candidates = {
                key: candidates_sents[idx]
                for key, candidates_sents in outputs.items()
                if key.startswith(self.CANDS_PREFIX)
            }
            mult_references = outputs[self.MREFS_KEY][idx]

            lines = "-" * 10
            width = 128

            local_main_metrics = {
                k: v[idx] for k, v in sents_scores.items() if "spider" in k
            }

            infos = {
                "datasubset": datasubset,
                "index": dset_index,
                "fname": fname,
            } | local_main_metrics

            pylog.info(
                f"\n"
                f"{lines}\nInfos\n{lines}\n{yaml.dump(infos, width=width, sort_keys=False)}"
                f"{lines}\nCandidates\n{lines}\n{yaml.dump(candidates, width=width, sort_keys=False)}"
                f"{lines}\nReferences\n{lines}\n{yaml.dump(mult_references, width=width, sort_keys=False)}"
            )

            # Log examples
            loggers = pl_module.loggers
            for logger in loggers:
                if isinstance(logger, TensorBoardLogger):
                    prefix = logger.name
                    logger.experiment.add_text(
                        f"{prefix}/{datasubset}_cands_{dset_index}",
                        yaml.dump(candidates, sort_keys=False),
                    )
                    logger.experiment.add_text(
                        f"{prefix}/{datasubset}_mrefs_{dset_index}",
                        yaml.dump(mult_references, sort_keys=False),
                    )

    def _save_outputs_to_csv(
        self,
        dpath: str,
        datasubset: str,
        outs: dict[str, list],
        sents_scores: dict[str, list[float]],
    ) -> None:
        # Sanity check
        lens = list(map(len, outs.values())) + list(map(len, sents_scores.values()))
        assert all_eq(lens), f"{lens=}"

        n_items = lens[0]
        csv_fname = f"{self._model_name}_outputs_{datasubset}.csv"
        csv_fpath = osp.join(dpath, csv_fname)

        def process(key: str, value: Any) -> Any:
            if isinstance(value, Tensor):
                return value.tolist()
            else:
                return value

        csv_all_values = outs | sents_scores

        with open(csv_fpath, "w") as file:
            keys = list(csv_all_values.keys())
            writer = csv.DictWriter(file, fieldnames=keys)
            writer.writeheader()

            for i in range(n_items):
                row = {key: values[i] for key, values in csv_all_values.items()}
                row = {key: process(key, value) for key, value in row.items()}
                writer.writerow(row)

        if self._save_dcase_csv_file:
            fnames = outs["fname"]
            mcands = {k: v for k, v in outs.items() if k.startswith(self.CANDS_PREFIX)}

            dcase_dpath = osp.join(dpath, "dcase")
            os.makedirs(dcase_dpath, exist_ok=True)

            for cands_name, cands in mcands.items():
                if len(mcands) == 1:
                    dcase_fname = (
                        f"submission_output_{self._model_name}_{datasubset}.csv"
                    )
                else:
                    dcase_fname = f"submission_output_{self._model_name}_{datasubset}_{cands_name}.csv"

                dcase_fpath = osp.join(dcase_dpath, dcase_fname)
                export_to_dcase_task6a_csv(dcase_fpath, fnames, cands)


def _get_datasubset_name(outputs: dict[str, Any]) -> str:
    datanames = list(sorted(set(map(str.lower, outputs["dataset"]))))
    subsets = list(sorted(set(map(str.lower, outputs["subset"]))))
    if len(datanames) == 1 and len(subsets) == 1:
        datasubset = f"{datanames[0]}_{subsets[0]}"
    else:
        datasubset = f"mix_{'_'.join(datanames)}_{'_'.join(subsets)}"
    return datasubset
