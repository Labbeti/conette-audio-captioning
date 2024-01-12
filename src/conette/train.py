#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

os.environ["MKL_NUM_THREADS"] = "2"
os.environ["NUMEXPR_NUM_THREADS"] = "2"
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

os.environ["TRANSFORMERS_OFFLINE"] = "TRUE"
os.environ["HF_HUB_OFFLINE"] = "TRUE"

import logging
import math
import os.path as osp
import sys
import time

from typing import Callable, Optional, Union

import colorlog
import hydra
import torch
import yaml

from hydra.utils import instantiate
from lightning_fabric.plugins.environments import LightningEnvironment
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.callbacks import (
    Callback,
    DeviceStatsMonitor,
    EarlyStopping,
    ModelCheckpoint,
    ModelSummary,
)
from transformers import logging as tfmers_logging

import conette

from conette.callbacks.aac_evaluator import AACEvaluator
from conette.callbacks.aac_validator import AACValidator
from conette.callbacks.debug import PrintDebug
from conette.callbacks.deepspeed import DeepSpeedCallback
from conette.callbacks.log import LogGCCallback, LogLRCallback, LogGradNorm, LogRngState
from conette.callbacks.resume import ResumeCallback
from conette.callbacks.stats_saver import StatsSaver
from conette.tokenization.aac_tokenizer import AACTokenizer
from conette.utils.custom_logger import CustomTensorboardLogger
from conette.utils.hydra import setup_resolvers, get_subrun_path, CustomFileHandler
from conette.utils.log_utils import set_loglevel
from conette.utils.misc import copy_slurm_logs, reset_seed


# Note: this function must be called globally
setup_resolvers()

pylog = logging.getLogger(__name__)


# Public functions
def setup_run(cfg: DictConfig) -> None:
    reset_seed(cfg.seed)
    OmegaConf.resolve(cfg)
    OmegaConf.set_readonly(cfg, True)

    # Print config
    subrun_path = get_subrun_path()
    if cfg.verbose >= 1:
        pylog.info(f"Subrun: {subrun_path}")
        pylog.info(f"Configuration:\n{OmegaConf.to_yaml(cfg, resolve=True)}")

    # Overwrite root logger formatter
    rank = os.getenv("SLURM_PROCID", 0)
    formatter = colorlog.ColoredFormatter(
        f"[%(purple)sRANK{rank}%(reset)s][%(cyan)s%(asctime)s%(reset)s][%(blue)s%(name)s%(reset)s][%(log_color)s%(levelname)s%(reset)s] - %(message)s",
        log_colors={
            "DEBUG": "purple",
            "INFO": "green",
            "WARNING": "yellow",
            "ERROR": "red",
            "CRITICAL": "bold_red",
        },
    )
    root_logger = logging.getLogger()
    handlers = root_logger.handlers
    for handler in handlers:
        handler.setFormatter(formatter)

    # Set log level in some packagers
    if cfg.debug or cfg.verbose >= 2:
        pkg_level = logging.DEBUG
        other_level = logging.WARNING
    elif cfg.verbose == 1:
        pkg_level = logging.INFO
        other_level = logging.ERROR
        tfmers_logging.set_verbosity_error()
    else:
        pkg_level = logging.WARNING
        other_level = logging.ERROR
        tfmers_logging.set_verbosity_error()

    set_loglevel(("sentence_transformers",), other_level)
    set_loglevel((conette,), pkg_level)
    pylog.setLevel(pkg_level)

    # Redirect PyTorch lightning outputs to a file
    os.makedirs(subrun_path, exist_ok=True)

    fpath_pl_outputs = osp.join(subrun_path, "logs", "lightning_outputs.log")
    handler = CustomFileHandler(fpath_pl_outputs)
    pl_pylog = logging.getLogger("pytorch_lightning")
    pl_pylog.addHandler(handler)

    # Set data dir for torch
    if cfg.path.torch_hub is not None:
        torch_hub_path = osp.expandvars(cfg.path.torch_hub)
        torch.hub.set_dir(torch_hub_path)

    # Set PyTorch sharing strategy if needed
    if cfg.sharing_strategy is not None:
        sharing_strategies = torch.multiprocessing.get_all_sharing_strategies()
        if cfg.sharing_strategy not in sharing_strategies:
            raise ValueError(
                f"Invalid argument {cfg.sharing_strategy=}. (expected one of {tuple(sharing_strategies)})"
            )
        if cfg.verbose >= 1:
            pylog.info(f"Set sharing strategy to {cfg.sharing_strategy}.")
        torch.multiprocessing.set_sharing_strategy(cfg.sharing_strategy)

    # Print debug info
    if cfg.verbose >= 2:
        overrides_fpath = osp.join(subrun_path, "hydra", "overrides.yaml")
        if osp.isfile(overrides_fpath):
            with open(overrides_fpath, "r") as file:
                overrides = yaml.safe_load(file)
            pylog.info(f"Overrides:\n{yaml.dump(overrides, sort_keys=False)}")

        varnames = (
            "SLURM_JOB_NAME",
            "SLURM_JOB_ID",
            "SLURM_NTASKS",
            "SLURM_PROCID",
            "SLURM_LOCALID",
            "SLURM_NODEID",
        )
        values = {}
        for name in varnames:
            values[name] = os.environ.get(name, None)
        pylog.debug(f"Env variables:\n{yaml.dump(values, sort_keys=True)}")


def teardown_run(cfg: DictConfig, run_start: float, run_end: float) -> None:
    total_duration_s = run_end - run_start
    total_duration_h = math.floor(total_duration_s / 3600.0)

    subrun_path = get_subrun_path()
    if cfg.verbose >= 1:
        total_duration_m = (total_duration_s / 60) % 60
        pylog.info(
            f"Results are saved in '{subrun_path}' in {total_duration_h:.0f}h{total_duration_m:02.0f}m."
        )

    fpaths = [
        cfg.get("slurm", {}).get("output", None),
        cfg.get("slurm", {}).get("error", None),
    ]
    copy_slurm_logs(fpaths, subrun_path)


def load_callbacks(
    cfg: DictConfig,
    tokenizers: dict[str, Optional[AACTokenizer]],
    datamodule: LightningDataModule,
    pl_module: LightningModule,
) -> dict[str, Callback]:
    callbacks = {}

    resume_callback = ResumeCallback(
        resume=cfg.resume,
        strict=cfg.strict_resume,
        ign_weights=cfg.ign_weights,
        verbose=cfg.verbose,
    )

    if cfg.resume_before_setup:
        resume_callback.load_checkpoint(pl_module)

    callbacks["resume"] = resume_callback

    # Add callback to stop training if monitor is NaN
    early_stop_callback = EarlyStopping(
        check_finite=True,
        mode=cfg.ckpts[0].mode,
        monitor=cfg.ckpts[0].monitor,
        patience=sys.maxsize,
    )
    callbacks["early_stop"] = early_stop_callback

    print_debug = PrintDebug(cfg.verbose)
    callbacks["print_debug"] = print_debug

    # Add Evaluator for compute test metrics scores at the end of the training (when trainer.test is called)
    evaluator = instantiate(
        cfg.evaluator,
        test_tokenizer=tokenizers["test_tokenizer"],
        verbose=cfg.verbose,
    )
    callbacks["evaluator"] = evaluator

    if hasattr(cfg.dm, "bsize"):
        bsize = cfg.dm.bsize
    else:
        if cfg.verbose >= 0:
            pylog.warning("Cannot detect batch size from data conf.")
        bsize = None

    log_lr = LogLRCallback(bsize=bsize)
    callbacks["log_lr"] = log_lr

    log_grad_norm = LogGradNorm(bsize=bsize)
    callbacks["log_grad_norm"] = log_grad_norm

    if cfg.debug:
        log_rng_state = LogRngState(bsize=bsize)
        callbacks["log_rng_state"] = log_rng_state

        log_gc = LogGCCallback(bsize=bsize)
        callbacks["log_gc"] = log_gc

    subrun_path = get_subrun_path()
    stats_saver = StatsSaver(
        subrun_path=subrun_path,
        on_end="none",
        tokenizers=tokenizers,
        git_hash=cfg.git_hash,
        cfg=cfg,
        verbose=cfg.verbose,
    )
    callbacks["stats_saver"] = stats_saver

    if "swa" in cfg.testing.run:
        if datamodule is not None:
            datamodule.setup("fit")
        pl_module.setup("fit")

        swa_callback = instantiate(cfg.testing.swa)
        callbacks["swa"] = swa_callback

    if "ema" in cfg.testing.run:
        ema_callback = instantiate(cfg.testing.ema)
        callbacks["ema"] = ema_callback

    if cfg.debug:
        device_stats_monitor = DeviceStatsMonitor()
        callbacks["device_stats_monitor"] = device_stats_monitor

    if cfg.debug or cfg.verbose >= 1:
        max_depth = 20
    elif cfg.verbose == 1:
        max_depth = 1
    else:
        max_depth = 0

    model_summary = ModelSummary(max_depth=max_depth)
    callbacks["model_summary"] = model_summary

    if cfg.enable_dspeed:
        deepspeed = DeepSpeedCallback(verbose=cfg.verbose)
        callbacks["deepspeed"] = deepspeed

    monitors = [ckpt_cfg.monitor for ckpt_cfg in cfg.ckpts]
    validator = AACValidator(monitors, cfg.val_metrics_keys)
    callbacks["validator"] = validator

    if cfg.trainer.enable_checkpointing:
        ckpts = instantiate(cfg.ckpts)
        for i, ckpt in enumerate(ckpts):
            callbacks[f"ckpt.{i}"] = ckpt

    callbacks = {
        name: callback for name, callback in callbacks.items() if callback is not None
    }
    return callbacks


def test_after_fit(
    cfg: DictConfig,
    datamodule: LightningDataModule,
    pl_module: LightningModule,
    evaluator: AACEvaluator,
    callbacks: dict[str, Callback],
    trainer: Trainer,
) -> None:
    testing_run = cfg.testing.run
    if isinstance(testing_run, str):
        testing_run = [testing_run]
    else:
        testing_run = list(testing_run)

    if "last" in testing_run:
        if cfg.verbose >= 1:
            pylog.info("Test using last model...")

        evaluator.set_model_name("last")
        trainer.test(pl_module, datamodule=datamodule, verbose=cfg.verbose >= 3)
        trainer.predict(pl_module, datamodule=datamodule)

    if "swa" in testing_run:
        if cfg.verbose >= 1:
            pylog.info("Using SWA weights for testing...")

        evaluator.set_model_name("swa")
        trainer.test(pl_module, datamodule=datamodule, verbose=cfg.verbose >= 3)
        trainer.predict(pl_module, datamodule=datamodule)

    if "best" in testing_run:
        ckpts = trainer.checkpoint_callbacks
        n_tests_done = 0

        for ckpt in ckpts:
            if not isinstance(ckpt, ModelCheckpoint) or ckpt.best_model_path == "":
                continue

            if cfg.verbose >= 1:
                pylog.info(
                    f"Test using best model file '{osp.basename(ckpt.best_model_path)}'..."
                )
            ckpt_data = torch.load(
                ckpt.best_model_path,
                map_location=pl_module.device,
            )
            pl_module.load_state_dict(ckpt_data["state_dict"])

            if ckpt.monitor is not None:
                monitor = ckpt.monitor  # type: ignore
                monitor = monitor[
                    monitor.rfind("/") + 1 :
                ]  # ex: "val/fense" -> "fense"
                model_name = f"best_{monitor}"
            else:
                model_name = "best"

            evaluator.set_model_name(model_name)
            trainer.test(pl_module, datamodule=datamodule, verbose=cfg.verbose >= 3)
            trainer.predict(pl_module, datamodule=datamodule)
            n_tests_done += 1

        if n_tests_done == 0:
            if "last" not in cfg.testing.run:
                pylog.warning(
                    "Cannot find best checkpoint callback, but testing will be done using last weights."
                )
                evaluator.set_model_name("last")
                trainer.test(pl_module, datamodule=datamodule, verbose=cfg.verbose >= 3)
                trainer.predict(pl_module, datamodule=datamodule)

            else:
                pylog.error("Cannot find best checkpoint callback.")


@hydra.main(
    version_base=None,
    config_path=osp.join("..", "conf"),
    config_name="train",
)
def main_train(cfg: DictConfig) -> Union[None, float]:
    """Train a model on data."""
    run_start = time.perf_counter()

    # --- 1/6 - Set seed, init loggers, print config.
    setup_run(cfg)

    # --- 2/6 - Build transforms & tokenizers
    audio_tfms_cfgs = {
        "train_audio_tfm": cfg.audio_t.train,
        "val_audio_tfm": cfg.audio_t.val,
        "test_audio_tfm": cfg.audio_t.test,
    }
    audio_tfms = {
        name: instantiate(trans_cfg) for name, trans_cfg in audio_tfms_cfgs.items()
    }
    audio_tfms: dict[str, Callable] = {
        name: trans for name, trans in audio_tfms.items() if trans is not None
    }

    train_tokenizers_cfgs = {
        "train_tokenizer": cfg.train_tok,
    }
    train_tokenizers = {
        name: instantiate(tok_cfg) for name, tok_cfg in train_tokenizers_cfgs.items()
    }
    train_tokenizers = {
        name: tokenizer
        for name, tokenizer in train_tokenizers.items()
        if tokenizer is not None
    }

    test_tokenizer = instantiate(cfg.test_tok)
    test_tokenizers = {"test_tokenizer": test_tokenizer}
    tokenizers = train_tokenizers | test_tokenizers

    # --- 3/6 - Build pytorch lightning modules & callbacks
    datamodule = instantiate(cfg.dm, **audio_tfms, **train_tokenizers)
    pl_module = instantiate(cfg.pl, **train_tokenizers)

    # Callbacks
    pl_loggers = []

    tb_logger = instantiate(cfg.logger)
    pl_loggers.append(tb_logger)

    callbacks = load_callbacks(cfg, tokenizers, datamodule, pl_module)

    # --- 4/6 - Build Trainer & run it
    fit_trainer: Trainer = instantiate(
        cfg.trainer,
        logger=pl_loggers,
        callbacks=list(callbacks.values()),
    )

    eval_trainer: Optional[Trainer]
    if fit_trainer.num_devices == 1:
        eval_trainer = fit_trainer
    elif fit_trainer.is_global_zero:
        eval_trainer = instantiate(
            cfg.trainer,
            logger=pl_loggers,
            num_nodes=1,
            devices=1,
            callbacks=list(callbacks.values()),
            plugins=LightningEnvironment(),
            strategy=None,
        )
    else:
        eval_trainer = None

    if cfg.trainer.auto_scale_batch_size is not None:
        # auto_scale_batch_size: None | "power" | "binsearch"
        if cfg.verbose >= 1:
            pylog.info(
                f"Start tuning batch size with mode={cfg.trainer.auto_scale_batch_size}..."
            )

        if not hasattr(datamodule, "TUNE_MODE"):
            raise ValueError("DM does not have 'TUNE_MODE' global param.")

        datamodule.TUNE_MODE = True
        # Setup dm et plm because tuner needs model to be built on start
        datamodule.setup("fit")
        pl_module.setup("fit")

        fit_trainer.tune(
            pl_module,
            datamodule=datamodule,
            scale_batch_size_kwargs=dict(init_val=8, batch_arg_name="_bsize"),
        )
        return None

    # Validate & test before fit
    evaluator: Optional[AACEvaluator] = callbacks.get("evaluator")  # type: ignore

    if (
        eval_trainer is not None
        and evaluator is not None
        and fit_trainer.max_epochs is not None
        and fit_trainer.max_epochs > 0
        and (
            fit_trainer.limit_train_batches is None
            or fit_trainer.limit_train_batches > 0
        )
    ):
        pylog.debug(f"Fit trainer = eval trainer? {fit_trainer is eval_trainer}")

        if cfg.val_on_start:
            pylog.debug("Validate on start...")
            eval_trainer.validate(pl_module, datamodule=datamodule, verbose=False)
            pylog.debug("Validate on start done.")

        if cfg.test_on_start and (cfg.resume is not None or cfg.resume_2 is not None):
            pylog.debug("Test on start...")
            evaluator.set_model_name("start")
            eval_trainer.test(pl_module, datamodule=datamodule, verbose=False)
            evaluator.set_model_name("unk")
            pylog.debug("Test on start done.")

    # Main training
    pylog.debug("Fit model...")
    fit_trainer.fit(pl_module, datamodule=datamodule)
    pylog.debug("Fit model done.")

    # --- 5/6 - Test checkpoints
    # Destroy group for testing on rank 0 after fit when using DDP
    if fit_trainer.num_devices > 1:
        torch.distributed.destroy_process_group()  # type: ignore

    if evaluator is not None and eval_trainer is not None:
        pylog.debug(f"Test after fit... (testing={cfg.testing.run})")
        test_after_fit(cfg, datamodule, pl_module, evaluator, callbacks, eval_trainer)
        pylog.debug(f"Test after fit done. (testing={cfg.testing.run})")
    else:
        pylog.info("Skip testing after fit.")

    # --- 6/6 - Close files and clean objects
    run_end = time.perf_counter()
    total_duration_s = run_end - run_start
    total_duration_h = total_duration_s / 3600.0

    stats_saver = callbacks.get("stats_saver")
    if isinstance(stats_saver, StatsSaver) and eval_trainer is not None:
        stats_saver.save_metrics_stats(
            eval_trainer,
            pl_module,
            datamodule,
            add_metrics=dict(total_duration_h=total_duration_h),
        )

    if cfg.out_crit is not None and isinstance(tb_logger, CustomTensorboardLogger):
        out = tb_logger.metrics.get(cfg.out_crit, cfg.out_default)
        if cfg.verbose >= 1:
            pylog.info(f"Training is finished with {cfg.out_crit}={out}.")
    else:
        out = cfg.out_default

    teardown_run(cfg, run_start, run_end)
    return out


if __name__ == "__main__":
    main_train()
