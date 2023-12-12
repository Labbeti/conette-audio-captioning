#!/usr/bin/env python
# -*- coding: utf-8 -*-

import datetime
import inspect
import logging
import os
import os.path as osp
import re
import shutil
import subprocess
import zipfile

from pathlib import Path
from subprocess import CalledProcessError
from typing import (
    Any,
    Callable,
    Iterable,
    Optional,
    TypeVar,
    Union,
)
from zipfile import ZipFile

import torch
import tqdm

from pytorch_lightning.utilities.seed import seed_everything


pylog = logging.getLogger(__name__)
T = TypeVar("T")


def get_none() -> None:
    # Returns None. Can be used for hydra instantiations.
    return None


def get_datetime(fmt: str = "%Y.%m.%d-%H.%M.%S") -> str:
    now = datetime.datetime.now()
    return now.strftime(fmt)


def reset_seed(seed: Optional[int]) -> Optional[int]:
    """Reset the seed of following packages for reproductibility :
    - random
    - numpy
    - torch
    - torch.cuda

    Also set deterministic behaviour for cudnn backend.

    :param seed: The seed to set. If None, this function does nothing.
    """
    if seed is not None and not isinstance(seed, int):
        raise TypeError(
            f"Invalid argument type {type(seed)=}. (expected NoneType or int)"
        )

    if seed is None:
        return seed

    seed = seed_everything(seed, workers=True)
    torch.backends.cudnn.benchmark = False  # type: ignore
    torch.backends.cudnn.deterministic = True  # type: ignore
    return seed


def save_conda_env(fpath: str, conda_path: str = "conda", verbose: int = 1) -> bool:
    try:
        cmd = [conda_path, "env", "export", "-f", fpath]
        _output = subprocess.check_output(cmd)
        return True
    except (CalledProcessError, PermissionError, FileNotFoundError) as err:
        if verbose >= 0:
            pylog.warning(f"Cannot save conda env in {fpath}. ({err=})")
        return False


def save_micromamba_env(
    fpath: str, micromamba_path: str = "micromamba", verbose: int = 1
) -> bool:
    try:
        cmd = [micromamba_path, "env", "export"]
        output = subprocess.check_output(cmd)
        output = output.decode()
        with open(fpath, "w") as file:
            file.writelines([output])
        return True
    except (CalledProcessError, PermissionError, FileNotFoundError) as err:
        if verbose >= 0:
            pylog.warning(f"Cannot save micromamba env in {fpath}. ({err=})")
        return False


def get_current_git_hash(
    cwd: str = osp.dirname(__file__),
    default: T = "UNKNOWN",
) -> Union[str, T]:
    """
    Return the current git hash in the current directory.

    :returns: The git hash. If an error occurs, returns 'UNKNOWN'.
    """
    try:
        git_hash = subprocess.check_output("git describe --always".split(" "), cwd=cwd)
        git_hash = git_hash.decode("UTF-8").replace("\n", "")
        return git_hash
    except (CalledProcessError, PermissionError) as err:
        pylog.warning(
            f"Cannot get current git hash from {cwd=}. (error message: '{err}')"
        )
        return default


def get_tags_version(cwd: str = osp.dirname(__file__)) -> str:
    """
    {LAST_TAG}-{NB_COMMIT_AFTER_LAST_TAG}-g{LAST_COMMIT_HASH}
    Example : v0.1.1-119-g40317c7

    :returns: The tag version with the git hash.
    """
    try:
        git_hash = subprocess.check_output("git describe --tags".split(" "), cwd=cwd)
        git_hash = git_hash.decode("UTF-8").replace("\n", "")
        return git_hash
    except (subprocess.CalledProcessError, PermissionError):
        return "UNKNOWN"


def get_obj_clsname(obj: Any) -> str:
    """Returns the full class name of an object."""
    class_ = obj.__class__
    module = class_.__module__
    if module == "builtins":
        return class_.__qualname__  # avoid outputs like 'builtins.str'
    return module + "." + class_.__qualname__


def save_code_to_zip(
    logdir: str,
    zip_fname: str = "source_code.zip",
    compression: int = zipfile.ZIP_LZMA,
    compresslevel: int = 1,
    verbose: int = 0,
) -> None:
    logdir = osp.expandvars(logdir)
    zip_fpath = osp.join(logdir, zip_fname)
    code_root_dpath = Path(__file__).parent.parent.parent

    suffixes_dnames = (
        ".ipynb_checkpoints",
        "old",
        "ign",
        "__pycache__",
        "/logs",
        "/data",
        ".egg-info",
    )

    include_fnames = [
        r".*\." + ext
        for ext in ("py", "yaml", "rst", "md", "sh", "txt", "cfg", "ini", "in")
    ]
    exclude_fnames = (r".*(\.ign|\.old|_ign|_old|\.egg-info).*",)

    include_fnames = list(map(re.compile, include_fnames))
    exclude_fnames = list(map(re.compile, exclude_fnames))

    tgt_fpaths = []
    for root, directories, files in tqdm.tqdm(
        os.walk(code_root_dpath),
        disable=verbose <= 1,
        desc="Searching files to save...",
    ):
        if any(root.endswith(suffix) for suffix in suffixes_dnames):
            directories[:] = []
            continue
        tgt_fnames = [
            fname
            for fname in files
            if any(re.match(p, fname) for p in include_fnames)
            and all(not re.match(p, fname) for p in exclude_fnames)
        ]
        if verbose >= 2 and len(tgt_fnames) > 0:
            pylog.debug(
                f"{root=} with {len(tgt_fnames)} python files. (ex={tgt_fnames[0]})"
            )
        tgt_fpaths += [osp.join(root, fname) for fname in tgt_fnames]

    with ZipFile(
        zip_fpath, "w", compression=compression, compresslevel=compresslevel
    ) as zfile:
        for fpath in tqdm.tqdm(
            tgt_fpaths, disable=verbose <= 1, desc=f"Writing {len(tgt_fpaths)} files..."
        ):
            zfile.write(fpath, arcname=osp.relpath(fpath, code_root_dpath))


def copy_slurm_logs(
    fpaths: Iterable[Optional[str]],
    subrun_dpath: Optional[str],
) -> None:
    if subrun_dpath is None:
        return None
    if "SLURM_JOB_ID" not in os.environ:
        return None

    subrun_dpath = osp.expandvars(subrun_dpath)
    fpaths = [fpath for fpath in fpaths if fpath is not None]

    job_id = os.environ["SLURM_JOB_ID"]
    replaces = {
        "%j": job_id,
        "%A": job_id,
    }
    for pattern, value in replaces.items():
        fpaths = [fpath.replace(pattern, value) for fpath in fpaths]
    fpaths = [fpath for fpath in fpaths if osp.isfile(fpath)]

    if len(fpaths) == 0:
        return None

    tgt_dpath = osp.join(subrun_dpath, "logs")
    os.makedirs(tgt_dpath, exist_ok=True)

    for fpath in fpaths:
        fname = osp.basename(fpath)
        tgt_fpath = osp.join(tgt_dpath, fname)
        shutil.copyfile(fpath, tgt_fpath)


def pass_filter(
    name: str,
    include: Optional[Iterable[str]] = None,
    exclude: Optional[Iterable[str]] = None,
) -> bool:
    """Returns True if name in include set and not in exclude set."""
    if include is not None and exclude is not None:
        return (name in include) and (name not in exclude)
    if include is not None:
        return name in include
    elif exclude is not None:
        return name not in exclude
    else:
        return True


def compose(*fns: Callable[[Any], Any]) -> Callable[[Any], Any]:
    def compose_impl(x):
        for fn in fns:
            x = fn(x)
        return x

    return compose_impl
