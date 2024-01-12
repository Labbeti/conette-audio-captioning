#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import os
import os.path as osp
import pickle

from functools import cache
from logging import FileHandler
from pathlib import Path
from typing import Any, Optional, Union

from hydra.core.hydra_config import HydraConfig
from hydra.types import RunMode
from omegaconf import DictConfig, OmegaConf
from omegaconf.errors import ConfigAttributeError

from conette.utils.collections import flat_dict_of_dict
from conette.utils.yaml_utils import load_yaml


pylog = logging.getLogger(__name__)


class CustomFileHandler(FileHandler):
    """FileHandler which builds intermediate directories.

    Used for export hydra logs to a file contained in a folder that does not exists yet at the start of the program.
    """

    def __init__(
        self,
        filename: str,
        mode: str = "a",
        encoding: Optional[str] = None,
        delay: bool = True,
    ) -> None:
        parent_dpath = osp.dirname(filename)
        if parent_dpath != "":
            try:
                os.makedirs(parent_dpath, exist_ok=True)
            except PermissionError:
                pass
        super().__init__(filename, mode, encoding, delay)


# Public functions
def setup_resolvers() -> None:
    """Prepare resolvers for hydra.

    This function should be called globally or before calling the function wrapped by hydra.main decorator.
    """
    resolvers = {
        "include_keys": include_keys_fn,
        "get_tag": get_tag_fn,
        "get_subtag": get_subtag_fn,
        "prod": lambda x, y: x * y,
    }
    for name, resolver in resolvers.items():
        if not OmegaConf.has_resolver(name):
            OmegaConf.register_new_resolver(name, resolver)


def include_keys_fn(prefix: str, _root_: DictConfig) -> list[str]:
    """Special function used by sweeper to determine the job override_dirname by including keys instead of excluding keys.

    To use it, you must register this function as resolver aat the beginning of your program (BEFORE build your config):
    ```
    >>> OmegaConf.register_new_resolver(name="include_keys", resolver=include_keys_fn)
    ```
    And you can call this function in your config to override dirname.

    As an example, you can use it to include the search space of hydra:
    ```
    sweep:
        hydra:
            job:
                config:
                    override_dirname:
                        exclude_keys: "${include_keys: hydra.sweeper.search_space}"
    ```
    """
    hydra_cfg = _load_hydra_cfg(_root_)
    overrides_dic = _get_overrides_from_cfg(hydra_cfg)
    included = OmegaConf.select(_root_, key=prefix).keys()
    excluded = [value for value in overrides_dic.keys() if value not in included]
    return excluded


def get_tag_fn(_root_: DictConfig) -> str:
    tagk = _root_.tagk
    if tagk == "auto":
        raise ValueError(
            "Cannot load 'multirun.yaml' automatically for tag interpolation."
        )

    join = "-"
    tagv = _get_tag_or_subtag(_root_, tagk, "NOTAG", False)

    pretag: str = _root_.pretag
    posttag: str = _root_.posttag

    if pretag != "":
        if not pretag.endswith(join):
            pretag = f"{pretag}{join}"

    if posttag != "":
        if not posttag.startswith(join):
            posttag = f"{join}{posttag}"

    tagv = f"{pretag}{tagv}{posttag}"

    return tagv


def get_subtag_fn(_root_: DictConfig) -> str:
    subtagk = _root_.subtagk

    if subtagk == "auto":
        hydra_cfg = _load_hydra_cfg(_root_)
        overrides = _get_overrides_from_file(hydra_cfg)
        subtagk = [k for k, v in overrides.items() if _is_sweep_value(v)]
        if _root_.verbose >= 2:
            pylog.debug(f"Auto-detect subtags: {subtagk}")

    subtagv = _get_tag_or_subtag(_root_, subtagk, "", True)
    return subtagv


def get_none(*args, **kwargs) -> None:
    """Returns None.

    Can be used for hydra instantiations with:
    ```
    _target_: "conette.utils.hydra.get_none"
    ```
    """
    return None


def get_pickle(
    fpath: Union[str, Path],
) -> Any:
    """Returns the pickled object from file.

    Can be used for hydra instantiations with:
    ```
    _target_: "conette.utils.hydra.get_pickle"
    fpath: "/path/to/file"
    ```

    :param fpath: The filepath to the pickled object.
    :returns: The pickled object.
    """
    if not isinstance(fpath, (str, Path)):
        raise TypeError(f"Invalid transform with pickle {fpath=}. (not a str or Path)")
    if not osp.isfile(fpath):
        raise FileNotFoundError(f"Invalid transform with pickle {fpath=}. (not a file)")

    with open(fpath, "rb") as file:
        data = pickle.load(file)
    return data


@cache
def get_subrun_path() -> str:
    hydra_cfg = HydraConfig.get()
    return osp.join(hydra_cfg.sweep.dir, hydra_cfg.sweep.subdir)


# Private functions
def _load_hydra_cfg(_root_: DictConfig) -> DictConfig:
    try:
        hydra_cfg = _root_.hydra
        return hydra_cfg
    except ConfigAttributeError:
        pass

    try:
        hydra_cfg = HydraConfig.get()
        return hydra_cfg  # type: ignore
    except ValueError as err:
        pylog.error(
            "Cannot get hydra cfg from root or from global HydraConfig instance."
        )
        raise err


def _get_tag_or_subtag(
    _root_: DictConfig,
    keys: Union[str, list[str]],
    default: str,
    accept_sweep_values: bool,
) -> str:
    if isinstance(keys, str):
        keys = [keys]

    hydra_cfg = _load_hydra_cfg(_root_)
    overrides = _get_overrides_from_cfg(hydra_cfg)
    overrides_clean = {
        k.replace("/", ".").split(".")[-1]: v for k, v in overrides.items()
    }
    choices = hydra_cfg.runtime.choices
    choices_clean = {k.replace("/", ".").split(".")[-1]: v for k, v in choices.items()}

    options = {}
    for key in keys:
        if key in overrides:
            options[key] = overrides[key]
        elif key in overrides_clean:
            options[key] = overrides_clean[key]
        elif key in choices:
            options[key] = choices[key]
        elif key in choices_clean:
            options[key] = choices_clean[key]
        else:
            value = OmegaConf.select(_root_, key, default="NOTFOUND")
            if value == "NOTFOUND":
                dic = OmegaConf.to_container(_root_)
                flatten = flat_dict_of_dict(dic)  # type: ignore
                matches = [k for k in flatten.keys() if k.endswith(key)]

                if len(matches) == 1:
                    value = OmegaConf.select(_root_, matches[0], default="NOTFOUND")
                    if value == "NOTFOUND":
                        pylog.error(
                            f"INTERNAL ERROR: Cannot find {matches[0]=} in config."
                        )
                        continue

                elif len(matches) == 0:
                    pylog.warning(f"Cannot find {key=} for tag.")
                    continue
                else:  # > 1
                    pylog.warning(
                        f"Found multiple candidates with {key=} for tag. ({matches=})"
                    )
                    continue

            if not isinstance(value, (int, float, str)):
                pylog.warning(
                    f"Ignore {key=} for tag. (expected type in (int, float, str))"
                )
                continue

            options[key] = value

    options_clean = {k.replace("/", ".").split(".")[-1]: v for k, v in options.items()}

    if len(options) != len(options_clean):
        raise ValueError(
            f"Found duplicated option name after dot. (found {tuple(options.keys())} != {tuple(options_clean.keys())})"
        )

    if not accept_sweep_values:
        sweep_values = {k: v for k, v in options_clean.items() if _is_sweep_value(v)}
        if len(sweep_values) > 0:
            raise ValueError(
                f"Invalid sweep values for main tag. (sweep keys: {tuple(sweep_values.keys())})"
            )

    tag = "-".join(f"{k}_{v}" for k, v in options_clean.items())
    tag = tag.replace(" ", "")
    if tag == "":
        tag = default
    else:
        tag = "-" + tag

    # Clean tag
    replaces = {
        "=": "_",
        ",": "_",
        " ": "_",
        "[": "",
        "]": "",
    }
    for p, v in replaces.items():
        tag = tag.replace(p, v)

    return tag


def _get_overrides_from_cfg(hydra_cfg: DictConfig) -> dict[str, Any]:
    overrides = hydra_cfg.overrides.task
    overrides_dic = {
        kv.split("=")[0].removeprefix("+"): kv.split("=")[1] for kv in overrides
    }

    output = {}
    for k, v in overrides_dic.items():
        if any(s in v for s in (".", "e", "E")):
            try:
                v = str(float(v))
            except ValueError:
                pass
        output[k] = v

    return output


def _get_overrides_from_file(hydra_cfg: DictConfig) -> dict[str, Any]:
    if hydra_cfg.mode != RunMode.MULTIRUN:
        return {}

    multirun_fpath = osp.join(hydra_cfg.sweep.dir, "multirun.yaml")
    if not osp.isfile(multirun_fpath):
        pylog.error(
            f"Cannot find automatically 'multirun.yaml' file in directory '{osp.dirname(multirun_fpath)}'."
        )
        return {}

    data = load_yaml(multirun_fpath)
    overrides: list[str] = data.get("hydra", {}).get("overrides", {}).get("task", [])
    overrides_dic = {
        kv.split("=")[0].removeprefix("+"): kv.split("=")[1] for kv in overrides
    }
    return overrides_dic


def _is_sweep_value(v: str) -> bool:
    """Returns true if the value is a hydra sweep argument value.

    >>> _is_sweep_value("1,2")
    ... True
    >>> _is_sweep_value("[1,2]")
    ... False
    >>> _is_sweep_value("a,b,c")
    ... True
    >>> _is_sweep_value("something")
    ... False
    """
    return (
        isinstance(v, str)
        and "," in v
        and not v.startswith("[")
        and not v.endswith("]")
    )


def load_overrides(fpath: str) -> dict[str, Any]:
    overrides: list[str] = load_yaml(fpath)
    overrides_dic = {}

    for override in overrides:
        idx = override.find("=")
        if idx == -1:
            raise RuntimeError(f"Cannot find character '=' in overrides. ({override=})")

        name = override[:idx].removeprefix("++").removeprefix("+")
        value = override[idx + 1 :]
        overrides_dic[name] = value

    return overrides_dic
