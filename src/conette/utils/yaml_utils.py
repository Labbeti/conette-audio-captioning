#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path
from typing import Any, Union

import yaml


def load_yaml(fpath: Union[str, Path]) -> Any:
    with open(fpath, "r") as file:
        data = yaml.safe_load(file)
    return data


def save_yaml(data: Any, fpath: Union[str, Path]) -> None:
    with open(fpath, "w") as file:
        yaml.dump(data, file)
    return data
