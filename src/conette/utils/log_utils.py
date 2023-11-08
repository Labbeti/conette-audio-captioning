#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging

from functools import cache
from logging import Logger
from types import ModuleType
from typing import (
    Iterable,
    Union,
)

pylog = logging.getLogger(__name__)


@cache
def warn_once(msg: str, logger: Union[Logger, ModuleType, None]) -> None:
    if logger is None:
        pylog = logging.root
    elif isinstance(logger, ModuleType):
        pylog: Logger = logger.root
    else:
        pylog = logger

    pylog.warning(msg)


def set_loglevel(
    packages: Union[str, ModuleType, Iterable[Union[str, ModuleType]]], level: int
) -> None:
    """Set main logger level for a list of packages."""
    if isinstance(packages, (str, ModuleType)):
        packages = [packages]
    packages = [pkg if isinstance(pkg, str) else pkg.__name__ for pkg in packages]

    for pkg in packages:
        pkg_pylog = logging.getLogger(pkg)
        pkg_pylog.setLevel(level)
