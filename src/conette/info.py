#!/usr/bin/env python
# -*- coding: utf-8 -*-

import platform
import sys

from pathlib import Path

import pytorch_lightning
import torch
import yaml

import conette

from conette import get_sample_path


def get_package_repository_path() -> str:
    """Return the absolute path where the source code of this package is installed."""
    return str(Path(__file__).parent.parent.parent)


def get_install_info() -> dict[str, str]:
    """Return local installation and paths."""
    return {
        "conette": conette.__version__,
        "python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "os": platform.system(),
        "architecture": platform.architecture()[0],
        "torch": str(torch.__version__),
        "lightning": pytorch_lightning.__version__,  # type: ignore
        "package_path": get_package_repository_path(),
        "sample_path": get_sample_path(),
    }


def print_install_info() -> None:
    """Show main packages versions and paths."""
    install_info = get_install_info()
    print(yaml.dump(install_info, sort_keys=False))


if __name__ == "__main__":
    print_install_info()
