#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""CoNeTTE model for Audio Captioning.
"""

__name__ = "conette"
__author__ = "Etienne Labbé (Labbeti)"
__author_email__ = "labbeti.pub@gmail.com"
__license__ = "MIT"
__maintainer__ = "Etienne Labbé (Labbeti)"
__status__ = "Development"
__version__ = "0.1.1"


from pathlib import Path
from typing import Any, Optional

from conette.huggingface.config import CoNeTTEConfig  # noqa: F401
from conette.huggingface.model import CoNeTTEModel  # noqa: F401


def conette(
    pretrained_model_name_or_path: Optional[str] = "Labbeti/conette",
    config_kwds: Optional[dict[str, Any]] = None,
    model_kwds: Optional[dict[str, Any]] = None,
) -> CoNeTTEModel:
    """Create pretrained CoNeTTEModel for inference."""
    if config_kwds is None:
        config_kwds = {}
    if model_kwds is None:
        model_kwds = {}

    if pretrained_model_name_or_path is None:
        config = CoNeTTEConfig(**config_kwds)
        model = CoNeTTEModel(**model_kwds)
    else:
        config = CoNeTTEConfig.from_pretrained(
            pretrained_model_name_or_path,
            **config_kwds,
        )
        model = CoNeTTEModel.from_pretrained(
            pretrained_model_name_or_path,
            config=config,
            **model_kwds,
        )
    return model  # type: ignore


def get_sample_path() -> str:
    """Returns audio sample absolute path."""
    path = Path(__file__).parent.joinpath("data", "sample.wav")
    return str(path)
