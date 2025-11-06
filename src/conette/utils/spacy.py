#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import subprocess
import sys
from subprocess import CalledProcessError
from typing import Any

import spacy

pylog = logging.getLogger(__name__)


def load_or_download_spacy_model(
    model_name: str,
    raise_if_cannot_dl: bool = True,
) -> Any:
    try:
        model = spacy.load(model_name)
        pylog.info(f"Model '{model_name}' for spacy is already downloaded.")

    except OSError:
        command = [sys.executable, "-m", "spacy", "download", model_name]
        try:
            subprocess.check_call(
                command,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            pylog.info(f"Model '{model_name}' for spacy has been downloaded.")
            model = spacy.load(model_name)

        except (CalledProcessError, PermissionError) as err:
            msg = f"Cannot download spaCy model '{model_name}' for tokenizer. (command '{command}' with error={err})"
            pylog.error(msg)
            if raise_if_cannot_dl:
                raise err
            model = None

    return model
