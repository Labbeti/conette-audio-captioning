#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import subprocess
import sys

from subprocess import CalledProcessError

import nltk


pylog = logging.getLogger(__name__)


def setup_other_models(offline: bool = False, verbose: int = 0) -> None:
    if offline:
        return None

    # Download spaCy model for AACTokenizer
    for model_name in ("en_core_web_sm",):
        command = f"{sys.executable} -m spacy download {model_name}".split(" ")
        try:
            subprocess.check_call(
                command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
            if verbose >= 1:
                pylog.info(f"Model '{model_name}' for spacy downloaded.")
        except (CalledProcessError, PermissionError) as err:  # type: ignore
            pylog.error(
                f"Cannot download spaCy model '{model_name}' for tokenizer. (command '{command}' with error={err})"
            )

    # Download stopwords list for constrained beam search
    nltk.download("stopwords")
