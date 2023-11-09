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
    spacy_model = "en_core_web_sm"
    cmd = [sys.executable, "-m", "spacy", "download", spacy_model]
    try:
        subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if verbose >= 1:
            pylog.info(f"Model '{spacy_model}' for spacy downloaded.")
    except (CalledProcessError, PermissionError) as err:  # type: ignore
        pylog.error(
            f"Cannot download spaCy model '{spacy_model}' for tokenizer. (command '{cmd}' with error={err})"
        )

    # Download stopwords list for constrained beam search
    nltk_model = "stopwords"
    nltk.download(nltk_model, quiet=verbose <= 0)
