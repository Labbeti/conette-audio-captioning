#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import ClassVar

from conette.tokenization.tokenizers.base import StrTokenizer

try:
    from conette.tokenization.tokenizers.ptb import PTBWordTokenizer
except ImportError:
    PTBWordTokenizer = None
from conette.tokenization.tokenizers.spacy import SpacyWordTokenizer
from conette.tokenization.tokenizers.wrapper import TokenizerWrapper


class WordTokenizer(TokenizerWrapper):
    """Tokenizer facade class for the following word tokenizers classes:

    - :class:`~conette.tokenization.tokenizers.ptb.PTBWordTokenizer`
    - :class:`~conette.tokenization.tokenizers.spacy.SpacyWordTokenizer`

    """

    BACKENDS: ClassVar[tuple[str, ...]] = ("spacy", "ptb")

    def __init__(self, backend: str = "spacy", *args, **kwargs) -> None:
        tokenizer = _word_tokenizer_factory(backend, *args, **kwargs)
        super().__init__(tokenizer)


def _word_tokenizer_factory(backend: str = "spacy", *args, **kwargs) -> StrTokenizer:
    if backend == "spacy":
        tokenizer = SpacyWordTokenizer(*args, **kwargs)
    elif backend == "ptb":
        if PTBWordTokenizer is None:
            raise RuntimeError(
                "Please install aac-metrics package to use ptb tokenizer backend. (found None PTBWordTokenizer)"
            )
        tokenizer = PTBWordTokenizer(*args, **kwargs)
    else:
        raise ValueError(
            f"Invalid argument {backend=}. (expected one of {WordTokenizer.BACKENDS})"
        )
    return tokenizer
