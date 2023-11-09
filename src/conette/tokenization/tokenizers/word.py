#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import ClassVar

from conette.tokenization.tokenizers.base import StrTokenizer
from conette.tokenization.tokenizers.spacy import SpacyWordTokenizer
from conette.tokenization.tokenizers.wrapper import TokenizerWrapper


class WordTokenizer(TokenizerWrapper):
    """Tokenizer facade class for the following word tokenizers classes:

    - :class:`~aac.tokenization.tokenizers.spacy.SpacyWordTokenizer`

    """

    BACKENDS: ClassVar[tuple[str, ...]] = ("spacy",)

    def __init__(self, backend: str = "spacy", *args, **kwargs) -> None:
        tokenizer = _word_tokenizer_factory(backend, *args, **kwargs)
        super().__init__(tokenizer)


def _word_tokenizer_factory(backend: str = "spacy", *args, **kwargs) -> StrTokenizer:
    if backend == "spacy":
        tokenizer = SpacyWordTokenizer(*args, **kwargs)
    else:
        raise ValueError(
            f"Invalid argument {backend=}. (expected one of {WordTokenizer.BACKENDS})"
        )
    return tokenizer
