#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path
from typing import Iterable, Union

from aac_metrics.utils.tokenization import ptb_tokenize_batch

from conette.tokenization.constants import SPECIAL_TOKENS
from conette.tokenization.tokenizers.base import StrTokenizer
from conette.tokenization.tokenizers.common import build_mappings_and_vocab


class PTBWordTokenizer(StrTokenizer):
    def __init__(
        self,
        cache_path: Union[str, Path, None] = None,
        java_path: Union[str, Path, None] = None,
        tmp_path: Union[str, Path, None] = None,
        special_tokens: Iterable[str] = SPECIAL_TOKENS,
    ) -> None:
        super().__init__()
        self._cache_path = cache_path
        self._java_path = java_path
        self._tmp_path = tmp_path
        self._special_tokens = special_tokens

    def detokenize_batch(self, sentences: Iterable[Iterable[str]]) -> list[str]:
        decoded_sentences = [" ".join(sentence) for sentence in sentences]
        return decoded_sentences

    def fit(self, sentences: Iterable[str]) -> tuple[list, dict, dict, dict]:
        encoded_sentences = self.tokenize_batch(sentences)
        itos, stoi, vocab = build_mappings_and_vocab(
            encoded_sentences, self._special_tokens
        )
        return encoded_sentences, itos, stoi, vocab

    def get_backend(self) -> str:
        return "ptb"

    def get_level(self) -> str:
        return "word"

    def tokenize_batch(self, sentences: Iterable[str], **kwargs) -> list[list[str]]:
        return ptb_tokenize_batch(
            sentences,
            cache_path=self._cache_path,
            java_path=self._java_path,
            tmp_path=self._tmp_path,
        )
