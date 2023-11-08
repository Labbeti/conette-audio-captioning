#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Callable, Iterable

from conette.tokenization.constants import BOS_TOKEN, EOS_TOKEN, PAD_TOKEN, UNK_TOKEN
from conette.tokenization.tokenizers.base import StrTokenizer
from conette.tokenization.tokenizers.common import build_mappings_and_vocab


class TokenizerWrapper(StrTokenizer):
    def __init__(self, tokenizer: StrTokenizer) -> None:
        super().__init__()
        self._tokenizer = tokenizer

    # Public implemented methods
    def detokenize_batch(
        self, sentences: Iterable[Iterable[str]], *args, **kwargs
    ) -> list[str]:
        return self._tokenizer.detokenize_batch(sentences, *args, **kwargs)

    def fit(self, sentences: Iterable[str]) -> tuple[list, dict, dict, dict]:
        return self._tokenizer.fit(sentences)

    def get_backend(self) -> str:
        return self._tokenizer.get_backend()

    def get_level(self) -> str:
        return self._tokenizer.get_level()

    def tokenize_batch(
        self, sentences: Iterable[str], *args, **kwargs
    ) -> list[list[str]]:
        return self._tokenizer.tokenize_batch(sentences, *args, **kwargs)

    # Public methods
    def unwrap(self, recursive: bool = True) -> StrTokenizer:
        tokenizer = self._tokenizer
        if not recursive:
            return tokenizer
        while isinstance(tokenizer, TokenizerWrapper):
            tokenizer = tokenizer._tokenizer
        return tokenizer

    # Properties
    @property
    def bos_token(self) -> str:
        return self._tokenizer.bos_token

    @property
    def eos_token(self) -> str:
        return self._tokenizer.eos_token

    @property
    def pad_token(self) -> str:
        return self._tokenizer.pad_token

    @property
    def unk_token(self) -> str:
        return self._tokenizer.unk_token

    @property
    def separator(self) -> str:
        return self._tokenizer.separator


class LambdaTokenizer(StrTokenizer):
    def __init__(
        self,
        level: str = "word",
        tokenizer: Callable[[str], list[str]] = str.split,
        detokenizer: Callable[[Iterable[str]], str] = " ".join,
        bos_token: str = BOS_TOKEN,
        eos_token: str = EOS_TOKEN,
        pad_token: str = PAD_TOKEN,
        unk_token: str = UNK_TOKEN,
        backend: str = "python",
    ) -> None:
        super().__init__()
        self._level = level
        self._tokenizer = tokenizer
        self._detokenizer = detokenizer
        self._bos_token = bos_token
        self._eos_token = eos_token
        self._pad_token = pad_token
        self._unk_token = unk_token
        self._backend = backend

    @property
    def bos_token(self) -> str:
        return self._bos_token

    @property
    def eos_token(self) -> str:
        return self._eos_token

    @property
    def pad_token(self) -> str:
        return self._pad_token

    @property
    def unk_token(self) -> str:
        return self._unk_token

    def detokenize_batch(self, sentences: Iterable[Iterable[str]]) -> list[str]:
        return [self._detokenizer(sentence) for sentence in sentences]

    def fit(self, sentences: Iterable[str]) -> tuple[list, dict, dict, dict]:
        encoded_sentences = self.tokenize_batch(sentences)
        itos, stoi, vocab = build_mappings_and_vocab(
            encoded_sentences, self.special_tokens
        )
        return encoded_sentences, itos, stoi, vocab

    def get_backend(self) -> str:
        return self._backend

    def get_level(self) -> str:
        return self._level

    def tokenize_batch(self, sentences: Iterable[str]) -> list[list[str]]:
        return [self._tokenizer(sentence) for sentence in sentences]
