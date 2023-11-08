#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Any, Iterable

import spacy

from conette.tokenization.constants import SPECIAL_TOKENS
from conette.tokenization.tokenizers.base import StrTokenizer
from conette.tokenization.tokenizers.common import build_mappings_and_vocab


class SpacyWordTokenizer(StrTokenizer):
    def __init__(
        self,
        model_name: str = "en_core_web_sm",
        special_tokens: Iterable[str] = SPECIAL_TOKENS,
    ) -> None:
        super().__init__()
        self._model_name = model_name
        self._special_tokens = special_tokens
        self._model = spacy.load(model_name)

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
        return "spacy"

    def get_level(self) -> str:
        return "word"

    def tokenize_batch(self, sentences: Iterable[str]) -> list[list[str]]:
        encoded_sentences = [self._model.tokenizer(sentence) for sentence in sentences]
        # Note : Spacy returns a list of spacy.tokens.token.Token object
        encoded_sentences = [
            [word.text for word in sentence] for sentence in encoded_sentences
        ]
        return encoded_sentences

    def __getstate__(self) -> dict[str, Any]:
        return {
            "model_name": self._model_name,
            "special_tokens": self._special_tokens,
        }

    def __setstate__(self, data: dict[str, Any]) -> None:
        self._model_name = data["model_name"]
        self._special_tokens = data.get("special_tokens", SPECIAL_TOKENS)
        self._model = spacy.load(data["model_name"])
