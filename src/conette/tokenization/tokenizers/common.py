#!/usr/bin/env python
# -*- coding: utf-8 -*-

from collections import Counter
from typing import Any, Iterable, TypeGuard


def build_mappings_and_vocab(
    encoded_sentences: list[list[str]],
    special_tokens: Iterable[str],
) -> tuple[dict[int, str], dict[str, int], dict[str, int]]:
    """Returns (itos, stoi, vocab) dictionaries."""
    tokens_counter = {token: 0 for token in special_tokens}
    tokens_counter |= dict(
        Counter(token for sentence in encoded_sentences for token in sentence)
    )
    itos = {i: token for i, token in enumerate(tokens_counter.keys())}
    stoi = {token: i for i, token in enumerate(tokens_counter.keys())}
    return itos, stoi, tokens_counter


def is_tokenized_sent_single(inputs: Any) -> TypeGuard[list[str]]:
    """Returns true if inputs is one of:
    - list[str]
    """
    return isinstance(inputs, list) and all(
        isinstance(inputs_i, str) for inputs_i in inputs
    )


def is_tokenized_sents_batch(inputs: Any) -> TypeGuard[list[list[str]]]:
    return isinstance(inputs, list) and all(
        is_tokenized_sent_single(inputs_i) for inputs_i in inputs
    )


def is_sent_single(inputs: Any) -> TypeGuard[str]:
    return isinstance(inputs, str)


def is_sents_batch(inputs: Any) -> TypeGuard[Iterable[str]]:
    return (
        isinstance(inputs, Iterable)
        and not isinstance(inputs, str)
        and all(is_sent_single(inputs_i) for inputs_i in inputs)
    )
