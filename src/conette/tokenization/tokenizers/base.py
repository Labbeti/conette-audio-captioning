#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pickle

from abc import ABC, abstractmethod
from typing import Any, Iterable, Union

from conette.tokenization.tokenizers.common import (
    is_tokenized_sent_single,
    is_tokenized_sents_batch,
    is_sent_single,
    is_sents_batch,
)
from conette.tokenization.constants import BOS_TOKEN, EOS_TOKEN, PAD_TOKEN, UNK_TOKEN


class StrTokenizer(ABC):
    """Abstract class for tokenizers to tokenize/detokenize sentences strings.

    Abstract methods:
    - detokenize_batch(self, sentences: Iterable[Iterable[str]]) -> list[str]
    - fit(self, sentences: Iterable[str]) -> tuple[list, dict, dict, dict]
    - get_backend(self) -> str
    - get_level(self) -> str
    - tokenize_batch(self, sentences: Iterable[str]) -> list[list[str]]

    """

    def __init__(self) -> None:
        super().__init__()
        # Note: trick to avoid child classes to have a __hash__ set to None if they redefined the __eq__ method
        # https://stackoverflow.com/questions/53518981/inheritance-hash-sets-to-none-in-a-subclass
        self.__class__.__hash__ = StrTokenizer.__hash__

    # Abstract Public methods
    @abstractmethod
    def detokenize_batch(self, sentences: Iterable[Iterable[str]]) -> list[str]:
        raise NotImplementedError("Abstract method")

    @abstractmethod
    def fit(self, sentences: Iterable[str]) -> tuple[list, dict, dict, dict]:
        raise NotImplementedError("Abstract method")

    @abstractmethod
    def get_backend(self) -> str:
        raise NotImplementedError("Abstract method")

    @abstractmethod
    def get_level(self) -> str:
        raise NotImplementedError("Abstract method")

    @abstractmethod
    def tokenize_batch(self, sentences: Iterable[str]) -> list[list[str]]:
        raise NotImplementedError("Abstract method")

    # Public methods
    def detokenize_rec(self, nested_sentences: Iterable, *args, **kwargs) -> Any:
        if is_tokenized_sent_single(nested_sentences):
            return self.detokenize_single(nested_sentences, *args, **kwargs)
        elif is_tokenized_sents_batch(nested_sentences):
            return self.detokenize_batch(nested_sentences, *args, **kwargs)
        elif isinstance(nested_sentences, Iterable):
            return [self.detokenize_rec(s, *args, **kwargs) for s in nested_sentences]
        else:
            raise TypeError(f"Invalid input type {type(nested_sentences)=}.")

    def detokenize_single(self, sentence: Iterable[str]) -> str:
        return self.detokenize_batch([sentence])[0]

    # Properties
    @property
    def bos_token(self) -> str:
        return BOS_TOKEN

    @property
    def eos_token(self) -> str:
        return EOS_TOKEN

    @property
    def pad_token(self) -> str:
        return PAD_TOKEN

    @property
    def unk_token(self) -> str:
        return UNK_TOKEN

    @property
    def separator(self) -> str:
        return " "

    @property
    def special_tokens(self) -> list[str]:
        special_tokens = [
            self.pad_token,
            self.bos_token,
            self.eos_token,
            self.unk_token,
        ]
        special_tokens = [token for token in special_tokens if token is not None]
        return special_tokens

    # Public methods
    def tokenize_rec(
        self,
        nested_sentences: Union[str, Iterable],
        *args,
        **kwargs,
    ) -> Any:
        if is_sent_single(nested_sentences):
            return self.tokenize_single(nested_sentences, *args, **kwargs)  # type: ignore
        elif is_sents_batch(nested_sentences):
            return self.tokenize_batch(nested_sentences, *args, **kwargs)
        elif isinstance(nested_sentences, Iterable):
            return [self.tokenize_rec(s, *args, **kwargs) for s in nested_sentences]
        else:
            raise TypeError(f"Invalid input type {type(nested_sentences)=}.")

    def tokenize_single(self, sentence: str, *args, **kwargs) -> list[str]:
        return self.tokenize_batch([sentence], *args, **kwargs)[0]

    # Magic methods
    def __eq__(self, __o: object) -> bool:
        return (
            isinstance(__o, StrTokenizer)
            and self.get_level() == __o.get_level()
            and pickle.dumps(self) == pickle.dumps(__o)
        )

    def __hash__(self) -> int:
        return sum(pickle.dumps(self))
