#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pickle
import re

from abc import ABC
from typing import Any, Iterable, Mapping, Optional

from conette.tokenization.constants import SPECIAL_TOKENS, EOS_TOKEN


class NormalizerI(ABC):
    """Base class for normalizers, which cleans sentences."""

    # Abstract public methods
    def normalize_batch(self, sentences: Iterable[str]) -> list[str]:
        raise NotImplementedError("Abstract method")

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> Any:
        raise NotImplementedError("Abstract class method")

    # Public methods
    def get_config(self) -> dict[str, Any]:
        return {
            "type": self.__class__.__name__,
        }

    def normalize_single(self, sentence: str) -> str:
        return self.normalize_batch([sentence])[0]

    # Magic methods
    def __call__(self, sentences: Iterable[str]) -> list[str]:
        return self.normalize_batch(sentences)

    def __eq__(self, __o: object) -> bool:
        return isinstance(__o, NormalizerI) and pickle.dumps(self) == pickle.dumps(__o)

    def __hash__(self) -> int:
        return sum(pickle.dumps(self))


class NormalizerList(NormalizerI, list[NormalizerI]):
    """Applies a list of normalizers sequentially."""

    def __init__(self, *normalizers: NormalizerI) -> None:
        NormalizerI.__init__(self)
        list.__init__(self, normalizers)

    @classmethod
    def from_iterable(cls, normalizers: Iterable[NormalizerI]) -> "NormalizerList":
        return NormalizerList(*normalizers)

    def get_config(self) -> dict[str, Any]:
        return {
            "type": self.__class__.__name__,
            "normalizers": [normalizer.get_config() for normalizer in self],
        }

    def normalize_batch(self, sentences: Iterable[str]) -> list[str]:
        normalized_sentences = list(sentences)
        for normalizer in self:
            normalized_sentences = normalizer.normalize_batch(normalized_sentences)
        return normalized_sentences


class Lowercase(NormalizerI):
    def normalize_batch(self, sentences: Iterable[str]) -> list[str]:
        return list(map(str.lower, sentences))


class Replace(NormalizerI):
    def __init__(self, pattern: str, repl: str) -> None:
        super().__init__()
        self._pattern = re.compile(pattern)
        self._repl = repl

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "Replace":
        return Replace(config["pattern"], config["repl"])

    def get_config(self) -> dict[str, Any]:
        return {
            "type": self.__class__.__name__,
            "pattern": str(self._pattern),
            "repl": self._repl,
        }

    def normalize_batch(self, sentences: Iterable[str]) -> list[str]:
        return [re.sub(self._pattern, self._repl, sentence) for sentence in sentences]


class Strip(NormalizerI):
    def normalize_batch(self, sentences: Iterable[str]) -> list[str]:
        return list(map(str.strip, sentences))


class CleanDoubleSpaces(Replace):
    def __init__(self) -> None:
        super().__init__(" +", " ")

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "CleanDoubleSpaces":
        return CleanDoubleSpaces()

    def get_config(self) -> dict[str, Any]:
        return {
            "type": self.__class__.__name__,
        }


class ReplaceRarePuncChars(NormalizerList):
    def __init__(self) -> None:
        super().__init__(
            Replace(r"“", '"'),
            Replace(r"”", '"'),
            Replace(r"`", "'"),
            Replace(r"’", "'"),
            Replace(r";", ","),
            Replace(r"…", "..."),
            Replace(r"&", " & "),
        )


class CleanPunctuation(Replace):
    PUNC_PATTERN: str = r"[,.!?;:\"“”’`\(\)\{\}\[\]\*\×\-#/+_~ʘ\\/]"

    def __init__(self, pattern: Optional[str] = None) -> None:
        if pattern is None:
            pattern = CleanPunctuation.PUNC_PATTERN
        super().__init__(pattern, " ")

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "CleanPunctuation":
        return CleanPunctuation(config["pattern"])

    def get_config(self) -> dict[str, Any]:
        return {
            "type": self.__class__.__name__,
            "pattern": str(self._pattern),
        }


class CleanSpacesBeforePunctuation(Replace):
    def __init__(self) -> None:
        pattern = r'\s+([,.!?;:"\'])'
        super().__init__(pattern, r"\1")

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "CleanSpacesBeforePunctuation":
        return CleanSpacesBeforePunctuation()

    def get_config(self) -> dict[str, Any]:
        return {
            "type": self.__class__.__name__,
        }


class CleanSpecialTokens(Replace):
    """Remove <bos>, <eos>, <pad>, <unk> by default."""

    def __init__(self, special_tokens: Iterable[str] = SPECIAL_TOKENS) -> None:
        # pattern: (<bos>|<eos>|<pad>|<unk>)
        super().__init__(f"({'|'.join(special_tokens)})", "")

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "CleanSpecialTokens":
        return CleanSpecialTokens()

    def get_config(self) -> dict[str, Any]:
        return {
            "type": self.__class__.__name__,
        }


class CleanHyphenSpaces(Replace):
    def __init__(self) -> None:
        super().__init__(r"(\s*)(\-)(\s*)", r"\2")

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "CleanHyphenSpaces":
        return CleanHyphenSpaces()

    def get_config(self) -> dict[str, Any]:
        return {
            "type": self.__class__.__name__,
        }


class TruncAtEos(NormalizerI):
    def __init__(self, eos: str = EOS_TOKEN) -> None:
        super().__init__()
        self._eos = eos

    def normalize_batch(self, sentences: Iterable[str]) -> list[str]:
        outputs = []
        for sentence in sentences:
            if self._eos in sentence:
                idx = sentence.index(self._eos)
                sentence = sentence[:idx]
            outputs.append(sentence)
        return outputs

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "TruncAtEos":
        return TruncAtEos(config["eos"])

    def get_config(self) -> dict[str, Any]:
        return {
            "type": self.__class__.__name__,
            "eos": self._eos,
        }
