#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import logging
import pickle
import sys

from functools import cache
from typing import Any, Iterable, Mapping, Sized, Union

import torch
import yaml

from torch import nn, Tensor

from conette.tokenization.normalizers import (
    CleanDoubleSpaces,
    CleanHyphenSpaces,
    CleanPunctuation,
    CleanSpacesBeforePunctuation,
    CleanSpecialTokens,
    Lowercase,
    NormalizerI,
    ReplaceRarePuncChars,
    Strip,
)
from conette.tokenization.tokenizers.wrapper import TokenizerWrapper
from conette.tokenization.tokenizers.common import is_tokenized_sent_single
from conette.tokenization.tokenizers.factory import (
    _pre_tokenizer_factory,
)


pylog = logging.getLogger(__name__)


class AACTokenizer(nn.Module, TokenizerWrapper):
    """Main tokenizer facade. Provide function to encode/decode sentences to Tensor.

    Contains normalizers which clean sentences, a tokenizer to split sentences to tokens and a map of token to index.
    The methods are similar to huggingface tokenizers library, but the implementation is different.

    Example 1
    ----------
    ```
    >>> tokenizer = AACTokenizer()
    >>> tokenizer.tokenize_single("a man is speaking")
    ... ["a", "man", "is", "speaking"]
    ```
    Example 2
    ----------
    ```
    >>> tokenizer = AACTokenizer()
    >>> sentences = ["A bird is singing." "a bird sings"]
    >>> _ = tokenizer.fit(sentences)
    >>> tokenizer.tokenize_batch(sentences, padding="batch")
    ... [["a", "bird", "is", "singing"], ["a", "bird", "sings", "<pad>"]]
    >>> tokenizer.encode_batch(sentences, padding="batch")
    ... tensor([[4, 5, 6, 7], [4, 5, 8, 0]])
    ```
    """

    PUNCTUATION_MODES = ("remove", "keep_comma", "keep", "keep_hyphen")
    OUT_TYPES: tuple[str, ...] = ("str", "int", "Tensor", "pt")
    VERSION = "2.2.0"

    # Initialization
    def __init__(
        self,
        level: str = "word",
        lowercase: bool = True,
        punctuation_mode: str = "remove",
        normalize: bool = True,
        **kwargs,
    ) -> None:
        """
        :param level: "word", "char", "affix", "bpe", "unigram" or "bert".
        :param lowercase: If True, encoded sentences will be converted to lowercase. defaults to True.
        :param punctuation_mode: If True, clean punctuation tokens. defaults to True.
        :param normalize: If True, enable pre and post normalization of the sentences. defaults to True.
        :param **kwargs: These optional values passed to the internal pre_tokenizer. The accepted values depends of the "level" argument.

            - If level == "affix":
                language: str = "english" (stemmer language)
                kwargs: Any arguments of level == "word" because it starts to tokenize to words before splitting into affixes.

            - If level == "bert":
                model_name: str = "bert-base-uncased"

            - If level == "bpe" or level == "unigram":
                vocab_size: int = 1000
                split_by_whitespace: bool = True
                character_coverage: float = 0.9995
                verbose: int = 1
                special_tokens: Iterable[str] = SPECIAL_TOKENS

            - If level == "char":
                special_tokens: Iterable[str] = SPECIAL_TOKENS

            - If level == "word":
                backend: str = "spacy" (one of "spacy", "nltk", "ptb", "python")
                - If backend == "spacy":
                    model_name: str = "en_core_web_sm"
                    special_tokens: Iterable[str] = SPECIAL_TOKENS

                - If backend == "nltk":
                    model_name: str = "english"
                    special_tokens: Iterable[str] = SPECIAL_TOKENS

                - If backend == "ptb":
                    cache_path: str = "~/.cache"
                    java_path: str = "java"
                    tmp_path: str = tempfile.gettempdir()
                    special_tokens: Iterable[str] = SPECIAL_TOKENS

                - If backend == "python":
                    separator: str
                    special_tokens: Iterable[str] = SPECIAL_TOKENS

        """
        hparams = {
            "level": level,
            "lowercase": lowercase,
            "punctuation_mode": punctuation_mode,
            "normalize": normalize,
        } | kwargs

        # Build normalizers
        pre_encoding_normalizers = _get_pre_encoding_normalizers(
            lowercase, punctuation_mode
        )
        post_decoding_normalizers = _get_post_decoding_normalizers(lowercase)
        tokenizer = _pre_tokenizer_factory(
            level=level,
            **kwargs,
        )

        nn.Module.__init__(self)
        TokenizerWrapper.__init__(self, tokenizer)

        # Set attributes
        self._hparams = hparams
        self._pre_encoding_normalizers = pre_encoding_normalizers
        self._post_decoding_normalizers = post_decoding_normalizers
        self._normalize = normalize

        self._added_special_tokens = []
        self._max_sentence_size = -1
        self._min_sentence_size = sys.maxsize
        self._n_sentences_fit = 0
        self._itos: dict[int, str] = {}
        self._stoi: dict[str, int] = {}
        self._vocab: dict[str, int] = {}

    # Properties
    @property
    def bos_token_id(self) -> int:
        return self.token_to_id(self.bos_token)

    @property
    def eos_token_id(self) -> int:
        return self.token_to_id(self.eos_token)

    @property
    def pad_token_id(self) -> int:
        return self.token_to_id(self.pad_token)

    @property
    def unk_token_id(self) -> int:
        return self.token_to_id(self.unk_token)

    @property
    def special_tokens_ids(self) -> list[int]:
        return [self.token_to_id(token) for token in self.special_tokens]

    @property
    def added_special_tokens(self) -> list[str]:
        return self._added_special_tokens

    @property
    def separator(self) -> str:
        return self._tokenizer.separator

    # nn.Module methods
    def extra_repr(self) -> str:
        return ", ".join(
            f"{name}={value}" for name, value in self.get_hparams().items()
        )

    def forward(self, nested_sentences: Any, *args, **kwargs) -> Any:
        return self.encode_rec(nested_sentences, *args, **kwargs)

    def get_extra_state(self) -> Any:
        return self.get_state()

    def set_extra_state(self, state: dict[str, Any]) -> None:
        return self.set_state(state)

    # StrTokenizer methods
    def detokenize_batch(
        self,
        sentences: Iterable[Iterable[str]],
        skip_special_tokens: bool = True,
    ) -> list[str]:
        sentences = self._tokenizer.detokenize_batch(sentences)  # type: ignore
        if self.is_normalization_enabled():
            for normalizer in self._post_decoding_normalizers:
                if skip_special_tokens or not isinstance(
                    normalizer, CleanSpecialTokens
                ):
                    sentences = normalizer.normalize_batch(sentences)
        return sentences

    def fit(
        self,
        sentences: Iterable[str],
    ) -> tuple[list[list[str]], dict[int, str], dict[str, int], dict[str, int]]:
        if self._n_sentences_fit > 0:
            raise RuntimeError(
                f"Cannot fit {self.__class__.__name__} twice. (found n_sentences_fit={self._n_sentences_fit} > 0)"
            )

        if self.is_normalization_enabled():
            for normalizer in self._pre_encoding_normalizers:
                sentences = normalizer.normalize_batch(sentences)

        encoded_sentences, itos, stoi, vocab = self._tokenizer.fit(sentences)
        del sentences

        self._itos |= itos
        self._stoi |= stoi
        self._vocab |= vocab

        if len(encoded_sentences) > 0:
            sentences_lens = list(map(len, encoded_sentences))
            self._max_sentence_size = max(self._max_sentence_size, max(sentences_lens))
            self._min_sentence_size = min(self._min_sentence_size, min(sentences_lens))
            self._n_sentences_fit += len(encoded_sentences)

        return encoded_sentences, itos, stoi, vocab

    def get_backend(self) -> str:
        return self._tokenizer.get_backend()

    def get_level(self) -> str:
        """Get token level of the tokenizer."""
        return self._tokenizer.get_level()

    def tokenize_batch(
        self,
        sentences: Iterable[str],
        add_bos_eos: bool = False,
        padding: Union[None, int, str] = None,
    ) -> list[list[str]]:
        """
        :param sentences: The input sentences to encode.
        :param add_bos_eos: If True, pre-pend BOS_TOKEN and append EOS_TOKEN to each sentence.
            defaults to False.
        :param padding: The padding size or mode.
            If a int, sentences will be padded until the specified size.
            If "batch", sentences will be padded until the maximal size in the batch.
            If "corpus", sentences will be padded until the maximal size stored by the corpus fit. (i.e. :meth:`~get_max_sentence_size`)
        """
        if self.is_normalization_enabled():
            for normalizer in self._pre_encoding_normalizers:
                sentences = normalizer.normalize_batch(sentences)

        tokenized_sentences = self._tokenizer.tokenize_batch(sentences)
        del sentences

        if add_bos_eos:
            tokenized_sentences = [
                [self.bos_token] + sentence + [self.eos_token]
                for sentence in tokenized_sentences
            ]

        if isinstance(padding, str):
            if padding == "batch":
                if len(tokenized_sentences) > 0:
                    padding = max(map(len, tokenized_sentences))
                else:
                    padding = 0
            elif padding == "corpus":
                padding = self._max_sentence_size
                if add_bos_eos:
                    padding += 2  # type: ignore
            else:
                PADDINGS = (None, "batch", "corpus", int)
                raise ValueError(
                    f"Invalid argument {padding=}. (expected one of {PADDINGS})"
                )
        elif padding is None:
            padding = 0

        assert isinstance(padding, int)
        if padding > 0:
            tokenized_sentences = [
                sentence + [self.pad_token] * (padding - len(sentence))
                for sentence in tokenized_sentences
            ]

        return tokenized_sentences

    # Other Public methods
    def add_special_token(self, token: str, count: int = 0) -> int:
        """Add a new token to the vocabulary.

        :returns: The id of the new token.
        """
        if token in self._vocab:
            raise ValueError(f"Invalid argument {token=}. (already in vocab)")

        idx_max = max(max(self._itos.keys()), max(self._stoi.values()))
        new_token_id = idx_max + 1
        self._itos[new_token_id] = token
        self._stoi[token] = new_token_id
        self._vocab[token] = count
        self._added_special_tokens.append(token)
        return new_token_id

    def clear(self) -> None:
        """Clear the internal state (frequencies, max sentence size, ...) and allow to call .fit() method again."""
        self._max_sentence_size = -1
        self._min_sentence_size = sys.maxsize
        self._n_sentences_fit = 0
        self._itos = {}
        self._stoi = {}
        self._vocab = {}

    def decode_batch(self, sentences: Union[Tensor, Iterable[Iterable]]) -> list[str]:
        """Decode a batch of encoded sentences to decoded sentences.

        :param sentences: The input sentences to decode.
        :returns: The decoded (joined) sentences.
        """
        if not isinstance(sentences, Sized):
            sentences = list(sentences)

        # Empty list
        if len(sentences) == 0:
            return []

        # list[list[str]]
        # Note: must be before the next condition to avoid inf recursion when sentences=[[]]
        elif all(
            isinstance(token, str) for sentence in sentences for token in sentence
        ):
            return self.detokenize_batch(sentences)  # type: ignore

        # Tensor ndim=2, list[Tensor ndim=1], list[list[Tensor ndim=0]] or list[list[int]]
        elif all(
            (isinstance(token, int) or (isinstance(token, Tensor) and token.ndim == 0))
            for sentence in sentences
            for token in sentence
        ):
            sentences = [
                [self.id_to_token(token) for token in sentence]
                for sentence in sentences
            ]
            return self.decode_batch(sentences)

        else:
            raise TypeError(
                "Invalid sentences type in decode_batch method. (expected Tensor with ndim=2, list[list[str]] or list[list[int]])"
            )

    def decode_rec(self, nested_sentences: Union[Tensor, Iterable]) -> Union[str, list]:
        """Decode recursively a tensor of sentences."""

        if not (
            (isinstance(nested_sentences, Tensor) and nested_sentences.ndim > 0)
            or isinstance(nested_sentences, Iterable)
        ):
            raise TypeError(
                f"Invalid {nested_sentences=} for decode_rec method in {self.__class__.__name__}. (expected a Tensor of ndim > 0 or a list)"
            )

        if isinstance(nested_sentences, Tensor):
            return self.decode_rec(nested_sentences.tolist())
        elif _is_encoded_sentence(nested_sentences):
            return self.decode_single(nested_sentences)
        elif isinstance(nested_sentences, Iterable) and all(
            map(_is_encoded_sentence, nested_sentences)
        ):
            return self.decode_batch(nested_sentences)
        else:
            return [self.decode_rec(sentences) for sentences in nested_sentences]

    def decode_single(self, sentence: Union[Tensor, Iterable]) -> str:
        """Decode a tensor representing a single sentence."""
        return self.decode_batch([sentence])[0]

    def encode_batch(
        self,
        sentences: Iterable[str],
        add_bos_eos: bool = True,
        out_type: str = "Tensor",
        default: Union[None, str, int] = None,
        padding: Union[None, int, str] = None,
        dtype: torch.dtype = torch.long,
        device: Union[str, torch.device, None] = "cpu",
    ) -> Union[Tensor, list]:
        """
        :param sentences: The input sentences to encode.
        :param add_bos_eos: If True, pre-pend BOS_TOKEN and append EOS_TOKEN to each sentence.
            defaults to False.
        :param out_type: The output type value. Can be "str", "int", "pt" or "Tensor".
            If "str", tokenize the sentence and return the output strings tokens.
            If "int", tokenize the sentence and map the tokens to their corresponding indices.
            If "pt" or "Tensor", tokenize the sentence, map the tokens to theirs corresponding indices and convert to tensor the result if possible.
            defaults to "str".
        :param default: The special unknown token to return if out_type in ("int", "pt", "Tensor") and the token is not in vocabulary.
            If None, it will raise a KeyError if a token cannot be found.
            If ... (ellipsis), it will uses the internal unk_token value.
            defaults to ....
        :param padding: The padding size or mode.
            If a int, sentences will be padded until the specified size.
            If "batch", sentences will be padded until the maximal size in the batch.
            If "corpus", sentences will be padded until the maximal size stored by the corpus fit. (i.e. :meth:`~get_max_sentence_size`)
        :param dtype: Torch output dtype. defaults to torch.long.
        :param device: Torch output device. defaults to "cpu".
        """
        if out_type == "str":
            pylog.warning(
                f"Argument value {out_type=} is depreciated. Please use tokenizer.tokenize_rec method instead."
            )

        tokenized_sentences = self.tokenize_batch(sentences, add_bos_eos, padding)

        if out_type == "str":
            pass
        elif out_type in ("int", "Tensor", "pt"):
            if default is None:
                invalid_tokens = [
                    token
                    for sentence in tokenized_sentences
                    for token in sentence
                    if token not in self._stoi
                ]
                if len(invalid_tokens) > 0:
                    pylog.error("Tokenizer hparams:")
                    pylog.error(self.get_hparams())
                    pylog.error(f"Tokenizer is fit: {self.is_fit()}")
                    pylog.error(f"Tokenizer vocab_size: {self.get_vocab_size()}")
                    raise ValueError(
                        f"Invalid sentences tokens (found tokens {invalid_tokens} not in vocabulary from {tokenized_sentences=}, {add_bos_eos=}, {out_type=}, {default=})."
                    )

            tokenized_sentences = [
                [self.token_to_id(token, default) for token in sentence]
                for sentence in tokenized_sentences
            ]
            if out_type in ("Tensor", "pt"):
                if device == "auto":
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                if isinstance(device, str):
                    device = torch.device(device)

                if len(tokenized_sentences) == 0 or all(
                    len(sentence) == len(tokenized_sentences[0])
                    for sentence in tokenized_sentences
                ):
                    tokenized_sentences = torch.as_tensor(
                        tokenized_sentences,
                        dtype=dtype,
                        device=device,
                    )
                else:
                    tokenized_sentences = [
                        torch.as_tensor(sentence, dtype=dtype, device=device)
                        for sentence in tokenized_sentences
                    ]
        else:
            raise ValueError(
                f"Invalid argument {out_type=}. (expected one of {AACTokenizer.OUT_TYPES})"
            )

        return tokenized_sentences

    def encode_rec(
        self,
        nested_sentences: Union[str, Iterable],
        add_bos_eos: bool = True,
        out_type: str = "Tensor",
        default: Union[None, str, int] = None,
        padding: Union[None, int, str] = None,
        dtype: torch.dtype = torch.long,
        device: Union[str, torch.device, None] = "cpu",
    ) -> Union[Tensor, list]:
        """Encode sentences recursively.

        :param nested_sentences: The input sentences to encode.
        :param add_bos_eos: If True, pre-pend BOS_TOKEN and append EOS_TOKEN to each sentence.
            defaults to False.
        :param out_type: The output type value. Can be "str", "int", "pt" or "Tensor".
            If "str", tokenize the sentence and return the output strings tokens.
            If "int", tokenize the sentence and map the tokens to their corresponding indices.
            If "pt" or "Tensor", tokenize the sentence, map the tokens to theirs corresponding indices and convert to tensor the result if possible.
            defaults to "str".
        :param default: The special unknown token to return if out_type in ("int", "pt", "Tensor") and the token is not in vocabulary.
            If None, it will raise a KeyError if a token cannot be found.
            If ... (ellipsis), it will uses the internal unk_token value.
            defaults to ....
        :param padding: The padding size or mode.
            If a int, sentences will be padded until the specified size.
            If "batch", sentences will be padded until the maximal size in the batch.
            If "corpus", sentences will be padded until the maximal size stored by the corpus fit. (i.e. :meth:`~get_max_sentence_size`)
        :param dtype: Torch output dtype. defaults to torch.long.
        :param device: Torch output device. defaults to "cpu".
        """
        kwds: dict[str, Any] = dict(
            add_bos_eos=add_bos_eos,
            out_type=out_type,
            default=default,
            padding=padding,
            dtype=dtype,
            device=device,
        )

        if isinstance(nested_sentences, str):
            return self.encode_single(nested_sentences, **kwds)
        elif all(isinstance(sentence, str) for sentence in nested_sentences):
            return self.encode_batch(nested_sentences, **kwds)
        else:
            if self.get_level() == "ptb":
                __warn_once(
                    "Using encore_rec(.) method with 'ptb' token level is slow on nested sentences because it will call the internal java program several times."
                    "It is highly recommended to flatten sentences before or use another tokenizer backend."
                )
            nested_sentences = [
                self.encode_rec(sentences, **kwds) for sentences in nested_sentences
            ]
            if out_type in ("pt", "Tensor"):
                dtype = torch.long
                device = torch.device("cpu")

                if len(nested_sentences) == 0:
                    nested_sentences = torch.as_tensor(
                        nested_sentences,
                        dtype=dtype,
                        device=device,
                    )
                elif all(isinstance(s, Tensor) and nested_sentences[0].shape == s.shape for s in nested_sentences):  # type: ignore
                    nested_sentences = torch.stack(nested_sentences)  # type: ignore
            return nested_sentences

    def encode_single(
        self,
        sentence: str,
        add_bos_eos: bool = True,
        out_type: str = "Tensor",
        default: Union[None, str, int] = None,
        padding: Union[None, int, str] = None,
        dtype: torch.dtype = torch.long,
        device: Union[str, torch.device, None] = "cpu",
    ) -> Tensor:
        """
        :param sentence: The input sentence to encode.
        :param add_bos_eos: If True, pre-pend BOS_TOKEN and append EOS_TOKEN to each sentence.
            defaults to True.
        :param out_type: The output type value. Can be "str", "int", "pt" or "Tensor".
            If "str", tokenize the sentence and return the output strings tokens.
            If "int", tokenize the sentence and map the tokens to their corresponding indices.
            If "pt" or "Tensor", tokenize the sentence, map the tokens to theirs corresponding indices and convert to tensor the result if possible.
            defaults to "str".
        :param default: The special unknown token to return if out_type in ("int", "pt", "Tensor") and the token is not in vocabulary.
            If None, it will raise a KeyError if a token cannot be found.
            If ... (ellipsis), it will uses the internal unk_token value.
            defaults to ....
        :param padding: The padding size or mode.
            If a int, sentences will be padded until the specified size.
            If "batch", it has no effect for this method.
            If "corpus", sentences will be padded until the maximal size stored by the corpus fit. (i.e. :meth:`~get_max_sentence_size`)
        :param dtype: Torch output dtype. defaults to torch.long.
        :param device: Torch output device. defaults to "cpu".
        """
        if padding == "batch":
            __warn_once(
                f"Argument {padding=} does nothing when encoding a single caption."
            )
        return self.encode_batch(
            [sentence], add_bos_eos, out_type, default, padding, dtype, device
        )[0]

    def get_counts(self) -> dict[str, int]:
        """Get the frequencies of each token in the fit sentences."""
        return self._vocab

    def get_hparams(self) -> dict[str, Any]:
        """Get tokenizer hyperparameters."""
        return self._hparams

    def get_max_sentence_size(self) -> int:
        """Returns the maximal sentence size. If the tokenizer is not fit, returns -1."""
        return self._max_sentence_size

    def get_min_sentence_size(self) -> int:
        """Returns the minimal sentence size. If the tokenizer is not fit, returns sys.maxsize."""
        return self._min_sentence_size

    def get_vocab(self) -> dict[str, int]:
        """Returns the vocabulary with the number of occurrence for each token."""
        return self._vocab

    def get_vocab_size(self) -> int:
        """Returns the size of the vocabulary, i.e. `len(tokenizer.get_vocab())`."""
        return len(self._vocab)

    def has(self, token: str) -> bool:
        return token in self._vocab

    def is_fit(self) -> bool:
        return self._n_sentences_fit > 0

    def is_normalization_enabled(self) -> bool:
        return self._normalize

    def id_to_token(self, index: Union[int, Tensor]) -> str:
        if isinstance(index, Tensor):
            if index.ndim != 0 or index.is_floating_point():
                raise ValueError(
                    f"Invalid argument {index=}. (expected an int or a scalar integer tensor)"
                )
            index = int(index.item())
        return self._itos[index]

    def set_count(self, token: str, freq: int) -> None:
        """Set the frequency count for a specific token.

        :param token: The token to modify.
        :param freq: The frequency count to set.
        :raises: KeyError if token is not in tokenizer vocab.
        """
        self._vocab[token] = freq

    def token_to_id(self, token: str, default: Union[None, str, int] = None) -> int:
        """Returns the correponding id of a token.

        If default is None and token is not in the tokenizer vocabulary, raises a KeyError.
        If default is not None and token is not in the tokenizer vocabulary, returns token_to_id(default).
        If default is ... (ellipsis) and token is not in the tokenizer vocabulary, it will use the internal unk_token_id as default output.
        Otherwise returns the id of the token.

        :param token: The input token.
        :param default: If not None, it is the default value of the token.
        :returns: The id of the token.
        """
        if default is ...:
            default = self.unk_token_id

        if default is None:
            return self._stoi[token]

        elif isinstance(default, str):
            if default in self._stoi:
                return self._stoi.get(token, self._stoi[default])
            else:
                raise KeyError(
                    f"Invalid default value {default=}. (not found in stoi map with vocab_size={self.get_vocab_size()})"
                )

        elif isinstance(default, int):
            return self._stoi.get(token, default)
        else:
            raise TypeError(
                f"Invalid argument type {type(default)=}. (expected type None, str or int)"
            )

    # Serialization
    @classmethod
    def from_file(cls, fpath: str) -> "AACTokenizer":
        """Save tokenizer from a pickle, yaml or json file."""

        if fpath.endswith(".pkl") or fpath.endswith(".pickle"):
            with open(fpath, "rb") as file:
                tokenizer = pickle.load(file)

        elif fpath.endswith(".yaml"):
            with open(fpath, "r") as file:
                state = yaml.safe_load(file)
            tokenizer = AACTokenizer()
            tokenizer.set_txt_state(state)

        elif fpath.endswith(".json"):
            with open(fpath, "r") as file:
                state = json.load(file)
            tokenizer = AACTokenizer()
            tokenizer.set_txt_state(state)

        else:
            raise ValueError(
                f"Invalid extension for {fpath=}. (expected pickle, yaml or json)"
            )

        return tokenizer

    @classmethod
    def from_txt_state(cls, state: Mapping[str, Any]) -> "AACTokenizer":
        tokenizer_data = state["tokenizer"]
        hparams = tokenizer_data["hparams"]

        tokenizer = AACTokenizer(**hparams)

        tokenizer._hparams = tokenizer_data["hparams"]
        tokenizer._normalize = tokenizer_data["normalize"]
        tokenizer._added_special_tokens = tokenizer_data["added_special_tokens"]
        tokenizer._max_sentence_size = tokenizer_data["max_sentence_size"]
        tokenizer._min_sentence_size = tokenizer_data["min_sentence_size"]
        tokenizer._n_sentences_fit = tokenizer_data["n_sentences_fit"]
        tokenizer._itos = tokenizer_data["itos"]
        tokenizer._stoi = tokenizer_data["stoi"]
        tokenizer._vocab = tokenizer_data["vocab"]
        return tokenizer

    def save_file(self, fpath: str) -> None:
        """Save tokenizer to a pickle, yaml or json file."""

        if fpath.endswith(".pkl") or fpath.endswith(".pickle"):
            with open(fpath, "wb") as file:
                pickle.dump(self, file)
        elif fpath.endswith(".yaml"):
            with open(fpath, "w") as file:
                yaml.dump(self.get_txt_state(), file)
        elif fpath.endswith(".json"):
            with open(fpath, "w") as file:
                json.dump(self.get_txt_state(), file)
        else:
            raise ValueError(
                f"Invalid extension for {fpath=}. (expected pickle, yaml or json)"
            )

    def get_state(self, type_: str = "txt") -> dict[str, Any]:
        if type_ == "txt":
            return self.get_txt_state()
        elif type_ == "bin":
            return self.get_bin_state()
        else:
            raise ValueError(f"Invalid argument {type_=}.")

    def set_state(self, state: Mapping[str, Any]) -> None:
        type_ = state.get("_type_", "bin")
        if type_ == "txt":
            return self.set_txt_state(state)
        elif type_ == "bin":
            return self.set_bin_state(state)
        else:
            raise ValueError(f"Invalid argument {type_=}.")

    def get_bin_state(self) -> dict[str, Any]:
        tokenizer_data = self.__dict__
        tokenizer_data = {k: v for k, v in tokenizer_data.items() if "hook" not in k}

        state = {
            "_target_": _get_full_class_name(self),
            "_version_": AACTokenizer.VERSION,
            "_type_": "bin",
            "tokenizer": tokenizer_data,
        }
        return state

    def set_bin_state(self, state: Mapping[str, Any]) -> None:
        if not isinstance(state, dict) or "tokenizer" not in state.keys():
            raise TypeError(
                f"Incompatible pickle value type {type(state)}. (expected dict with key 'tokenizer')"
            )

        source_version = state.get("_version_", "1.0.0")
        state_version = source_version

        if state_version == "1.0.0":
            # 1.0.0 -> 2.0.0
            state["tokenizer"] = {
                k.removeprefix("_AACTokenizer_"): v
                for k, v in state["tokenizer"].items()
            }
            state_version = "2.0.0"

        if state_version == "2.0.0":
            # 2.0.0 -> 2.1.0

            hparams = state["tokenizer"].get("_hparams", {})
            if "punctuation_mode" in hparams:
                pass
            else:
                clean_punctuation = hparams.pop("clean_punctuation", None)

                # note: keep '==' here since we could have non-boolean values
                if clean_punctuation == True:  # noqa: E712
                    punctuation_mode = "remove"
                elif clean_punctuation == False:  # noqa: E712
                    punctuation_mode = "keep"
                else:
                    raise ValueError(
                        f"Invalid value {clean_punctuation=}."
                        f"(with {source_version=} and expected 'clean_punctuation' in {tuple(hparams.keys())})"
                    )

                state["tokenizer"]["_hparams"]["punctuation_mode"] = punctuation_mode

            state_version = "2.1.0"

        if state_version == "2.1.0":
            state["_normalize"] = True
            state["_added_special_tokens"] = []
            state_version = "2.2.0"

        tokenizer_state: dict[str, Any] = state["tokenizer"]
        OLD_BOS = "<sos>"
        NEW_BOS = "<bos>"
        if OLD_BOS in tokenizer_state["_stoi"]:
            idx = tokenizer_state["_stoi"].pop(OLD_BOS)
            tokenizer_state["_stoi"][NEW_BOS] = idx
            tokenizer_state["_itos"][idx] = NEW_BOS
            tokenizer_state["_vocab"][NEW_BOS] = tokenizer_state["_vocab"].pop(OLD_BOS)

        if state_version != self.VERSION:
            raise RuntimeError(
                f"Invalid tokenizer version. (from {state_version=}, {source_version=} and {self.VERSION=})"
            )

        AACTokenizer.__init__(self)
        for k, v in tokenizer_state.items():
            self.__setattr__(k, v)

    def get_txt_state(self) -> dict[str, Any]:
        tokenizer_data = {
            "hparams": self._hparams,
            "normalize": self._normalize,
            "added_special_tokens": self._added_special_tokens,
            "max_sentence_size": self._max_sentence_size,
            "min_sentence_size": self._min_sentence_size,
            "n_sentences_fit": self._n_sentences_fit,
            "itos": self._itos,
            "stoi": self._stoi,
            "vocab": self._vocab,
        }
        state = {
            "_target_": _get_full_class_name(self),
            "_version_": AACTokenizer.VERSION,
            "_type_": "txt",
            "tokenizer": tokenizer_data,
        }
        return state

    def set_txt_state(self, state: Mapping[str, Any]) -> None:
        tokenizer = AACTokenizer.from_txt_state(state)
        for k, v in tokenizer.__dict__.items():
            self.__setattr__(k, v)

    # Magic methods
    def __contains__(self, __o: object) -> bool:
        return isinstance(__o, str) and self.has(__o)

    def __eq__(self, __o: object) -> bool:
        return isinstance(__o, AACTokenizer) and pickle.dumps(self) == pickle.dumps(__o)

    def __getitem__(self, input_: str) -> int:
        return self.token_to_id(input_)

    def __getstate__(self) -> dict[str, Any]:
        return self.get_state()

    def __len__(self) -> int:
        return self.get_vocab_size()

    def __setstate__(self, state: dict[str, Any]) -> None:
        return self.set_state(state)


def _get_full_class_name(obj: Any) -> str:
    """Returns the classname of an object with parent modules.

    Example 1
    ----------
    >>> _get_obj_fullname(torch.nn.Linear(10, 10))
    'torch.nn.modules.linear.Linear'
    """
    class_ = obj.__class__
    module = class_.__module__
    if module == "builtins":
        # avoid outputs like 'builtins.str'
        return class_.__qualname__
    return module + "." + class_.__qualname__


def _is_encoded_sentence(inputs: Any) -> bool:
    """Returns true if inputs is one of:
    - list[int]
    - list[str]
    - list[Tensor with ndim=0]
    - Tensor with ndim=1
    """
    return (
        is_tokenized_sent_single(inputs)  # type: ignore
        or (
            isinstance(inputs, list)
            and all(
                (
                    isinstance(inputs_i, int)
                    or (isinstance(inputs_i, Tensor) and inputs_i.ndim == 0)
                )
                for inputs_i in inputs
            )
        )
        or (isinstance(inputs, Tensor) and inputs.ndim == 1)
    )


@cache
def __warn_once(msg: str) -> None:
    pylog.warning(msg)


def _get_pre_encoding_normalizers(
    lowercase: bool, punctuation_mode: str
) -> list[NormalizerI]:
    pre_encoding_normalizers: list[NormalizerI] = [
        CleanSpecialTokens(),
        ReplaceRarePuncChars(),
    ]
    if lowercase:
        pre_encoding_normalizers.append(Lowercase())

    if punctuation_mode == "remove":
        pre_encoding_normalizers.append(CleanPunctuation())

    elif punctuation_mode == "keep_comma":
        pattern = CleanPunctuation.PUNC_PATTERN
        pattern = pattern.replace(",", "")
        pre_encoding_normalizers.append(CleanPunctuation(pattern))
        pre_encoding_normalizers.append(CleanSpacesBeforePunctuation())

    elif punctuation_mode == "keep_comma_dot":
        pattern = CleanPunctuation.PUNC_PATTERN
        pattern = pattern.replace(",", "").replace(".", "")
        pre_encoding_normalizers.append(CleanPunctuation(pattern))
        pre_encoding_normalizers.append(CleanSpacesBeforePunctuation())

    elif punctuation_mode == "keep_hyphen":
        pattern = CleanPunctuation.PUNC_PATTERN
        pattern = pattern.replace(r"\-", r"")
        pre_encoding_normalizers.append(CleanPunctuation(pattern))

    elif punctuation_mode == "keep":
        pre_encoding_normalizers.append(CleanSpacesBeforePunctuation())

    else:
        raise ValueError(
            f"Invalid argument {punctuation_mode=}. (expected one of {AACTokenizer.PUNCTUATION_MODES})"
        )

    pre_encoding_normalizers += [
        CleanDoubleSpaces(),
        Strip(),
    ]
    return pre_encoding_normalizers


def _get_post_decoding_normalizers(lowercase: bool) -> list[NormalizerI]:
    post_decoding_normalizers: list[NormalizerI] = [
        CleanSpecialTokens(),
        CleanSpacesBeforePunctuation(),
        Strip(),
        CleanDoubleSpaces(),
        CleanHyphenSpaces(),
    ]
    if lowercase:
        post_decoding_normalizers.append(Lowercase())
    return post_decoding_normalizers
