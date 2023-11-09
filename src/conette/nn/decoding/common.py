#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Any, Optional, Protocol

from torch import Tensor


class AACDecoder(Protocol):
    """Protocol for aac decoders. Similar to `torch.nn.TransformerDecoder` inputs."""

    def __call__(
        self,
        frame_embs: Tensor,
        frame_embs_pad_mask: Optional[Tensor],
        caps_in: Tensor,
        caps_in_pad_mask: Optional[Tensor],
        caps_in_sq_mask: Tensor,
    ) -> Tensor:
        """Decode audio embeddings + previous captions tokens to next token logits.

        :param frame_embs: (n_frames, bsize, emb_size)
        :param frame_embs_pad_mask: (bsize, n_frames) or None
        :param caps_in: (caps_in_len, bsize)
        :param caps_in_pad_mask: (caps_in_len, bsize) or None
        :param caps_in_sq_mask: (caps_in_len, caps_in_len)
        :returns: logits of shape (caps_in_len, bsize, vocab_size)
        """
        raise NotImplementedError("Protocol abstract method.")


class AACDecoderExpt(Protocol):
    """Protocol for aac decoders. Similar to `torch.nn.TransformerDecoder` inputs."""

    def __call__(
        self,
        frame_embs: Tensor,
        frame_embs_pad_mask: Optional[Tensor],
        caps_in: Tensor,
        caps_in_pad_mask: Optional[Tensor],
        caps_in_sq_mask: Tensor,
        **kwargs,
    ) -> Any:
        """Decode audio embeddings + previous captions tokens to next token logits.

        :param frame_embs: (n_frames, bsize, emb_size)
        :param frame_embs_pad_mask: (bsize, n_frames) or None
        :param caps_in: (caps_in_len, bsize)
        :param caps_in_pad_mask: (caps_in_len, bsize) or None
        :param caps_in_sq_mask: (caps_in_len, caps_in_len)
        :returns: logits of shape (caps_in_len, bsize, vocab_size)
        """
        raise NotImplementedError("Protocol abstract method.")
