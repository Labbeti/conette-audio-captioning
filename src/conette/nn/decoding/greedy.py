#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math

from typing import Any, Optional

import torch

from torch import Tensor

from conette.nn.decoding.common import AACDecoder
from conette.nn.functional.label import ints_to_multihots
from conette.nn.functional.mask import generate_square_subsequent_mask


@torch.no_grad()
def greedy_search(
    decoder: AACDecoder,
    pad_id: int,
    bos_id: int,
    eos_id: int,
    vocab_size: int,
    frame_embs: Tensor,
    frame_embs_pad_mask: Tensor,
    min_pred_size: int = 0,
    max_pred_size: int = 20,
    forbid_rep_mask: Optional[Tensor] = None,
) -> Tensor:
    """Greedy search for Transformer decoder.

    :param decoder: The decoder part of the model.
    :param pad_id: Padding token id.
    :param bos_id: Begin-of-Sentence token id.
    :param eos_id: End-of-Sentence token id.
    :param vocab_size: Vocabulary size of the model.
    :param frame_embs: (bsize, frame_emb_size, n_frames)
    :param frame_embs_pad_mask: (bsize, audio_seq_size)
    :param min_pred_size: Minimal number of tokens in the output sentences. defaults to 0.
    :param max_pred_size: Maximal number of tokens in the output sentences. defaults to 20.
    :param forbid_rep_mask: TODO
    :returns: logits of shape (bsize, vocab_size, max_pred_size or less)
    """
    assert min_pred_size >= 0

    bsize = frame_embs.shape[0]
    device = frame_embs.device
    bkwds: dict[str, Any] = dict(dtype=torch.bool, device=device)
    fkwds: dict[str, Any] = dict(dtype=frame_embs.dtype, device=device)
    ikwds: dict[str, Any] = dict(dtype=torch.long, device=device)

    # (bsize, emb_size, n_frames) -> (n_frames, bsize, emb_size)
    frame_embs = frame_embs.permute(2, 0, 1)

    batch_idxs = torch.arange(bsize, **ikwds)

    preds = torch.full(
        (bsize, max_pred_size + 1),
        pad_id,
        **ikwds,
    )
    preds[:, 0] = bos_id

    global_logits_out = torch.full(
        (bsize, vocab_size, max_pred_size),
        -math.inf,
        **fkwds,
    )
    global_logits_out[:, pad_id, :] = 0

    caps_in_sq_mask = generate_square_subsequent_mask(max_pred_size, device)
    if forbid_rep_mask is None:
        forbid_rep_mask = torch.zeros((vocab_size,), **bkwds)
    use_forbid_rep = forbid_rep_mask.eq(True).any()

    # unfinished sentence mask
    # unfinished_mask = torch.full((bsize,), True, device=device, dtype=torch.bool)
    pred_size = max_pred_size

    for i in range(max_pred_size):
        preds_in_i = preds[:, : i + 1].transpose(0, 1)
        caps_in_sq_mask_i = caps_in_sq_mask[: i + 1, : i + 1]

        full_logits_i = decoder(
            frame_embs.contiguous(),
            frame_embs_pad_mask.contiguous(),
            preds_in_i.contiguous(),
            None,
            caps_in_sq_mask_i.contiguous(),
        )
        # full_logits_i : (i+1, cur_size, vocab_size)
        logits_i = full_logits_i[-1]
        del full_logits_i
        # logits_i : (cur_size, vocab_size)

        if i < min_pred_size:
            logits_i[:, eos_id] = -math.inf

        if use_forbid_rep:
            prev_preds = preds[:, : i + 1]
            prev_preds_ohot = ints_to_multihots(prev_preds, vocab_size, **bkwds)
            prev_preds_ohot = prev_preds_ohot.logical_and_(
                forbid_rep_mask.unsqueeze(dim=0)
            )
            logits_i[prev_preds_ohot] = -math.inf

        next_toks_i = logits_i.argmax(dim=-1)
        # next_toks_i shape: (cur_size,)
        preds[:, i + 1] = next_toks_i

        if i < max_pred_size - 1:
            is_unfinished_i = next_toks_i != eos_id
        else:
            is_unfinished_i = torch.full((logits_i.shape[0],), False, **bkwds)

        global_logits_out[batch_idxs, :, i] = logits_i

        preds = preds[is_unfinished_i]
        batch_idxs = batch_idxs[is_unfinished_i]
        frame_embs = frame_embs[:, is_unfinished_i]
        frame_embs_pad_mask = frame_embs_pad_mask[is_unfinished_i]

        if preds.nelement() <= 0:
            pred_size = i + 1
            break

    if pred_size < max_pred_size:
        global_logits_out = global_logits_out[:, :, :pred_size].contiguous()

    # logits shape: (bsize, vocab_size, pred_size)
    return global_logits_out
