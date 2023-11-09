#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import math

from typing import Any, Optional, Union

import torch

from torch import nn, Tensor

from conette.nn.decoding.common import AACDecoder
from conette.nn.functional.label import ints_to_multihots
from conette.nn.functional.mask import (
    generate_square_subsequent_mask,
    tensor_to_lengths,
)
from conette.nn.functional.repeat import repeat_interleave_nd


pylog = logging.getLogger(__name__)


@torch.no_grad()
def generate(
    decoder: AACDecoder,
    pad_id: int,
    bos_id: Union[int, Tensor],
    eos_id: int,
    vocab_size: int,
    frame_embs: Tensor,
    frame_embs_pad_mask: Tensor,
    beam_size: int = 2,
    min_pred_size: int = 0,
    max_pred_size: int = 20,
    forbid_rep_mask: Optional[Tensor] = None,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Generate sentences using beam search algorithm with some additional options.

    Note: works per batch, which can requires a lot of memory if batch_size and beam_size increases.

    :param decoder: The decoder part of the model.
    :param pad_id: Padding token id.
    :param bos_id: Begin-of-Sentence token id.
    :param eos_id: End-of-Sentence token id.
    :param vocab_size: Vocabulary size of the model.
    :param frame_embs: (bsize, frame_emb_size, n_frames)
    :param frame_embs_pad_mask: (bsize, audio_seq_size)
    :param beam_size: The number of beams for the search. defaults to 2.
    :param min_pred_size: Minimal number of tokens in the output sentences. defaults to 0.
    :param max_pred_size: Maximal number of tokens in the output sentences. defaults to 20.
    :param forbid_rep_mask: (vocab_size,) or None
    :returns: A tuple containing 4 tensors: (best_preds_out, best_avg_lprobs, global_preds_out, global_avg_lprobs) with the following shapes:
        best_preds_out: (bsize, max_best_pred_size)
        best_avg_lprobs: (bsize,)
        global_preds_out: (bsize, beam_size, max_global_pred_size)
        global_avg_lprobs: (bsize, beam_size)
    """
    assert beam_size > 0
    assert min_pred_size >= 0

    bsize = frame_embs.shape[0]
    device = frame_embs.device
    bkwds: dict[str, Any] = dict(dtype=torch.bool, device=device)
    fkwds: dict[str, Any] = dict(dtype=frame_embs.dtype, device=device)
    ikwds: dict[str, Any] = dict(dtype=torch.long, device=device)

    # frame_embs: (bsize, frame_emb_size, n_frames) -> (bsize*beam_size, frame_emb_size, n_frames)
    # frame_embs_pad_mask: (bsize, audio_seq_size) -> (bsize*beam_size, audio_seq_size)
    frame_embs = repeat_interleave_nd(frame_embs, beam_size)
    frame_embs_pad_mask = repeat_interleave_nd(frame_embs_pad_mask, beam_size)

    if isinstance(bos_id, int):
        bos_ids = torch.full((bsize,), bos_id, **ikwds)
    else:
        bos_ids = bos_id

    if bos_ids.is_floating_point():
        raise ValueError(f"Invalid argument {bos_id=}. (expected integer tensor)")
    if tuple(bos_ids.shape) != (bsize,):
        raise ValueError(
            f"Invalid argument {bos_id=}. (expected int or tensor of bsize values but found {bos_ids.shape=})"
        )
    del bos_id

    bod_ids = repeat_interleave_nd(bos_ids, beam_size)

    # frame_embs: (bsize*beam_size, frame_emb_size, n_frames) -> (n_frames, bsize*beam_size, frame_emb_size)
    frame_embs = frame_embs.permute(2, 0, 1)

    preds = torch.full((bsize * beam_size, max_pred_size + 1), pad_id, **ikwds)
    preds[:, 0] = bod_ids
    # batch_idxs example: [0, 0, 0, 0, 1, 1, 1, 1]
    batch_idxs = torch.as_tensor(
        [i for i in range(bsize) for _j in range(beam_size)], **ikwds
    )
    # beam_idxs example: [0, 1, 2, 3, 0, 1, 2, 3]
    beam_idxs = torch.as_tensor(
        [j for _i in range(bsize) for j in range(beam_size)], **ikwds
    )
    sum_lprobs = torch.zeros((bsize * beam_size,), **fkwds)

    global_preds_out = torch.full((bsize * beam_size, max_pred_size), pad_id, **ikwds)
    global_is_finished = torch.full((bsize * beam_size,), False, **bkwds)
    global_avg_lprobs = torch.zeros((bsize * beam_size,), **fkwds)

    arange = torch.arange(bsize, **ikwds)
    caps_in_sq_mask = generate_square_subsequent_mask(max_pred_size, device)
    if forbid_rep_mask is None:
        forbid_rep_mask = torch.zeros((vocab_size,), **bkwds)
    use_forbid_rep = forbid_rep_mask.eq(True).any().item()

    pred_size = max_pred_size

    for i in range(max_pred_size):
        preds_in_i = preds[:, : i + 1].transpose(0, 1)
        caps_in_sq_mask_i = caps_in_sq_mask[: i + 1, : i + 1]

        full_logits_i = decoder(
            frame_embs=frame_embs.contiguous(),
            frame_embs_pad_mask=frame_embs_pad_mask.contiguous(),
            caps_in=preds_in_i.contiguous(),
            caps_in_pad_mask=None,
            caps_in_sq_mask=caps_in_sq_mask_i.contiguous(),
        )
        # full_logits_i shape: (i+1, cur_size, vocab_size)
        logits_i = full_logits_i[-1]
        del full_logits_i
        # logits_i shape: (cur_size, vocab_size)

        if i < min_pred_size:
            logits_i[:, eos_id] = -math.inf

        mask_i = batch_idxs.unsqueeze(dim=0).eq(arange.unsqueeze(dim=1))
        # mask_i shape: (bsize, cur_size)
        # mask_i[A][B] is a bool tensor which indicates if example B is from batch A
        indexes = arange[mask_i.sum(dim=1) > 0]

        is_finished_i = torch.full((preds.shape[0],), False, **bkwds)
        # is_finished_i shape: (cur_size,)

        for j in indexes:
            mask_ij = mask_i[j]
            logits_ij = logits_i[mask_ij]
            sum_lprobs_ij = sum_lprobs[mask_ij]
            # logits_ij shape: (beam_size_ij, vocab_size)

            if use_forbid_rep:
                prev_preds = preds[mask_ij, : i + 1]
                # prev_preds shape: (beam_size_ij, i+1)
                prev_preds_mult_hot = ints_to_multihots(prev_preds, vocab_size, **bkwds)
                # prev_preds_ohot shape: (beam_size_ij, vocab_size)
                prev_preds_mult_hot = prev_preds_mult_hot.logical_and_(
                    forbid_rep_mask.unsqueeze(dim=0)
                )
                logits_ij[prev_preds_mult_hot] = -math.inf

            prev_beam_idxs, next_word_idxs, sum_lprobs_ij = _select_k_next_toks(
                logits_i=logits_ij,
                prev_sum_lprobs=sum_lprobs_ij,
                is_first=i == 0,
            )

            # Update lprob sum
            sum_lprobs[mask_ij] = sum_lprobs_ij
            # Update previous preds with previous beams selected by prev_beam_idxs
            preds[mask_ij, : i + 1] = preds[mask_ij][prev_beam_idxs, : i + 1]
            # Add next word for each beam
            preds[mask_ij, i + 1] = next_word_idxs.unsqueeze(dim=0)

            # mask_ij shape: (cur_size,)

            if i < max_pred_size - 1:
                is_finished_i[mask_ij] = next_word_idxs == eos_id
            else:
                is_finished_i[mask_ij] = True

        if is_finished_i.any():
            finished_global_idxs_i = (
                beam_idxs[is_finished_i] + batch_idxs[is_finished_i] * beam_size
            )

            # Update global outputs
            global_preds_out[finished_global_idxs_i, : i + 1] = preds[
                is_finished_i, 1 : i + 2
            ]
            global_is_finished[finished_global_idxs_i] = True
            global_avg_lprobs[finished_global_idxs_i] = sum_lprobs[is_finished_i].div(
                i + 1
            )

            if global_is_finished.eq(True).all():
                pred_size = i + 1
                break

        is_unfinished_i = is_finished_i.logical_not()

        frame_embs = frame_embs[:, is_unfinished_i]
        frame_embs_pad_mask = frame_embs_pad_mask[is_unfinished_i]
        preds = preds[is_unfinished_i]
        batch_idxs = batch_idxs[is_unfinished_i]
        beam_idxs = beam_idxs[is_unfinished_i]
        sum_lprobs = sum_lprobs[is_unfinished_i]

    global_preds_out = global_preds_out.reshape(bsize, beam_size, max_pred_size)
    global_is_finished = global_is_finished.reshape(bsize, beam_size)
    global_avg_lprobs = global_avg_lprobs.reshape(bsize, beam_size)

    if pred_size < max_pred_size:
        # Trunc global preds
        assert global_preds_out[:, :, pred_size:].eq(pad_id).all().item()
        global_preds_out = global_preds_out[:, :, :pred_size].contiguous()

    # global_preds_out shape: (bsize, beam_size, pred_size)
    # global_avg_lprobs shape: (bsize, beam_size)

    # Get best scores for each batch
    best_avg_lprobs, best_beams = global_avg_lprobs.max(dim=1)
    indexes = best_beams[:, None, None].expand(bsize, beam_size, pred_size)
    best_preds_out = global_preds_out.gather(1, indexes)[:, 0]

    # Trunc best preds
    best_preds_lens = tensor_to_lengths(best_preds_out, end_value=eos_id)
    best_preds_maxlen = best_preds_lens.max() + 1
    best_preds_out = best_preds_out[:, :best_preds_maxlen].contiguous()

    return best_preds_out, best_avg_lprobs, global_preds_out, global_avg_lprobs


def _select_k_next_toks(
    logits_i: Tensor,
    prev_sum_lprobs: Tensor,
    is_first: bool,
) -> tuple[Tensor, Tensor, Tensor]:
    """
    :param logits_i: (beam_size, vocab_size)
    :param prev_sum_lprobs: (beam_size,)
    :param is_first: Indicate if this is the first word predicted to avoid predict the same word at the beginning.
    """
    beam_size, vocab_size = logits_i.shape
    log_activation = nn.LogSoftmax(dim=1)

    if is_first:
        logits_i = logits_i[0].unsqueeze(dim=0)
        sum_lprobs = log_activation(logits_i)
        # sum_lprobs shape: (1, vocab_size)
    else:
        prev_sum_lprobs = prev_sum_lprobs.unsqueeze(dim=1).expand(
            beam_size,
            vocab_size,
        )
        lprobs_i = log_activation(logits_i)
        sum_lprobs = prev_sum_lprobs + lprobs_i
        # sum_lprobs shape: (beam_size, vocab_size)

    sum_lprobs_flat = sum_lprobs.view(-1)
    new_sum_lprobs, next_token_idxs_flat = torch.topk(sum_lprobs_flat, beam_size)

    prev_beam_idxs = next_token_idxs_flat.div(
        vocab_size,
        rounding_mode="trunc",
    )
    next_word_idxs = next_token_idxs_flat % vocab_size

    # prev_beam_idxs: shape is (beam_size,), values in [0, beam_size[
    # next_word_idxs: shape is (beam_size,), values in [0, vocab_size[
    # sum_lprobs_selected: shape is (beam_size,), values in ]-inf, 0]

    return prev_beam_idxs, next_word_idxs, new_sum_lprobs
