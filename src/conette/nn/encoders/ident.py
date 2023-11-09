#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Optional

from torch import nn, Tensor


class FrameIdentEncoder(nn.Module):
    def __init__(self, time_dim: int = 1) -> None:
        super().__init__()
        self.time_dim = time_dim

    def forward(
        self,
        x: Tensor,
        x_shape: Optional[Tensor],
    ) -> dict[str, Optional[Tensor]]:
        if x.ndim == 4:
            # Squeeze channel dim from (bsize, channel_size, seq_size, emb_size)
            x = x.squeeze(dim=1)

        if x_shape is None:
            x_lens = None
        else:
            x_lens = x_shape[:, self.time_dim]

        # x: (bsize, max_seq_size, emb_size)
        # x_lens: (bsize,) or None
        outs = {
            "frame_embs": x,
            "frame_embs_lens": x_lens,
        }
        return outs
