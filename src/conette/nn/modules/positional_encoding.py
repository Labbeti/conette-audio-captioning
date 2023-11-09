#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math

import torch

from torch import nn, Tensor
from torch.nn.init import uniform_
from torch.nn.parameter import Parameter


class PositionalEncoding(nn.Module):
    # BASED ON PYTORCH TUTORIAL : https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    def __init__(
        self,
        emb_size: int,
        dropout_p: float,
        maxlen: int = 5000,
    ) -> None:
        super().__init__()
        den = torch.exp(-torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout_p)
        self.register_buffer("pos_embedding", pos_embedding)
        self.pos_embedding: Tensor

    def forward(self, token_embedding: Tensor) -> Tensor:
        pos_embedding_value = self.pos_embedding[: token_embedding.size(0), :]
        output = self.dropout(token_embedding + pos_embedding_value)
        return output


class LearnablePositionalEncoding(nn.Module):
    def __init__(
        self,
        emb_size: int,
        dropout_p: float,
        pos_emb_init: str = "sin",
        maxlen: int = 5000,
    ) -> None:
        super().__init__()

        if pos_emb_init == "sin":
            den = torch.exp(-torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
            pos = torch.arange(0, maxlen).reshape(maxlen, 1)
            pos_embedding = torch.zeros((maxlen, emb_size))
            pos_embedding[:, 0::2] = torch.sin(pos * den)
            pos_embedding[:, 1::2] = torch.cos(pos * den)
            pos_embedding = pos_embedding.unsqueeze(-2)

        elif pos_emb_init == "uniform":
            pos_embedding = torch.empty((maxlen, emb_size))
            # based on https://discuss.pytorch.org/t/how-are-layer-weights-and-biases-initialized-by-default/13073
            stdv = 1.0 / math.sqrt(emb_size)
            uniform_(pos_embedding, -stdv, stdv)
            pos_embedding = pos_embedding.unsqueeze(-2)

        else:
            POS_EMB_INITS = ("sin", "uniform")
            raise ValueError(
                f"Invalid argument {pos_emb_init=}. (expected one of {POS_EMB_INITS})"
            )

        pos_embedding = Parameter(pos_embedding)

        self.dropout = nn.Dropout(dropout_p)
        self.register_parameter("pos_embedding", pos_embedding)
        self.pos_embedding: Tensor

    def forward(self, x: Tensor) -> Tensor:
        pos_embedding_value = self.pos_embedding[: x.size(0), :]
        output = self.dropout(x + pos_embedding_value)
        return output
