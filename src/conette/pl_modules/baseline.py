#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging

from typing import Any, Optional

import torch

from torch import nn, Tensor

from conette.nn.decoders.aac_tfmer import AACTransformerDecoder
from conette.nn.decoding.beam import generate
from conette.nn.decoding.forcing import teacher_forcing
from conette.nn.decoding.greedy import greedy_search
from conette.nn.encoders.ident import FrameIdentEncoder
from conette.nn.functional.indexes import randperm_diff
from conette.nn.functional.mask import (
    lengths_to_pad_mask,
    tensor_to_pad_mask,
)
from conette.nn.loss.ce_mean import CrossEntropyLossMean
from conette.pl_modules.base import AACLightningModule
from conette.pl_modules.common import (
    build_proj_lin,
    get_forbid_rep_mask,
    TrainBatch,
    ValBatch,
    TestBatch,
)
from conette.tokenization.aac_tokenizer import AACTokenizer
from conette.transforms.mixup import sample_lambda


pylog = logging.getLogger(__name__)


class BaselinePLM(AACLightningModule):
    def __init__(
        self,
        # Model params
        label_smoothing: float = 0.1,
        gen_val_cands: str = "generate",
        mixup_alpha: float = 0.4,
        # Encoder params
        proj_name: str = "lin768",
        # Decoder params
        min_pred_size: int = 3,
        max_pred_size: Optional[int] = None,
        beam_size: int = 10,
        nhead: int = 8,
        d_model: int = 256,
        num_decoder_layers: int = 6,
        decoder_dropout_p: float = 0.2,
        dim_feedforward: int = 2048,
        acti_name: str = "gelu",
        # Optimizer params
        optim_name: str = "AdamW",
        lr: float = 5e-4,
        weight_decay: float = 2.0,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        use_custom_wd: bool = True,
        # Scheduler params
        sched_name: str = "cos_decay",
        sched_n_steps: Optional[int] = None,
        sched_interval: str = "epoch",
        sched_freq: int = 1,
        # Other params
        train_tokenizer: Optional[AACTokenizer] = None,
        verbose: int = 0,
    ) -> None:
        super().__init__(train_tokenizer)

        self.train_criterion: nn.Module = nn.Identity()
        self.encoder: nn.Module = nn.Identity()
        self.decoder: AACTransformerDecoder = None  # type: ignore
        self.projection: nn.Module = nn.Identity()

        self.save_hyperparameters(ignore=("train_tokenizer",))

        if self.tokenizer.is_fit():
            self.setup()

    # --- Setup methods
    def build_model(self) -> None:
        if self.is_built():
            raise RuntimeError("Cannot build model twice.")

        if not self.tokenizer.is_fit():
            raise RuntimeError(
                f"AACTokenizer is not fit for {self.__class__.__name__}."
            )

        tok_max_sent_size = self.tokenizer.get_max_sentence_size()
        if self.hp.max_pred_size is None:
            self.hparams.max_pred_size = tok_max_sent_size  # type: ignore
            if self.hp.verbose >= 1:
                pylog.info(f"Auto-detect value {self.hp.max_pred_size=}.")
        else:
            if self.hp.verbose >= 1:
                pylog.info(
                    f"Set {self.hp.max_pred_size=}. (with tokenizer max={tok_max_sent_size})"
                )

        self.train_criterion = nn.CrossEntropyLoss(
            ignore_index=self.pad_id,
            label_smoothing=self.hp.label_smoothing,
        )
        self.val_criterion = CrossEntropyLossMean(ignore_index=self.pad_id, dim=1)
        self.encoder = FrameIdentEncoder()

        if self.hp.proj_name == "lin2048":
            self.projection = build_proj_lin(2048, self.hp.d_model, False)
        elif self.hp.proj_name == "lin768":
            self.projection = build_proj_lin(768, self.hp.d_model, False)
        else:
            raise ValueError(f"Invalid argument {self.hp.proj_name=}.")

        self.decoder = AACTransformerDecoder(
            vocab_size=self.tokenizer.get_vocab_size(),
            bos_id=self.bos_id,
            eos_id=self.eos_id,
            pad_id=self.pad_id,
            acti_name=self.hp.acti_name,
            d_model=self.hp.d_model,
            dim_feedforward=self.hp.dim_feedforward,
            dropout=self.hp.decoder_dropout_p,
            nhead=self.hp.nhead,
            num_decoder_layers=self.hp.num_decoder_layers,
        )

        forbid_rep_mask = get_forbid_rep_mask(
            "content_words",
            self.tokenizer,
            self.device,
            self.hp.verbose,
        )

        self.forbid_rep_mask: Optional[Tensor]
        self.register_buffer("forbid_rep_mask", forbid_rep_mask)

    def is_built(self) -> bool:
        return self.decoder is not None

    # --- Train, val and test methods
    def training_step(self, batch: TrainBatch, *args, **kwargs) -> Tensor:
        audio = batch["audio"]
        audio_shape = batch["audio_shape"]
        captions = batch["captions"]

        bsize = captions.shape[0]
        # if precomputed, audio: (bsize, n_channels=1, time_steps=31, emb_size=2048)
        # captions : (bsize, max_cap_size)

        # Apply mixup on audio and input token embs
        indexes = randperm_diff(bsize, device=self.device)
        audio, audio_shape, lbd = self.mix_audio(audio, audio_shape, indexes)

        caps_in = captions[:, :-1]
        caps_out = captions[:, 1:]
        del captions

        caps_in_pad_mask = tensor_to_pad_mask(caps_in, pad_value=self.pad_id)

        caps_in = self.decoder.emb_layer(caps_in)
        caps_in = caps_in * lbd + caps_in[indexes] * (1.0 - lbd)

        # Forward
        encoder_outs = self.encode_audio(audio, audio_shape)
        logits = self.decode_audio(
            encoder_outs,
            "forcing",
            caps_in=caps_in,
            caps_in_pad_mask=caps_in_pad_mask,
        )  # note: use mixed prev tokens

        loss = self.train_criterion(logits, caps_out)  # note: use unmixed target

        with torch.no_grad():
            scores = {
                "loss": loss,
            }
            prefix = "train"
            scores = {f"{prefix}/{k}": v for k, v in scores.items()}
            self.log_dict(
                scores,
                batch_size=bsize,
            )

        return loss

    def validation_step(self, batch: ValBatch, *args, **kwargs) -> Any:
        audio = batch["audio"]
        audio_shape = batch["audio_shape"]
        mult_captions = batch["mult_captions"]
        bsize, n_caps_per_audio, _ = mult_captions.shape

        losses = torch.empty(
            size=(bsize, n_caps_per_audio),
            dtype=audio.dtype,
            device=audio.device,
        )

        encoder_outs = self.encode_audio(audio, audio_shape)

        for i in range(n_caps_per_audio):
            caps_in = mult_captions[:, i, :-1]
            caps_out = mult_captions[:, i, 1:]

            # logits : (bsize, vocab_size, capt_len)
            logits_i = self.decode_audio(encoder_outs, "forcing", caps_in=caps_in)
            losses_i = self.val_criterion(logits_i, caps_out)
            losses[:, i] = losses_i

        loss = losses.mean()

        if self.hp.gen_val_cands in ("none", None):
            output = None

        elif self.hp.gen_val_cands in ("greedy", "generate"):
            # Compute beam search results
            outs = self.decode_audio(encoder_outs, self.hp.gen_val_cands)
            if self.hp.gen_val_cands == "greedy":
                preds = outs.argmax(dim=1)
            else:
                preds = outs[0]

            cands = self.decode_text(preds)
            mrefs = batch["mult_references"]
            output = {
                f"cands_{self.hp.gen_val_cands}": cands,
                "mrefs": mrefs,
            }
        else:
            raise ValueError(f"Invalid argument {self.hp.gen_val_cands=}.")

        bar_scores = {"loss": loss}
        non_bar_scores = {}

        prefix = "val"
        bar_scores = {f"{prefix}/{k}": v for k, v in bar_scores.items()}
        non_bar_scores = {f"{prefix}/{k}": v for k, v in non_bar_scores.items()}

        log_kwargs: dict[str, Any] = dict(batch_size=bsize)
        self.log_dict(bar_scores, prog_bar=True, **log_kwargs)
        self.log_dict(non_bar_scores, prog_bar=False, **log_kwargs)

        return output

    def test_step(self, batch: TestBatch, *args, **kwargs) -> dict[str, Any]:
        audio = batch["audio"]
        audio_shape = batch["audio_shape"]
        mult_captions = batch["mult_captions"]

        bsize, n_caps_per_audio, _ = mult_captions.shape
        encoder_outs = self.encode_audio(audio, audio_shape)

        # Compute test loss
        losses = torch.empty(
            size=(bsize, n_caps_per_audio),
            dtype=audio.dtype,
            device=audio.device,
        )

        for i in range(n_caps_per_audio):
            caps_in = mult_captions[:, i, :-1]
            caps_out = mult_captions[:, i, 1:]
            logits_i = self.decode_audio(encoder_outs, "forcing", caps_in=caps_in)
            losses_i = self.val_criterion(logits_i, caps_out)
            losses[:, i] = losses_i

        loss = losses.mean()

        dataname = batch["dataset"][0]
        subset = batch["subset"][0]
        scores = {
            f"test/{dataname}_{subset}.loss": loss,
        }
        self.log_dict(scores, batch_size=bsize)

        # Compute beam search results
        preds, lprobs, mult_preds, mult_lprobs = self.decode_audio(
            encoder_outs, "generate"
        )
        outs = {
            "losses": losses,
            "preds": preds,
            "lprobs": lprobs,
            "mpreds": mult_preds,
            "mlprobs": mult_lprobs,
        }

        # Decode beam search results
        keys = [key for key in outs.keys() if "preds" in key]
        for key in keys:
            cands_key = key.replace("preds", "cands")

            preds = outs[key]
            cands = self.tokenizer.decode_rec(preds)

            outs[cands_key] = cands

        if "mult_references" in batch:
            outs["mrefs"] = batch["mult_references"]
        return outs

    def forward(
        self,
        batch: dict[str, Any],
        decode_method: str = "generate",
        **kwargs,
    ) -> dict[str, Tensor]:
        audio: Tensor = batch["audio"]
        audio_shape: Tensor = batch["audio_shape"]
        encoder_outs = self.encode_audio(audio, audio_shape)
        if decode_method == "forcing" and "captions" in batch:
            kwargs["caps_in"] = batch["captions"][:, :-1]
        outs = self.decode_audio(encoder_outs, decode_method, **kwargs)

        if decode_method == "generate":
            preds, lprobs, mult_preds, mult_lprobs = outs
            cands = self.decode_text(preds)
            mult_cands = self.decode_text(mult_preds)
            return {
                "cands": cands,
                "preds": preds,
                "lprobs": lprobs,
                "mult_cands": mult_cands,
                "mult_preds": mult_preds,
                "mult_lprobs": mult_lprobs,
            }
        else:
            return outs

    # --- Other methods
    def decode_audio(
        self,
        encoder_outs: dict[str, Tensor],
        decode_method: str,
        **kwargs,
    ) -> Any:
        if decode_method == "forcing":
            if "caps_in" not in kwargs.keys():
                raise ValueError(
                    f"Please provide a 'caps_in' keyword argument with {decode_method=}. (found {tuple(kwargs.keys())})"
                )
            forcing_hp: dict[str, Any] = {
                "pad_id": self.pad_id,
                "bos_id": self.bos_id,
                "eos_id": self.eos_id,
                "vocab_size": self.tokenizer.get_vocab_size(),
            }
            kwargs = forcing_hp | kwargs
            outs = teacher_forcing(
                self.decoder,
                **encoder_outs,
                **kwargs,
            )
        elif decode_method == "greedy":
            greedy_hp = {
                "pad_id": self.pad_id,
                "bos_id": self.bos_id,
                "eos_id": self.eos_id,
                "vocab_size": self.tokenizer.get_vocab_size(),
                "min_pred_size": self.hp.min_pred_size,
                "max_pred_size": self.hp.max_pred_size,
                "forbid_rep_mask": self.forbid_rep_mask,
            }
            kwargs = greedy_hp | kwargs
            outs = greedy_search(
                self.decoder,
                **encoder_outs,
                **kwargs,
            )

        elif decode_method == "generate":
            generate_hp = {
                "pad_id": self.pad_id,
                "bos_id": self.bos_id,
                "eos_id": self.eos_id,
                "vocab_size": self.tokenizer.get_vocab_size(),
                "min_pred_size": self.hp.min_pred_size,
                "max_pred_size": self.hp.max_pred_size,
                "forbid_rep_mask": self.forbid_rep_mask,
                "beam_size": self.hp.beam_size,
            }
            kwargs = generate_hp | kwargs
            outs = generate(
                self.decoder,
                **encoder_outs,
                **kwargs,
            )
        else:
            DECODE_METHODS = ("forcing", "greedy", "generate")
            raise ValueError(
                f"Unknown argument {decode_method=}. (expected one of {DECODE_METHODS})"
            )
        return outs

    def encode_audio(self, audio: Tensor, audio_shape: Tensor) -> dict[str, Tensor]:
        encoder_outs = self.encoder(audio, audio_shape)
        frame_embs = encoder_outs["frame_embs"]
        frame_embs_lens = encoder_outs.pop("frame_embs_lens")

        frame_embs = self.projection(frame_embs)
        # frame_embs shape: (bsize, emb_size, time_size)

        time_dim = -1
        frame_embs_max_len = max(frame_embs_lens.max(), frame_embs.shape[time_dim])
        frame_embs_pad_mask = lengths_to_pad_mask(frame_embs_lens, frame_embs_max_len)

        encoder_outs["frame_embs"] = frame_embs
        encoder_outs["frame_embs_pad_mask"] = frame_embs_pad_mask

        return encoder_outs

    def mix_audio(
        self,
        audio: Tensor,
        audio_shape: Tensor,
        indexes: Optional[Tensor],
    ) -> tuple[Tensor, Tensor, Tensor]:
        if indexes is None:
            return audio, audio_shape, torch.full((), 1.0)

        lbd = sample_lambda(
            self.hp.mixup_alpha,
            asymmetric=True,
            size=(),
        )
        mixed_audio = audio * lbd + audio[indexes] * (1.0 - lbd)
        mixed_audio_shape = torch.max(audio_shape, audio_shape[indexes])
        return mixed_audio, mixed_audio_shape, lbd
