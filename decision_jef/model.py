"""Decision-Jef.

    h      = encoder(state + question blocks)
    query  = q_proj(h[DEC])        one per question
    keys   = o_proj(h[OPT_i])      one per option, in order
    logits = <query, key_i> * scale

The query is read after the whole option list, so the decision sees every
option. The keys come from the same pass, so the options are read together
rather than scored in isolation.

The answer space is exactly the options supplied at call time. There is no
classification head over a fixed label set, so a value you did not offer is
not representable.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel
from transformers.models.modernbert.modeling_modernbert import (
    create_bidirectional_mask,
    create_bidirectional_sliding_window_mask,
)

from decision_jef.pack import Packed, segment_attention_mask

KINDS = ("choice", "noul", "score")


class DecisionJef(nn.Module):
    def __init__(self, model_name: str = "jhu-clsp/mmBERT-base",
                 proj_dim: int = 256, dropout: float = 0.0,
                 trainable_layers: int = -1, isolate_questions: bool = True,
                 noul_head: bool = True, act_head: bool = True, pretrained: bool = False,
                 encoder_config: Optional[dict] = None):
        super().__init__()
        if encoder_config is not None:
            # Carried in the checkpoint, so loading touches no other
            # repository and prints no warning about unauthenticated requests.
            cfg = AutoConfig.for_model(**encoder_config)
        else:
            cfg = AutoConfig.from_pretrained(model_name)
        cfg.attention_dropout = dropout
        if pretrained:
            self.encoder = AutoModel.from_pretrained(model_name, config=cfg)
        else:
            # Inference replaces every weight from the checkpoint a moment
            # later, so fetching the backbone first downloads a gigabyte to
            # throw away -- and prints a load report about the MLM head this
            # model does not have. Build the architecture from the config and
            # let load_state_dict fill it.
            self.encoder = AutoModel.from_config(cfg)
        self.config = self.encoder.config
        d = self.config.hidden_size
        self.isolate_questions = isolate_questions

        # Separate projections for the question and the option side.
        self.q_proj = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, proj_dim))
        self.o_proj = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, proj_dim))
        # Learned temperature on normalised vectors.
        self.logit_scale = nn.Parameter(torch.tensor(2.996))
        # A yes/no question's two options are restatements of its
        # instructions, so its two logits are read straight off [DEC].
        # The outcome space is still exactly two.
        self.noul_head = nn.Sequential(
            nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, 2)
        ) if noul_head else None
        # Answer, or hand this decision to a person. Cost-sensitive, because
        # the two mistakes are not equal: escalating a case the model would
        # have got right wastes a person's time, and answering one it gets
        # wrong is the expensive failure. Read off [DEC] like the rest.
        self.act_head = nn.Sequential(
            nn.LayerNorm(d), nn.Linear(d, d // 4), nn.GELU(),
            nn.Linear(d // 4, 1)
        ) if act_head else None
        self.model_name = model_name
        self.proj_dim = proj_dim
        self.use_noul_head = noul_head
        self.set_trainable(trainable_layers)

    def set_trainable(self, n: int):
        """n < 0 unfreezes the whole encoder; n >= 0 keeps the top n layers."""
        if n < 0:
            for p in self.encoder.parameters():
                p.requires_grad_(True)
            return
        for p in self.encoder.parameters():
            p.requires_grad_(False)
        for layer in list(self.encoder.layers)[len(self.encoder.layers) - n:]:
            for p in layer.parameters():
                p.requires_grad_(True)
        if hasattr(self.encoder, "final_norm"):
            for p in self.encoder.final_norm.parameters():
                p.requires_grad_(True)

    def _masks(self, batch: Packed, hidden) -> Dict[str, torch.Tensor]:
        """The encoder's own masks, restricted to the per-question segments."""
        kw = dict(config=self.config, inputs_embeds=hidden,
                  attention_mask=batch.attention_mask)
        masks = {"full_attention": create_bidirectional_mask(**kw),
                 "sliding_attention": create_bidirectional_sliding_window_mask(**kw)}
        if not self.isolate_questions:
            return masks

        allowed = segment_attention_mask(batch.segment_ids, batch.attention_mask)
        # Built by hand rather than intersected with the encoder's own masks.
        # `create_bidirectional_mask` contributes only the padding restriction,
        # which segment_attention_mask already applies, and the sliding
        # constructor's band is indexed by absolute position -- exactly what
        # _window has to replace for the isolation promise to hold.
        return {"full_attention": allowed,
                "sliding_attention": allowed & self._window(batch)}

    def _window(self, batch: Packed) -> torch.Tensor:
        """The local-attention band the sliding layers use, [1, 1, L, L] bool."""
        half = self.config.local_attention // 2
        # Distance in restarted positions, not in absolute indices. Measured on
        # the released weights: with an index-based band, an 84-token question
        # placed in front pushed the next question's block past the 128-token
        # window and cost it sight of part of the shared state, moving its
        # answer by 2.1e-02 even with attention isolation exact. Positions
        # restart per question (see pack.py), so this band reproduces the
        # single-question geometry for every question in the request.
        pos = batch.position_ids
        band = (pos.unsqueeze(2) - pos.unsqueeze(1)).abs() <= half
        return band.unsqueeze(1)

    def escalate_logit(self, batch: Packed) -> Optional[torch.Tensor]:
        """One logit per question: how much this decision wants a person.

        Separate from forward() so the answer path costs nothing when a caller
        does not ask for it.
        """
        if self.act_head is None:
            return None
        hidden = self.encoder.embeddings(input_ids=batch.input_ids)
        h = self.encoder(input_ids=batch.input_ids,
                         position_ids=batch.position_ids,
                         attention_mask=self._masks(batch, hidden)).last_hidden_state
        return self.act_head(h[batch.batch_idx, batch.dec_pos]).squeeze(-1)

    def forward(self, batch: Packed, restrict: bool = True,
                with_escalation: bool = False):
        hidden = self.encoder.embeddings(input_ids=batch.input_ids)
        h = self.encoder(input_ids=batch.input_ids,
                         position_ids=batch.position_ids,
                         attention_mask=self._masks(batch, hidden)).last_hidden_state

        flat = batch.batch_idx
        q = h[flat, batch.dec_pos]                                  # [Nq, D]
        qv = F.normalize(self.q_proj(q), dim=-1)

        Nq, mo = batch.opt_pos.shape
        rows = flat.unsqueeze(1).expand(Nq, mo)
        o = h[rows.reshape(-1), batch.opt_pos.reshape(-1)]          # [Nq*mo, D]
        ov = F.normalize(self.o_proj(o), dim=-1).view(Nq, mo, -1)

        logits = (qv.unsqueeze(1) * ov).sum(-1) * self.logit_scale.exp()

        if self.noul_head is not None and batch.kinds:
            is_noul = torch.tensor([k == "noul" for k in batch.kinds],
                                   device=logits.device)
            if bool(is_noul.any()):
                direct = self.noul_head(q)                      # [Nq, 2]
                pad = logits.shape[1] - 2
                if pad > 0:
                    direct = torch.cat(
                        [direct, direct.new_full((direct.shape[0], pad), -1e4)], 1)
                logits = torch.where(is_noul.unsqueeze(1), direct[:, : logits.shape[1]],
                                     logits)

        if restrict:
            logits = logits.masked_fill(~batch.opt_mask, -1e4)
        if with_escalation and self.act_head is not None:
            # h is already computed, so this reuses the same pass.
            return logits, self.act_head(q).squeeze(-1)
        return logits

    def trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def temperature_bucket(kind: str, n_options: int) -> str:
    """Which temperature applies to a question of this type and size."""
    if kind == "noul":
        return "noul"
    if kind == "score":
        return f"score:{n_options}"

    if n_options <= 3:
        return "choice:2-3"
    if n_options <= 6:
        return "choice:4-6"
    return "choice:7+"
