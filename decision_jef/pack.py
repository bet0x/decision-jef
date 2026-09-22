"""Turn a request into one input sequence.

Layout:

    <bos> state ... <eos>                                    segment 0, shared
    [Q] instructions ... [OPT] opt0 [OPT] opt1 ... [DEC]     segment 1
    [Q] instructions ... [OPT] opt0 [OPT] opt1 ... [DEC]     segment 2

The options of a question sit inline, in the order given. The [DEC] marker
closes each question's block. Every question attends to the state and to
itself only, and the state attends to neither, so adding a question never
changes another question's answer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import torch

from decision_jef.wire import Question, Request

OPT_MARKER = "<unused0>"
DEC_MARKER = "<unused1>"
Q_MARKER = "<unused2>"

# Reserve room for the questions first, then give the state whatever is left,
# so a large schema is never silently truncated away.
MIN_STATE_TOKENS = 16


@dataclass
class Packed:
    input_ids: torch.Tensor        # [B, L]
    attention_mask: torch.Tensor   # [B, L] 1 = real token
    segment_ids: torch.Tensor      # [B, L] 0 = state, i+1 = question i
    dec_pos: torch.Tensor          # [Nq] index of each question's [DEC]
    opt_pos: torch.Tensor          # [Nq, max_opts] index of each [OPT]
    opt_mask: torch.Tensor         # [Nq, max_opts] True = real option
    batch_idx: torch.Tensor        # [Nq] which row of the batch
    kinds: List[str]               # [Nq]
    qids: List[str]                # [Nq]
    keys: List[List[str]]          # [Nq][n_opts] answer keys, in order
    dropped: int = 0               # questions that did not fit

    def to(self, device) -> "Packed":
        d = {k: (v.to(device) if torch.is_tensor(v) else v)
             for k, v in self.__dict__.items()}
        return Packed(**d)


def _question_block(tok, q: Question, kind_prefix: bool = True) -> Tuple[List[int], List[int], int]:
    """Token ids for one question, plus the offsets of its [OPT] markers and
    its [DEC] marker, relative to the block start."""
    opt_id = tok.convert_tokens_to_ids(OPT_MARKER)
    dec_id = tok.convert_tokens_to_ids(DEC_MARKER)
    q_id = tok.convert_tokens_to_ids(Q_MARKER)

    enc = lambda s: tok(s, add_special_tokens=False)["input_ids"]
    # The question type is named in the text rather than carried by a separate
    # type embedding.
    head = f"{q.kind} question: {q.instructions}" if kind_prefix else q.instructions
    ids = [q_id] + enc(head)
    opts = []
    for i, desc in enumerate(q.descriptions):
        # Ordinal rank is written into the level text, so the levels carry
        # their order.
        text = f"level {i}: {desc}" if q.kind == "score" else desc
        opts.append(len(ids))
        ids.append(opt_id)
        ids.extend(enc(text)[:48])          # option text cap
    dec = len(ids)
    ids.append(dec_id)
    return ids, opts, dec


def pack(requests: Sequence[Request], tok, max_len: int = 1024,
         kinds: Optional[Sequence[str]] = None) -> Packed:
    bos = tok.bos_token_id if tok.bos_token_id is not None else tok.cls_token_id
    eos = tok.eos_token_id if tok.eos_token_id is not None else tok.sep_token_id
    pad = tok.pad_token_id
    enc = lambda s: tok(s, add_special_tokens=False)["input_ids"]

    rows, seg_rows, per_q, dropped = [], [], [], 0
    for b, req in enumerate(requests):
        blocks = []
        for qid, q in req.questions.items():
            if len(q.keys) < 2:
                dropped += 1
                continue
            blocks.append((qid, q) + _question_block(tok, q))
        budget = max_len - 2 - sum(len(ids) for *_, ids, _, _ in
                                   [(0, 0, b[2], b[3], b[4]) for b in blocks])
        state = enc(req.state)[:max(MIN_STATE_TOKENS, budget)]

        ids = [bos] + state + [eos]
        seg = [0] * len(ids)
        for qi, (qid, q, blk, opts, dec) in enumerate(blocks):
            if len(ids) + len(blk) > max_len:
                dropped += 1
                continue
            base = len(ids)
            ids.extend(blk)
            seg.extend([qi + 1] * len(blk))
            per_q.append(dict(batch=b, qid=qid, kind=q.kind, keys=q.keys,
                              dec=base + dec, opts=[base + o for o in opts]))
        rows.append(ids)
        seg_rows.append(seg)

    B, L = len(rows), max(len(r) for r in rows)
    input_ids = torch.full((B, L), pad, dtype=torch.long)
    attn = torch.zeros((B, L), dtype=torch.long)
    segs = torch.zeros((B, L), dtype=torch.long)
    for i, (r, s) in enumerate(zip(rows, seg_rows)):
        input_ids[i, : len(r)] = torch.tensor(r)
        attn[i, : len(r)] = 1
        segs[i, : len(s)] = torch.tensor(s)

    Nq = len(per_q)
    mo = max((len(p["opts"]) for p in per_q), default=1)
    dec_pos = torch.zeros(Nq, dtype=torch.long)
    opt_pos = torch.zeros((Nq, mo), dtype=torch.long)
    opt_mask = torch.zeros((Nq, mo), dtype=torch.bool)
    batch_idx = torch.zeros(Nq, dtype=torch.long)
    for i, p in enumerate(per_q):
        dec_pos[i] = p["dec"]
        batch_idx[i] = p["batch"]
        for j, o in enumerate(p["opts"]):
            opt_pos[i, j] = o
            opt_mask[i, j] = True

    return Packed(input_ids=input_ids, attention_mask=attn, segment_ids=segs,
                  dec_pos=dec_pos, opt_pos=opt_pos, opt_mask=opt_mask,
                  batch_idx=batch_idx,
                  kinds=[p["kind"] for p in per_q],
                  qids=[p["qid"] for p in per_q],
                  keys=[p["keys"] for p in per_q],
                  dropped=dropped)


def segment_attention_mask(segment_ids: torch.Tensor,
                           attention_mask: torch.Tensor) -> torch.Tensor:
    """[B, 1, L, L] boolean: True where attention is allowed.

    A position may attend to the state (segment 0) and to its own segment.
    The state may not attend to any question, which is what makes the state
    encoding identical no matter which questions accompany it.
    """
    sq = segment_ids.unsqueeze(2)                     # querying position
    sk = segment_ids.unsqueeze(1)                     # key position
    allowed = ((sk == 0) | (sq == sk)) & (sq != 0) | ((sq == 0) & (sk == 0))
    allowed = allowed & attention_mask.bool().unsqueeze(1)
    L = segment_ids.shape[-1]
    eye = torch.arange(L, device=segment_ids.device)
    allowed[:, eye, eye] = True                       # never mask a row entirely
    return allowed.unsqueeze(1)
