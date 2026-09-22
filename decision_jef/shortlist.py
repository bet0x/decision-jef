"""Coarse-to-fine choice for large option sets.

Accuracy on a choice question falls toward chance past about twenty options.
Measured zero-shot on a 65-label intent set: 0.430 at five options, 0.195 at
ten, 0.095 at twenty, 0.030 at forty. Nothing is truncated -- eighty options
fit in 898 tokens -- so this is a coverage limit, not a budget one, and it
does not go away by making the window larger.

Shortlisting narrows the set first, then asks the question once on what
survives. Ranking is the part that has to come from somewhere else. Using this model's
own logits to rank was tried and does not work: at fifty options they are the
broken signal, so they rank as badly as they answer -- on a fifty-one-label
intent set the obviously correct label did not make the top eight. The default
here is lexical overlap between the state and each option's text, which needs
no model and no dependency and puts "the customer wants a duplicate charge
refunded" first for a state about being billed twice. Pass `embed_fn` to rank
with your own embeddings instead.

    answers = shortlist_decide(decider, state, questions, k=16)

A question at or below `k` options passes through untouched and costs nothing.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch

from decision_jef.pack import pack
from decision_jef.wire import Answer, Question, Request

DEFAULT_K = 16


_WORD = re.compile(r"[a-z0-9]+")
_STOP = frozenset("""a an and are as at be by for from has have in is it its of on or
that the this to was were will with you your do does did not no yes""".split())


def _terms(text: str) -> Counter:
    return Counter(w for w in _WORD.findall(str(text).lower())
                   if w not in _STOP and len(w) > 2)


def _lexical_scores(state: str, options: List[str]) -> List[float]:
    """Overlap between the state and each option, weighted so a term common to
    every option counts for little -- the idf half of tf-idf, which is what
    separates "refund" from "customer" when every label says customer."""
    docs = [_terms(o) for o in options]
    n = len(docs)
    df = Counter()
    for d in docs:
        df.update(d.keys())
    q = _terms(state)
    out = []
    for d in docs:
        score = 0.0
        for term, count in d.items():
            if term in q:
                idf = math.log(1 + n / (1 + df[term]))
                score += math.sqrt(count) * idf * math.sqrt(q[term])
        norm = math.sqrt(sum(v for v in d.values())) or 1.0
        out.append(score / norm)
    return out


def rank_options(decider, state: str, question: Question,
                 k: int = DEFAULT_K,
                 embed_fn: Optional[Callable[[Sequence[str]], object]] = None
                 ) -> Tuple[List[str], Optional[List[float]]]:
    """The `k` best option keys for this state, best first.

    Returns every key in its original order, and None for the scores, when the
    question already has `k` options or fewer -- so a caller can tell a real
    ranking from a pass-through.

    `embed_fn` maps a list of strings to an array of shape (len(texts), dim).
    It is called once with the state first and then one string per option. Left
    at None, ranking is lexical.
    """
    keys = question.keys
    if len(keys) <= k:
        return list(keys), None

    texts = question.descriptions
    if embed_fn is None:
        scores = _lexical_scores(state, [f"{a} {b}" for a, b in zip(keys, texts)])
    else:
        import numpy as np
        vecs = np.asarray(embed_fn([state] + list(texts)), dtype="float64")
        q, opts = vecs[0], vecs[1:]
        qn = np.linalg.norm(q) or 1.0
        on = np.linalg.norm(opts, axis=1)
        on[on == 0] = 1.0
        scores = (opts @ q / (on * qn)).tolist()

    # A tie keeps the earlier label, so the order a caller gave is the
    # tiebreak rather than whatever sort happens to do.
    order = sorted(range(len(keys)), key=lambda i: (-scores[i], i))[:k]
    return [keys[i] for i in order], [scores[i] for i in order]


def narrow(question: Question, keys: List[str]) -> Question:
    """The same question restricted to `keys`, in the order given."""
    if question.kind != "choice":
        return question
    return Question("choice", question.instructions,
                    {key: question.criteria[key] for key in keys
                     if key in question.criteria})


@torch.no_grad()
def shortlist_decide(decider, state: str, questions: Dict[str, Question],
                     k: int = DEFAULT_K, calibrated: bool = True,
                     embed_fn: Optional[Callable[[Sequence[str]], object]] = None
                     ) -> Tuple[Dict[str, Answer], Dict[str, dict]]:
    """Answer `questions`, narrowing any choice with more than `k` options.

    Returns the answers and, per question, what the narrowing did: the kept
    keys in rank order, their scores, and whether it passed through. The
    probabilities on a narrowed question are over the kept options only, which
    is the honest reading -- the dropped ones were never scored against the
    survivors.
    """
    if k < 2:
        raise ValueError("k must be at least 2")

    narrowed: Dict[str, Question] = {}
    meta: Dict[str, dict] = {}
    for qid, q in questions.items():
        if q.kind != "choice" or len(q.keys) <= k:
            narrowed[qid] = q
            meta[qid] = {"passthrough": True, "n": len(q.keys), "k": k,
                         "kept": list(q.keys), "scores": None}
            continue
        kept, scores = rank_options(decider, state, q, k, embed_fn)
        narrowed[qid] = narrow(q, kept)
        meta[qid] = {"passthrough": False, "n": len(q.keys), "k": k,
                     "kept": kept, "scores": scores}

    return decider.decide(state, narrowed, calibrated=calibrated), meta
