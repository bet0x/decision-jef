"""The request and response types.

  choice  criteria is an ordered map of option key -> description, up to 255.
  noul    yes/no. The answer is one probability.
  score   criteria is an ordered array of 2 to 10 level descriptions. The
          answer adds a legend and a probability-weighted score.

Every container preserves insertion order: option order is part of the
question, and nothing here sorts keys.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Union

KINDS = ("choice", "noul", "score")
MAX_CHOICE_OPTIONS = 255          # 2**8 - 1
MAX_SCORE_LEVELS = 10
MIN_SCORE_LEVELS = 2


def as_instructions(value) -> str:
    """The instructions as text, whatever shape the caller sent them in.

    A real Jev client may send an object here rather than a string: a browser
    agent driving a game loop sends `{"task": "Choose navigation for this
    tick.", ...}` with its standing policy and the tick's constraints
    alongside. The model reads instructions as text, so an object has to be
    flattened rather than rejected, and `str(dict)` would feed it Python repr
    punctuation.
    """
    if isinstance(value, str):
        return value
    if value is None:
        return ""
    if isinstance(value, dict):
        # `task` first when present: it is the instruction, the rest is context.
        keys = (["task"] if "task" in value else []) + [
            k for k in value if k != "task"]
        # Strip a trailing stop from each part before joining, or a value that
        # already ends in one produces "tick.. policy:".
        parts = [f"{k}: {as_instructions(value[k])}".strip().rstrip(".").strip()
                 for k in keys if value[k] is not None]
        return ". ".join(p for p in parts if p) + ("." if parts else "")
    if isinstance(value, (list, tuple)):
        return ". ".join(as_instructions(v) for v in value if v is not None)
    return str(value)


def as_state_text(value) -> str:
    """The state as text. A structured state arrives as an object.

    Serialised as JSON rather than flattened: a caller that sent nested
    objects meant their structure, and the state is the thing every question
    asks about. Key order is the caller's.
    """
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=False)


def _text(value) -> str:
    """Option text, or "" when there is none.

    str(None) is "None", which is truthy, so a plain `str(v).strip() or key`
    silently labels an option "None" instead of falling back to its key.
    """
    return "" if value is None else str(value).strip()


@dataclass
class Question:
    kind: str
    instructions: str
    # choice: ordered {key: description}. score: ordered [description, ...].
    # noul: empty.
    criteria: Union[Dict[str, str], List[str], None] = None

    def __post_init__(self):
        if self.kind not in KINDS:
            raise ValueError(f"unknown kind {self.kind!r}; expected one of {KINDS}")
        if self.kind == "choice":
            if not isinstance(self.criteria, dict) or len(self.criteria) < 2:
                raise ValueError("choice needs an ordered map of at least 2 options")
            if len(self.criteria) > MAX_CHOICE_OPTIONS:
                raise ValueError(f"choice accepts at most {MAX_CHOICE_OPTIONS} options")
        elif self.kind == "score":
            if not isinstance(self.criteria, (list, tuple)):
                raise ValueError("score needs an ordered array of level descriptions")
            if not MIN_SCORE_LEVELS <= len(self.criteria) <= MAX_SCORE_LEVELS:
                raise ValueError(
                    f"score needs {MIN_SCORE_LEVELS} to {MAX_SCORE_LEVELS} levels")
        elif self.criteria is not None:
            # A noul question may optionally carry descriptions for false and
            # true; anything else is rejected.
            if not isinstance(self.criteria, dict) or \
                    {k.lower() for k in self.criteria} != {"false", "true"}:
                raise ValueError("noul criteria must be exactly false and true")

    @property
    def keys(self) -> List[str]:
        """The answer keys, in order. Score levels are indexed from 0 as strings,
        which is how the API returns them in `probabilities` and `legend`."""
        if self.kind == "choice":
            return list(self.criteria)
        if self.kind == "score":
            return [str(i) for i in range(len(self.criteria))]
        return ["false", "true"]        # noul reports p(true) as `noul`

    @property
    def descriptions(self) -> List[str]:
        """Option text in the same order as `keys`. A noul question has no
        option text at all -- its two outcomes are implicit."""
        if self.kind == "choice":
            # A description of None or "" is a legitimate wire form: the key is
            # the label and needs no gloss. Fall back to the key rather than
            # handing the tokenizer nothing, which raises "You need to specify
            # either `text` or `text_target`".
            return [_text(v) or str(k) for k, v in self.criteria.items()]
        if self.kind == "score":
            return [_text(v) or f"level {i}"
                    for i, v in enumerate(self.criteria)]
        if self.criteria:
            low = {k.lower(): v for k, v in self.criteria.items()}
            return [_text(low["false"]) or "no", _text(low["true"]) or "yes"]
        return ["no", "yes"]


@dataclass
class Request:
    state: str
    questions: Dict[str, Question]          # ordered
    model: str = "jev-latest"

    def __post_init__(self):
        if not self.questions:
            raise ValueError("a request needs at least one question")
        for qid, q in self.questions.items():
            if isinstance(q, Question):
                continue
            # A plain dict is the obvious thing to pass, and without this check
            # it reaches the packer, where `q.keys` resolves to dict.keys and
            # the failure reads "object of type 'builtin_function_or_method'
            # has no len()". Say what to do instead.
            if isinstance(q, dict):
                raise TypeError(
                    f"question {qid!r} is a plain dict; wrap it as "
                    "Question(kind=..., instructions=..., criteria=...), or "
                    "build the whole request with Request.from_json()")
            raise TypeError(
                f"question {qid!r} is {type(q).__name__}, expected Question")

    def to_json(self) -> dict:
        out = {"model": self.model, "state": self.state, "questions": {}}
        for qid, q in self.questions.items():
            d = {"type": q.kind, "instructions": q.instructions}
            if q.kind == "choice":
                d["criteria"] = dict(q.criteria)
            elif q.kind == "score":
                d["criteria"] = list(q.criteria)
            out["questions"][qid] = d
        return out

    @classmethod
    def from_json(cls, d: dict) -> "Request":
        qs = {}
        for qid, q in d["questions"].items():
            qs[qid] = Question(q["type"], as_instructions(q["instructions"]),
                               q.get("criteria"))
        return cls(state=as_state_text(d["state"]), questions=qs,
                   model=d.get("model", "jev-latest"))


@dataclass
class Answer:
    kind: str
    probabilities: Dict[str, float]         # ordered, sums to 1
    choice: Optional[str] = None
    score: Optional[float] = None
    legend: Optional[Dict[str, str]] = None
    confidence: Optional[float] = None
    # Present only when decide(with_escalation=True): the model's own estimate
    # that this answer is wrong, the cost-derived threshold, and the verdict.
    escalation: Optional[Dict[str, object]] = None

    @classmethod
    def from_json(cls, d: dict) -> "Answer":
        kind = d["type"]
        if kind == "noul":
            # The API returns a single `noul` probability, not a distribution.
            # Expand it so every kind reads the same downstream.
            p = float(d["noul"])
            probs = {"false": 1.0 - p, "true": p}
        else:
            probs = {k: float(v) for k, v in d["probabilities"].items()}
        return cls(kind=kind, probabilities=probs, choice=d.get("choice"),
                   score=d.get("score"), legend=d.get("legend"),
                   confidence=d.get("confidence"))

    def p(self, key: str) -> float:
        return self.probabilities.get(key, 0.0)

    def log_odds(self, a: str, b: str, floor: float = 5e-3) -> float:
        """Log odds of one key against another.

        The API rounds probabilities to two decimals, so a reported 0.00 is a
        reporting floor and could be anything under 0.005. Clamping to that
        floor keeps the ratio finite and states the assumption in one place.
        """
        import math
        return math.log(max(self.p(a), floor) / max(self.p(b), floor))


def answers_from_response(d: dict) -> Dict[str, Answer]:
    return {qid: Answer.from_json(a) for qid, a in d["answers"].items()}

