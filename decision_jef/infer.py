"""Inference API.

The answer space is built from the request, so the model cannot return a value
you did not offer. Probabilities are temperature-scaled per (type, cardinality)
bucket, with the temperatures fitted on a held-out calibration split.
"""

from __future__ import annotations

import json
import os
from typing import Dict, Optional, Sequence

import torch

from decision_jef.model import DecisionJef
from decision_jef.pack import pack
from decision_jef.wire import Answer, Question, Request

# Two decimal places is what a reported probability is worth here, so a
# floor well below that costs no information and avoids claiming certainty.
PROB_FLOOR = 1e-4


class Decider:
    def __init__(self, model: DecisionJef, tokenizer, temperatures=None,
                 max_len: int = 1024):
        self.model = model.eval()
        self.tok = tokenizer
        self.temperatures = temperatures or {}
        self.max_len = max_len

    @classmethod
    def from_pretrained(cls, repo_or_path: str, device: str = "auto",
                        max_len: int = 1024) -> "Decider":
        from huggingface_hub import hf_hub_download
        from transformers import AutoTokenizer

        def grab(name):
            if os.path.isdir(repo_or_path):
                p = os.path.join(repo_or_path, name)
                return p if os.path.exists(p) else None
            try:
                return hf_hub_download(repo_or_path, name)
            except Exception:
                return None

        weights = grab("model.pt")
        if weights is None:
            raise FileNotFoundError(
                f"No model.pt in {repo_or_path!r}.\n"
                "The weights are published separately from this package. If the "
                "repository is private or gated, authenticate first with "
                "`hf auth login`, or pass a local directory containing model.pt.\n"
                "See https://huggingface.co/BarraHome/Decision-Jef-0.1")
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        dev = torch.device(device)
        ck = torch.load(weights, map_location=dev, weights_only=False)
        cfg = dict(ck["config"])
        # Read any unrecorded geometry flag off the weights rather than
        # trusting a default that may have changed.
        if "noul_head" not in cfg:
            cfg["noul_head"] = any(k.startswith("noul_head.") for k in ck["model"])
        model = DecisionJef(**cfg, pretrained=False,
                            encoder_config=ck.get("encoder_config")).to(dev)
        model.load_state_dict(ck["model"])
        # Prefer a tokenizer vendored beside the weights, so a load needs no
        # second repository and no network call once the model is cached.
        try:
            tok = AutoTokenizer.from_pretrained(repo_or_path)
        except Exception:
            tok = AutoTokenizer.from_pretrained(cfg["model_name"])
        temps = ck.get("temperatures") or {}
        tf = grab("temperatures.json")
        if not temps and tf:
            with open(tf) as fh:
                temps = json.load(fh)
        return cls(model, tok, temps, max_len)

    def _temp(self, kind: str, n: int) -> float:
        from decision_jef.model import temperature_bucket
        b = temperature_bucket(kind, n)
        return float(self.temperatures.get(b, self.temperatures.get("_global", 1.0)))

    @torch.no_grad()
    def decide(self, state: str, questions: Dict[str, Question],
               calibrated: bool = True) -> Dict[str, Answer]:
        """One forward pass for every question. Questions cannot see each
        other, so adding one never changes another's answer."""
        req = Request(state, questions)
        dev = next(self.model.parameters()).device
        batch = pack([req], self.tok, self.max_len).to(dev)
        logits = self.model(batch).float()
        out: Dict[str, Answer] = {}
        counts = batch.opt_mask.sum(-1).tolist()
        for i, (qid, kind, keys) in enumerate(
                zip(batch.qids, batch.kinds, batch.keys)):
            row = logits[i]
            if calibrated:
                row = row / self._temp(kind, int(counts[i]))
            p = torch.softmax(row[: len(keys)], -1).cpu().tolist()
            # Never report certainty. Sharpening temperatures (around 0.34
            # here) can drive a confident logit gap to a rounded 1.0, and a
            # probability of exactly 1 claims the answer cannot be wrong.
            # Clamp and renormalise so the distribution still sums to 1.
            p = [min(max(v, PROB_FLOOR), 1.0 - PROB_FLOOR) for v in p]
            total = sum(p)
            p = [v / total for v in p]
            d = {k: round(float(v), 4) for k, v in zip(keys, p)}
            best = max(d, key=d.get)
            out[qid] = Answer(
                kind=kind, probabilities=d,
                choice=best if kind != "noul" else None,
                score=round(sum(j * v for j, v in enumerate(p)), 4)
                if kind == "score" else None,
                legend={str(j): t for j, t in enumerate(questions[qid].descriptions)}
                if kind == "score" else None,
                confidence=round(max(p), 4))
        return out

    def to_wire(self, answers: Dict[str, Answer]) -> dict:
        """The response body, ready to serialise."""
        body = {"model": "Decision-Jef-0.1", "answers": {}}
        for qid, a in answers.items():
            d = {"type": a.kind, "probabilities": a.probabilities,
                 "confidence": a.confidence}
            if a.kind == "noul":
                d["noul"] = a.probabilities.get("true", 0.0)
            else:
                d["choice"] = a.choice
                if a.kind == "score":
                    d["score"] = a.score
                    d["legend"] = a.legend
            body["answers"][qid] = d
        return body
