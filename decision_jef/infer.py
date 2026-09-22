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
                 max_len: int = 1024, cost_escalate: float = 0.5,
                 cost_wrong: float = 3.0):
        self.model = model.eval()
        self.tok = tokenizer
        self.temperatures = temperatures or {}
        self.max_len = max_len
        # The two costs the escalation threshold comes from: hand a decision to
        # a person when the chance of being wrong exceeds
        # cost_escalate / cost_wrong. At 0.5 and 3.0 that is one in six.
        self.cost_escalate = cost_escalate
        self.cost_wrong = cost_wrong

    @classmethod
    def from_pretrained(cls, repo_or_path: str, device: str = "auto",
                        max_len: int = 1024) -> "Decider":
        """Load from safetensors plus JSON, or from a legacy .pt checkpoint.

        Weights come from `model.safetensors`, geometry from `config.json` and
        the fitted temperatures from `temperatures.json`. None of the three can
        execute code on load.

        A `model.pt` is read only when no safetensors file is present, and it
        needs `weights_only=False` because it carries non-tensor entries --
        which means unpickling whatever the file contains. That is the reason
        this format is no longer the one published.
        """
        from transformers import AutoTokenizer

        def grab(name, required=False):
            import os
            if os.path.isdir(repo_or_path):
                path = os.path.join(repo_or_path, name)
                return path if os.path.exists(path) else None
            from huggingface_hub import hf_hub_download
            try:
                return hf_hub_download(repo_or_path, name)
            except Exception:
                if required:
                    raise
                return None

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        dev = torch.device(device)

        weights = grab("model.safetensors")
        if weights is not None:
            from safetensors.torch import load_file

            cfg_path = grab("config.json") or grab("decision_jef.json")
            if cfg_path is None:
                raise FileNotFoundError(
                    f"{repo_or_path} has model.safetensors but no config.json")
            with open(cfg_path) as fh:
                raw = json.load(fh)
            # config.json is a transformers config carrying the encoder's own
            # fields, so this model's settings live under one key rather than
            # at the top level. Older bundles put them at the top level.
            cfg = dict(raw.get("decision_jef") or raw)
            if "encoder_config" in raw:
                cfg["encoder_config"] = raw["encoder_config"]
            elif "model_type" in raw:
                # The published config is the encoder's own, so the encoder
                # config is that file minus this model's own keys. Building it
                # here is what lets a load touch no other repository.
                cfg["encoder_config"] = {
                    k: v for k, v in raw.items()
                    if k not in ("decision_jef", "architectures",
                                 "transformers_version", "_name_or_path",
                                 "torch_dtype")}
            state = load_file(weights, device=str(dev))
            # The published tensors carry the encoder's canonical names so a
            # plain AutoModel load works; this model expects them under
            # `encoder.`. Put the prefix back on everything that is not one of
            # this model's own heads.
            heads = ("q_proj.", "o_proj.", "logit_scale", "type_emb.",
                     "noul_head.", "act_head.")
            if not any(k.startswith("encoder.") for k in state):
                state = {(k if k.startswith(heads) else "encoder." + k): v
                         for k, v in state.items()}
            temps = {}
            tf = grab("temperatures.json")
            if tf:
                with open(tf) as fh:
                    temps = json.load(fh)
        else:
            legacy = grab("model.pt")
            if legacy is None:
                raise FileNotFoundError(
                    f"No model.safetensors in {repo_or_path!r}.\n"
                    "The weights are published separately from this package. "
                    "If the repository is private or gated, authenticate first "
                    "with `hf auth login`, or pass a local directory "
                    "containing model.safetensors and config.json.\n"
                    "See https://huggingface.co/BarraHome/Decision-Jef-0.1")
            # Legacy path: this unpickles the file. Kept so checkpoints written
            # before the format change still load, not because it is safe.
            ck = torch.load(legacy, map_location=dev, weights_only=False)
            cfg = dict(ck["config"])
            state = ck["model"]
            temps = ck.get("temperatures") or {}
            if "encoder_config" in ck:
                cfg["encoder_config"] = ck["encoder_config"]

        # Read any unrecorded geometry flag off the weights rather than
        # trusting a default that may have changed. Every optional head goes in
        # this list; a default that flips silently is how a checkpoint stops
        # loading a version later.
        for flag in ("noul_head", "act_head"):
            if flag not in cfg:
                cfg[flag] = any(k.startswith(flag + ".") for k in state)

        known = {"model_name", "proj_dim", "dropout", "trainable_layers",
                 "isolate_questions", "noul_head", "act_head", "encoder_config"}
        model = DecisionJef(**{k: v for k, v in cfg.items() if k in known},
                            pretrained=False).to(dev)
        model.load_state_dict(state)
        try:
            tok = AutoTokenizer.from_pretrained(repo_or_path)
        except Exception:
            tok = AutoTokenizer.from_pretrained(cfg["model_name"])
        return cls(model, tok, temps, max_len)

    def _temp(self, kind: str, n: int) -> float:
        from decision_jef.model import temperature_bucket
        b = temperature_bucket(kind, n)
        return float(self.temperatures.get(b, self.temperatures.get("_global", 1.0)))

    @property
    def escalate_threshold(self) -> float:
        """Escalate above this probability of being wrong."""
        return self.cost_escalate / self.cost_wrong

    @torch.no_grad()
    def decide(self, state: str, questions: Dict[str, Question],
               calibrated: bool = False,
               with_escalation: bool = False) -> Dict[str, Answer]:
        """One forward pass for every question. Questions cannot see each
        other, so adding one never changes another's answer."""
        req = Request(state, questions)
        dev = next(self.model.parameters()).device
        batch = pack([req], self.tok, self.max_len).to(dev)
        want = with_escalation and self.model.act_head is not None
        out = self.model(batch, with_escalation=want)
        logits, esc = out if want else (out, None)
        logits = logits.float()
        p_wrong = torch.sigmoid(esc.float()).tolist() if esc is not None else None
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
            escalate = None
            if p_wrong is not None:
                escalate = {"probability_wrong": round(float(p_wrong[i]), 4),
                            "threshold": round(self.escalate_threshold, 4),
                            "escalate": p_wrong[i] > self.escalate_threshold}
            out[qid] = Answer(
                kind=kind, probabilities=d, escalation=escalate,
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
