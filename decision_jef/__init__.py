"""Decision-Jef: typed decisions with calibrated probabilities.

    from decision_jef import Decider, Question

    d = Decider.from_pretrained("BarraHome/Decision-Jef-0.1")
    out = d.decide(
        state="Payouts have failed three times. The bank says everything is fine.",
        questions={
            "queue": Question("choice", "Which team should handle this?",
                              {"payments": "Payout failures",
                               "account": "Login and account access",
                               "other": "Something else"}),
            "escalate": Question("noul", "Does this need urgent human attention?"),
            "mood": Question("score", "How frustrated is the customer?",
                             ["Calm", "Frustrated", "Very angry"]),
        },
    )
"""
__version__ = "0.8.0"

from decision_jef import email
from decision_jef.infer import Decider
from decision_jef.shortlist import (DEFAULT_K, narrow, rank_options,
                                    shortlist_decide)
from decision_jef.wire import Answer, Question, Request, as_instructions

# `serve` is deliberately not imported here. It pulls in http.server for every
# caller that only wants to decide, and importing it eagerly also makes
# `python -m decision_jef.serve` warn about a double import.
def __getattr__(name):
    if name == "serve":
        from decision_jef.serve import serve as _serve
        return _serve
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["Decider", "Question", "Request", "Answer", "email", "serve",
           "as_instructions",
           "shortlist_decide", "rank_options", "narrow", "DEFAULT_K",
           "__version__"]
