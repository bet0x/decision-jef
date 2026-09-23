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
__version__ = "0.5.0"

from decision_jef import email
from decision_jef.infer import Decider
from decision_jef.shortlist import (DEFAULT_K, narrow, rank_options,
                                    shortlist_decide)
from decision_jef.serve import serve
from decision_jef.wire import Answer, Question, Request, as_instructions

__all__ = ["Decider", "Question", "Request", "Answer", "email", "serve",
           "as_instructions",
           "shortlist_decide", "rank_options", "narrow", "DEFAULT_K",
           "__version__"]
