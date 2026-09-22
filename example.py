"""Runnable example: one state, four questions, one forward pass.

    pip install -e .
    python example.py
"""
import json
import time

from decision_jef import Decider, Question

MODEL = "/tmp/jef-local"

state = """from: user@acme.com
subject: Duplicate charge on invoice #4411
body: Hi, we were billed twice for March. Please refund the duplicate today
or we will cancel our plan."""

questions = {
    "department": Question("choice", "Which department should handle this request?", {
        "billing": "invoices, payments, refunds",
        "technical": "bugs, outages, system errors",
        "sales": "pricing, new contracts",
        "other": "everything else",
    }),
    "urgency": Question("score", "How urgent is this request?",
                        ["not urgent", "soon", "critical deadline or blocking issue"]),
    # A noul question works far better with a description per outcome. Measured
    # on this very example: bare yes/no gives p(true) 0.096 and 0.309, both
    # wrong; the descriptions below give 0.850 and 0.912, both right. The
    # option text is what the model reads, so say what each outcome means.
    "churn_risk": Question("noul", "Does the user threaten to cancel or leave?", {
        "false": "The user makes no threat to stop using the service.",
        "true": "The user threatens to cancel, churn or leave.",
    }),
    "refund_requested": Question("noul", "Does the user explicitly request a refund?", {
        "false": "The user does not ask for money back.",
        "true": "The user asks for a refund or a reversal of a charge.",
    }),
}

d = Decider.from_pretrained(MODEL)

d.decide(state, questions)          # warm up the kernels first
t0 = time.perf_counter()
answers = d.decide(state, questions)
ms = (time.perf_counter() - t0) * 1000

for qid, a in answers.items():
    top = a.choice if a.kind != "noul" else ("yes" if a.p("true") > 0.5 else "no")
    extra = f"  score={a.score}" if a.kind == "score" else ""
    print(f"{qid:18} {a.kind:6} -> {str(top):10} confidence {a.confidence:.2f}{extra}")
    print(f"{'':18} {a.probabilities}")

print(f"\none forward pass for all {len(questions)} questions: {ms:.1f} ms")
print("\nwire format:")
print(json.dumps(d.to_wire(answers), indent=2)[:600])

# Adding a question cannot change another question's answer.
subset = {k: questions[k] for k in ("department", "urgency")}
again = d.decide(state, subset)
same = all(abs(again[k].p(o) - answers[k].p(o)) < 1e-3
           for k in subset for o in again[k].probabilities)
print(f"\nquestion isolation holds when the other two are removed: {same}")
