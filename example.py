"""Runnable example: one state, four questions, one forward pass.

    pip install -e .
    python example.py
"""
import json
import time

from decision_jef import Decider, Question, email, shortlist_decide

MODEL = "BarraHome/Decision-Jef-0.1"

# A real reply: the thread, a signature and a disclaimer hang off the bottom,
# and none of it is the message. `as_state` cuts them and formats the fields.
RAW = """Hi, we were billed twice for March. Please refund the duplicate today
or we will cancel our plan.

On Tue, 3 Jun 2025 at 14:02, Support <help@acme.com> wrote:
> Thanks for reaching out, could you confirm the invoice number?

--
Dana Ruiz
VP Finance

This e-mail is confidential and intended solely for the addressee."""

state = email.as_state("dana@acme.com", "Duplicate charge on invoice #4411", RAW)
print("state the model actually sees:")
print("  " + state.replace("\n", "\n  "))

questions = {
    "department": Question("choice", "Which department should handle this?", {
        "billing": "invoices, payments, refunds",
        "technical": "bugs, outages, system errors",
        "sales": "pricing, new contracts",
        "other": "everything else",
    }),
    "urgency": Question("score", "How urgent is this request?",
                        ["not urgent", "soon", "critical deadline or blocking issue"]),
    # A yes/no question works far better with a description per outcome.
    # Measured on this very example: bare yes/no gives p(true) 0.096 and 0.309,
    # both wrong; the descriptions below give 0.850 and 0.912, both right. The
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
answers = d.decide(state, questions, with_escalation=True)
ms = (time.perf_counter() - t0) * 1000

print("\nanswers")
for qid, a in answers.items():
    top = a.choice if a.kind != "noul" else ("yes" if a.p("true") > 0.5 else "no")
    extra = f"  score={a.score}" if a.kind == "score" else ""
    print(f"  {qid:18} {a.kind:6} -> {str(top):10} confidence {a.confidence:.2f}{extra}")
    print(f"  {'':18} {a.probabilities}")
    if a.escalation:
        e = a.escalation
        verdict = "hand to a person" if e["escalate"] else "answer it"
        print(f"  {'':18} wrong with p={e['probability_wrong']:.2f}, "
              f"threshold {e['threshold']:.2f} -> {verdict}")

print(f"\none forward pass for all {len(questions)} questions: {ms:.1f} ms")

# Raw probabilities by default. The shipped temperatures sharpen, which lowers
# aggregate ECE and makes a genuinely ambiguous input worse; the model card
# has the measurement.
tuned = d.decide(state, questions, calibrated=True)
print(f"\nchurn_risk p(true): {answers['churn_risk'].p('true'):.3f} raw, "
      f"{tuned['churn_risk'].p('true'):.3f} with the shipped temperatures")

print("\nwire format:")
print(json.dumps(d.to_wire(answers), indent=2)[:420])

# Adding a question cannot change another question's answer.
subset = {k: questions[k] for k in ("department", "urgency")}
again = d.decide(state, subset)
same = all(abs(again[k].p(o) - answers[k].p(o)) < 1e-3
           for k in subset for o in again[k].probabilities)
print(f"\nquestion isolation holds when the other two are removed: {same}")

# A large option set: narrow it, then ask once. Past about twenty options the
# model is extrapolating, so the ranking matters more than the window.
labels = {f"intent_{i:02d}": f"the customer wants outcome number {i}"
          for i in range(60)}
labels["billing_refund"] = "the customer wants a duplicate charge refunded"
wide = {"intent": Question("choice", "What does the customer want?", labels)}
got, meta = shortlist_decide(d, state, wide, k=8)
m = meta["intent"]
print(f"\nshortlist: {m['n']} options narrowed to {m['k']}")
print(f"  kept: {m['kept']}")
print(f"  answer: {got['intent'].choice} at {got['intent'].confidence:.2f}")
if got["intent"].choice != "billing_refund":
    # Worth leaving in rather than picking a k that hides it. The ranking put
    # the right label first and the model still picked another one: training
    # never showed a choice question with more than ten options, so this is
    # the documented coverage limit, not a shortlist failure. Narrowing is
    # necessary here and it is not sufficient.
    print("  the ranking put billing_refund first and the model still missed it")
    print("  -- past ten options this model extrapolates; see the model card")
