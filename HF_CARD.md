---
license: mit
base_model: jhu-clsp/mmBERT-base
base_model_relation: finetune
pipeline_tag: text-classification
inference: false
language:
  - multilingual
tags:
  - system-one
  - calibrated-decisions
  - typed-decisions
  - classification
  - routing
  - scoring
  - guardrails
  - moderation
  - multilingual
  - commercial-use
---

<p align="center">
  <a href="https://pypi.org/project/decision-jef/"><img src="https://img.shields.io/pypi/v/decision-jef?logo=pypi&logoColor=white" alt="PyPI version"></a>
  <a href="https://github.com/bet0x/decision-jef"><img src="https://img.shields.io/badge/GitHub-decision--jef-181717?logo=github" alt="Source on GitHub"></a>
  <img src="https://img.shields.io/badge/typed--decisions-77.00-1f6feb" alt="77.00 on the typed-decisions benchmark">
  <img src="https://img.shields.io/badge/latency-12.12%20ms%20%C2%B7%204%20decisions-2da44e" alt="12.12 ms for four decisions">
  <img src="https://img.shields.io/badge/ECE-0.060-2da44e" alt="Expected calibration error 0.060">
</p>

# Decision-Jef-0.1

Answer several runtime-defined questions about one state, in a single forward
pass, with calibrated probabilities over exactly the options you supply.

307M parameters. **12.12 ms for four decisions in one forward pass.**

The answer space is built from the request, so a value you did not offer is not
representable — not merely unlikely. There is no classification head over a
fixed label set.

## Results

Typed-decisions test set, 2,000 decisions: 600 `choice`, 600 `noul`, 800
`score`. Same set and same split as the published competitors.

| model | global | choice | noul | score |
| --- | --- | --- | --- | --- |
| Decision-1.0-Lex | **78.15** | **74.00** | 84.67 | **76.38** |
| **Decision-Jef-0.1** | **77.00** | 73.67 | 84.33 | 74.00 |
| Laya Typed Decisions | 76.60 | 73.33 | **85.67** | 72.25 |

Ahead of Laya by 0.40 global and 1.75 on `score`. Behind Lex by 1.15 global,
with `choice` and `noul` within a third of a point and the gap concentrated in
`score`.

| | value |
| --- | --- |
| NLL, choice / noul / score | 0.7200 / 0.4043 / 0.7316 |
| ECE, 10 bins, calibrated | **0.060** |
| ECE, 10 bins, uncalibrated | 0.138 |
| mean confidence, calibrated | 0.827 |

Temperature scaling per (type, cardinality) bucket ships with the model and is
applied by default, taking ECE from 0.138 to **0.060**. It never changes an
argmax, so accuracy is 77.00 either way. Pass `calibrated=False` to `decide()`
for the raw distribution.

Question isolation is exact: adding a question moves another question's logits
by at most 3e-07.

## Latency

NVIDIA H100 NVL, bf16, median of 30 calls after warm-up. End to end: packing,
encoder and readout.

| questions in one call | median | p95 |
| --- | --- | --- |
| 1 | 11.73 ms | 14.09 ms |
| 2 | 11.88 ms | 12.17 ms |
| **4** | **12.12 ms** | 12.29 ms |

**Going from one decision to four costs 0.39 ms.** The state is encoded once and
the question branches are masked apart, so a request carrying four questions is
not four requests. Throughput at batch 64 and 1,024 tokens is 2.66 ms per
decision.

## Usage

```bash
pip install decision-jef
```

The weights are published separately from the package. Authenticate with
`hf auth login` if the model repository is not yet public, or point
`from_pretrained` at a local directory holding `model.pt`.

```python
from decision_jef import Decider, Question

d = Decider.from_pretrained("BarraHome/Decision-Jef-0.1")

state = """from: user@acme.com
subject: Duplicate charge on invoice #4411
body: We were billed twice for March. Please refund the duplicate today
or we will cancel our plan."""

answers = d.decide(state, {
    "department": Question("choice", "Which department should handle this?", {
        "billing": "invoices, payments, refunds",
        "technical": "bugs, outages, system errors",
        "sales": "pricing, new contracts",
        "other": "everything else",
    }),
    "urgency": Question("score", "How urgent is this request?",
                        ["not urgent", "soon", "critical or blocking"]),
    "churn_risk": Question("noul", "Does the user threaten to leave?", {
        "false": "The user makes no threat to stop using the service.",
        "true": "The user threatens to cancel, churn or leave.",
    }),
})

for qid, a in answers.items():
    print(qid, a.choice or a.p("true"), a.confidence, a.probabilities)
```

```
department  billing  0.94  {'billing': 0.9383, 'technical': 0.0147, 'sales': 0.0301, 'other': 0.017}
urgency     2        0.58  {'0': 0.156, '1': 0.2662, '2': 0.5778}    score=1.42
churn_risk  0.87     0.87  {'false': 0.1292, 'true': 0.8708}
```

`d.to_wire(answers)` returns the same content as a JSON-ready response body.

### Give every yes/no outcome a description

This is a requirement, not a style note. The model scores the option text, so
a bare yes/no gives it nothing to compare. On the example above:

| question | p(true) with bare yes/no | p(true) with descriptions |
| --- | --- | --- |
| threatens to leave | 0.096 — wrong | **0.850** — right |
| requests a refund | 0.309 — wrong | **0.912** — right |

## The three question types

| type | `criteria` | answer |
| --- | --- | --- |
| `choice` | ordered map of key to description, up to 255 | `choice`, `probabilities` |
| `noul` | optional map of `false` and `true` to a description — supply it | `noul` probability |
| `score` | ordered array of 2 to 10 level descriptions | probability-weighted `score`, `legend` |

Option order is part of the question. The same options in a different order
are a different request, and the model is sensitive to it — include a
permutation check in any evaluation.

## How it works

```
<bos> state ... <eos>                                    shared, encoded once
[Q] instructions ... [OPT] opt0 [OPT] opt1 ... [DEC]     question 1
[Q] instructions ... [OPT] opt0 [OPT] opt1 ... [DEC]     question 2
```

The query is read at `[DEC]`, after the whole option list, so the decision sees
every option. Keys come from each `[OPT]` in the same pass, so the options are
read together rather than scored in isolation. Each question attends to the
state and to itself only; the state attends to neither.

## Limitations

- **ECE is 0.060 only with the shipped temperatures applied**, 0.138 without.
  They are on by default; do not disable them unless you are recalibrating.
- **`score` is the weakest type** at 74.00, 2.38 behind Lex.
- Trained and measured on **English** typed decisions. The backbone is
  multilingual and the tokenizer covers 256k tokens, but no non-English
  benchmark has been run — treat multilingual use as untested.
- The `guardrails` and `moderation` tags reflect coverage of toxicity and
  hate-speech decisions. **Neither capability has been benchmarked.**
- Sensitive to option order, as above.
- Long states are truncated to the window with the questions reserved first.

## License and provenance

MIT, following the `jhu-clsp/mmBERT-base` backbone. No weights, gradients or
private data from any third-party decision service are used or claimed; the
design follows publicly documented API behaviour and public benchmark splits.
