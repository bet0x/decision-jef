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
  <img src="https://img.shields.io/badge/typed--decisions-77.80-1f6feb" alt="77.80 on the typed-decisions benchmark">
  <img src="https://img.shields.io/badge/permuted-77.70-1f6feb" alt="77.70 with the options permuted">
  <img src="https://img.shields.io/badge/latency-12.2%20ms%20%C2%B7%203%20decisions-2da44e" alt="12.2 ms for three decisions">
  <img src="https://img.shields.io/badge/ECE-0.043%20raw-2da44e" alt="Expected calibration error 0.043 as shipped">
</p>

# Decision-Jef-0.1

Answer several runtime-defined questions about one state, in a single forward
pass, with a probability over exactly the options you supply.

307M parameters. **12.2 ms for three decisions in one forward pass.**

The answer space is built from the request, so a value you did not offer is not
representable — not merely unlikely. There is no classification head over a
fixed label set.

## Results

Typed-decisions test set, 2,000 decisions: 600 `choice`, 600 `noul`, 800
`score`. Same set and same split as the published models below. Figures are as
shipped, with no temperature applied.

| model | global | choice | noul | score |
| --- | --- | --- | --- | --- |
| Decision-1.0-Lex | **78.15** | 74.00 | 84.67 | **76.38** |
| **Decision-Jef-0.1** | 77.80 | **75.00** | **84.80** | 74.60 |
| Laya Typed Decisions | 76.60 | 73.33 | 85.67 | 72.25 |

Ahead of the best published figure on `choice` and level with it on `noul`.
`score` is the whole of the remaining 0.35-point deficit: 1.78 points behind on
the 800 decisions that carry 40% of the benchmark.

| | this model | Laya | Jev |
| --- | --- | --- | --- |
| soft accuracy | **0.607** | 0.471 | 0.580 |
| Brier, summed over classes | **0.083** | 0.061 † | 0.148 |
| ECE, 10 bins | **0.043** | 0.213 | 0.144 |
| score MAE | **0.244** | 0.242 | 0.391 |
| majority class on this set | 0.457 | | |
| random guess on this set | 0.318 | | |

Soft accuracy is the label mass on the answer we pick, which matters on a
benchmark whose labels are annotator averages rather than single verdicts.

† Published Brier figures use two conventions. Laya reports 0.061 averaged per
class and attributes 0.148 to Jev summed over classes, which reads as a 2.4x
gap and is a unit mismatch. The 0.083 above is summed, the same convention as
the 0.148, so it is comparable to Jev's figure and **not** to Laya's.

### Option order

Every `choice` question in this benchmark presents its options in one fixed
order, and the gold answer sits in position 2 in 39.7% of them against 25% for
a uniform draw. A model can therefore score on the benchmark by learning the
position. This one does not:

| | native order | options permuted | change |
| --- | --- | --- | --- |
| global | 77.80 | 77.70 | **+0.10** |
| choice | 75.00 | 74.67 | +0.33 |
| noul | 84.80 | 84.80 | +0.00 |
| score | 74.60 | 74.60 | +0.00 |

Over 1,800 permutations of the benchmark's `choice` questions, the answer
changes on **5.7%** of them. Jev is measured at 0.13 and Laya at 0.15 on the
same kind of check.

### Calibration

ECE is **0.043 as shipped**, which is already inside a 0.10 criterion without
any post-hoc correction. Temperature scaling per (type, cardinality) bucket
ships with the model and is **off by default**. Applying it trades one metric
for the other:

| | as shipped | with the temperatures |
| --- | --- | --- |
| ECE, 10 bins | 0.043 | **0.008** |
| Brier, summed | **0.083** | 0.108 |
| score MAE | **0.244** | 0.263 |
| global accuracy | 77.80 | 77.75 |

The two disagree because they ask different questions. ECE asks whether stated
confidence matches hit rate; Brier asks whether the reported distribution
matches the annotator average. The fitted temperatures sharpen, at 0.789
globally, which helps the first and hurts the second. Accuracy does not move
either way, because a per-bucket temperature never changes an argmax.

Leave them off unless you specifically need the confidence figure to track the
hit rate, and fit your own on your own data if you do. Laya's published pair —
Brier 0.061 with ECE 0.213 — sits at the opposite end of the same trade-off
rather than at a better point on it.

### Where the probabilities are honest

Measured against a generator whose true posterior is known by construction: a
ticket that cues one department has a true answer, one that cues two is a
genuine coin flip between them.

| case | true answer | this model |
| --- | --- | --- |
| one department cued | 1.00 | **1.000** |
| two departments cued | 0.50 | **0.552** |
| escalation implied by priority | 0.90 | **0.912** |
| mass on the two cued departments | 1.00 | **1.000** |

The model recovers the true posterior when the state settles the question, puts
all of its mass on the departments the state allows, and reports close to an
even split where the answer is genuinely undetermined. For scale on the third
row, a 0.90 truth is the case where a stated rate rather than a verdict is the
right answer.

### Question isolation

Adding a question does not change another question's answer. This holds
exactly, not approximately, and it holds against the *length* of the other
questions as well as their content — each question's block is packed with the
geometry it would have if it were alone in the request.

Checked across all 2,000 benchmark decisions with one, two, three and four
questions sharing each sequence: in fp32 the four runs agree to every digit
reported, globally and per type. In bf16 they spread by about 0.15, which is
padded-shape arithmetic rather than information crossing between questions.

## Files

| file | what |
| --- | --- |
| `model.safetensors` | the weights, 143 tensors. No pickle is published, so a load executes nothing. |
| `config.json` | the encoder's own config, verbatim from `jhu-clsp/mmBERT-base`, plus one `decision_jef` key for this model's geometry. |
| `temperatures.json` | the fitted per-bucket temperatures. Off by default; see above. |
| `metrics.json` | this checkpoint's benchmark figures, raw and calibrated. |
| `tokenizer.json`, `tokenizer_config.json`, `special_tokens_map.json` | vendored, so a load needs one repository and no second download. |

The encoder tensors carry their canonical ModernBERT names, so the fine-tuned
encoder can be loaded on its own:

```python
from transformers import AutoModel
encoder = AutoModel.from_pretrained("BarraHome/Decision-Jef-0.1")
```

That returns a `ModernBertModel` with every one of its 134 weights loaded and
the nine decision-head tensors skipped. Verified tensor by tensor: both paths
give bit-identical encoder weights.

## Latency

NVIDIA H100 NVL, fp32, median of 30 calls after warm-up, measured end to end
through this package: packing, encoder and readout. fp32 is what
`from_pretrained` gives you; bf16 roughly halves these numbers.

| questions in one call | median | p95 |
| --- | --- | --- |
| 1 | 11.67 ms | 12.44 ms |
| 2 | 12.01 ms | 13.14 ms |
| 3 | 12.19 ms | 12.85 ms |
| 4 | 12.30 ms | 12.57 ms |

Three extra questions cost 0.63 ms in total, about 5% over a single question.
The state is encoded once and the question branches are masked apart, so a
request carrying four questions is not four requests — but it is not free
either, and the trend across these four rows is monotonic rather than noise.

Throughput at batch 64 and 1,024 tokens is 5.35 ms per decision in fp32.

## Usage

```bash
pip install decision-jef
```

The weights are published separately from the package.

```python
from decision_jef import Decider, Question, email

d = Decider.from_pretrained("BarraHome/Decision-Jef-0.1")

state = email.as_state(
    "user@acme.com", "Duplicate charge on invoice #4411",
    "We were billed twice for March. Please refund the duplicate today "
    "or we will cancel our plan.")

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
department  billing  1.00  {'billing': 0.9997, 'technical': 0.0001, 'sales': 0.0001, 'other': 0.0001}
urgency     1        0.92  {'0': 0.0576, '1': 0.9151, '2': 0.0273}    score=0.97
churn_risk  0.9726   0.97  {'false': 0.0274, 'true': 0.9726}
```

`decide` returns raw probabilities. Pass `calibrated=True` to apply the shipped
temperatures, and read the calibration section first.

`d.to_wire(answers)` returns the same content as a JSON-ready response body.

A question must be a `Question`, not a plain dict. Passing a dict raises a
`TypeError` that says so.

### Give every yes/no outcome a description

This is a requirement, not a style note. The model scores the option text, so
a bare yes/no gives it nothing to compare, and the answer collapses toward an
uninformative half. On the example above:

| question | p(true) with bare yes/no | p(true) with descriptions |
| --- | --- | --- |
| threatens to leave | 0.564 | **0.973** |
| requests a refund | 0.539 | **0.961** |

## Large option sets

Accuracy falls as the option list grows, and the window is not the reason —
eighty options fit in 898 tokens with nothing truncated.

| options | accuracy | random |
| --- | --- | --- |
| 5 | 0.900 | 0.200 |
| 10 | 0.815 | 0.100 |
| 20 | 0.750 | 0.050 |
| 40 | 0.610 | 0.025 |
| 65 | 0.535 | 0.015 |

Laya reports 0.425 on a 77-label set and attributes it to an option-text
budget; Jev is reported at 0.870 and is ahead of this model here. Narrow the
set first if you are above about twenty options:

```python
from decision_jef import shortlist_decide

answers, meta = shortlist_decide(decider, state, questions, k=16)
meta["intent"]["kept"]     # the options that survived, best first
```

Ranking is lexical overlap between the state and each option's text, weighted
so a term every option shares counts for little. It needs no second model.
Pass `embed_fn` to rank with your own embeddings instead. A question at or
below `k` options passes through and costs nothing.

Ranking with this model's own logits was tried and does not work: at fifty
options they are the broken signal, so they rank as badly as they answer.

## Email

A reply carries the thread below it, a signature, and often a legal
disclaimer. None of that is the message, and it is often the longest part.

```python
from decision_jef import email

state = email.as_state(sender, subject, body)
```

`email.clean(body)` cuts the quoted thread, the signature block and the
disclaimer, and `as_state` formats the fields the way the rest of this package
feeds the model.

## The three question types

| type | `criteria` | answer |
| --- | --- | --- |
| `choice` | ordered map of key to description, up to 255 | `choice`, `probabilities` |
| `noul` | optional map of `false` and `true` to a description — supply it | `noul` probability |
| `score` | ordered array of 2 to 10 level descriptions | probability-weighted `score`, `legend` |

Option order is part of the request: the same options in a different order are
a different request. The answer changes on 5.7% of `choice` questions under
permutation, so average over permutations if you need a stable answer.

## How it works

```
<bos> state ... <eos>                                    shared, encoded once
[Q] instructions ... [OPT] opt0 [OPT] opt1 ... [DEC]     question 1
[Q] instructions ... [OPT] opt0 [OPT] opt1 ... [DEC]     question 2
```

The query is read at `[DEC]`, after the whole option list, so the decision sees
every option. Keys come from each `[OPT]` in the same pass, so the options are
read together rather than scored in isolation. Each question attends to the
state and to itself only; the state attends to neither. Position ids restart at
the end of the state for every question, and the local-attention band is
measured in those restarted positions, which is what makes isolation hold
against the other questions' length and not only their content.

## No escalation head in this release

Earlier releases shipped a head that predicted whether the model's own answer
was wrong, so a caller could hand the flagged cases to a person. It does not
work in these weights and has been removed rather than shipped with a caveat.

Measured on the benchmark's 2,000 decisions before removal: mean predicted
P(wrong) was 0.045 where the answer was right and 0.083 where it was wrong.
Escalating the worst half by predicted P(wrong) raised accuracy on the rest
from 0.777 to 0.795 — not enough separation to set a threshold on, and not
something a different threshold recovers.

`decide(..., with_escalation=True)` therefore returns `escalation=None`. The
field is documented and will carry a value again when a head earns it.

## Limitations

- **`score` is the weakest type** at 74.60, 1.78 behind the best published
  figure, and it is the whole of this model's remaining deficit.
- **Large option sets degrade steadily.** 0.750 at twenty options and 0.535 at
  sixty-five, against a random baseline of 0.050 and 0.015. Jev is reported at
  0.870 at sixty-five and is ahead here. Use `shortlist_decide` above twenty.
- **A bare yes/no is close to uninformative.** 0.564 where the described form
  gives 0.973. Supply a description for `false` and for `true`.
- **The shipped temperatures sharpen and are off by default.** They take ECE
  from 0.043 to 0.008 and take Brier from 0.083 to 0.108.
- **No escalation head**, see above.
- **Option order still changes the answer on 5.7% of `choice` questions.** That
  is ahead of both published figures but it is not zero.
- Trained and measured on **English** typed decisions. The backbone is
  multilingual and the tokenizer covers 256k tokens, but no non-English
  benchmark has been run — treat multilingual use as untested.
- The `guardrails` and `moderation` tags reflect coverage of toxicity and
  hate-speech decisions. **Neither capability has been benchmarked.**
- Reported global figures carry about ±0.15 of bf16 arithmetic noise. The
  77.80 above is a single measurement, not a mean over seeds.
- Long states are truncated to the window with the questions reserved first.

## License and provenance

MIT, following the `jhu-clsp/mmBERT-base` backbone. No weights, gradients or
private data from any third-party decision service are used or claimed; the
design follows publicly documented API behaviour and public benchmark splits.
