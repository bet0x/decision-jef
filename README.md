<div align="center">
  <p>
    <a href="https://pypi.org/project/decision-jef/"><img src="https://img.shields.io/pypi/v/decision-jef?logo=pypi&logoColor=white" alt="PyPI version"></a>
    <a href="https://pypi.org/project/decision-jef/"><img src="https://img.shields.io/pypi/pyversions/decision-jef?logo=python&logoColor=white" alt="Supported Python versions"></a>
    <a href="https://github.com/bet0x/decision-jef/actions/workflows/ci.yml"><img src="https://github.com/bet0x/decision-jef/actions/workflows/ci.yml/badge.svg?branch=main" alt="CI"></a>
    <a href="https://huggingface.co/BarraHome/Decision-Jef-0.1"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Decision--Jef--0.1-yellow" alt="Model on Hugging Face"></a>
    <a href="./LICENSE"><img src="https://img.shields.io/github/license/bet0x/decision-jef" alt="MIT License"></a>
    <a href="https://github.com/bet0x/decision-jef"><img src="https://img.shields.io/github/stars/bet0x/decision-jef?style=flat&logo=github" alt="GitHub stars"></a>
  </p>
  <p>
    <img src="https://img.shields.io/badge/typed--decisions-73.90-1f6feb" alt="73.90 on the typed-decisions benchmark">
    <img src="https://img.shields.io/badge/latency-11.6%20ms%20%C2%B7%203%20decisions-2da44e" alt="11.6 ms for three decisions">
    <img src="https://img.shields.io/badge/ECE-0.084%20raw%20%C2%B7%200.027%20tuned-2da44e" alt="Expected calibration error">
    <img src="https://img.shields.io/badge/parameters-307M-8250df" alt="307 million parameters">
  </p>
</div>

# Decision-Jef-0.1

Answer several runtime-defined questions about one state, in a single forward
pass, with a probability over exactly the options you supply. Read the
calibration section before you threshold on those probabilities.

307M parameters. **11.6 ms for three decisions in one forward pass.**

The answer space is built from the request, so a value you did not offer is not
representable — not merely unlikely. There is no classification head over a
fixed label set.

## Results

Typed-decisions test set, 2,000 decisions: 600 `choice`, 600 `noul`, 800
`score`. Same set and same split as the published competitors.

| model | global | choice | noul | score |
| --- | --- | --- | --- | --- |
| Decision-1.0-Lex | **78.15** | **74.00** | 84.67 | **76.38** |
| Laya Typed Decisions | 76.60 | 73.33 | **85.67** | 72.25 |
| **Decision-Jef-0.1** | 73.90 | 71.17 | 82.50 | 69.50 |

Behind both on this benchmark, and the reason is worth stating rather than
hiding: an earlier checkpoint of this model scored **77.00** here, ahead of
Laya. It is still in this repository's history at revision `36e88a3`. The
current weights trade 3.1 points of in-domain accuracy for two things this
benchmark cannot see at all, because every choice question in it has three to
five options in one fixed order:

| | earlier checkpoint | current |
| --- | --- | --- |
| answer changes when the options are permuted | 0.193 | **0.095** |
| accuracy at 20 options | 0.095 | **0.950** |
| accuracy at 65 options | 0.025 | **0.840** |

For scale on the second: Laya reports 0.425 on a 77-label set and attributes it
to an option-text budget; Jev is reported at 0.870. And on permutation, 0.095
is ahead of both published figures -- Jev measured at 0.13, Laya at 0.15.

| | value |
| --- | --- |
| soft accuracy | 0.593 |
| ECE, 10 bins, with the shipped temperatures | 0.027 |
| ECE, 10 bins, as shipped (no temperature) | **0.084** |
| Brier, summed over classes | **0.079** |
| score MAE | 0.286 |
| NLL, choice / noul / score | 0.7666 / 0.3901 / 0.7415 |
| majority class on this set | 0.457 |
| random guess on this set | 0.318 |

Soft accuracy is the label mass on the answer we pick, which matters on a
benchmark whose labels are annotator averages rather than single verdicts. On
that measure and on Brier and ECE this model is ahead of both published
figures; on hard accuracy it is behind.

One caution about published Brier numbers: they use two different conventions.
Laya reports 0.061 averaged per class and attributes 0.148 to Jev summed over
classes, which reads as a 2.4x gap and is a unit mismatch. The 0.079 above is
summed, the same convention as the 0.148.

Temperature scaling per (type, cardinality) bucket ships with the model but is
**off by default**, and the reason is below. It never changes an argmax, so
accuracy is 73.90 either way.

### The calibration number needs a caveat

ECE on this benchmark improves from 0.084 to 0.027 with the fitted
temperatures, and that improvement is misleading. The temperatures sharpen
(around 0.34), and sharpening is right for a question the state answers and
wrong for one it does not.

Measured against a generator whose true posterior is known by construction --
a ticket that cues one department has a true answer, one that cues two is a
genuine coin flip between them:

| case | true answer | as shipped | with the fitted temperatures |
| --- | --- | --- | --- |
| one department cued | 1.00 | **1.000** | 1.000 |
| **two departments cued** | **0.50** | 0.895 | **0.930** |
| escalation implied by priority | 0.90 | 0.980 | 0.991 |
| mass on the two cued departments | 1.00 | **1.000** | 1.000 |

Read the first and last rows first: the model recovers the true posterior
**exactly** when the state settles the question, and puts all of its mass on
the two departments the state allows. Neither was true of the earlier
checkpoint, which reported 0.660 where the truth was 1.0 and leaked a third of
its mass onto departments the state ruled out.

The second row is the failure that remains. Where the truth is an even split
the model reports 0.895, and the fitted temperatures push that to 0.930.
Aggregate ECE does not see it, because most benchmark rows are not coin flips
and sharpening helps on those, so the average moves the wrong way for the right
reason.

So: trust a high probability from this model when the state contains the
answer, and do not read a high probability as evidence that the state contains
it. The shipped temperatures were fitted against labels that are annotator
averages, which is not the same target as the truth; fit your own on your own
data if you need one.

Question isolation is exact: adding a question moves another question's logits
by at most 3e-07.

## Files

| file | what |
| --- | --- |
| `model.safetensors` | the weights. No pickle is published, so a load executes nothing. |
| `config.json` | the encoder's own config, verbatim from `jhu-clsp/mmBERT-base`, plus one `decision_jef` key for this model's geometry. |
| `temperatures.json` | the fitted per-bucket temperatures. Off by default; see below. |
| `metrics.json` | the benchmark numbers of this checkpoint. |
| `tokenizer.json`, `tokenizer_config.json`, `special_tokens_map.json` | vendored, so a load needs one repository and no second download. |

The encoder tensors carry their canonical ModernBERT names, so the fine-tuned
encoder can be loaded on its own:

```python
from transformers import AutoModel
encoder = AutoModel.from_pretrained("BarraHome/Decision-Jef-0.1")
```

That returns a `ModernBertModel` with every one of its 134 weights loaded and
the nine decision-head tensors skipped. Both paths give bit-identical encoder
weights.

## Latency

NVIDIA H100 NVL, fp32, median of 30 calls after warm-up. End to end: packing,
encoder and readout. fp32 is what `from_pretrained` gives you; bf16 roughly
halves these numbers.

| questions in one call | median | p95 |
| --- | --- | --- |
| 1 | 11.83 ms | 13.59 ms |
| 2 | 12.03 ms | 12.48 ms |
| 3 | 11.38 ms | 12.19 ms |
| 4 | 11.96 ms | 12.99 ms |

**One question costs the same as four.** The spread across the four rows is
smaller than the spread between repeats of the same row, so the honest reading
is that the extra questions are free at this scale rather than that they cost
some small amount. The state is encoded once and the question branches are
masked apart, so a request carrying four questions is not four requests.

Throughput at batch 64 and 1,024 tokens is 5.21 ms per decision in fp32.

## Usage

```bash
pip install decision-jef
```

The weights are published separately from the package. Authenticate with
`hf auth login` if the model repository is not yet public, or point
`from_pretrained` at a local directory holding `model.pt`.

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

`decide` returns raw probabilities. Pass `calibrated=True` to apply the shipped
temperatures, and read the calibration section first -- they sharpen, which is
not what every input wants.

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

## Large option sets

Accuracy falls toward chance past about twenty options, and the window is not
the reason -- eighty options fit in 898 tokens with nothing truncated. The
model simply never saw more than ten during training. Narrow the set first:

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

## Escalation

When the checkpoint carries the escalation head, it will say how likely its own
answer is to be wrong, and whether that exceeds the threshold the two costs
imply:

```python
answers = decider.decide(state, questions, with_escalation=True)
answers["department"].escalation
# {"probability_wrong": 0.21, "threshold": 0.1667, "escalate": True}
```

The threshold is `cost_escalate / cost_wrong`, 0.5 over 3.0 by default: hand
the decision to a person when being wrong is more expensive than interrupting
someone. Set both on the `Decider` to match your own costs. The field is
`None` when the checkpoint has no such head.

The head is trained to predict its own errors, so it needs no extra labels.
Measured on the benchmark's 2,000 decisions:

| | value |
| --- | --- |
| mean predicted P(wrong) when the answer is right | 0.378 |
| mean predicted P(wrong) when the answer is wrong | 0.602 |
| share of cases escalated at the default threshold | 46.6% |
| accuracy on the cases it answers | **0.821** |
| accuracy answering everything | 0.739 |

Escalating about half the cases raises accuracy on the rest from 0.739 to
0.821. For comparison, Laya reports 0.803 accuracy while acting on everything.

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

- **Overconfident where the answer is genuinely undetermined.** Against a
  known posterior it is exact when the state settles the question and reports
  0.895 where the truth is an even 0.50. The benchmark's own ECE cannot show
  this, because its labels are annotator averages rather than truth.
- **The shipped temperatures sharpen and are off by default.** They take
  aggregate ECE from 0.084 to 0.027 and make the ambiguous case worse, 0.895 to
  0.930 against a true 0.50. Fit your own on your own data.
- **`score` is the weakest type** at 69.50, 6.88 behind Lex.
- Trained and measured on **English** typed decisions. The backbone is
  multilingual and the tokenizer covers 256k tokens, but no non-English
  benchmark has been run — treat multilingual use as untested.
- The `guardrails` and `moderation` tags reflect coverage of toxicity and
  hate-speech decisions. **Neither capability has been benchmarked.**
- **Option order still changes the answer on 9.5% of choice questions**, over
  1,800 permutations of the benchmark. That is ahead of both published figures
  but it is not zero: average over permutations if you need a stable answer.
- **Large option sets work now but degrade past about forty.** 0.950 at twenty
  options, 0.910 at forty, 0.840 at sixty-five, against a random baseline of
  0.050, 0.025 and 0.015. Use `shortlist_decide` above that.
- **Behind both published models on hard accuracy** on the typed-decisions
  benchmark, 73.90 against 78.15 and 76.60, while ahead of both on soft
  accuracy, Brier and ECE.
- Long states are truncated to the window with the questions reserved first.

## License and provenance

MIT, following the `jhu-clsp/mmBERT-base` backbone. No weights, gradients or
private data from any third-party decision service are used or claimed; the
design follows publicly documented API behaviour and public benchmark splits.
