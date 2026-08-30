# Third-party scorer provenance

## `gqa_eval.py` — the official GQA evaluation script

**Not written by this project and not redistributed by this repository.** Download it from the
recorded upstream URL before official scoring. A local "improvement" would silently make results
non-comparable, so `scripts/71_score_testdev_official.py` verifies its SHA-256 before execution.

| | |
|---|---|
| retrieved from | `https://raw.githubusercontent.com/ronghanghu/gqa_eval_script/master/eval.py` |
| source snapshot verified on | 2026-08-07 |
| sha256 | `de6426214a886d6baf98b797ae1dbfd1ecdfed4a66fd647104f8febbc70caf9b` |
| bytes | 20,297 |
| upstream | GQA (Hudson & Manning, CVPR 2019), <https://cs.stanford.edu/people/dorarad/gqa/evaluate.html> |

Place the downloaded file at `scripts/vendor/gqa_eval.py`, then verify:

```bash
sha256sum scripts/vendor/gqa_eval.py
```

The required digest is the one in the table above.

### Why this source rather than Stanford's `eval.zip`

Stanford distributes the script inside `eval.zip` (784 MB), which also carries the `*_choices.json`
files needed for Validity and Plausibility. This source is the same script with documented fixes to
the original's bugs. We use it for **Accuracy, Binary, Open and Distribution only** — none of which
depend on the files in the zip. If Validity or Plausibility are ever reported, download the zip and
record that decision here.

### What it does, verbatim

Accuracy is raw string equality, with no normalisation on either side (`gqa_eval.py:358-362`):

```python
gold = question["answer"]
predicted = predictions[qid]

correct = (predicted == gold)
score = toScore(correct)
```

Predictions are loaded with no transformation (`gqa_eval.py:141`):

```python
predictions = {p["questionId"]: p["prediction"] for p in predictions}
```

Binary vs open is a property of the QUESTION, not of the answer (`gqa_eval.py:375`):

```python
answerType = "open" if question["types"]["structural"] == "query" else "binary"
```

### Test-Dev limitations are upstream, not ours

The script sets `scenes = None  # for testdev` and `choices = None  # for testdev`. Grounding,
Validity and Plausibility are therefore undefined on Test-Dev in the official tooling itself. When
`choices` is absent the script does not skip those metrics — it scores every question `False`, so
**a printed `Validity: 0.00%` means "not computed", never "the model scored zero"**. Do not report
those two numbers from a Test-Dev run.

### Agreement with our metric

Run on all eight corrected confirmatory cells, this script reproduces our `exact_full` exactly:

| cell | `exact_full` | official Accuracy |
|---|---:|---:|
| clip + mlp256 | 51.33% | 51.33% |
| clip + pool32 | 44.30% | 44.30% |
| clip + qformer32 | 44.47% | 44.47% |
| dinov2 + mlp256 | 49.93% | 49.93% |
| dinov2 + qformer32 | 43.83% | 43.83% |
| ijepa + mlp256 | 45.37% | 45.37% |
| ijepa + pool32 | 41.80% | 41.80% |
| ijepa + qformer32 | 41.37% | 41.37% |

8 of 8 exact, verified at export time on the source experiment checkout. This release does not ship
the scorer, so the agreement is not re-executed in CI; `scripts/71_score_testdev_official.py`
fails closed until the hash-verified upstream file is installed, and then re-checks every cell it scores.

Our `exact_full` is strictly more permissive (it lowercases, strips punctuation and drops
articles, symmetrically). Under the corrected protocol the model emits clean short answers, so
that leniency is never exercised — hence the exact agreement. On the legacy layer the two differ
by +0.20 points, which is one reason legacy numbers are never reported as GQA accuracy.
