# Complete Test-Dev results

This folder contains all 12,578 QIDs in the frozen Test-Dev endpoint and all six
CLIP/DINOv2/I-JEPA × MLP256/Q-Former32 prediction cells.

| File | Purpose |
|---|---|
| `predictions.csv` | Flat, GitHub-friendly table: QID, image ID, category/types, six answers, six correctness flags |
| `predictions.json` | The same 12,578 records with nested per-cell metadata |
| `category_summary.csv` | Category counts and per-cell full-string accuracy |
| `provenance.json` | SHA-256 for every raw input and derived output |
| [`testdev_category_distribution.png`](../../docs/figures/testdev_category_distribution.png) | Number of questions in each reasoning category |
| [`testdev_category_accuracy.png`](../../docs/figures/testdev_category_accuracy.png) | Six-cell accuracy by category |

No raw image, question text, or gold answer is committed because the official page
does not state a clear redistribution licence for republishing the whole corpus.
The QID and `image_id` columns join directly to the official GQA question and image
downloads. Place them at:

```text
data/gqa/questions/testdev_balanced_questions.json
data/gqa/images/<image_id>.jpg
```

The packet is rebuilt by `scripts/build_testdev_results.py`, which rejects partial
QID sets, duplicate QIDs, gold mismatches, metric mismatches, or incorrect endpoint
metadata.

After downloading the official questions/images, create a local joined CSV with
question text, gold/full answers, and image paths:

```bash
python scripts/materialize_testdev_examples.py --require-images
```

This writes `results/testdev/examples/examples_with_questions.csv`. The folder is
gitignored because it contains upstream question/gold text and machine-local paths.
