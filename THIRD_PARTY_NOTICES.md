# Third-party notices

The MIT licence in `LICENSE` covers original project software. It does not
relicense upstream datasets, model weights, or third-party evaluation code.

## Datasets

This repository does not redistribute dataset images or the complete upstream
question/answer corpora. Users download each source themselves and must follow
the terms published by its owner:

- GQA — https://cs.stanford.edu/people/dorarad/gqa/download.html
- VQAv2 — https://visualqa.org/download.html
- COCO — https://cocodataset.org/#termsofuse
- Visual Genome — https://homes.cs.washington.edu/~ranjay/visualgenome/api.html
- LLaVA-Instruct-150K — https://github.com/haotian-liu/LLaVA/blob/main/docs/Data.md

The committed QID lists, exclusions, predictions, summaries, and plots are
project-generated reproducibility artifacts. They remain keyed to upstream GQA
identifiers and do not grant rights to the underlying images or annotations.

## Upstream models

Qwen2, CLIP, DINOv2, I-JEPA, and OWLv2 remain subject to their respective model
cards and upstream licences. This repository records exact revision hashes but
does not redistribute their weights. See `MODELS.md` and
`artifacts/model_revisions.json`.

## Official GQA evaluator

The official evaluator is not redistributed because its source page does not
state a clear licence for republication. `scripts/vendor/PROVENANCE.md` records
the upstream URL, expected byte count, and required SHA-256. Download it before
running `scripts/71_score_testdev_official.py`.
