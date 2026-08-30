# Data acquisition and layout

The repository does not redistribute datasets. Download each source from its
official page and comply with its terms of use.

For the full transformation from official downloads to cached features, frozen
surfaces, official scoring, category plots, and the all-example results packet, see
[PREPROCESSING.md](PREPROCESSING.md).

| Source | Official location | Used for |
|---|---|---|
| GQA | https://cs.stanford.edu/people/dorarad/gqa/download.html | Evaluation questions, images, scene graphs |
| VQAv2 | https://visualqa.org/download.html | Short-answer bridge training |
| COCO | https://cocodataset.org/ | VQAv2 and LLaVA images |
| Visual Genome | https://homes.cs.washington.edu/~ranjay/visualgenome/api.html | Scene-graph augmentation resources |
| LLaVA-Instruct-150K | https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K | Instruction bridge training |

Expected layout:

```text
data/
├── coco/
│   ├── train2014/
│   └── train2017/
├── gqa/
│   ├── images/
│   ├── questions/
│   │   ├── val_balanced_questions.json
│   │   └── testdev_balanced_questions.json
│   └── val_sceneGraphs.json
├── llava/
│   └── llava_instruct_150k.json
├── visual_genome/
└── vqa/
    ├── v2_OpenEnded_mscoco_train2014_questions.json
    └── v2_mscoco_train2014_annotations.json
```

## Leakage exclusions

Two small artifacts are included because training is not reproducible or safe
without them:

| File | IDs | SHA-256 |
|---|---:|---|
| `data/vqa/gqa_val_coco_exclude.json` | 4,921 | `4016f67eda0f3f8258a24fe4b22f3cbd8a2868fcd73ad0559824fac4d8f6e3e3` |
| `data/gqa/vg_exclude_gqa_eval.json` | 10,234 | `3c073d451d799b72e575bbd441854894300a517ca8e0ed53cb896027ea6ffcc3` |

The loaders apply the COCO exclusion to both LLaVA and VQAv2. First verify the
included Visual Genome exclusion, then audit the actual generator training IDs:

```bash
python scripts/13_vg_leakage_audit.py
python scripts/13_vg_leakage_audit.py --train-ids path/to/generator_train_image_ids.json
```

The first command never overwrites an existing exclusion file. Do not continue
unless the second command reports a zero residual.

## Frozen evaluation artifacts

Every QID list and companion manifest under `data/gqa/` is covered by
`release/RELEASE_MANIFEST.sha256` and by the release contract test. Never edit
these files in place. A supplementary surface must use a new filename and cannot
be merged into a locked headline table.
