# Reproducibility guide

## Reproducibility boundary

This repository makes the code, frozen endpoint definitions, leakage exclusions,
configuration, and curated result tables inspectable. It does not redistribute
third-party datasets or model weights. Trained bridge checkpoints are also not
included; exact score replay therefore requires either owner-provided checkpoints
or retraining under the recorded recipe.

Hardware can affect bf16 greedy generations. Treat results produced on a different
GPU architecture as a replication, not as a byte-reproduction claim.

## Invariants

- Never edit, reorder, extend, or overwrite a file under `data/gqa/`.
- Use a new `--ckpt-tag` for every training run.
- Run `scripts/13_vg_leakage_audit.py` before a Visual Genome-derived experiment;
  the required result is `residual = 0`.
- Corrected-layer primary metric: `exact_full`.
- Never combine corrected and legacy rows in a table, average, or contrast.
- Parse CSV with Python's `csv` module or another real CSV parser.
- Stop on any manifest or raw-result mismatch.

## Environment profiles

### Legacy CUDA 12.4

```bash
python3.12 -m venv .venv
source env.sh
python -m pip install --upgrade pip
python -m pip install -r requirements/legacy-cu124.txt
python -m pip check
```

### Corrected Blackwell CUDA 13.0

```bash
python3.12 -m venv .venv
source env.sh
python -m pip install --upgrade pip
python -m pip install -r requirements/corrected-cu130.txt
python -m pip check
```

The corrected profile records the environment used for the final bf16 runs. If
the exact wheel index is no longer available, preserve the versions in a container
rather than silently substituting packages.

## Verification before compute

```bash
source env.sh
python -m pytest -q
python -m compileall -q src scripts
bash -n env.sh cluster/train.sbatch cluster/evaluate.sbatch
sha256sum -c release/RELEASE_MANIFEST.sha256
```

Before any training or evaluation process:

```bash
nvidia-smi
pgrep -af '06c_train_bridge|07_evaluate' || true
```

Do not start if the assigned GPU is already occupied.

## Corrected bridge recipes

Shared values: 150,000 examples, seed 42, five epochs, flat learning rate
`1e-4`, weight decay `0.01`, batch size 8, bf16 frozen LLM, ChatML v1, and
supervised EOS.

| Encoder | Connector | Required geometry flags |
|---|---|---|
| CLIP | MLP256 | `--bridge-type mlp --mlp-hidden 5888 --strip-cls` |
| DINOv2 | MLP256 | `--bridge-type mlp --mlp-hidden 5888 --strip-cls` |
| I-JEPA | MLP256 | `--bridge-type mlp --mlp-hidden 5700` |
| any | Q-Former32 | `--bridge-type qformer` plus encoder-specific CLS handling |

Example:

```bash
python scripts/06c_train_bridge.py \
  --encoder ijepa \
  --bridge-type mlp \
  --mlp-hidden 5700 \
  --limit 150000 \
  --epochs 5 \
  --seed 42 \
  --lr 1e-4 \
  --weight-decay 0.01 \
  --batch-size 8 \
  --grad-accum 1 \
  --llm-precision bf16 \
  --prompt-format chatml_v1 \
  --supervise-eos \
  --ckpt-tag reproduction_ijepa_mlp_s42
```

On a smaller GPU, lower `--batch-size` and increase `--grad-accum` so their
product remains 8. Record both values in the run provenance.

## Evaluation surfaces

| File | Role |
|---|---|
| `tune_500_qids.json` | Development and selection |
| `confirm_3000_qids.json` | Confirmatory surface; no selection |
| `eval_2000_qids.json` | Locked endpoint |
| `testdev_12578_qids.json` | Official GQA Test-Dev surface |

Example:

```bash
python scripts/07_evaluate.py \
  --qids data/gqa/confirm_3000_qids.json \
  --checkpoint outputs/checkpoints/<checkpoint>.pt \
  --out-dir outputs/eval \
  --num-examples 0
```

For inference-time augmentation, use `scripts/09_augment_eval.py` and give a
fresh `--stem-suffix` whenever the condition differs from the plain checkpoint.

## Result inspection

The `results/` directory is a frozen catalogue snapshot, not a training input.
Validate it with a real CSV parser:

```bash
python - <<'PY'
from pathlib import Path
import csv

for path in sorted(Path("results").glob("*.csv")):
    with path.open(newline="") as handle:
        rows = list(csv.reader(handle))
    assert rows and all(len(row) == len(rows[0]) for row in rows)
    print(path, len(rows) - 1, "rows")
PY
```
