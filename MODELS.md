# Models and checkpoints

## Upstream frozen models

The configuration uses Hugging Face model IDs and downloads them through
Transformers on first use.

| Component | Model ID | Source |
|---|---|---|
| Language model | `Qwen/Qwen2-7B-Instruct` | https://huggingface.co/Qwen/Qwen2-7B-Instruct |
| CLIP encoder | `openai/clip-vit-large-patch14` | https://huggingface.co/openai/clip-vit-large-patch14 |
| DINOv2 encoder | `facebook/dinov2-large` | https://huggingface.co/facebook/dinov2-large |
| I-JEPA encoder | `facebook/ijepa_vith14_1k` | https://huggingface.co/facebook/ijepa_vith14_1k |

Review each upstream model card and license before use. Model downloads are kept
under `.cache/huggingface/` after `source env.sh`.

Every known upstream model is loaded with an immutable `revision=` value from
`src/models/revisions.py`. The audited values are also recorded in
[`artifacts/model_revisions.json`](artifacts/model_revisions.json). Historical
checkpoints stored model IDs but not revisions, so these hashes are evidenced by
the source experiment checkout's local cache refs rather than checkpoint metadata.

The machine-readable [model inventory](artifacts/model_inventory.json) records all
three feature geometries and matched connector parameter counts. Q-Former module
dumps are provided for [CLIP](artifacts/bridge_architecture_clip.txt),
[DINOv2](artifacts/bridge_architecture_dinov2.txt), and
[I-JEPA](artifacts/bridge_architecture_ijepa.txt). DINOv2 and CLIP share the same
257-token, 1024-dimensional bridge geometry; I-JEPA uses 256 tokens at 1280 dimensions.

## Trained bridge checkpoints

Bridge checkpoints are not included or hosted. They are hundreds of megabytes
each, and the complete local collection is tens of gigabytes. This is a source,
protocol, and result-artifact release; exact model replay requires owner-supplied
checkpoints or retraining.

To reproduce from scratch, use the recipes in
[REPRODUCIBILITY.md](REPRODUCIBILITY.md). Checkpoints are written to
`outputs/checkpoints/` with names of the form:

```text
bridge_<encoder>_<ckpt-tag>_ep<epoch>.pt
```

Never reuse a tag that already exists. The training script records the encoder,
connector geometry, prompt protocol, EOS supervision, precision, seed, data draw,
and optimiser settings inside the checkpoint. Evaluation validates those fields.

If checkpoints are published later, the bundle should include:

- one SHA-256 per file;
- exact source release commit;
- upstream model revisions;
- GPU architecture and dependency profile;
- training command and endpoint role;
- a statement that checkpoints contain no third-party dataset samples.
