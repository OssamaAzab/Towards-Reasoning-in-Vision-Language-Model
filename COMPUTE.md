# Compute, memory, and runtime

## Environment

Final corrected evaluations used Python 3.12, a CUDA 13.0 PyTorch build, bf16
Qwen2-7B-Instruct, greedy decoding, and NVIDIA RTX PRO 6000 Blackwell GPUs. The
bridge is the only trained component.

## Connector size

| Connector | Visual tokens | Trainable parameters |
|---|---:|---:|
| Q-Former32 | 32 | 60.28M CLIP/DINOv2; 60.48M I-JEPA |
| MLP256 | 256 | 61.82M CLIP/DINOv2; 60.23M I-JEPA |
| Pool32 | 32 | 61.82M CLIP; 60.23M I-JEPA |

Exact integers are in `artifacts/model_inventory.json`.

## Controlled inference benchmark

The benchmark used one RTX PRO 6000, bf16, greedy decoding, batch size 1,
15 warm-up examples, and three repetitions of 200 examples per available
configuration. Times are medians over 600 measurements.

| Configuration | Visual tokens | Vision | Connector | Prefill | Decode | GPU model inference | Peak allocated | Peak reserved |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| CLIP / MLP256 | 256 | 8.45 ms | 0.70 ms | 21.37 ms | 12.75 ms | 43.29 ms | 15.79 GiB | 16.46 GiB |
| CLIP / Q-Former32 | 32 | 8.37 | 1.28 | 14.55 | 12.58 | 36.80 | 15.67 | 16.55 |
| CLIP / Pool32 | 32 | 8.45 | 0.33 | 14.52 | 12.58 | 35.89 | 15.67 | 16.43 |
| DINOv2 / MLP256 | 256 | 8.49 | 0.70 | 21.38 | 12.76 | 43.33 | 15.80 | 16.46 |
| DINOv2 / Q-Former32 | 32 | 8.43 | 1.28 | 14.53 | 12.57 | 36.83 | 15.67 | 16.57 |
| I-JEPA / MLP256 | 256 | 15.62 | 0.76 | 21.36 | 12.76 | 50.50 | 17.07 | 17.77 |
| I-JEPA / Q-Former32 | 32 | 15.61 | 1.27 | 14.52 | 12.57 | 43.98 | 16.95 | 17.89 |
| I-JEPA / Pool32 | 32 | 15.62 | 0.36 | 14.53 | 12.57 | 43.11 | 16.94 | 17.74 |

DINOv2 / Pool32 was not constructed and is not estimated. Reducing 256 visual
tokens to 32 was associated with approximately 1.47× lower prefill latency and
1.15×–1.21× lower GPU model-inference latency across the five available
compressed configurations. This is descriptive engineering evidence on one
hardware/protocol surface, not a causal or cross-hardware claim.

**FLOPs were not measured.** Job wall-clock durations are not used as latency
comparisons because queueing, setup, evaluation, and checkpoint behavior differ.

## Storage planning

- Repository clone: approximately 15 MB.
- Qwen2-7B-Instruct download: approximately 15 GB.
- One trained bridge checkpoint: approximately 0.7 GB for corrected MLP/Q-Former.
- Complete local historical checkpoint collection: tens of GB.
- Frozen feature caches: dataset- and encoder-dependent; plan for tens of GB per
  encoder on the 150K/500K training mixtures.

The exact benchmark values are in `docs/figures/data/efficiency.json`.
