# Portable Slurm templates

These templates deliberately omit partition, account, node, and site-specific
filesystem directives. Supply scheduler policy on the `sbatch` command line.
Both jobs request one GPU and fail unless repository and virtual-environment paths
are explicit.

Training example:

```bash
sbatch \
  --partition=<gpu-partition> \
  --account=<account> \
  --export=ALL,REPO_ROOT="$PWD",VENV_PATH="$PWD/.venv",ENCODER=clip,CKPT_TAG=reproduction_clip_mlp_s42,BRIDGE_TYPE=mlp,MLP_HIDDEN=5888,STRIP_CLS=1 \
  cluster/train.sbatch
```

Evaluation example:

```bash
sbatch \
  --partition=<gpu-partition> \
  --account=<account> \
  --export=ALL,REPO_ROOT="$PWD",VENV_PATH="$PWD/.venv",CHECKPOINT="$PWD/outputs/checkpoints/bridge_clip_reproduction_clip_mlp_s42_ep5.pt",QIDS="$PWD/data/gqa/confirm_3000_qids.json" \
  cluster/evaluate.sbatch
```

Before submission, inspect local scheduler rules and adjust memory and walltime.
Do not increase the GPU count: independent single-GPU cells should be submitted as
separate jobs with unique checkpoint tags.
