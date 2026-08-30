# Limitations

- **External data:** datasets and photographs are not included. Full evaluation
  requires the official GQA question and image downloads.
- **Checkpoint availability:** trained bridge weights are not hosted by this
  release. Scores can be inspected immediately; model replay requires owner-
  supplied checkpoints or retraining.
- **Hardware sensitivity:** corrected bf16 greedy generations were measured on
  RTX PRO 6000 Blackwell GPUs. Other GPU architectures are replications rather
  than byte-reproduction claims.
- **Training-seed uncertainty:** CLIP and I-JEPA have three-seed coverage for the
  primary connector comparison; not every encoder/ablation has equal seed depth.
- **Endpoint roles:** tune-500 is a development surface. Confirmatory, locked,
  and Test-Dev results must not be pooled or used retrospectively for selection.
- **Metric scope:** `exact_full` is primary for corrected ChatML+EOS runs. Legacy
  VQA-soft values document an older protocol and are never mixed with corrected
  results.
- **Category coverage:** the published Test-Dev and development surfaces contain
  zero `count` questions, so the project makes no counting claim.
- **Efficiency scope:** latency and memory are descriptive batch-1 measurements
  on one GPU. They do not generalise automatically to batched serving, other
  hardware, precisions, sequence lengths, or answer lengths.
- **Coverage:** CPU tests exercise release contracts and reusable logic, but do
  not execute every data/GPU branch. Passing CI is not proof of score replay.
- **Quoted-study code:** the structural-text (geometry/lexical) Test-Dev study
  and the image-clustered bootstrap are reported from frozen records with source
  hashes; their code is not in this release and those rows cannot be regenerated
  from it.
- **Third-party rights:** the project licence does not cover external datasets,
  weights, or the official GQA evaluator.
