# Figure provenance

Every README figure is either a repository-native SVG schematic or a PNG regenerated from a
values JSON under `data/` by `scripts/generate_portfolio_plots.py` (six figures, one style) or
`scripts/build_testdev_results.py` (the two Test-Dev packet plots). No raw GQA or Visual Genome
photograph is redistributed. The values JSONs name the release row or lock entry each number
comes from; nothing is recomputed for display.

| File | What it shows | Provenance | SHA-256 |
|---|---|---|---|
| `architecture.svg` | Three frozen encoder options, connector choices, and frozen Qwen2 stack | Repository-native schematic built from `artifacts/model_inventory.json` | `f0cdb97bd24943be70e30a40148f5885f3da014c55856f283b1489f0cca6e13a` |
| `data_pipeline.svg` | Official downloads through leakage gates, feature caching, training, Test-Dev, and public artifacts | Repository-native schematic tied to `PREPROCESSING.md` | `54731a30bbbd7be1312c6407a83ae728f29a9b40609d73a068353a6f8db11006` |
| `evaluation_surfaces.svg` | The six frozen question sets, their sizes, roles, and order of use | Repository-native schematic; counts from `data/gqa/*_MANIFEST.json` and the README surfaces table | `75b12bf38f410a12950add16c38e2937d0da8009b6d0cf885704dc582050fd02` |
| `encoder_connector_interaction.png` | Six-cell CLIP/DINOv2/I-JEPA × MLP256/Q-Former32 Test-Dev result with image-clustered intervals | `scripts/generate_portfolio_plots.py` from `data/encoder_connector_interaction.json` | `0732c590b85c98fcc6ae6da202794e2877e3f830057f5e3f37618e723bfe69fa` |
| `confirmatory_results.png` | Eight confirmatory-surface cells (three encoders × three connectors) against the text-only floor | `scripts/generate_portfolio_plots.py` from `data/confirmatory_results.json`, transcribed from `results/results_corrected_eos.csv` by `result_id` | `8314ec6c8f4ef8a7d190f1f233401364205d12225497f62e4e982290c9214cfa` |
| `training_convergence.png` | Corrected MLP256 tune-500 full-string accuracy over epochs 1–5 for CLIP, DINOv2, and I-JEPA | `scripts/generate_portfolio_plots.py` from `data/training_convergence.json`; 15 raw evaluator record hashes are pinned there | `8004ffd0f97c725f8eb73bc50def199ac907a36dfe52a0e8e99487f6bfc4d5a8` |
| `efficiency.png` | Per-stage batch-1 inference latency for the eight available configurations | `scripts/generate_portfolio_plots.py` from `data/efficiency.json`; one GPU, 600 measurements per configuration | `6cfd6d135f4fc7081a85c1fdd2e14920c744a51f5446359386b79dd4764f935c` |
| `structural_augmentation.png` | Primary and secondary structural-text contrasts on GQA Test-Dev | `scripts/generate_portfolio_plots.py` from `data/structural_augmentation.json` | `cf6a5505cb6c056f5664e039135f3b882cfb3d677d8152fc0710ec0038c4c1e7` |
| `surface_composition.png` | Reasoning-category share of the tune-500, confirm-3000, and Test-Dev surfaces | `scripts/generate_portfolio_plots.py` from `data/surface_composition.json`, transcribed from `results/results_per_category.csv` and the Test-Dev manifest | `65de88d5fc646d66a100a351df2080744e637fa2ed2dedf8128f9d3b31ddefe3` |
| `testdev_category_distribution.png` | Category counts for all 12,578 Test-Dev questions | `scripts/build_testdev_results.py` from the frozen QIDs and official questions | `463a604b1f65e4ae682ed84040a5d2fad66444dc78e302e7ed50b1387c26b818` |
| `testdev_category_accuracy.png` | Six-cell full-string accuracy by Test-Dev category | `scripts/build_testdev_results.py` from all six evaluator record files | `b07cdc01e177a5a97cec07014817020e645c100e37e4f95b72754f729e1cd292` |

## Style contract

- One sans-serif face (DejaVu Sans, bundled with matplotlib) and one chrome palette in every PNG.
- Encoder identity is always CLIP `#2a78d6` (circle), DINOv2 `#e07020` (square), I-JEPA `#1a9c6b`
  (triangle). The trio was validated for colour-vision deficiency (adjacent and all-pairs,
  light and dark surfaces); marker shape carries identity in greyscale.
- Ordered quantities (pipeline stages, surface sizes) use one-hue blue ramps; non-detections
  are drawn as intervals spanning zero; cells that were never trained are drawn as absent,
  never as zero. Truncated accuracy axes are stated in the figure subtitle.
- Regeneration is byte-deterministic under `requirements/ci.txt`; `python
  scripts/generate_portfolio_plots.py` reproduces the six PNG hashes above.

Plot images are presentation artifacts; machine-readable result tables under `results/`
remain the source for analysis.
