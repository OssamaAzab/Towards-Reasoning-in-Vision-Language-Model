# Machine-readable results

These files are curated catalogue snapshots. They are evidence and inspection
inputs, not model-training inputs.

| File | Purpose |
|---|---|
| `results_all_index.csv` | Measured result cells across execution layers and surfaces |
| `results_corrected_eos.csv` | Corrected ChatML + EOS result cells |
| `results_legacy.csv` | Historical raw-prompt result cells |
| `results_confidence_intervals.csv` | Stored contrasts and confidence intervals |
| `results_per_category.csv` | Category-level result cells |
| `CATALOGUE_SUMMARY.json` | Row-count, column-count, and SHA-256 contract |

Do not join corrected and legacy rows into a pooled estimate. For corrected runs,
`exact_full` is primary. Parse every CSV with a real CSV parser because labels can
contain quoted commas.

## Bootstrap units

`results_confidence_intervals.csv` records the resampling unit of every interval in
`bootstrap_unit`. GQA Test-Dev questions nest within 398 images, so the reported
Test-Dev intervals (README key-results table, `docs/figures/data/*.json`) are
**image-clustered**. The `question`-unit Test-Dev rows in this catalogue keep their
point estimates and official-scorer contrasts, but their intervals are narrower than
the clustered ones and are superseded for interval reporting. Never quote a
question-unit Test-Dev interval as the headline interval.
