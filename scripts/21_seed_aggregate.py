"""Step 21: aggregate seed-replicated runs (B-seeds) into mean +/- sd per condition.

Seed replicas share the run stem with an `_s<seed>` tag (e.g. bridge_clip_150k_s43_ep3);
the seedless locked run is the seed-42 member of its group. For every condition with
more than one seed present, reports mean and sample sd of model VQA-soft and of the
effect, so training variance sits next to the question-sampling CIs of scripts/12.
Runs on whatever exists — no seeds yet means it just says so and exits cleanly.

    python scripts/21_seed_aggregate.py     # writes outputs/eval/results_seeds.csv
"""
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import load_config  # noqa: E402


# The locked 150K runs predate per-epoch stem naming ("150k_3ep" vs the replicas'
# "150k_s<seed>_ep3"), so they are aliased into the seed-stripped condition their
# replicas produce; without this the seed-42 member would silently drop out.
_ALIASES = {
    "bridge_clip_150k_3ep": ("bridge_clip_150k_ep3", 42),
    "bridge_ijepa_150k_3ep": ("bridge_ijepa_150k_ep3", 42),
}


def seed_key(run: str):
    """Split a run stem into (condition-without-seed, seed); seedless runs are seed 42."""
    if run in _ALIASES:
        return _ALIASES[run]
    m = re.search(r"_s(\d+)(_|$)", run)
    if m:
        return run[:m.start()] + run[m.end() - len(m.group(2)):], int(m.group(1))
    return run, 42


def main():
    """Group results_summary rows by seed-stripped condition; report mean +/- sd."""
    eval_dir = Path(load_config()["paths"]["outputs"]) / "eval"
    rows = list(csv.DictReader(open(eval_dir / "results_summary.csv")))
    groups = defaultdict(dict)
    for r in rows:
        cond, seed = seed_key(r["run"])
        groups[cond][seed] = r

    out = []
    for cond, by_seed in sorted(groups.items()):
        if len(by_seed) < 2:
            continue
        vqa = [float(r["model_vqa_soft_%"]) for r in by_seed.values()]
        eff = [float(r["effect_vqa_soft_%"]) for r in by_seed.values()]
        any_r = next(iter(by_seed.values()))
        out.append({
            "condition": cond, "encoder": any_r["encoder"],
            "augmentation": any_r["augmentation"], "seeds": len(by_seed),
            "seed_list": " ".join(str(s) for s in sorted(by_seed)),
            "model_vqa_soft_mean": round(mean(vqa), 2),
            "model_vqa_soft_sd": round(stdev(vqa), 2),
            "effect_vqa_soft_mean": round(mean(eff), 2),
            "effect_vqa_soft_sd": round(stdev(eff), 2),
        })

    if not out:
        print("no condition has >1 seed yet — nothing to aggregate (run B-seeds first)")
        return
    path = eval_dir / "results_seeds.csv"
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)
    print(f"wrote {path} ({len(out)} seed-replicated conditions)")


if __name__ == "__main__":
    main()
