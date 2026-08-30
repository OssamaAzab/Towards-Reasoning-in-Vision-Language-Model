"""Diagnostic probes on frozen representations, before and after each connector. CPU only.

    python scripts/53_visual_audit_probes.py --features outputs/visual_audit/<run>/probe_features.json

WHAT A PROBE HERE DOES AND DOES NOT SHOW. A probe that recovers a property from a frozen
representation shows the property is **decodable** at that point in the pipeline. It does NOT show
that Qwen2 uses it. Decodability is an upper bound on use: information that is not decodable
cannot be used, but information that is decodable may still be ignored. Every table this script
prints repeats that, because the distinction is the whole point of the audit.

NOTHING IS UPDATED EXCEPT THE PROBE. The encoder, all three connectors and the LLM are never
loaded here — this reads cached, pooled representations from disk. The only trained parameters
are a single linear layer per (representation, property) pair.

WHY TORCH AND NOT SCIKIT-LEARN. Neither environment has scikit-learn, and installing it would
resolve numpy/scipy underneath the frozen torch stack that every reported number in this project
depends on. A multinomial logistic regression is a linear layer and a cross-entropy loss, so it is
implemented directly rather than risking the production environment for a convenience import.

THE COMPARISON IS ONLY FAIR IF THE PIPELINE IS IDENTICAL. Pre-connector vectors are 1024-d (CLIP)
and post-connector vectors are 3584-d (Qwen2 hidden size). Raw dimensionality alone can move probe
accuracy, so every representation goes through the SAME pipeline: standardise, project to the same
number of PCA components, then a linear probe with the same regularisation and the same folds.

EVERY ACCURACY IS PRINTED NEXT TO ITS MAJORITY-CLASS BASELINE. On this slice the position target's
majority class is 40.6% and the relation target's is 55.2%; an accuracy below those is worse than
a constant guess, and a probe number without its baseline is unreadable.

*** THE POOLING MAKES THE SPATIAL AND ATTRIBUTE NULLS UNINTERPRETABLE. READ THIS FIRST. ***
scripts/51 stores each representation as a MEAN over tokens. Mean-pooling is permutation-invariant:
it discards where anything was, by construction, before the probe ever runs. A null on `position`
or `relation` therefore says nothing about the encoder or the connector — it is a property of the
pooling this pipeline chose. `colour` and `count` are damaged the same way, being per-object
properties averaged over a whole scene into one vector.

**These nulls MUST NOT be used to localise information loss.** Doing so would attribute to the
connector a loss that the pooling caused. Answering "where is the information lost" with probes
needs representations that retain the token axis (per-patch or attention-weighted), which this
run does not produce. The behavioural conditions in scripts/54 carry the evidence instead.
`object` is the one target not inherently destroyed by averaging, and on this slice it survives
class-frequency filtering with only 26 examples across 2 classes — limited exploratory evidence,
nothing more.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SEED = 42
FOLDS = 5
PCA_DIM = 64          # same for every representation, so dimensionality cannot drive the result
MIN_PER_CLASS = 8     # classes rarer than this cannot be learned or evaluated at n~261
EPOCHS, LR, WEIGHT_DECAY = 300, 0.05, 1e-2
PROPERTIES = ("object", "colour", "count", "position", "relation")


def sha256_file(p: Path) -> str:
    """Hash an input so the probe run pins what it read."""
    return hashlib.sha256(p.read_bytes()).hexdigest()


def stratified_folds(y: np.ndarray, k: int, seed: int = SEED) -> list[np.ndarray]:
    """Class-balanced fold assignment, so a rare class cannot vanish from a training split."""
    rng = np.random.default_rng(seed)
    fold = np.empty(len(y), dtype=int)
    for cls in np.unique(y):
        idx = np.where(y == cls)[0]
        rng.shuffle(idx)
        fold[idx] = np.arange(len(idx)) % k
    return [np.where(fold == i)[0] for i in range(k)]


def fit_linear_probe(xtr, ytr, xte, yte, n_classes, seed=SEED):
    """One multinomial logistic regression. Returns test accuracy."""
    g = torch.Generator().manual_seed(seed)
    lin = torch.nn.Linear(xtr.shape[1], n_classes)
    with torch.no_grad():                       # deterministic init, independent of global state
        lin.weight.normal_(0, 0.01, generator=g)
        lin.bias.zero_()
    opt = torch.optim.Adam(lin.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    xtr_t, ytr_t = torch.tensor(xtr, dtype=torch.float32), torch.tensor(ytr, dtype=torch.long)
    for _ in range(EPOCHS):
        opt.zero_grad()
        torch.nn.functional.cross_entropy(lin(xtr_t), ytr_t).backward()
        opt.step()
    with torch.no_grad():
        pred = lin(torch.tensor(xte, dtype=torch.float32)).argmax(1).numpy()
    return float((pred == yte).mean())


def evaluate(X: np.ndarray, y: np.ndarray) -> dict:
    """Cross-validated probe accuracy against the majority-class baseline."""
    classes, y_idx = np.unique(y, return_inverse=True)
    folds = stratified_folds(y_idx, FOLDS)
    accs, base = [], []
    for i in range(FOLDS):
        te = folds[i]
        tr = np.concatenate([folds[j] for j in range(FOLDS) if j != i])
        if len(te) == 0 or len(tr) == 0:
            continue
        # Standardise and project using TRAIN statistics only; fitting them on all rows would
        # leak test information into the representation and inflate every number below.
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
        ztr, zte = (X[tr] - mu) / sd, (X[te] - mu) / sd
        d = min(PCA_DIM, ztr.shape[0] - 1, ztr.shape[1])
        _, _, V = np.linalg.svd(ztr, full_matrices=False)
        P = V[:d].T
        accs.append(fit_linear_probe(ztr @ P, y_idx[tr], zte @ P, y_idx[te], len(classes)))
        # The baseline is also computed per fold from the TRAIN split, so it is the accuracy of
        # the best constant predictor a probe could have learned, not an oracle over the test set.
        maj = Counter(y_idx[tr]).most_common(1)[0][0]
        base.append(float((y_idx[te] == maj).mean()))
    return {"accuracy": float(np.mean(accs)), "accuracy_std": float(np.std(accs)),
            "majority_baseline": float(np.mean(base)),
            "above_baseline": float(np.mean(accs) - np.mean(base)),
            "n": int(len(y)), "n_classes": int(len(classes)), "folds": FOLDS}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--features", required=True)
    ap.add_argument("--coverage", default="outputs/visual_audit/coverage.json")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    feat_p, cov_p = Path(args.features), ROOT / args.coverage
    for p in (feat_p, cov_p):
        if not p.is_file():
            raise SystemExit(f"FAIL required input absent: {p}")
    store = json.loads(feat_p.read_text())
    labels = json.loads(cov_p.read_text())["probe_labels"]

    reps = [("pre-connector (CLIP patches, mean-pooled)", store["pre"])]
    for name, d in store["post"].items():
        reps.append((f"post-{name}", d))

    print("=== diagnostic probes on FROZEN representations ===")
    print("DECODABILITY ONLY. A probe recovering a property shows the information is present in")
    print("the representation. It does NOT show the frozen Qwen2 uses it. Information that is not")
    print("decodable cannot be used; information that is decodable may still be ignored.\n")
    print(f"pipeline: standardise -> PCA({PCA_DIM}) -> linear probe, {FOLDS}-fold stratified CV,")
    print(f"          identical for every representation so dimensionality cannot drive the result")
    print(f"          (pre is 1024-d, post is 3584-d); classes with <{MIN_PER_CLASS} examples dropped\n")

    results: dict = {}
    for rep_name, vecs in reps:
        imgs_all = sorted(vecs)
        print(f"--- {rep_name}  ({len(imgs_all)} images, "
              f"{len(vecs[imgs_all[0]])}-d before PCA) ---")
        print(f"  {'property':10s} {'n':>5s} {'cls':>4s} {'probe':>8s} {'baseline':>9s} {'delta':>8s}")
        results[rep_name] = {}
        for prop in PROPERTIES:
            imgs = [i for i in imgs_all if prop in labels.get(i, {})]
            y_all = np.array([labels[i][prop] for i in imgs])
            keep_cls = {c for c, n in Counter(y_all).items() if n >= MIN_PER_CLASS}
            sel = [j for j, i in enumerate(imgs) if y_all[j] in keep_cls]
            if len(sel) < FOLDS * 2 or len(keep_cls) < 2:
                print(f"  {prop:10s} {'—':>5s} {'—':>4s}   insufficient data after dropping "
                      f"classes with <{MIN_PER_CLASS} examples")
                results[rep_name][prop] = {"skipped": "insufficient data",
                                           "n_after_filter": len(sel),
                                           "n_classes_after_filter": len(keep_cls)}
                continue
            X = np.array([vecs[imgs[j]] for j in sel], dtype=np.float64)
            y = y_all[sel]
            r = evaluate(X, y)
            results[rep_name][prop] = r
            print(f"  {prop:10s} {r['n']:5d} {r['n_classes']:4d} {100 * r['accuracy']:7.1f}% "
                  f"{100 * r['majority_baseline']:8.1f}% {100 * r['above_baseline']:+7.1f}")
        print()

    payload = {
        "probe": "linear, on frozen pooled representations",
        "interpretation": "DECODABILITY, not causal use by Qwen2",
        "pipeline": {"standardise": True, "pca_dim": PCA_DIM, "folds": FOLDS, "seed": SEED,
                     "epochs": EPOCHS, "lr": LR, "weight_decay": WEIGHT_DECAY,
                     "min_per_class": MIN_PER_CLASS,
                     "fit_on_train_only": "standardisation and PCA are fit on the training "
                                          "split of each fold"},
        "features": str(feat_p), "features_sha256": sha256_file(feat_p),
        "coverage": args.coverage, "coverage_sha256": sha256_file(cov_p),
        "results": results,
        "models_updated": "none — encoder, connectors and LLM are not loaded here",
    }
    out = Path(args.out) if args.out else feat_p.with_name("probe_results.json")
    if out.exists():
        raise SystemExit(f"FAIL refusing to overwrite {out}")
    out.write_text(json.dumps(payload, indent=1) + "\n")
    print(f"wrote {out}")
    print("\nREAD THESE AS DECODABILITY. A drop from pre- to post-connector means the connector")
    print("discarded information. A high post-connector number does NOT mean Qwen2 used it —")
    print("that question is answered by the six behavioural conditions, not by a probe.")


if __name__ == "__main__":
    main()
