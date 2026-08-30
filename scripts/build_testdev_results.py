#!/usr/bin/env python3
"""Build the public all-example GQA Test-Dev results packet from six evaluator records."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.gqa import category_of  # noqa: E402
from src.eval.metrics import exact_full  # noqa: E402


CATEGORY_ORDER = ("relate", "compare", "count", "exist", "choose", "other")
CELL_LABELS = (
    "clip_mlp256",
    "clip_qformer32",
    "dinov2_mlp256",
    "dinov2_qformer32",
    "ijepa_mlp256",
    "ijepa_qformer32",
)
DISPLAY_NAMES = {
    "clip_mlp256": "CLIP + MLP256",
    "clip_qformer32": "CLIP + Q-Former32",
    "dinov2_mlp256": "DINOv2 + MLP256",
    "dinov2_qformer32": "DINOv2 + Q-Former32",
    "ijepa_mlp256": "I-JEPA + MLP256",
    "ijepa_qformer32": "I-JEPA + Q-Former32",
}
IMAGE_ID = re.compile(r"^[A-Za-z0-9_-]+$")


def sha256(path: Path) -> str:
    """Return the SHA-256 digest for one source or derived file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def index_records(records: list[dict], slug: str, expected_qids: set[str]) -> dict[str, dict]:
    """Index one evaluator cell, rejecting duplicate, partial, or foreign QIDs."""
    indexed: dict[str, dict] = {}
    for record in records:
        qid = str(record.get("qid", ""))
        if not qid or qid in indexed:
            raise SystemExit(f"FAIL {slug} has a missing or duplicate qid: {qid!r}")
        indexed[qid] = record
    if set(indexed) != expected_qids:
        missing = sorted(expected_qids - set(indexed))[:5]
        extra = sorted(set(indexed) - expected_qids)[:5]
        raise SystemExit(
            f"FAIL qid set mismatch for {slug}: got {len(indexed)}, expected {len(expected_qids)}, "
            f"missing={missing}, extra={extra}"
        )
    return indexed


def assemble_records(
    questions: dict[str, dict],
    qids: list[str],
    cell_records: dict[str, list[dict]],
) -> list[dict]:
    """Join questions and six evaluator cells into safe public result records."""
    if len(qids) != len(set(qids)):
        raise SystemExit("FAIL endpoint qid list contains duplicates")
    if set(cell_records) != set(CELL_LABELS):
        raise SystemExit(
            f"FAIL cell set mismatch: got {sorted(cell_records)}, expected {sorted(CELL_LABELS)}"
        )
    missing_questions = [qid for qid in qids if qid not in questions]
    if missing_questions:
        raise SystemExit(f"FAIL {len(missing_questions)} endpoint qids lack questions")

    expected_qids = set(qids)
    indexed = {
        slug: index_records(cell_records[slug], slug, expected_qids) for slug in CELL_LABELS
    }
    output: list[dict] = []
    for qid in qids:
        question = questions[qid]
        gold = str(question.get("answer", ""))
        image_id = str(question.get("imageId", ""))
        if not IMAGE_ID.fullmatch(image_id):
            raise SystemExit(f"FAIL unsafe image id for qid {qid}: {image_id!r}")
        types = question.get("types", {})
        category = category_of(
            str(question.get("question", "")),
            str(types.get("structural", "")),
            str(types.get("semantic", "")),
            str(types.get("detailed", "")),
        )

        predictions: dict[str, dict] = {}
        for slug in CELL_LABELS:
            record = indexed[slug][qid]
            if str(record.get("gold", "")) != gold:
                raise SystemExit(
                    f"FAIL gold mismatch for qid {qid}/{slug}: "
                    f"question={gold!r}, record={record.get('gold')!r}"
                )
            answer = str(record.get("bridge", ""))
            correct = bool(exact_full(answer, gold))
            if "bridge_exact_full" in record and bool(record["bridge_exact_full"]) != correct:
                raise SystemExit(f"FAIL exact_full mismatch for qid {qid}/{slug}")
            predictions[slug] = {
                "label": DISPLAY_NAMES[slug],
                "answer": answer,
                "correct": correct,
            }

        output.append(
            {
                "qid": qid,
                "image_id": image_id,
                "image_relpath": f"../../data/gqa/images/{image_id}.jpg",
                "question": str(question.get("question", "")),
                "gold": gold,
                "gold_full_answer": str(question.get("fullAnswer", "")),
                "category": category,
                "types": {
                    "structural": str(types.get("structural", "")),
                    "semantic": str(types.get("semantic", "")),
                    "detailed": str(types.get("detailed", "")),
                },
                "predictions": predictions,
            }
        )
    return output


def summarize_categories(records: list[dict]) -> list[dict]:
    """Return category counts and exact-full accuracy for each of the six cells."""
    rows: list[dict] = []
    for category in CATEGORY_ORDER:
        subset = [record for record in records if record["category"] == category]
        row: dict[str, object] = {"category": category, "n": len(subset)}
        for slug in CELL_LABELS:
            correct = sum(bool(record["predictions"][slug]["correct"]) for record in subset)
            row[f"{slug}_accuracy"] = round(100.0 * correct / len(subset), 6) if subset else None
        rows.append(row)
    if sum(int(row["n"]) for row in rows) != len(records):
        raise SystemExit("FAIL category summary does not cover every result record")
    return rows


def public_records(records: list[dict]) -> list[dict]:
    """Remove upstream question/gold text while retaining our derived predictions."""
    allowed = ("qid", "image_id", "image_relpath", "category", "types", "predictions")
    return [{key: record[key] for key in allowed} for record in records]


def write_summary_csv(rows: list[dict], path: Path) -> None:
    """Write category counts and accuracies with a stable column order."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["category", "n", *(f"{slug}_accuracy" for slug in CELL_LABELS)]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_predictions_csv(records: list[dict], path: Path) -> None:
    """Write one flat, GitHub-friendly row per QID with all six predictions."""
    fields = [
        "qid", "image_id", "image_relpath", "category", "structural", "semantic", "detailed",
        *(field for slug in CELL_LABELS for field in (f"{slug}_answer", f"{slug}_correct")),
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for record in records:
            row: dict[str, object] = {
                "qid": record["qid"],
                "image_id": record["image_id"],
                "image_relpath": record["image_relpath"],
                "category": record["category"],
                **record["types"],
            }
            for slug in CELL_LABELS:
                row[f"{slug}_answer"] = record["predictions"][slug]["answer"]
                row[f"{slug}_correct"] = str(record["predictions"][slug]["correct"]).lower()
            writer.writerow(row)


def write_plots(rows: list[dict], figure_dir: Path) -> None:
    """Write Test-Dev category-distribution and six-cell accuracy plots."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    figure_dir.mkdir(parents=True, exist_ok=True)
    categories = [str(row["category"]) for row in rows]
    counts = [int(row["n"]) for row in rows]

    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    bars = ax.bar(categories, counts, color="#2f6fad")
    ax.bar_label(bars, padding=3, fontsize=9)
    ax.set_title("GQA balanced Test-Dev reasoning categories (n=12,578)")
    ax.set_ylabel("questions")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(figure_dir / "testdev_category_distribution.png", dpi=160)
    plt.close(fig)

    matrix = np.array(
        [[next(item for item in rows if item["category"] == category)[f"{slug}_accuracy"]
          for category in categories] for slug in CELL_LABELS],
        dtype=float,
    )
    fig, ax = plt.subplots(figsize=(10.2, 5.2))
    image = ax.imshow(matrix, cmap="Blues", vmin=np.nanmin(matrix), vmax=np.nanmax(matrix), aspect="auto")
    ax.set_xticks(range(len(categories)), categories)
    ax.set_yticks(range(len(CELL_LABELS)), [DISPLAY_NAMES[slug] for slug in CELL_LABELS])
    ax.set_title("Full-string exact match by Test-Dev category")
    for y in range(matrix.shape[0]):
        for x in range(matrix.shape[1]):
            value = matrix[y, x]
            ax.text(x, y, "—" if np.isnan(value) else f"{value:.1f}", ha="center", va="center",
                    color="white" if value > np.nanmean(matrix) else "#111827", fontsize=9)
    fig.colorbar(image, ax=ax, label="accuracy (%)", shrink=0.85)
    fig.tight_layout()
    fig.savefig(figure_dir / "testdev_category_accuracy.png", dpi=160)
    plt.close(fig)


def parse_cell_args(values: list[str]) -> dict[str, Path]:
    """Parse repeated `slug=path` arguments and require the exact six-cell set."""
    parsed: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise SystemExit(f"FAIL --cell must be slug=path, got {value!r}")
        slug, raw_path = value.split("=", 1)
        if slug in parsed:
            raise SystemExit(f"FAIL duplicate --cell slug: {slug}")
        parsed[slug] = Path(raw_path)
    if set(parsed) != set(CELL_LABELS):
        raise SystemExit(f"FAIL --cell set must be exactly: {', '.join(CELL_LABELS)}")
    return parsed


def validate_cell_metadata(slug: str, blob: dict, n_questions: int) -> None:
    """Require the final corrected Test-Dev endpoint metadata for every raw cell."""
    meta = blob.get("_meta") if isinstance(blob, dict) else None
    if not isinstance(meta, dict):
        raise SystemExit(f"FAIL {slug} lacks an _meta object")
    required = {
        "evidence_layer": "corrected_chatml_v1_eos",
        "prompt_format": "chatml_v1",
        "supervise_eos": True,
        "split": "testdev_12578_qids",
        "metric_version": "exact_full/1.0.0",
        "checkpoint_epoch": 5,
        "seed": 42,
        "n_questions": n_questions,
        "eval_llm_precision": "bf16",
    }
    mismatches = {key: (meta.get(key), value) for key, value in required.items() if meta.get(key) != value}
    if mismatches:
        raise SystemExit(f"FAIL metadata mismatch for {slug}: {mismatches}")
    encoder = slug.split("_", 1)[0]
    if encoder not in str(meta.get("checkpoint_encoder", "")).lower():
        raise SystemExit(f"FAIL checkpoint encoder mismatch for {slug}")
    tag = str(meta.get("checkpoint_tag", "")).lower()
    if ("qformer32" in slug) != ("qf" in tag):
        raise SystemExit(f"FAIL checkpoint connector mismatch for {slug}: tag={tag!r}")


def main() -> None:
    """Build prediction files, summary CSV, provenance, and public plots."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--qids", type=Path, required=True)
    parser.add_argument("--cell", action="append", default=[], help="slug=raw_records.json")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results/testdev")
    parser.add_argument("--figure-dir", type=Path, default=ROOT / "docs/figures")
    args = parser.parse_args()

    cell_paths = parse_cell_args(args.cell)
    inputs = [args.questions, args.qids, *cell_paths.values()]
    missing = [str(path) for path in inputs if not path.is_file()]
    if missing:
        raise SystemExit(f"FAIL missing input files: {missing}")

    questions = json.loads(args.questions.read_text())
    qids = [str(qid) for qid in json.loads(args.qids.read_text())]
    blobs = {slug: json.loads(path.read_text()) for slug, path in cell_paths.items()}
    for slug, blob in blobs.items():
        validate_cell_metadata(slug, blob, len(qids))
    cell_records = {
        slug: blob["records"] if isinstance(blob, dict) and "records" in blob else blob
        for slug, blob in blobs.items()
    }
    records = assemble_records(questions, qids, cell_records)
    summary = summarize_categories(records)
    overall_accuracies = {
        slug: round(
            100.0 * sum(record["predictions"][slug]["correct"] for record in records) / len(records),
            6,
        )
        for slug in CELL_LABELS
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "format": "gqa-testdev-results/1.0.0",
        "n_questions": len(records),
        "n_images": len({record["image_id"] for record in records}),
        "cells": {slug: DISPLAY_NAMES[slug] for slug in CELL_LABELS},
        "category_counts": {row["category"]: row["n"] for row in summary},
        "image_policy": "images are not redistributed; image_relpath resolves after official GQA download",
        "question_policy": "questions and golds are not redistributed; join the official file locally by qid",
    }
    predictions_json_path = args.out_dir / "predictions.json"
    predictions_json_path.write_text(
        json.dumps({"_meta": metadata, "records": public_records(records)}, separators=(",", ":"))
    )
    predictions_csv_path = args.out_dir / "predictions.csv"
    write_predictions_csv(public_records(records), predictions_csv_path)
    summary_path = args.out_dir / "category_summary.csv"
    write_summary_csv(summary, summary_path)
    write_plots(summary, args.figure_dir)
    distribution_path = args.figure_dir / "testdev_category_distribution.png"
    accuracy_path = args.figure_dir / "testdev_category_accuracy.png"

    provenance = {
        "format": "gqa-testdev-results-provenance/1.0.0",
        "questions": {"file": args.questions.name, "sha256": sha256(args.questions)},
        "qids": {"file": args.qids.name, "sha256": sha256(args.qids)},
        "cells": {
            slug: {
                "label": DISPLAY_NAMES[slug],
                "file": cell_paths[slug].name,
                "sha256": sha256(cell_paths[slug]),
                "accuracy": overall_accuracies[slug],
            }
            for slug in CELL_LABELS
        },
        "outputs": {
            "predictions.json": {
                "sha256": sha256(predictions_json_path), "bytes": predictions_json_path.stat().st_size
            },
            "predictions.csv": {
                "sha256": sha256(predictions_csv_path), "bytes": predictions_csv_path.stat().st_size
            },
            "category_summary.csv": {"sha256": sha256(summary_path), "bytes": summary_path.stat().st_size},
            "testdev_category_distribution.png": {
                "sha256": sha256(distribution_path), "bytes": distribution_path.stat().st_size
            },
            "testdev_category_accuracy.png": {
                "sha256": sha256(accuracy_path), "bytes": accuracy_path.stat().st_size
            },
        },
    }
    (args.out_dir / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"wrote {predictions_json_path} and {predictions_csv_path} "
          f"({len(records):,} QIDs, {metadata['n_images']} images)")
    print(f"wrote {summary_path} and Test-Dev category plots")


if __name__ == "__main__":
    main()
