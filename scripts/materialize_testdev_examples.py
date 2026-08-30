#!/usr/bin/env python3
"""Join public predictions with locally downloaded GQA text and image paths."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def materialize(
    predictions_path: Path,
    questions: dict[str, dict],
    image_dir: Path,
    output_path: Path,
    *,
    require_images: bool,
) -> None:
    """Write a local-only joined CSV, failing on QID/image identity mismatches."""
    with predictions_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        prediction_fields = list(reader.fieldnames or [])
        rows = list(reader)
    if not rows or "qid" not in prediction_fields or "image_id" not in prediction_fields:
        raise SystemExit("FAIL predictions CSV lacks qid/image_id rows")
    qids = [row["qid"] for row in rows]
    if len(qids) != len(set(qids)):
        raise SystemExit("FAIL predictions CSV has duplicate qids")
    if set(qids) != set(questions):
        raise SystemExit(
            f"FAIL qid set mismatch: predictions={len(qids)}, questions={len(questions)}"
        )

    joined: list[dict[str, str]] = []
    missing_images: list[str] = []
    for row in rows:
        question = questions[row["qid"]]
        image_id = str(question.get("imageId", ""))
        if image_id != row["image_id"]:
            raise SystemExit(
                f"FAIL image id mismatch for qid {row['qid']}: "
                f"predictions={row['image_id']!r}, questions={image_id!r}"
            )
        image_path = image_dir / f"{image_id}.jpg"
        if not image_path.is_file():
            missing_images.append(image_id)
        joined.append(
            {
                **row,
                "question": str(question.get("question", "")),
                "gold": str(question.get("answer", "")),
                "gold_full_answer": str(question.get("fullAnswer", "")),
                "image_path": str(image_path.resolve()) if image_path.is_file() else "",
            }
        )
    if require_images and missing_images:
        raise SystemExit(
            f"FAIL missing images: {len(missing_images)} QIDs; first={missing_images[:5]}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [*prediction_fields, "question", "gold", "gold_full_answer", "image_path"]
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(joined)
    print(f"wrote {output_path} ({len(joined):,} rows; missing images={len(missing_images):,})")


def main() -> None:
    """Materialize the ignored local examples CSV."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, default=ROOT / "results/testdev/predictions.csv")
    parser.add_argument("--questions", type=Path,
                        default=ROOT / "data/gqa/questions/testdev_balanced_questions.json")
    parser.add_argument("--image-dir", type=Path, default=ROOT / "data/gqa/images")
    parser.add_argument("--out", type=Path,
                        default=ROOT / "results/testdev/examples/examples_with_questions.csv")
    parser.add_argument("--require-images", action="store_true")
    args = parser.parse_args()
    for path in (args.predictions, args.questions):
        if not path.is_file():
            raise SystemExit(f"FAIL required input does not exist: {path}")
    questions = json.loads(args.questions.read_text())
    materialize(args.predictions, questions, args.image_dir, args.out,
                require_images=args.require_images)


if __name__ == "__main__":
    main()
