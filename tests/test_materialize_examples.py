"""Tests for the local-only Test-Dev question/gold/image join."""

from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/materialize_testdev_examples.py"


def load_module():
    """Import the materializer."""
    spec = importlib.util.spec_from_file_location("materialize_testdev_examples", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def write_predictions(path: Path) -> None:
    """Write two minimal public prediction rows."""
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["qid", "image_id", "category", "clip_mlp256_answer"])
        writer.writeheader()
        writer.writerows([
            {"qid": "1", "image_id": "img1", "category": "relate", "clip_mlp256_answer": "table"},
            {"qid": "2", "image_id": "img2", "category": "exist", "clip_mlp256_answer": "yes"},
        ])


def test_materializer_joins_official_text_and_local_image_paths(tmp_path: Path) -> None:
    """The local artifact adds questions/golds without changing public predictions."""
    module = load_module()
    predictions = tmp_path / "predictions.csv"
    write_predictions(predictions)
    questions = {
        "1": {"question": "What is next to the chair?", "answer": "table", "fullAnswer": "A table.", "imageId": "img1"},
        "2": {"question": "Is there a dog?", "answer": "yes", "fullAnswer": "Yes.", "imageId": "img2"},
    }
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    (image_dir / "img1.jpg").write_bytes(b"image-one")
    output = tmp_path / "examples.csv"
    module.materialize(predictions, questions, image_dir, output, require_images=False)
    with output.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    assert rows[0]["question"] == "What is next to the chair?"
    assert rows[0]["gold"] == "table"
    assert rows[0]["image_path"].endswith("img1.jpg")
    assert rows[1]["image_path"] == ""


def test_materializer_fails_on_qid_or_image_mismatch(tmp_path: Path) -> None:
    """Cross-surface questions and required missing images fail closed."""
    module = load_module()
    predictions = tmp_path / "predictions.csv"
    write_predictions(predictions)
    questions = {"1": {"question": "q", "answer": "a", "fullAnswer": "a", "imageId": "img1"}}
    with pytest.raises(SystemExit, match="qid set mismatch"):
        module.materialize(predictions, questions, tmp_path, tmp_path / "out.csv", require_images=False)

    questions["2"] = {"question": "q2", "answer": "yes", "fullAnswer": "yes", "imageId": "img2"}
    with pytest.raises(SystemExit, match="missing images"):
        module.materialize(predictions, questions, tmp_path, tmp_path / "out.csv", require_images=True)
