"""Contracts for the public six-cell Test-Dev results packet."""

from __future__ import annotations

import importlib.util
import csv
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/build_testdev_results.py"


def load_builder():
    """Import the results builder from its public entry point."""
    spec = importlib.util.spec_from_file_location("testdev_results_builder", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def question(image_id: str, text: str, gold: str, *, structural: str, semantic: str,
             detailed: str) -> dict:
    """Return one minimal GQA question fixture."""
    return {
        "imageId": image_id,
        "question": text,
        "answer": gold,
        "fullAnswer": f"The answer is {gold}.",
        "types": {"structural": structural, "semantic": semantic, "detailed": detailed},
    }


def raw_record(qid: str, gold: str, prediction: str) -> dict:
    """Return one standard evaluator record fixture."""
    return {
        "qid": qid,
        "gold": gold,
        "bridge": prediction,
        "bridge_exact_full": prediction == gold,
        "image_path": f"/private/mount/{qid}.jpg",
    }


def fixtures():
    """Return a complete two-question, six-cell fixture."""
    questions = {
        "1": question("img_1", "What is beside the chair?", "table",
                      structural="query", semantic="rel", detailed="queryRel"),
        "2": question("img_2", "Is there a dog?", "yes",
                      structural="verify", semantic="obj", detailed="existObj"),
    }
    predictions = {
        "clip_mlp256": ("table", "yes"),
        "clip_qformer32": ("desk", "yes"),
        "dinov2_mlp256": ("table", "yes"),
        "dinov2_qformer32": ("desk", "no"),
        "ijepa_mlp256": ("table", "no"),
        "ijepa_qformer32": ("chair", "yes"),
    }
    cells = {
        slug: [raw_record("1", "table", values[0]), raw_record("2", "yes", values[1])]
        for slug, values in predictions.items()
    }
    return questions, ["1", "2"], cells


def test_builder_assembles_all_six_cells_and_safe_image_paths() -> None:
    """Every public record is complete, categorized, and free of private mounts."""
    module = load_builder()
    records = module.assemble_records(*fixtures())
    assert len(records) == 2
    assert set(records[0]["predictions"]) == set(module.CELL_LABELS)
    assert records[0]["category"] == "relate"
    assert records[1]["category"] == "exist"
    assert records[0]["image_relpath"] == "../../data/gqa/images/img_1.jpg"
    assert "/private/" not in str(records)


def test_builder_rejects_missing_qids_and_gold_mismatch() -> None:
    """Partial or cross-surface inputs fail instead of silently shrinking the results."""
    module = load_builder()
    questions, qids, cells = fixtures()
    missing = {slug: list(records) for slug, records in cells.items()}
    missing["clip_mlp256"] = missing["clip_mlp256"][:1]
    with pytest.raises(SystemExit, match="qid set mismatch"):
        module.assemble_records(questions, qids, missing)

    cells["dinov2_qformer32"][0]["gold"] = "chair"
    with pytest.raises(SystemExit, match="gold mismatch"):
        module.assemble_records(questions, qids, cells)


def test_category_summary_reports_counts_and_per_cell_accuracy() -> None:
    """Category processing is reproducible from the same all-example records."""
    module = load_builder()
    records = module.assemble_records(*fixtures())
    summary = {row["category"]: row for row in module.summarize_categories(records)}
    assert summary["relate"]["n"] == 1
    assert summary["relate"]["clip_mlp256_accuracy"] == 100.0
    assert summary["relate"]["clip_qformer32_accuracy"] == 0.0
    assert summary["exist"]["dinov2_mlp256_accuracy"] == 100.0


def test_public_records_do_not_redistribute_question_or_gold_text() -> None:
    """Git ships derived predictions; official questions/golds are joined locally."""
    module = load_builder()
    records = module.assemble_records(*fixtures())
    public = module.public_records(records)
    assert len(public) == len(records)
    for record in public:
        assert "question" not in record
        assert "gold" not in record
        assert "gold_full_answer" not in record
        assert set(record) == {"qid", "image_id", "image_relpath", "category", "types", "predictions"}


def test_flat_predictions_csv_contains_every_cell(tmp_path: Path) -> None:
    """The GitHub-friendly CSV exposes one row per QID and six prediction pairs."""
    module = load_builder()
    public = module.public_records(module.assemble_records(*fixtures()))
    path = tmp_path / "predictions.csv"
    module.write_predictions_csv(public, path)
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    assert rows[0]["qid"] == "1"
    assert rows[0]["clip_mlp256_answer"] == "table"
    assert rows[0]["clip_mlp256_correct"] == "true"
    assert rows[0]["dinov2_qformer32_answer"] == "desk"
