"""Contracts for the final CV/public-release completion pass."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]

REQUIRED = (
    "LICENSE",
    "THIRD_PARTY_NOTICES.md",
    "LIMITATIONS.md",
    "COMPUTE.md",
    "CONTRIBUTING.md",
    "RELEASING.md",
    "artifacts/model_revisions.json",
    "src/models/revisions.py",
    "scripts/generate_portfolio_plots.py",
    "scripts/materialize_testdev_examples.py",
    "docs/figures/data/training_convergence.json",
    "docs/figures/data/efficiency.json",
    "docs/figures/training_convergence.png",
    "docs/figures/efficiency.png",
    "docs/figures/data/confirmatory_results.json",
    "docs/figures/data/surface_composition.json",
    "docs/figures/confirmatory_results.png",
    "docs/figures/surface_composition.png",
)

EXPECTED_REVISIONS = {
    "Qwen/Qwen2-7B-Instruct": "f2826a00ceef68f0f2b946d945ecc0477ce4450c",
    "openai/clip-vit-large-patch14": "32bd64288804d66eefd0ccbe215aa642df71cc41",
    "facebook/dinov2-large": "47b73eefe95e8d44ec3623f8890bd894b6ea2d6c",
    "facebook/ijepa_vith14_1k": "f157467ea509bc356ff9f61fd3c0d840eec5e04e",
    "facebook/ijepa_vith14_22k": "ba3c4513ca2b0f0c010f80ae2265b3dbe1083039",
    "google/owlv2-base-patch16-ensemble": "cfd3195ba4ea9592eec887ded089f4c08eff231d",
}


def load_module(relative: str, name: str):
    """Import one release module by path."""
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_final_public_release_files_exist() -> None:
    """Every recommended non-hosting release surface is present."""
    assert [relative for relative in REQUIRED if not (ROOT / relative).is_file()] == []


def test_exact_model_revisions_are_recorded_and_enforced() -> None:
    """Known upstream models never float on a mutable main branch."""
    artifact = json.loads((ROOT / "artifacts/model_revisions.json").read_text())
    assert artifact["revisions"] == EXPECTED_REVISIONS
    module = load_module("src/models/revisions.py", "model_revisions")
    assert module.MODEL_REVISIONS == EXPECTED_REVISIONS
    for model_id, revision in EXPECTED_REVISIONS.items():
        assert module.revision_for(model_id) == revision
    assert module.revision_for("owner/new-model") is None

    for relative in ("src/models/encoders.py", "src/models/llm.py", "src/data/predicted_graph.py"):
        source = (ROOT / relative).read_text()
        assert "revision=" in source


def test_training_convergence_values_cover_three_encoders_and_five_epochs() -> None:
    """The added curve is the matched corrected tune-500 accuracy curve."""
    payload = json.loads((ROOT / "docs/figures/data/training_convergence.json").read_text())
    assert payload["metric"] == "exact_full"
    assert payload["surface"] == "tune_500_qids"
    expected = {
        "clip": [44.6, 50.0, 50.4, 52.8, 51.0],
        "dinov2": [43.4, 47.6, 47.0, 46.4, 47.2],
        "ijepa": [40.8, 43.4, 45.6, 46.0, 46.2],
    }
    assert {encoder: [row["accuracy"] for row in rows] for encoder, rows in payload["encoders"].items()} == expected
    assert all([row["epoch"] for row in rows] == [1, 2, 3, 4, 5]
               for rows in payload["encoders"].values())


def test_efficiency_values_are_bounded_engineering_measurements() -> None:
    """Latency evidence names hardware, population, stages, and the missing cell."""
    payload = json.loads((ROOT / "docs/figures/data/efficiency.json").read_text())
    assert payload["hardware"] == "NVIDIA RTX PRO 6000 Blackwell"
    assert payload["batch_size"] == 1
    assert payload["measured_examples_per_configuration"] == 600
    assert payload["flops"] == "not measured"
    rows = payload["configurations"]
    assert len(rows) == 9
    missing = [row for row in rows if row["status"] == "not constructed"]
    assert [row["configuration"] for row in missing] == ["dinov2:pool32"]
    measured = [row for row in rows if row["status"] == "measured"]
    assert len(measured) == 8
    assert all(abs(row["model_inference_ms"] - (
        row["vision_ms"] + row["connector_ms"] + row["prefill_ms"] + row["decode_ms"]
    )) <= 0.031 for row in measured), "published totals use unrounded stage medians"


def test_new_plots_are_readable_and_linked_from_readme() -> None:
    """The two recommended plots are visible without opening internal documents."""
    readme = (ROOT / "README.md").read_text()
    for name in ("training_convergence.png", "efficiency.png"):
        path = ROOT / "docs/figures" / name
        assert f"docs/figures/{name}" in readme
        with Image.open(path) as image:
            assert image.format == "PNG"
            assert image.width >= 900 and image.height >= 500


def test_readme_has_key_results_and_three_reproduction_tiers() -> None:
    """A recruiter can inspect quickly and a researcher can find the full path."""
    readme = (ROOT / "README.md").read_text()
    assert "## Key results" in readme
    assert "## Reproduction tiers" in readme
    for label in ("Two-minute inspection", "Ten-minute CPU verification", "Full GPU reproduction"):
        assert label in readme


def test_public_docs_state_compute_and_claim_boundaries() -> None:
    """The portfolio does not turn descriptive timing or missing coverage into claims."""
    compute = (ROOT / "COMPUTE.md").read_text().lower()
    limitations = (ROOT / "LIMITATIONS.md").read_text().lower()
    notices = (ROOT / "THIRD_PARTY_NOTICES.md").read_text().lower()
    assert "flops were not measured" in compute
    assert "one rtx pro 6000" in compute
    assert "hardware" in limitations and "training-seed" in limitations
    assert "gqa" in notices and "visual genome" in notices and "qwen" in notices
    assert "hugging face checkpoint hosting" not in (ROOT / "README.md").read_text().lower()


def test_figure_generator_rebuilds_all_readme_figures_from_frozen_values(tmp_path) -> None:
    """Every README PNG is reproducible from the shipped values JSON with one style."""
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/generate_portfolio_plots.py"), "--out-dir", str(tmp_path)],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    for name in (
        "training_convergence.png", "efficiency.png", "encoder_connector_interaction.png",
        "structural_augmentation.png", "confirmatory_results.png", "surface_composition.png",
    ):
        with Image.open(tmp_path / name) as image:
            assert image.format == "PNG"
            assert image.width >= 900 and image.height >= 500


def test_figure_values_name_their_catalogue_rows() -> None:
    """New figure values are transcribed from release files, never recomputed."""
    import csv

    confirmatory = json.loads((ROOT / "docs/figures/data/confirmatory_results.json").read_text())
    rows = {row["result_id"]: row for row in csv.DictReader(
        (ROOT / "results/results_corrected_eos.csv").open(newline=""))}
    assert len(confirmatory["cells"]) == 8
    for cell in confirmatory["cells"]:
        row = rows[cell["result_id"]]
        assert float(row["model_score"]) == cell["value"]
        assert row["encoder"] == cell["encoder"] and row["connector"] == cell["connector"]
        assert float(row["comparator_score"]) == confirmatory["floor"]["value"]
    assert confirmatory["unavailable"] == [{"encoder": "dinov2", "connector": "pool32", "status": "not trained"}]

    composition = json.loads((ROOT / "docs/figures/data/surface_composition.json").read_text())
    for surface in composition["surfaces"]:
        assert sum(surface["counts"].values()) == surface["n"]
        assert surface["counts"]["count"] == 0
    manifest = json.loads((ROOT / "data/gqa/testdev_12578_qids_MANIFEST.json").read_text())
    testdev = next(s for s in composition["surfaces"] if s["key"] == "testdev_12578_qids")
    assert {k: v for k, v in testdev["counts"].items() if k != "count"} == manifest["category_mix"]


def test_encoder_palette_is_the_validated_cvd_safe_trio() -> None:
    """The one encoder colour mapping used in every figure does not drift silently."""
    spec = importlib.util.spec_from_file_location("portfolio_plots", ROOT / "scripts/generate_portfolio_plots.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.COLOURS == {"clip": "#2a78d6", "dinov2": "#e07020", "ijepa": "#1a9c6b"}
    assert module.MARKERS == {"clip": "o", "dinov2": "s", "ijepa": "^"}
