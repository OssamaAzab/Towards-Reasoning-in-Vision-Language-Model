"""Fail-closed contract for the clean reproducibility export."""

from __future__ import annotations

import csv
import hashlib
import os
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

REQUIRED_FILES = {
    ".gitignore",
    ".github/workflows/ci.yml",
    "README.md",
    "REPRODUCIBILITY.md",
    "DATA.md",
    "MODELS.md",
    "PREPROCESSING.md",
    "CITATION.cff",
    "artifacts/bridge_architecture_dinov2.txt",
    "artifacts/model_inventory.json",
    "docs/figures/README.md",
    "docs/figures/architecture.svg",
    "docs/figures/encoder_connector_interaction.png",
    "docs/figures/surface_composition.png",
    "docs/figures/confirmatory_results.png",
    "docs/figures/evaluation_surfaces.svg",
    "docs/figures/structural_augmentation.png",
    "docs/figures/data_pipeline.svg",
    "docs/figures/testdev_category_accuracy.png",
    "docs/figures/testdev_category_distribution.png",
    "docs/figures/data/encoder_connector_interaction.json",
    "docs/figures/data/structural_augmentation.json",
    "docs/figures/data/confirmatory_results.json",
    "docs/figures/data/surface_composition.json",
    "results/testdev/README.md",
    "results/testdev/category_summary.csv",
    "results/testdev/predictions.csv",
    "results/testdev/predictions.json",
    "results/testdev/provenance.json",
    "env.sh",
    "config/default.yaml",
    "config/gqa_synonyms_v1.json",
    "requirements/legacy-cu124.txt",
    "requirements/corrected-cu130.txt",
    "requirements/ci.txt",
    "cluster/README.md",
    "cluster/train.sbatch",
    "cluster/evaluate.sbatch",
    "release/ALLOWLIST.txt",
    "release/EXCLUSIONS.md",
    "release/RELEASE_MANIFEST.sha256",
    "release/SOURCE_PROVENANCE.json",
    "results/README.md",
    "results/CATALOGUE_SUMMARY.json",
    "scripts/build_testdev_results.py",
}

PROTECTED_HASHES = {
    "config/gqa_synonyms_v1.json": "87a648353e44334937a0593ac13567650f877078d17f446bc4c0a5ce59c86e71",
    "data/vqa/gqa_val_coco_exclude.json": "4016f67eda0f3f8258a24fe4b22f3cbd8a2868fcd73ad0559824fac4d8f6e3e3",
    "data/gqa/confirm_3000_qids.json": "d24eaa49f1e27cbc68a8767330340017f74783c1d4fcaf83ef5ab361a6331401",
    "data/gqa/eval_2000_qids.json": "57148adb377383705d497d0db4a3aff03b7aebefb5cef02608765e5479be9bd9",
    "data/gqa/graph_3000_qids.json": "0c4e54cd51b6b86556d9e9dc4993199906b126d1183df60bcd216a55aff20536",
    "data/gqa/graph_3000_qids_MANIFEST.json": "b3dd687be26f905c6baf2a9a0d8b96795a10fb0bea45191a032d1a34e681d50c",
    "data/gqa/vg_exclude_gqa_eval.json": "3c073d451d799b72e575bbd441854894300a517ca8e0ed53cb896027ea6ffcc3",
    "data/gqa/objective_4000_qids.json": "dbc0e49061092c1d738df942d8fcb25440568dfe5b2672104a7c9b6035081d46",
    "data/gqa/objective_4000_qids_MANIFEST.json": "972c0dd005db7fc9d7dd1237162570eea19b292101b5e6682c65002cd77f26cd",
    "data/gqa/testdev_12578_qids.json": "6344c4cd1e3f3e952be156fd52381a043d8595ce111d1da3ceffaa58cbc45cb6",
    "data/gqa/testdev_12578_qids_MANIFEST.json": "8d9fb612faad740afa5df4bf4bbf215cd4cc359282b7885e946022e4b0692c99",
    "data/gqa/tune_500_qids.json": "6ddd9ca773a7ffdf09734a396dcf5436adb43cd0199e54833da6f3aa54e58a8e",
    "data/gqa/tune_500_reltr_clean_qids.json": "6b938d2f2e708f8867a2e5d54380cb02340c4e668dffb40eed8a0835ef3e6f4e",
    "data/gqa/tune_500_reltr_clean_qids_MANIFEST.json": "3659aaf227e0ebef4e96278650a8c43bb5edbc3efe8864d5e3b2971fa4300859",
    "results/results_all_index.csv": "27853793688cee43b33ae3872242d002083bb4034d1f2b08ef6c058c503b471f",
    "results/results_confidence_intervals.csv": "103cd21f116c2537bbcb96f20f2f51a46081e377446e8afe270a97eff3201897",
    "results/results_corrected_eos.csv": "c3358dc56a306fa629d433947deb63843e8d8b134aafdd107c1eeefbbb639322",
    "results/results_legacy.csv": "1a1f12f85c7038407538779f87a6eac131c2852baace69a2afb45443dafc16ea",
    "results/results_per_category.csv": "dfca2a5e10c63279a8489152c4e96a67b0b5083beffdca18033f25c93f03ffe7",
}

FORBIDDEN_TOP_LEVEL = {
    ".review",
    "AGENTS.md",
    "CODEX_CONTEXT.md",
    "CODEX_PROMPT.md",
    "FINAL_PROJECT_LOCK",
    "MANUSCRIPT_OUTLINE.md",
    "REPORT.md",
    "REPORT_CORRECTED.md",
    "coordination",
    "research",
    "submit",
}

FORBIDDEN_CONTENT = (
    "/scratch/" + "p2602",
    "/mnt/" + "fast",
    "/vol/" + "vssp",
    "oa" + "01808",
    "ai" + "surrey",
    "surrey.ac.uk",
)

SECRET_PATTERNS = (
    re.compile("gh" + r"[pousr]_[A-Za-z0-9_]{20,}"),
    re.compile("github_" + r"pat_[A-Za-z0-9_]{20,}"),
    re.compile("sk" + r"-[A-Za-z0-9_-]{20,}"),
    re.compile("AK" + r"IA[0-9A-Z]{16}"),
    re.compile("BEGIN " + r"(?:RSA |OPENSSH |EC )?PRIVATE KEY"),
)


def sha256(path: Path) -> str:
    """Return a file's SHA-256 digest."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def release_files() -> set[str]:
    """Return release files, excluding generated caches and Git metadata."""
    return {
        path.relative_to(ROOT).as_posix()
        for path in ROOT.rglob("*")
        if path.is_file()
        and ".git" not in path.parts
        and ".pytest_cache" not in path.parts
        and "__pycache__" not in path.parts
        and not path.name.endswith((".pyc", ".pyo"))
    }


def text_files() -> list[Path]:
    """Return text surfaces that may carry private paths or credentials."""
    suffixes = {"", ".cff", ".csv", ".json", ".md", ".py", ".sbatch", ".sh", ".txt", ".yaml"}
    return [path for path in ROOT.rglob("*") if path.is_file() and path.suffix in suffixes]


def test_required_public_surface_exists() -> None:
    """The export exposes one concise, runnable documentation surface."""
    missing = sorted(path for path in REQUIRED_FILES if not (ROOT / path).is_file())
    assert missing == []
    if (ROOT / ".git").exists():
        # Fresh history: no commit on any branch may ever have tracked working-only material,
        # and the source experiment commit must not be an object in this repository.
        import json

        touched = subprocess.run(
            ["git", "log", "--all", "--format=%H", "--name-only", "--", *sorted(FORBIDDEN_TOP_LEVEL)],
            cwd=ROOT, capture_output=True, text=True, check=False,
        )
        assert touched.returncode == 0, touched.stderr
        assert touched.stdout.strip() == "", "release history must never have tracked excluded material"
        source_commit = json.loads((ROOT / "release/SOURCE_PROVENANCE.json").read_text())["source_commit"]
        present = subprocess.run(
            ["git", "cat-file", "-e", f"{source_commit}^{{commit}}"],
            cwd=ROOT, capture_output=True, text=True, check=False,
        )
        assert present.returncode != 0, "the source experiment history must not be reachable from the release"


def test_ci_runs_the_clean_release_gates_with_read_only_permissions() -> None:
    """GitHub Actions proves the same test and manifest contracts on a clean checkout."""
    import yaml

    workflow = yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text())
    assert workflow["permissions"] == {"contents": "read"}
    job = workflow["jobs"]["verify"]
    steps = job["steps"]
    checkout = next(step for step in steps if step.get("uses", "").startswith("actions/checkout@"))
    assert checkout["with"]["fetch-depth"] == 0
    setup = next(step for step in steps if step.get("uses", "").startswith("actions/setup-python@"))
    assert setup["with"]["python-version"] == "3.12"
    commands = "\n".join(str(step.get("run", "")) for step in steps)
    for required in (
        "requirements/ci.txt",
        "python -m pytest -q -p no:cacheprovider",
        "sha256sum -c release/RELEASE_MANIFEST.sha256",
        "bash -n env.sh cluster/train.sbatch cluster/evaluate.sbatch",
    ):
        assert required in commands


def test_private_and_study_surfaces_are_absent() -> None:
    """Internal working documents never enter the fresh export."""
    present = sorted(name for name in FORBIDDEN_TOP_LEVEL if (ROOT / name).exists())
    assert present == []
    assert not list(ROOT.rglob("*study*"))
    assert not list(ROOT.rglob("*supervisor*"))
    assert not list(ROOT.rglob("*.html")), "the release uses ordinary result files, not a website"


def test_release_contains_no_institutional_paths_or_secrets() -> None:
    """Portable text contains neither local identities nor common secret formats."""
    findings: list[str] = []
    for path in text_files():
        if path == Path(__file__) or path.name == "EXCLUSIONS.md":
            continue
        text = path.read_text(errors="replace")
        frozen_provenance = path.parent == ROOT / "data/gqa" and path.name.endswith("_MANIFEST.json")
        for marker in FORBIDDEN_CONTENT:
            if marker.lower() in text.lower() and not frozen_provenance:
                findings.append(f"{path.relative_to(ROOT)}: private marker {marker!r}")
        for pattern in SECRET_PATTERNS:
            if pattern.search(text):
                findings.append(f"{path.relative_to(ROOT)}: secret pattern {pattern.pattern!r}")
    assert findings == []


def test_protected_protocol_files_are_byte_exact() -> None:
    """Copied evaluation slices and manifests retain their frozen bytes."""
    actual = {relative: sha256(ROOT / relative) for relative in PROTECTED_HASHES}
    assert actual == PROTECTED_HASHES


def test_portable_config_has_no_institutional_defaults() -> None:
    """The runtime config is repository-relative and does not assume one cluster."""
    config = (ROOT / "config/default.yaml").read_text()
    assert "${PROJECT_ROOT}" in config
    assert all(marker.lower() not in config.lower() for marker in FORBIDDEN_CONTENT)


def test_cluster_launchers_are_single_gpu_and_fail_closed() -> None:
    """Portable Slurm entry points require explicit paths and one GPU each."""
    for name in ("train.sbatch", "evaluate.sbatch"):
        script = (ROOT / "cluster" / name).read_text()
        assert "set -euo pipefail" in script
        assert "#SBATCH --gpus=1" in script
        assert "#SBATCH --partition=" not in script
        assert "#SBATCH --nodelist=" not in script
        assert "#SBATCH --account=" not in script
        assert "REPO_ROOT" in script
        assert "VENV_PATH" in script
        assert all(marker.lower() not in script.lower() for marker in FORBIDDEN_CONTENT)


def test_cluster_launchers_reject_missing_environment_before_gpu_access() -> None:
    """A bare launcher invocation fails before it can inspect or claim a GPU."""
    clean_env = {"PATH": os.environ.get("PATH", "")}
    for name in ("train.sbatch", "evaluate.sbatch"):
        result = subprocess.run(
            ["bash", str(ROOT / "cluster" / name)],
            cwd=ROOT,
            env=clean_env,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode != 0
        assert "REPO_ROOT" in result.stderr
        assert "nvidia-smi" not in result.stdout


def test_documented_core_clis_expose_help_without_writing_outputs() -> None:
    """The two public entry points have a side-effect-free help path."""
    outputs_existed = (ROOT / "outputs").exists()
    for relative in ("scripts/06c_train_bridge.py", "scripts/07_evaluate.py"):
        result = subprocess.run(
            [sys.executable, str(ROOT / relative), "--help"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        assert "usage:" in result.stdout
    assert (ROOT / "outputs").exists() is outputs_existed


def test_leakage_audit_never_overwrites_an_existing_exclusion() -> None:
    """The public leakage gate verifies frozen bytes instead of rewriting them."""
    script = (ROOT / "scripts/13_vg_leakage_audit.py").read_text()
    assert "if out.exists():" in script
    assert "existing exclusion does not match" in script
    assert "verified existing" in script


def test_result_catalogues_are_real_csv() -> None:
    """Every exported result CSV parses with a real CSV reader and has a header."""
    csv_files = sorted((ROOT / "results").glob("*.csv"))
    assert csv_files
    for path in csv_files:
        with path.open(newline="") as handle:
            rows = list(csv.reader(handle))
        assert rows and rows[0] and all(len(row) == len(rows[0]) for row in rows)


def test_catalogue_summary_matches_csv_bytes_and_shape() -> None:
    """The compact public catalogue contract matches every shipped table."""
    import json

    summary = json.loads((ROOT / "results/CATALOGUE_SUMMARY.json").read_text())
    assert summary["format"] == "vlm-results-catalogue/1.0.0"
    assert set(summary["files"]) == {path.name for path in (ROOT / "results").glob("*.csv")}
    for name, expected in summary["files"].items():
        path = ROOT / "results" / name
        with path.open(newline="") as handle:
            rows = list(csv.reader(handle))
        assert expected == {
            "rows": len(rows) - 1,
            "columns": len(rows[0]),
            "sha256": sha256(path),
        }


def test_portfolio_visuals_cover_dataset_architecture_and_all_encoders() -> None:
    """A GitHub visitor sees the data, system, and three-encoder result immediately."""
    import json
    from PIL import Image

    readme = (ROOT / "README.md").read_text()
    for relative in (
        "docs/figures/architecture.svg",
        "docs/figures/encoder_connector_interaction.png",
        "docs/figures/structural_augmentation.png",
        "docs/figures/surface_composition.png",
        "docs/figures/confirmatory_results.png",
        "docs/figures/evaluation_surfaces.svg",
        "docs/figures/testdev_category_distribution.png",
        "docs/figures/testdev_category_accuracy.png",
    ):
        assert relative in readme

    architecture = (ROOT / "docs/figures/architecture.svg").read_text()
    for label in ("CLIP", "DINOv2", "I-JEPA", "Q-Former32", "MLP256", "Qwen2-7B"):
        assert label in architecture
    pipeline = (ROOT / "docs/figures/data_pipeline.svg").read_text()
    for label in ("Official data", "Leakage gates", "Frozen features", "Bridge training", "Test-Dev", "Artifacts"):
        assert label in pipeline
    assert "explorer" not in pipeline, "the HTML explorer was removed; the schematic must not advertise it"
    surfaces = (ROOT / "docs/figures/evaluation_surfaces.svg").read_text()
    for label in ("Tune-500", "Confirm-3000", "Test-Dev", "Objective-4000", "Graph-3000", "Locked-2000"):
        assert label in surfaces

    inventory = json.loads((ROOT / "artifacts/model_inventory.json").read_text())
    assert set(inventory["encoders"]) == {"clip", "dinov2", "ijepa"}
    assert inventory["encoders"]["dinov2"]["feature_shape"] == [257, 1024]
    assert inventory["encoders"]["ijepa"]["feature_shape"] == [256, 1280]
    dinov2_dump = (ROOT / "artifacts/bridge_architecture_dinov2.txt").read_text()
    assert "encoder=dinov2 (encoder_dim=1024)" in dinov2_dump
    assert "TOTAL              60,280,064" in dinov2_dump

    for name in (
        "encoder_connector_interaction.png",
        "structural_augmentation.png",
        "surface_composition.png",
        "confirmatory_results.png",
        "testdev_category_distribution.png",
        "testdev_category_accuracy.png",
    ):
        with Image.open(ROOT / "docs/figures" / name) as image:
            assert image.format == "PNG"
            assert image.width >= 800
            assert image.height >= 400


def test_testdev_results_packet_is_complete_and_matches_results() -> None:
    """The public results folder contains the endpoint and reproduces all six scores."""
    import json

    raw = (ROOT / "results/testdev/predictions.json").read_text()
    assert "/scratch/" not in raw and "/mnt/" not in raw and "/vol/" not in raw
    payload = json.loads(raw)
    records = payload["records"]
    assert all("question" not in record and "gold" not in record and "gold_full_answer" not in record
               for record in records)
    qids = [str(value) for value in json.loads((ROOT / "data/gqa/testdev_12578_qids.json").read_text())]
    assert [record["qid"] for record in records] == qids
    assert payload["_meta"]["n_questions"] == len(records) == 12578
    assert payload["_meta"]["n_images"] == len({record["image_id"] for record in records}) == 398
    assert set(payload["_meta"]["cells"]) == {
        "clip_mlp256", "clip_qformer32", "dinov2_mlp256",
        "dinov2_qformer32", "ijepa_mlp256", "ijepa_qformer32",
    }

    manifest = json.loads((ROOT / "data/gqa/testdev_12578_qids_MANIFEST.json").read_text())
    expected_categories = {**manifest["category_mix"], "count": 0}
    assert payload["_meta"]["category_counts"] == {
        category: expected_categories.get(category, 0)
        for category in ("relate", "compare", "count", "exist", "choose", "other")
    }

    provenance = json.loads((ROOT / "results/testdev/provenance.json").read_text())
    result_rows = list(csv.DictReader((ROOT / "results/results_corrected_eos.csv").open(newline="")))
    for slug, cell in provenance["cells"].items():
        encoder, connector = slug.split("_", 1)
        matches = [
            row for row in result_rows
            if row["split"] == "testdev_12578_qids"
            and row["encoder"] == encoder
            and row["connector"] == connector
            and row["metric"] == "exact_full"
            and row["input_mode"] == "image_only"
            and row["reason_mode"] == "direct"
        ]
        assert len(matches) == 1
        assert float(matches[0]["model_score"]) == cell["accuracy"]
    with (ROOT / "results/testdev/predictions.csv").open(newline="") as handle:
        prediction_rows = list(csv.DictReader(handle))
    assert len(prediction_rows) == 12578
    assert [row["qid"] for row in prediction_rows] == qids
    assert not list((ROOT / "results/testdev").glob("*.jpg"))


def test_allowlist_matches_export_exactly() -> None:
    """Nothing can enter the export through an unreviewed copy or glob."""
    allowlist = {
        line.strip()
        for line in (ROOT / "release/ALLOWLIST.txt").read_text().splitlines()
        if line.strip() and not line.startswith("#")
    }
    assert allowlist == release_files()


def test_release_manifest_verifies_every_other_file() -> None:
    """The SHA manifest is complete and self-excluding."""
    manifest_path = ROOT / "release/RELEASE_MANIFEST.sha256"
    entries: dict[str, str] = {}
    for line in manifest_path.read_text().splitlines():
        if not line.strip():
            continue
        digest, relative = line.split("  ", 1)
        entries[relative] = digest

    expected_files = release_files() - {"release/RELEASE_MANIFEST.sha256"}
    assert set(entries) == expected_files
    assert {relative: sha256(ROOT / relative) for relative in entries} == entries
