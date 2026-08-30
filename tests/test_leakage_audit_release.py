"""Unit coverage for the public leakage-audit file contract."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/13_vg_leakage_audit.py"


def load_module():
    """Import the numerically prefixed audit script."""
    spec = importlib.util.spec_from_file_location("vg_leakage_audit", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_missing_exclusion_is_created(tmp_path: Path) -> None:
    """A new checkout can create the derived exclusion once."""
    module = load_module()
    path = tmp_path / "exclude.json"
    status = module.verify_or_write_exclusion(path, [3, 7, 9])
    assert status == "created"
    assert json.loads(path.read_text()) == [3, 7, 9]


def test_matching_exclusion_is_not_rewritten(tmp_path: Path) -> None:
    """An existing frozen exclusion retains its exact bytes."""
    module = load_module()
    path = tmp_path / "exclude.json"
    original = "[3, 7, 9]"
    path.write_text(original)
    status = module.verify_or_write_exclusion(path, [3, 7, 9])
    assert status == "verified"
    assert path.read_text() == original


def test_mismatched_exclusion_fails_closed(tmp_path: Path) -> None:
    """A mismatch is reported and never reconciled automatically."""
    module = load_module()
    path = tmp_path / "exclude.json"
    path.write_text("[3, 7]")
    with pytest.raises(SystemExit, match="existing exclusion does not match"):
        module.verify_or_write_exclusion(path, [3, 7, 9])
    assert path.read_text() == "[3, 7]"
