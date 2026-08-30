"""Protocol-aware result artifacts: structured metadata and a no-overwrite guard.

Two defects this closes, both verified in the repository:

1. `scripts/07_evaluate.py` derives its output stem purely from the checkpoint
   filename and calls `write_text` with no existence check, while `/outputs/` is
   gitignored. Evaluating a legacy checkpoint under a corrected protocol therefore
   silently destroys un-versioned result records, with no recovery.
2. `scripts/10_collect_results.py` infers a run's condition from filename suffixes
   (`stem.endswith("cot")`), which collides as soon as conditions multiply. Every
   artifact now carries its condition explicitly, so collection reads metadata
   instead of guessing from a name.
"""
from __future__ import annotations

import json
from pathlib import Path

ARTIFACT_VERSION = "artifact/1.0.0"


def artifact_stem(checkpoint_stem: str, *, evidence_layer: str, split: str,
                  input_mode: str, reason_mode: str) -> str:
    """Deterministic, protocol-aware stem — distinct conditions can never collide."""
    return (f"{checkpoint_stem}__{evidence_layer}__{split}"
            f"__{input_mode}__{reason_mode}")


def build_meta(*, spec, checkpoint: dict, split: str, metric_version: str,
               extra: dict | None = None) -> dict:
    """Assemble the provenance block embedded in every result artifact."""
    meta = {
        "artifact_version": ARTIFACT_VERSION,
        "evidence_layer": spec.evidence_layer,
        "prompt_format": spec.prompt_format,
        "supervise_eos": spec.supervise_eos,
        "input_mode": spec.input_mode,
        "reason_mode": spec.reason_mode,
        "split": split,
        "metric_version": metric_version,
        "prompt_module_version": __import__("src.prompt", fromlist=["x"]).MODULE_VERSION,
        "checkpoint_encoder": checkpoint.get("encoder"),
        "checkpoint_tag": checkpoint.get("tag"),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_prompt_format": checkpoint.get("prompt_format", "raw"),
        "checkpoint_supervise_eos": bool(checkpoint.get("supervise_eos", False)),
        "llm_precision": checkpoint.get("llm_precision"),
        "seed": checkpoint.get("seed"),
    }
    meta.update(extra or {})
    return meta


def write_json(path: str | Path, payload, *, meta: dict, overwrite: bool = False) -> Path:
    """Write a JSON artifact with its metadata, refusing to clobber by default."""
    path = Path(path)
    if path.exists() and not overwrite:
        raise SystemExit(
            f"refusing to overwrite {path}. Corrected evaluations must never reuse a "
            "legacy artifact name — outputs/ is gitignored, so an overwrite is "
            "unrecoverable. Use a protocol-aware stem, or pass --overwrite deliberately."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    body = {"_meta": meta, "records": payload} if isinstance(payload, list) else dict(payload)
    if not isinstance(payload, list):
        body["_meta"] = meta
    path.write_text(json.dumps(body, indent=2))
    return path


def write_text(path: str | Path, text: str, *, meta: dict, overwrite: bool = False) -> Path:
    """Write a markdown artifact with a metadata header, refusing to clobber."""
    path = Path(path)
    if path.exists() and not overwrite:
        raise SystemExit(f"refusing to overwrite {path}; pass --overwrite deliberately")
    path.parent.mkdir(parents=True, exist_ok=True)
    banner = ("Corrected ChatML/EOS evaluation; primary protocol."
              if meta.get("evidence_layer", "").startswith("corrected")
              else "Legacy raw-prompt/no-EOS evaluation; exploratory only.")
    header = f"<!-- {json.dumps(meta)} -->\n\n> **{banner}**\n\n"
    path.write_text(header + text)
    return path
