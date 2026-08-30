"""Immutable upstream model revisions used by the released experiments."""

from __future__ import annotations


MODEL_REVISIONS = {
    "Qwen/Qwen2-7B-Instruct": "f2826a00ceef68f0f2b946d945ecc0477ce4450c",
    "openai/clip-vit-large-patch14": "32bd64288804d66eefd0ccbe215aa642df71cc41",
    "facebook/dinov2-large": "47b73eefe95e8d44ec3623f8890bd894b6ea2d6c",
    "facebook/ijepa_vith14_1k": "f157467ea509bc356ff9f61fd3c0d840eec5e04e",
    "facebook/ijepa_vith14_22k": "ba3c4513ca2b0f0c010f80ae2265b3dbe1083039",
    "google/owlv2-base-patch16-ensemble": "cfd3195ba4ea9592eec887ded089f4c08eff231d",
}


def revision_for(model_id: str) -> str | None:
    """Return the frozen revision for a known model, else None for explicit extensions."""
    return MODEL_REVISIONS.get(model_id)
