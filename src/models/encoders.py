"""Frozen vision encoders: load a HuggingFace image encoder, frozen, on the GPU.

In this project the vision encoder is NEVER trained — only the bridge is. So every
encoder is loaded frozen (`requires_grad=False`, `eval()`), run under `no_grad` to
produce patch features, and those features get cached to disk. This module is the
single place that knows how to load an encoder, so swapping encoders for RQ2
(CLIP / I-JEPA / DINOv2 / I-JEPA IN22K) is a one-line config change, not new code.

The completed comparison uses CLIP, DINOv2 and I-JEPA (IN1K, and IN22K as a checkpoint
intervention). V-JEPA 2 was scoped in an earlier plan and never used; no result in the
project rests on it.

I-JEPA and DINOv2 load through the generic AutoModel path; CLIP needs its vision tower only
(CLIPVisionModel) because AutoModel would load the full text+vision CLIPModel, whose
config has no single `hidden_size` and whose forward expects text input.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers import AutoImageProcessor, AutoModel

from src.models.revisions import revision_for


@dataclass
class VisionEncoder:
    """A frozen vision encoder bundled with its processor and key dimensions."""
    name: str                  # config key: clip | ijepa | dinov2 | ijepa_in22k
    model_id: str              # HuggingFace model id
    model: torch.nn.Module     # frozen, in eval mode, on `device`
    processor: object          # matching image processor
    device: str
    hidden_dim: int            # feature width per patch — the bridge's input size
    revision: str | None       # immutable upstream revision when known


def load_vision_encoder(name: str, cfg: dict, device: str | None = None) -> VisionEncoder:
    """Load encoder `name` (a key under config models.encoders); freeze, eval, move to GPU."""
    model_id = cfg["models"]["encoders"][name]
    revision = revision_for(model_id)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    processor = AutoImageProcessor.from_pretrained(model_id, revision=revision)
    if name == "clip":
        # Vision tower only: hidden_size lives on CLIPVisionConfig, and its forward takes
        # pixel_values alone (the full CLIPModel would also demand text input_ids).
        from transformers import CLIPVisionModel
        model = CLIPVisionModel.from_pretrained(model_id, revision=revision)
    else:
        model = AutoModel.from_pretrained(model_id, revision=revision)

    # Freeze every parameter: this encoder is inference-only, never optimized.
    model.requires_grad_(False)
    model.eval()
    model.to(device)

    hidden_dim = model.config.hidden_size
    return VisionEncoder(name=name, model_id=model_id, model=model,
                         processor=processor, device=device, hidden_dim=hidden_dim,
                         revision=revision)


@torch.no_grad()
def encode_image(encoder: VisionEncoder, image) -> torch.Tensor:
    """Encode one PIL image into patch features of shape [1, num_patches, hidden_dim]."""
    inputs = encoder.processor(images=image, return_tensors="pt").to(encoder.device)
    outputs = encoder.model(**inputs)
    return outputs.last_hidden_state
