"""Step 3: prove the frozen vision encoder works — encode ONE real GQA image.

Loads the active encoder (I-JEPA for now), runs a single GQA image through it, and
reports the feature shape, hidden_dim (the bridge's input size), and peak VRAM, then
saves the feature tensor as a sanity artifact. No LLM, no bridge yet — this only
proves one image goes in and patch features come out, comfortably within 20 GB.

    python scripts/03_encode_one_image.py
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless: render figures to file, no display needed
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

# Make the repo root importable when run as `python scripts/03_encode_one_image.py`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.gqa import GQADataset  # noqa: E402
from src.models.encoders import encode_image, load_vision_encoder  # noqa: E402
from src.utils import ensure_dir, load_config, set_seed, vram_report  # noqa: E402


def save_feature_preview(image, features, out_path, encoder_name) -> None:
    """Save a side-by-side: the input image and a heatmap of its patch features.

    Each patch is a high-dimensional vector, which cannot be shown directly. We
    reduce every patch to a single number via its first principal component (PCA
    over the patches), place those numbers back on the patch grid, and draw them as
    a heatmap. This makes the otherwise-invisible feature tensor easy to see: the
    heatmap highlights where the encoder's dominant feature responds across the image.
    """
    feats = features[0].float().cpu().numpy()        # [num_patches, hidden_dim]
    n_patches = feats.shape[0]
    side = int(round(n_patches ** 0.5))              # e.g. 256 patches -> 16 x 16
    centered = feats - feats.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)  # principal axes
    grid = (centered @ vt[0]).reshape(side, side)    # project each patch onto PC1

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(image)
    axes[0].set_title("Input image")
    axes[0].axis("off")
    heat = axes[1].imshow(grid, cmap="viridis")
    axes[1].set_title(f"Patch features (PCA component 1)\n{side}x{side} grid — {encoder_name}")
    axes[1].axis("off")
    fig.colorbar(heat, ax=axes[1], fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Encode one GQA image with the frozen active encoder and report shape/dim/VRAM."""
    cfg = load_config()
    set_seed(cfg.get("seed", 42))
    name = cfg["models"].get("active_encoder", "ijepa")  # RQ2 swap point

    # One real GQA image that exists on disk (first with a resolved path).
    ds = GQADataset(cfg["gqa"]["questions"]["val_balanced"], cfg["gqa"]["image_dirs"])
    ex = next(e for e in ds.examples() if e.image_path is not None)
    image = Image.open(ex.image_path).convert("RGB")
    print(f"Image : {ex.image_path}  ({image.size[0]}x{image.size[1]})")
    print(f"        qid={ex.qid}  Q: {ex.question}  A: {ex.answer}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    print(f"\nLoading frozen encoder '{name}' (downloads on first run) ...")
    enc = load_vision_encoder(name, cfg, device)
    n_params = sum(p.numel() for p in enc.model.parameters())
    n_trainable = sum(p.numel() for p in enc.model.parameters() if p.requires_grad)
    print(f"  model_id  : {enc.model_id}")
    print(f"  device    : {enc.device}")
    print(f"  params    : {n_params/1e6:.0f}M total, {n_trainable} trainable (expect 0 — frozen)")
    assert n_trainable == 0, "encoder is not fully frozen!"

    feats = encode_image(enc, image)

    print("\n--- RESULT ---")
    print(f"Feature shape : {tuple(feats.shape)}   # [batch, num_patches, hidden_dim]")
    print(f"hidden_dim    : {enc.hidden_dim}   # <-- bridge input size (record this)")
    print(f"dtype         : {feats.dtype}")
    if device == "cuda":
        peak = torch.cuda.max_memory_allocated() / 1024**3
        print(f"VRAM peak     : {vram_report()}")

    out = ensure_dir(cfg["paths"]["features"]) / f"sanity_{name}_{ex.image_id}.pt"
    torch.save({"image_id": ex.image_id, "encoder": enc.model_id,
                "hidden_dim": enc.hidden_dim, "features": feats.cpu()}, out)
    print(f"\nSaved sanity artifact -> {out}")

    # Visual aid: the input image beside a heatmap of its patch features.
    fig_dir = ensure_dir(Path(cfg["paths"]["outputs"]) / "figures")
    preview = fig_dir / f"encoding_preview_{name}_{ex.image_id}.png"
    save_feature_preview(image, feats, preview, enc.model_id)
    print(f"Saved encoding preview -> {preview}")


if __name__ == "__main__":
    main()
