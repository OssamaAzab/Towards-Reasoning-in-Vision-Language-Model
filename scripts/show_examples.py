"""Show qualitative examples: a real GQA image with the model's answer, with vs without vision.

Saves a figure (image + question + gold + the bridge's answer WITH the image + the
text-only answer WITHOUT it) so the model's behaviour on a real image is visible.
A utility for inspection/report figures, not a numbered roadmap step.

    python scripts/show_examples.py --n 3
"""
import argparse
import random
import sys
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from PIL import Image

# Make the repo root importable when run as `python scripts/show_examples.py`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.gqa import GQADataset  # noqa: E402
from src.models.bridge import QFormerBridge  # noqa: E402
from src.models.encoders import encode_image, load_vision_encoder  # noqa: E402
from src.models.llm import load_llm  # noqa: E402
from src.models.vlm import text_only_generate, vlm_generate  # noqa: E402
from src import prompt as P  # noqa: E402
from src.utils import ensure_dir, load_config, set_seed  # noqa: E402

SUFFIX = P.SHORT_CUE   # single-sourced; a private copy here would drift from eval


def main() -> None:
    """Generate a few (image, question, with-image answer, no-image answer) panels."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=3, help="number of examples to show")
    ap.add_argument("--split", default="val_balanced")
    args = ap.parse_args()

    cfg = load_config()
    set_seed(cfg.get("seed", 42))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    name = cfg["models"].get("active_encoder", "ijepa")

    enc = load_vision_encoder(name, cfg, device)
    llm = load_llm(cfg)
    bcfg = cfg["bridge"]
    bridge = QFormerBridge(
        encoder_dim=enc.hidden_dim, llm_dim=llm.model.config.hidden_size,
        num_query_tokens=bcfg["num_query_tokens"], hidden_size=bcfg["hidden_size"],
        num_layers=bcfg["num_hidden_layers"], num_heads=bcfg["num_attention_heads"],
    ).to(device)
    ckpt = torch.load(Path(cfg["paths"]["checkpoints"]) / f"bridge_{name}.pt",
                      map_location=device, weights_only=False)
    bridge.load_state_dict(ckpt["state_dict"])
    bridge.eval()

    # Pick a few examples with short, concrete gold answers for a clean demo.
    ds = GQADataset(cfg["gqa"]["questions"][args.split], cfg["gqa"]["image_dirs"])
    chosen = []
    for qid in random.sample(ds.qids, 400):
        ex = ds.get(qid)
        if ex.image_path and ex.image_path.exists() and 1 <= len(ex.answer.split()) <= 2:
            chosen.append(ex)
        if len(chosen) >= args.n:
            break

    rows = []
    for ex in chosen:
        image = Image.open(ex.image_path).convert("RGB")
        feats = encode_image(enc, image)
        with_img = vlm_generate(bridge, llm, feats, ex.question + SUFFIX, device,
                                max_new_tokens=12).strip()
        no_img = text_only_generate(llm, ex.question + SUFFIX, device,
                                    max_new_tokens=12).strip()
        rows.append((image, ex.question, ex.answer, with_img, no_img))
        print(f"\nQ: {ex.question}\n  gold        : {ex.answer}"
              f"\n  WITH image  : {with_img}\n  NO image    : {no_img}")

    fig, axes = plt.subplots(len(rows), 2, figsize=(13, 4.2 * len(rows)),
                             gridspec_kw={"width_ratios": [1, 1.25]})
    axes = axes.reshape(len(rows), 2)
    for r, (image, q, gold, with_img, no_img) in enumerate(rows):
        axes[r, 0].imshow(image)
        axes[r, 0].axis("off")
        text = (f"Q: {textwrap.fill(q, 40)}\n\n"
                f"gold answer:  {gold}\n\n"
                f"WITH image (bridge):\n  {textwrap.fill(with_img, 44)}\n\n"
                f"NO image (text-only):\n  {textwrap.fill(no_img, 44)}")
        axes[r, 1].axis("off")
        axes[r, 1].text(0.0, 1.0, text, va="top", ha="left", fontsize=11, family="monospace")
    fig.suptitle("Same question, answered WITH the image (bridge) vs WITHOUT it (text-only)",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    out = ensure_dir(Path(cfg["paths"]["outputs"]) / "figures") / "qualitative_examples.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
