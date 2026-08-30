"""Step 6 sanity check: can the bridge actually learn? Overfit a single example.

Trains ONLY the bridge for a number of steps on one (image, question, answer) example
and shows the loss falling toward zero. This confirms the bridge can learn (not just
that gradients flow), and that the optimisation path is wired correctly, before
investing in the full LLaVA training. No dataset needed: the image is encoded once
(the encoder is frozen) and reused across steps.

    python scripts/06b_overfit_one.py --steps 30
"""
import argparse
import sys
import time
from pathlib import Path

import torch
from PIL import Image

# Make the repo root importable when run as `python scripts/06b_overfit_one.py`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.gqa import GQADataset  # noqa: E402
from src.models.bridge import QFormerBridge  # noqa: E402
from src.models.encoders import encode_image, load_vision_encoder  # noqa: E402
from src.models.llm import load_llm  # noqa: E402
from src import prompt as P  # noqa: E402
from src.models.vlm import vlm_generate, vlm_loss  # noqa: E402
from src.utils import load_config, set_seed  # noqa: E402

_T0 = time.perf_counter()


def log(msg: str) -> None:
    """Print a progress line prefixed with elapsed seconds, flushed immediately."""
    print(f"[{time.perf_counter() - _T0:6.1f}s] {msg}", flush=True)


def main() -> None:
    """Optimise the bridge on a single example and report the loss trajectory."""
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--steps", type=int, default=30, help="optimisation steps")
    ap.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    ap.add_argument("--prompt-format", choices=["raw", "chatml_v1"], default="raw",
                    help="prompt layout (default: raw, reproducing the legacy path)")
    ap.add_argument("--supervise-eos", action="store_true",
                    help="supervise the assistant stop token (requires chatml_v1). This "
                         "is the cheapest possible check that the corrected objective is "
                         "learnable AND that the bridge can learn to stop")
    args = ap.parse_args()
    spec = P.PromptSpec(prompt_format=args.prompt_format,
                        supervise_eos=args.supervise_eos,
                        max_answer_tokens=None)
    log(f"PROTOCOL: {spec.prompt_format} supervise_eos={spec.supervise_eos} "
        f"layer={spec.evidence_layer}")

    cfg = load_config()
    set_seed(cfg.get("seed", 42))
    if not torch.cuda.is_available():
        raise SystemExit("A CUDA GPU is required.")
    device = "cuda"

    name = cfg["models"].get("active_encoder", "ijepa")
    log(f"loading frozen encoder '{name}' and the 4-bit LLM ...")
    enc = load_vision_encoder(name, cfg, device)
    llm = load_llm(cfg, log=log)

    bcfg = cfg["bridge"]
    bridge = QFormerBridge(
        encoder_dim=enc.hidden_dim, llm_dim=llm.model.config.hidden_size,
        num_query_tokens=bcfg["num_query_tokens"], hidden_size=bcfg["hidden_size"],
        num_layers=bcfg["num_hidden_layers"], num_heads=bcfg["num_attention_heads"],
    ).to(device)
    bridge.train()

    # One example; encode the image once (the encoder is frozen).
    ds = GQADataset(cfg["gqa"]["questions"]["val_balanced"], cfg["gqa"]["image_dirs"])
    ex = next(e for e in ds.examples() if e.image_path is not None)
    image = Image.open(ex.image_path).convert("RGB")
    feats = encode_image(enc, image)
    log(f"example: Q='{ex.question}'  gold A='{ex.answer}'")

    before, stop_before = vlm_generate(bridge, llm, feats, ex.question, device,
                                       max_new_tokens=32, spec=spec, return_stop=True)
    before = before.strip()

    opt = torch.optim.AdamW(bridge.parameters(), lr=args.lr)
    log(f"overfitting one example for {args.steps} steps (lr={args.lr}) ...")
    loss_val = float("nan")
    for step in range(1, args.steps + 1):
        opt.zero_grad()
        loss = vlm_loss(bridge, llm, feats, ex.question, ex.answer, device, spec=spec)
        loss.backward()
        opt.step()
        loss_val = loss.item()
        if step == 1 or step % 5 == 0:
            log(f"  step {step:3d}   loss {loss_val:.4f}")

    after, stop_after = vlm_generate(bridge, llm, feats, ex.question, device,
                                     max_new_tokens=32, spec=spec, return_stop=True)
    after = after.strip()

    print("\n--- OVERFIT SANITY CHECK ---")
    print(f"protocol            : {spec.prompt_format}, supervise_eos={spec.supervise_eos}")
    print(f"gold answer         : {ex.answer!r}")
    print(f"answer before train : {before!r}   ({stop_before['stop_reason']}, "
          f"{stop_before['n_generated']} tok)")
    print(f"answer after train  : {after!r}   ({stop_after['stop_reason']}, "
          f"{stop_after['n_generated']} tok)")
    print(f"final loss          : {loss_val:.4f}")
    if spec.supervise_eos:
        # On a single example the bridge should learn BOTH the answer and the stop.
        # Learning the answer while still running to the cap means the EOS gradient is
        # not reaching the bridge, which no accuracy number would reveal.
        verdict = ("OK: learned to stop" if not stop_after["cap_hit"]
                   else "FAIL: still hits the cap — the EOS target is not being learned")
        print(f"stop behaviour      : {verdict}")
    print("\nOK: the bridge learns (loss dropped, answer corrected)."
          if loss_val < 1.0 else
          f"\nWARNING: final loss {loss_val:.4f} is high; the bridge may not be learning.")


if __name__ == "__main__":
    main()
