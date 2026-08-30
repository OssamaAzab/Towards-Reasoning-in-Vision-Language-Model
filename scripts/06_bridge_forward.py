"""Step 6 (skeleton): prove the full image->answer path produces a trainable loss.

Wires the frozen encoder -> trainable bridge -> frozen 4-bit LLM on ONE example:
encode an image, summarise it into visual tokens via the bridge, prepend them to a
question, and compute the language-modelling loss on a target answer. Verifies the
loss is finite and that gradients reach the bridge ONLY (encoder and LLM stay
frozen). No dataset yet — this proves the architecture before training on LLaVA.

    python scripts/06_bridge_forward.py
"""
import argparse
import sys
import time
from pathlib import Path

import torch
from PIL import Image

# Make the repo root importable when run as `python scripts/06_bridge_forward.py`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import prompt as P  # noqa: E402
from src.data.gqa import GQADataset  # noqa: E402
from src.models.bridge import QFormerBridge  # noqa: E402
from src.models.encoders import encode_image, load_vision_encoder  # noqa: E402
from src.models.llm import load_llm  # noqa: E402
from src.models.vlm import assemble  # noqa: E402
from src.utils import load_config, set_seed, vram_report  # noqa: E402

_T0 = time.perf_counter()


def log(msg: str) -> None:
    """Print a progress line prefixed with elapsed seconds, flushed immediately."""
    print(f"[{time.perf_counter() - _T0:6.1f}s] {msg}", flush=True)


def main() -> None:
    """Run one image+question+answer through encoder->bridge->LLM and check gradients."""
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prompt-format", choices=["raw", "chatml_v1"], default="raw",
                    help="prompt layout (default: raw, reproducing the legacy path)")
    ap.add_argument("--supervise-eos", action="store_true",
                    help="supervise the assistant stop token (requires chatml_v1)")
    args = ap.parse_args()

    cfg = load_config()
    set_seed(cfg.get("seed", 42))
    if not torch.cuda.is_available():
        raise SystemExit("A CUDA GPU is required.")
    device = "cuda"
    torch.cuda.reset_peak_memory_stats()

    name = cfg["models"].get("active_encoder", "ijepa")
    log(f"loading frozen encoder '{name}' and the 4-bit LLM ...")
    enc = load_vision_encoder(name, cfg, device)
    llm = load_llm(cfg, log=log)
    llm_dim = llm.model.config.hidden_size
    log(f"encoder hidden_dim={enc.hidden_dim}  ->  llm hidden_dim={llm_dim}")

    # Build the bridge (the only trainable module) from config.
    bcfg = cfg["bridge"]
    bridge = QFormerBridge(
        encoder_dim=enc.hidden_dim, llm_dim=llm_dim,
        num_query_tokens=bcfg["num_query_tokens"], hidden_size=bcfg["hidden_size"],
        num_layers=bcfg["num_hidden_layers"], num_heads=bcfg["num_attention_heads"],
    ).to(device)
    n_bridge = sum(p.numel() for p in bridge.parameters() if p.requires_grad)
    log(f"bridge: {n_bridge / 1e6:.1f}M trainable params, "
        f"{bcfg['num_query_tokens']} query tokens")

    # One (image, question, answer) example.
    ds = GQADataset(cfg["gqa"]["questions"]["val_balanced"], cfg["gqa"]["image_dirs"])
    ex = next(e for e in ds.examples() if e.image_path is not None)
    image = Image.open(ex.image_path).convert("RGB")
    log(f"example: Q='{ex.question}'  A='{ex.answer}'")

    # 1) Frozen encoder -> patch features (no gradient).
    feats = encode_image(enc, image)                       # [1, P, encoder_dim]
    # 2) Trainable bridge -> visual tokens (requires gradient).
    visual = bridge(feats.float())                         # [1, Q, llm_dim]

    # 3) Lay the sequence out with src.prompt — the SAME builder training uses.
    #    This script previously assembled the sequence inline, which made it a fifth
    #    independent copy of the layout logic AND omitted the short-answer cue, so it
    #    was demonstrating a sequence training never actually sees. A plumbing check
    #    that checks different plumbing is worse than no check.
    spec = P.PromptSpec(prompt_format=args.prompt_format,
                        supervise_eos=args.supervise_eos, max_answer_tokens=None)
    log(f"PROTOCOL: {spec.prompt_format} supervise_eos={spec.supervise_eos} "
        f"layer={spec.evidence_layer}")
    built = P.build(spec, ex.question, llm.tokenizer, answer=ex.answer,
                    n_visual=visual.size(1))
    inputs_embeds, labels = assemble(built, visual, llm.model.get_input_embeddings(), device)

    # 4) Labels: everything before the answer is masked with -100, so the loss is
    #    computed on the answer span only. Under the corrected protocol the supervised
    #    span also carries the stop token, which is what teaches the bridge to finish.
    n_vis = built.n_visual
    n_q = len(built.prefix_ids) + len(built.body_ids) + len(built.mid_ids)
    n_a = len(built.target_ids)
    assert (labels != -100).sum().item() == n_a, "supervised span is not the answer span"

    # 5) Forward through the frozen LLM -> language-modelling loss on the answer.
    out = llm.model(inputs_embeds=inputs_embeds, labels=labels)
    loss = out.loss
    log(f"forward done. loss = {loss.item():.4f}  (finite: {bool(torch.isfinite(loss))})")

    # 6) Backward: gradients must reach the bridge only.
    loss.backward()
    bridge_grad = sum(p.grad.abs().sum().item()
                      for p in bridge.parameters() if p.grad is not None)
    enc_grad = any(p.grad is not None for p in enc.model.parameters())
    llm_grad = any(p.grad is not None for p in llm.model.parameters())

    print("\n--- PLUMBING CHECK ---")
    print(f"loss                  : {loss.item():.4f}")
    print(f"bridge gradient sum   : {bridge_grad:.3e}   (must be > 0  -> it learns)")
    print(f"encoder has gradients : {enc_grad}            (must be False -> frozen)")
    print(f"LLM has gradients     : {llm_grad}            (must be False -> frozen)")
    print(f"sequence length       : {inputs_embeds.size(1)} tokens "
          f"({n_vis} visual + {n_q} question + {n_a} answer)")
    print(f"peak VRAM             : {vram_report()}")
    ok = bridge_grad > 0 and not enc_grad and not llm_grad and bool(torch.isfinite(loss))
    print(f"\n{'OK' if ok else 'FAILED'}: image -> encoder -> bridge -> LLM -> loss; "
          "gradients flow to the bridge only.")


if __name__ == "__main__":
    main()
