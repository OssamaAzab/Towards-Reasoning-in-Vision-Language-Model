"""Step 4: load the frozen 4-bit language model and generate text (no image yet).

Proves the language-model half of the pipeline on its own: load Qwen2-7B-Instruct
in 4-bit, then answer a few text-only prompts. Progress is printed with elapsed
time at each stage, and peak VRAM is reported, so you can watch loading and
generation happen and confirm the model fits within 20 GB. The bridge that turns
an image into tokens for this LLM comes in a later step.

    python scripts/04_generate_text.py
"""
import sys
import time
from pathlib import Path

import torch

# Make the repo root importable when run as `python scripts/04_generate_text.py`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.llm import generate, load_llm  # noqa: E402
from src.utils import load_config, set_seed, vram_report  # noqa: E402

_T0 = time.perf_counter()


def log(msg: str) -> None:
    """Print a progress line prefixed with elapsed seconds, flushed immediately."""
    print(f"[{time.perf_counter() - _T0:6.1f}s] {msg}", flush=True)


def main() -> None:
    """Load the 4-bit LLM, run a few prompts, and report latency and VRAM."""
    cfg = load_config()
    set_seed(cfg.get("seed", 42))

    if not torch.cuda.is_available():
        raise SystemExit("A CUDA GPU is required for 4-bit loading.")
    torch.cuda.reset_peak_memory_stats()

    log("step 4: load frozen 4-bit LLM, then generate text")
    llm = load_llm(cfg, log=log)
    vram = torch.cuda.memory_allocated() / 1024**3
    log(f"loaded {llm.model_id} in 4-bit: {vram:.2f} GB VRAM, "
        f"max_new_tokens={llm.max_new_tokens}")

    prompts = [
        "In one sentence, what is compositional visual reasoning?",
        "Answer with a single word. What colour is a typical banana?",
        "Is the cat to the left of the dog if the dog is to the right of the cat? "
        "Answer yes or no and explain briefly.",
        "What is the capital of Egypt and tell me about it?",
    ]

    for i, prompt in enumerate(prompts, 1):
        log(f"generating ({i}/{len(prompts)}) ...")
        t = time.perf_counter()
        answer = generate(llm, prompt)
        dt = time.perf_counter() - t
        n_new = len(llm.tokenizer(answer)["input_ids"])
        print(f"\n  Q: {prompt}\n  A: {answer.strip()}"
              f"\n  ({dt:.1f}s, ~{n_new / dt:.1f} tok/s)\n", flush=True)

    peak = torch.cuda.max_memory_allocated() / 1024**3
    log(f"done. peak VRAM: {vram_report()}")


if __name__ == "__main__":
    main()
