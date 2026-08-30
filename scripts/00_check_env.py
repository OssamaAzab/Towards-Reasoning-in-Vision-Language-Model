"""Step 0: repeatable end-to-end environment smoke test.

Checks Python, cache locations, PyTorch + CUDA + GPU/VRAM, and that transformers
and bitsandbytes import. Exits non-zero if any required check fails, so it is safe
to run in CI or before every work session.

Run after `source env.sh` and `pip install -r requirements.txt`:
    python scripts/00_check_env.py
"""
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _report(name: str, ok: bool, detail: str) -> bool:
    """Print one PASS/FAIL line and return ok unchanged."""
    print(f"  [{'PASS' if ok else 'FAIL'}] {name:24}{detail}")
    return ok


def main() -> int:
    """Run all checks; return 0 if every required check passes, else 1."""
    results: list[bool] = []

    print(f"Python     : {sys.version.split()[0]}")
    print(f"venv prefix: {sys.prefix}\n")

    cache_root = str(PROJECT_ROOT / ".cache")
    print(f"Cache locations (should all be under {cache_root}):")
    for var in ("HF_HOME", "HUGGINGFACE_HUB_CACHE", "HF_DATASETS_CACHE",
                "TORCH_HOME", "PIP_CACHE_DIR", "XDG_CACHE_HOME"):
        val = os.environ.get(var, "<unset>")
        results.append(_report(var, val.startswith(cache_root), val))

    print("\nDeep-learning stack:")
    try:
        import torch
        results.append(_report("torch import", True, torch.__version__))
    except Exception as e:
        results.append(_report("torch import", False, repr(e)))
        return _summary(results)

    cuda_ok = torch.cuda.is_available()
    results.append(_report("cuda available", cuda_ok, str(cuda_ok)))

    if cuda_ok:
        name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        results.append(_report("gpu visible", True, name))
        results.append(_report("vram", vram_gb > 0, f"{vram_gb:.1f} GB"))
    else:
        results.append(_report("gpu visible", False, "CUDA not available"))

    try:
        import transformers
        results.append(_report("transformers import", True, transformers.__version__))
    except Exception as e:
        results.append(_report("transformers import", False, repr(e)))

    try:
        import bitsandbytes as bnb
        results.append(_report("bitsandbytes import", True,
                               getattr(bnb, "__version__", "unknown")))
    except Exception as e:
        results.append(_report("bitsandbytes import", False, repr(e)))

    return _summary(results)


def _summary(results: list[bool]) -> int:
    """Print an overall PASS/FAIL line and return the process exit code."""
    ok = all(results)
    print(f"\n{'ALL CHECKS PASSED' if ok else 'SOME CHECKS FAILED'} "
          f"({sum(results)}/{len(results)})")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
