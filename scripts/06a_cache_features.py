"""Step 6 (caching): precompute frozen-encoder features for LLaVA images to disk.

The encoder is frozen, so its features never change — cache them once and let training
read them instead of re-encoding every epoch. Stores fp16 features per unique image
under outputs/features/llava_<encoder>/. Safe to re-run: already-cached images are
skipped. Training (06c) also fills the cache on demand, so this script is an optional
but faster way to pre-build it.

    python scripts/06a_cache_features.py --limit 5000
"""
import argparse
import hashlib
import sys
import time
from pathlib import Path

import torch
from PIL import Image

# Make the repo root importable when run as `python scripts/06a_cache_features.py`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.feature_cache import cache_dir_for, has_features, save_features  # noqa: E402
from src.data.llava import LLaVADataset  # noqa: E402
from src.data.vqa import VQADataset  # noqa: E402
from src.models.encoders import encode_image, load_vision_encoder  # noqa: E402
from src.utils import load_config, set_seed  # noqa: E402

_T0 = time.perf_counter()


def log(msg: str) -> None:
    """Print a progress line prefixed with elapsed seconds, flushed immediately."""
    print(f"[{time.perf_counter() - _T0:7.1f}s] {msg}", flush=True)


def select_image_paths(llava, vqa, source: str, limit: int, short_frac: float) -> list[Path]:
    """Return file-order image paths for LLaVA-only or the locked mixed-data draw."""
    if limit < 0:
        raise SystemExit("--limit must be non-negative")
    if not 0.0 <= short_frac <= 1.0:
        raise SystemExit("--short-frac must be in [0, 1]")
    if source == "llava":
        return [llava.get(index).image_path for index in range(min(limit, len(llava)))]
    n_short = int(limit * short_frac)
    n_llava = limit - n_short
    if n_llava > len(llava) or n_short > len(vqa):
        raise SystemExit(
            "requested mixed draw exceeds usable data: "
            f"LLaVA {n_llava}/{len(llava)}, VQAv2 {n_short}/{len(vqa)}"
        )
    paths = [llava.get(index).image_path for index in range(n_llava)]
    paths.extend(vqa.get(index).image_path for index in range(n_short))
    return paths


def unique_image_paths(paths: list[Path]) -> list[Path]:
    """Deduplicate paths by the filename-stem key used by the feature cache."""
    unique = {}
    for path in paths:
        unique.setdefault(path.stem, path)
    return list(unique.values())


def validate_manifest(paths: list[Path], expected_count: int, expected_sha256: str) -> dict:
    """Fail closed unless cache keys match the preregistered count and digest."""
    keys = sorted(path.stem for path in unique_image_paths(paths))
    digest = hashlib.sha256(("\n".join(keys) + "\n").encode()).hexdigest()
    if len(keys) != expected_count or digest != expected_sha256:
        raise SystemExit(
            "manifest mismatch: "
            f"expected count={expected_count} sha256={expected_sha256}, "
            f"got count={len(keys)} sha256={digest}"
        )
    return {"count": len(keys), "sha256": digest}


def main() -> None:
    """Encode and cache features for the unique images in the first --limit examples."""
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=5000, help="cover the first N LLaVA examples")
    ap.add_argument("--encoder", help="encoder key under models.encoders (default: active_encoder)")
    ap.add_argument("--source", choices=("llava", "mixed"), default="llava",
                    help="LLaVA-only or the locked VQAv2/LLaVA training draw")
    ap.add_argument("--short-frac", type=float,
                    help="VQAv2 fraction for --source mixed (default: config train.short_frac)")
    ap.add_argument("--expected-keys", type=int,
                    help="fail unless the selected unique-key count matches")
    ap.add_argument("--expected-keyset-sha256",
                    help="fail unless the sorted selected-key digest matches")
    ap.add_argument("--expected-tokens", type=int,
                    help="fail unless encoded features have this token count")
    ap.add_argument("--expected-dim", type=int,
                    help="fail unless encoded features have this feature width")
    args = ap.parse_args()

    cfg = load_config()
    set_seed(cfg.get("seed", 42))
    if not torch.cuda.is_available():
        raise SystemExit("A CUDA GPU is required.")
    device = "cuda"

    name = args.encoder or cfg["models"].get("active_encoder", "ijepa")
    if name not in cfg["models"]["encoders"]:
        raise SystemExit(f"unknown encoder '{name}'")
    short_frac = args.short_frac
    if short_frac is None:
        short_frac = cfg["train"]["short_frac"]
    if (args.expected_keys is None) != (args.expected_keyset_sha256 is None):
        raise SystemExit("--expected-keys and --expected-keyset-sha256 must be supplied together")

    llava = LLaVADataset(
        cfg["llava"]["annotations"],
        cfg["llava"]["image_dir"],
        cfg["llava"].get("exclude_ids"),
    )
    vqa = None
    if args.source == "mixed":
        vqa = VQADataset(
            cfg["vqa"]["questions"],
            cfg["vqa"]["annotations"],
            cfg["vqa"]["image_dir"],
            cfg["vqa"].get("exclude_ids"),
        )
    selected = select_image_paths(llava, vqa, args.source, args.limit, short_frac)
    paths = unique_image_paths(selected)
    if args.expected_keys is not None:
        manifest = validate_manifest(paths, args.expected_keys, args.expected_keyset_sha256)
        log(f"manifest PASS: {manifest['count']} keys sha256={manifest['sha256']}")
    missing_images = [path for path in paths if not path.is_file()]
    if missing_images:
        raise SystemExit(
            f"selected manifest has {len(missing_images)} missing images; first={missing_images[:5]}"
        )

    log(f"loading frozen encoder '{name}' ...")
    enc = load_vision_encoder(name, cfg, device)
    cache_dir = cache_dir_for(cfg["paths"]["features"], name)

    todo = [path for path in paths if not has_features(cache_dir, path)]
    log(f"{len(paths)} unique images in selected manifest; {len(todo)} to encode")

    t0 = time.perf_counter()
    for i, p in enumerate(todo, 1):
        feats = encode_image(enc, Image.open(p).convert("RGB"))
        if args.expected_tokens is not None and feats.shape[1] != args.expected_tokens:
            raise SystemExit(
                f"feature token mismatch for {p}: expected {args.expected_tokens}, got {feats.shape[1]}"
            )
        if args.expected_dim is not None and feats.shape[2] != args.expected_dim:
            raise SystemExit(
                f"feature width mismatch for {p}: expected {args.expected_dim}, got {feats.shape[2]}"
            )
        save_features(cache_dir, p, feats)
        if i % 500 == 0 or i == len(todo):
            log(f"  cached {i}/{len(todo)}")
    dt = time.perf_counter() - t0

    missing_cache = [path.stem for path in paths if not has_features(cache_dir, path)]
    if missing_cache:
        raise SystemExit(
            f"cache incomplete after encoding: {len(missing_cache)} missing; first={missing_cache[:5]}"
        )
    zero_cache = [
        path.stem for path in paths
        if (cache_dir / f"{path.stem}.npy").stat().st_size == 0
    ]
    if zero_cache:
        raise SystemExit(
            f"cache has {len(zero_cache)} zero-byte files; first={zero_cache[:5]}"
        )
    files = list(cache_dir.glob("*.npy"))
    size_gb = sum(f.stat().st_size for f in files) / 1e9
    per = (dt / len(todo) * 1000) if todo else 0.0
    log(f"done: encoded {len(todo)} in {dt:.1f}s ({per:.1f} ms/image)")
    log(f"cache PASS {cache_dir}: required={len(paths)} files={len(files)} size={size_gb:.2f} GB")


if __name__ == "__main__":
    main()
