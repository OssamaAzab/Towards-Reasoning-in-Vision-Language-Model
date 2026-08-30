"""Disk cache for frozen-encoder features.

The encoder is frozen, so its features for a given image never change. Caching them
to disk lets bridge training read features instead of re-running the encoder every
epoch — essential for training on thousands of images, and far cheaper than holding
everything in RAM. Features are stored as fp16 NumPy arrays, one file per image
(keyed by the image filename stem), so repeated images are encoded only once.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


def cache_dir_for(features_root, encoder_name: str) -> Path:
    """Return (creating if needed) the feature-cache directory for an encoder."""
    d = Path(features_root) / f"llava_{encoder_name}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _file(cache_dir, image_path) -> Path:
    """Cache file path for an image, keyed by its filename stem."""
    return Path(cache_dir) / f"{Path(image_path).stem}.npy"


def has_features(cache_dir, image_path) -> bool:
    """True if features for this image are already cached on disk."""
    return _file(cache_dir, image_path).exists()


def save_features(cache_dir, image_path, features) -> None:
    """Save features ([1, P, D] or [P, D]) as an fp16 .npy keyed by image stem."""
    arr = features.detach().squeeze(0).to(torch.float16).cpu().numpy()
    np.save(_file(cache_dir, image_path), arr)


def load_features(cache_dir, image_path, device="cpu") -> torch.Tensor:
    """Load cached features as a [1, P, D] float tensor on `device`."""
    arr = np.load(_file(cache_dir, image_path))
    return torch.from_numpy(arr).float().unsqueeze(0).to(device)
