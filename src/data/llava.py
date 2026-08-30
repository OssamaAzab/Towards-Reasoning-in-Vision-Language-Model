"""LLaVA-Instruct-150K loader: (image, instruction, answer) triples for bridge training.

Each entry is a multi-turn conversation grounded in a COCO image. For training the
bridge we use the first turn: the human instruction (with the "<image>" placeholder
removed) and the assistant's reply. Images are COCO train2017, resolved from the
configured image directory.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils import resolve_dir  # noqa: E402


@dataclass
class LLaVAExample:
    """One training example: a COCO image path, an instruction, and a target answer."""
    image_path: Path
    instruction: str
    answer: str


class LLaVADataset:
    """Loads LLaVA-Instruct-150K and yields first-turn (image, instruction, answer) triples."""

    def __init__(self, annotations_path: str, image_dir: str | list[str],
                 exclude_ids_path: str | None = None):
        """Read the annotations JSON; drop conversations whose COCO image is in the exclude set.

        LLaVA images are COCO (filename stem == COCO id). To prevent train/eval leakage
        (report Section 9.4), any image that is also a GQA-eval image is excluded.

        `image_dir` accepts a list of candidate directories (tried in order) so the same
        config works on this workstation and on AI@Surrey, which mount COCO at different
        paths. A plain string behaves exactly as before.
        """
        self.image_dir = resolve_dir(image_dir)
        exclude = set()
        if exclude_ids_path and Path(exclude_ids_path).exists():
            exclude = set(json.load(open(exclude_ids_path)))
        self.n_excluded = 0
        with open(annotations_path) as f:
            raw = json.load(f)
        if not exclude:
            self._raw: list = raw
            return
        kept = []
        for item in raw:
            try:
                coco_id = int(Path(item["image"]).stem)
            except (ValueError, KeyError):
                coco_id = None
            if coco_id is not None and coco_id in exclude:
                self.n_excluded += 1
                continue
            kept.append(item)
        self._raw = kept

    def __len__(self) -> int:
        """Number of conversations in the dataset."""
        return len(self._raw)

    def get(self, i: int) -> LLaVAExample:
        """Build a first-turn example from conversation i."""
        item = self._raw[i]
        convo = item["conversations"]
        instruction = convo[0]["value"].replace("<image>", "").strip()
        answer = convo[1]["value"].strip()
        return LLaVAExample(
            image_path=self.image_dir / item["image"],
            instruction=instruction,
            answer=answer,
        )

    def examples(self, n: int | None = None) -> Iterator[LLaVAExample]:
        """Yield up to n examples (all if n is None), in file order."""
        count = len(self._raw) if n is None else min(n, len(self._raw))
        for i in range(count):
            yield self.get(i)
