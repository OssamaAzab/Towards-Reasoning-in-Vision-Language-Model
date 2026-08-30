"""VQAv2 loader: short-answer (image, question, answer) triples over COCO train2014.

Mixed into bridge training to teach terse answers (LLaVA alone makes the bridge
verbose). Images are COCO train2014. To prevent train/eval leakage (report Section 9.4),
any COCO image that is also a GQA evaluation image is excluded; the exclusion set is
precomputed (GQA image ids are Visual Genome ids, mapped to coco_id via VG metadata).
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
class VQAExample:
    """One short-answer VQA example: a COCO image path, a question, and a short answer."""
    image_path: Path
    question: str
    answer: str


class VQADataset:
    """Loads VQAv2 train2014 and yields short-answer triples, excluding GQA-eval images."""

    def __init__(self, questions_path: str, annotations_path: str,
                 image_dir: str | list[str], exclude_ids_path: str | None = None):
        """Read questions + answers; drop any example whose COCO image is in the exclude set.

        `image_dir` accepts a list of candidate directories (tried in order) so one config
        serves both this workstation and AI@Surrey. A plain string behaves as before.
        """
        self.image_dir = resolve_dir(image_dir)
        exclude = set()
        if exclude_ids_path and Path(exclude_ids_path).exists():
            exclude = set(json.load(open(exclude_ids_path)))
        self.n_excluded = 0

        questions = json.load(open(questions_path))["questions"]
        answers = {a["question_id"]: a["multiple_choice_answer"]
                   for a in json.load(open(annotations_path))["annotations"]}
        self._items = []
        for item in questions:
            if item["image_id"] in exclude:
                self.n_excluded += 1
                continue
            self._items.append((item["image_id"], item["question"],
                                answers[item["question_id"]]))

    def __len__(self) -> int:
        """Number of (non-excluded) VQAv2 examples."""
        return len(self._items)

    def get(self, i: int) -> VQAExample:
        """Build a VQAExample for item i."""
        image_id, question, answer = self._items[i]
        fname = f"COCO_train2014_{image_id:012d}.jpg"
        return VQAExample(self.image_dir / fname, question, answer)

    def examples(self, n: int | None = None) -> Iterator[VQAExample]:
        """Yield up to n examples (all if n is None), in file order."""
        count = len(self._items) if n is None else min(n, len(self._items))
        for i in range(count):
            yield self.get(i)
